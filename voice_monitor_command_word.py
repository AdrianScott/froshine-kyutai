import argparse
import asyncio
import inspect
import logging
import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from difflib import SequenceMatcher

import msgpack
import numpy as np
import pyaudio
import websockets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

file_handler = logging.FileHandler("voice_commands.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Audio capture configuration
SAMPLE_RATE = 24000
FRAME_DURATION_MS = 80  # Kyutai reference blocksize (1920 samples @ 24kHz)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Kyutai streaming defaults
DEFAULT_WS_URL = os.getenv(
    "FROSHINE_KYUTAI_WS_URL", "ws://127.0.0.1:8908/api/asr-streaming"
)
DEFAULT_API_KEY = os.getenv("FROSHINE_KYUTAI_API_KEY", "public_token")
SEMANTIC_VAD_HEAD = int(os.getenv("FROSHINE_KYUTAI_VAD_HEAD", "2"))
SEMANTIC_VAD_THRESHOLD = float(os.getenv("FROSHINE_KYUTAI_VAD_THRESHOLD", "0.5"))
AUTO_SERVER = os.getenv("FROSHINE_AUTO_SERVER", "1").lower() not in ("0", "false", "no")
AUTO_SERVER_STOP = os.getenv("FROSHINE_AUTO_SERVER_STOP", "1").lower() not in (
    "0",
    "false",
    "no",
)
AUTO_SERVER_READY_TIMEOUT = float(os.getenv("FROSHINE_MOSHI_READY_TIMEOUT", "30"))
AUTO_SERVER_READY_INTERVAL = float(os.getenv("FROSHINE_MOSHI_READY_INTERVAL", "0.5"))
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "config1-stt-en-hf.toml"
)
DEFAULT_EN_CONFIG = DEFAULT_CONFIG_PATH
DEFAULT_EN_FR_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "config1-stt-en_fr-hf.toml"
)
DEFAULT_MIN_FREE_VRAM_EN_MB = 12000

# Command configuration
COMMAND_WORD = os.getenv("FROSHINE_COMMAND_WORD", "flow").lower()
WAKE_WORD_RATIO_THRESHOLD = float(
    os.getenv("FROSHINE_COMMAND_WORD_SIMILARITY", "0.7")
)
ENV_REQUIRE_WAKE_WORD = os.getenv("FROSHINE_REQUIRE_WAKE_WORD", "1").lower() not in (
    "0",
    "false",
    "no",
)
USE_WAKE_WORD = ENV_REQUIRE_WAKE_WORD
aliases_env = os.getenv("FROSHINE_COMMAND_WORD_ALIASES", "")
COMMAND_WORD_ALIASES = {
    alias.strip().lower()
    for alias in aliases_env.split(",")
    if alias.strip()
}
# COMMAND_WORD_ALIASES.update({"flo", "glow", "flowe", "fro", "hello", "helo"})
COMMAND_WORD_ALIASES.add(COMMAND_WORD)
COMMAND_SYNONYMS = {
    "pause": ["pause", "paws", "paus", "pawz"],
    "unpause": ["unpause", "onpause", "on pause", "un pause", "continue"],
    "enter": ["enter", "inner"],
    "quit": ["quit", "quick", "stop"],
    "switch to browser": ["switch to browser", "open browser"],
    "save file": ["save file", "save document"],
    "enter command mode": ["mode command", "flow mode command"],
    "exit command mode": ["mode stop", "flow mode stop"],
}
COMMANDS = {
    "enter": ["enter"],
    "switch to browser": ["switch to browser"],
    "save file": ["save file"],
    "pause": ["pause"],
    "unpause": ["unpause", "continue"],
    "quit": ["quit"],
    "enter command mode": ["mode command"],
    "exit command mode": ["mode stop"],
}

running = True
typed_history = ""
is_paused = False
command_mode = False
pending_command_word = None
pending_command_tokens = []
pending_command_words = []
pending_mode_tokens = []
pending_mode_words = []
server_process = None
shutdown_event = None
queue_ref = None
loop_ref = None
output_manager = None

CONTROL_PHRASES = {
    ("pause",): "pause",
    ("paws",): "pause",
    ("paus",): "pause",
    ("pawz",): "pause",
    ("unpause",): "unpause",
    ("on", "pause"): "unpause",
    ("onpause",): "unpause",
    ("un", "pause"): "unpause",
    ("continue",): "unpause",
    ("enter",): "enter",
    ("inner",): "enter",
    ("quit",): "quit",
    ("quick",): "quit",
    ("stop",): "quit",
    ("switch", "to", "browser"): "switch to browser",
    ("open", "browser"): "switch to browser",
    ("save", "file"): "save file",
    ("save", "document"): "save file",
    ("mode", "command"): "enter command mode",
    ("mode", "stop"): "exit command mode",
    ("flow", "mode", "command"): "enter command mode",
    ("flow", "mode", "stop"): "exit command mode",
}

SAVE_AUDIO_ENV = parse_bool_env("FROSHINE_SAVE_AUDIO", False)
SAVE_TRANSCRIPT_ENV = parse_bool_env("FROSHINE_SAVE_TRANSCRIPT", False)
DEFAULT_OUTPUT_DIR = os.getenv("FROSHINE_OUTPUT_DIR", "outputs")
DEFAULT_AUDIO_ROTATE_HOURS = float(os.getenv("FROSHINE_AUDIO_ROTATE_HOURS", "1"))
DEFAULT_AUDIO_FORMAT = os.getenv("FROSHINE_AUDIO_FORMAT", "opus").strip().lower()
if DEFAULT_AUDIO_FORMAT not in ("wav", "opus"):
    logging.warning(
        "Invalid FROSHINE_AUDIO_FORMAT=%s; falling back to 'opus'.",
        DEFAULT_AUDIO_FORMAT,
    )
    DEFAULT_AUDIO_FORMAT = "opus"
try:
    DEFAULT_AUDIO_BITRATE_KBPS = int(os.getenv("FROSHINE_AUDIO_BITRATE_KBPS", "16"))
except ValueError:
    logging.warning(
        "Invalid FROSHINE_AUDIO_BITRATE_KBPS value; falling back to 16."
    )
    DEFAULT_AUDIO_BITRATE_KBPS = 16


def build_flow_commands():
    flow_commands = {}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        flow_commands[(ch,)] = ("text", ch)
    digit_words = {
        "zero": "0",
        "oh": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    for word, digit in digit_words.items():
        flow_commands[(word,)] = ("text", digit)
    for digit in "0123456789":
        flow_commands[(digit,)] = ("text", digit)
    punctuation = {
        "comma": ",",
        "period": ".",
        "dot": ".",
        "colon": ":",
        "semicolon": ";",
        "dash": "-",
        "quote": "'",
    }
    for word, mark in punctuation.items():
        flow_commands[(word,)] = ("text", mark)
    flow_commands[("double", "quote")] = ("text", '"')
    flow_commands[("question", "mark")] = ("text", "?")
    flow_commands[("question",)] = ("text", "?")
    flow_commands[("exclamation", "mark")] = ("text", "!")
    flow_commands[("exclamation",)] = ("text", "!")
    flow_commands[("newline",)] = ("key", "Return")
    flow_commands[("space",)] = ("text", " ")
    flow_commands[("tab",)] = ("key", "Tab")
    flow_commands[("backspace",)] = ("key", "BackSpace")
    flow_commands[("escape",)] = ("key", "Escape")
    flow_commands[("esc",)] = ("key", "Escape")
    flow_commands[("up",)] = ("key", "Up")
    flow_commands[("down",)] = ("key", "Down")
    flow_commands[("left",)] = ("key", "Left")
    flow_commands[("right",)] = ("key", "Right")
    flow_commands[("click", "left")] = ("click", 1)
    flow_commands[("click", "right")] = ("click", 3)
    return flow_commands


FLOW_COMMANDS = build_flow_commands()
CONTROL_PHRASES_BY_LEN = sorted(CONTROL_PHRASES.keys(), key=len, reverse=True)
FLOW_COMMANDS_BY_LEN = sorted(FLOW_COMMANDS.keys(), key=len, reverse=True)
MACROS = {
    ("quick", "save"): [
        ("click", 3),
        ("sleep", 0.15),
        ("text", "v"),
        ("sleep", 0.1),
        ("key", "Return"),
        ("sleep", 0.1),
    ],
}
MACRO_PHRASES_BY_LEN = sorted(MACROS.keys(), key=len, reverse=True)
PHRASE_PREFIXES = {
    phrase[:idx]
    for phrase in (*CONTROL_PHRASES.keys(), *FLOW_COMMANDS.keys(), *MACROS.keys())
    for idx in range(1, len(phrase))
}


def parse_args():
    parser = argparse.ArgumentParser(description="Froshine Voice Commander (Kyutai)")
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help="Kyutai STT websocket endpoint (default from FROSHINE_KYUTAI_WS_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Kyutai API key header (default from FROSHINE_KYUTAI_API_KEY)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("FROSHINE_AUDIO_DEVICE"),
        help="Preferred audio input device name or index (overrides env)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Execute commands without requiring the wake word",
    )
    parser.add_argument(
        "--word",
        help="Override the command word (default from FROSHINE_COMMAND_WORD)",
    )
    parser.add_argument(
        "--no-auto-server",
        action="store_true",
        help="Do not auto-start moshi-server when the websocket is unreachable",
    )
    parser.add_argument(
        "--save-audio",
        action=argparse.BooleanOptionalAction,
        default=SAVE_AUDIO_ENV,
        help="Save microphone audio to file (default from FROSHINE_SAVE_AUDIO)",
    )
    parser.add_argument(
        "--save-transcript",
        action=argparse.BooleanOptionalAction,
        default=SAVE_TRANSCRIPT_ENV,
        help="Save transcription text to file (default from FROSHINE_SAVE_TRANSCRIPT)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved audio/transcripts (default from FROSHINE_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--audio-rotate-hours",
        type=float,
        default=DEFAULT_AUDIO_ROTATE_HOURS,
        help=(
            "Rotate saved audio file every N hours; 0 disables "
            "(default from FROSHINE_AUDIO_ROTATE_HOURS)"
        ),
    )
    parser.add_argument(
        "--audio-format",
        choices=("wav", "opus"),
        default=DEFAULT_AUDIO_FORMAT,
        help="Audio output format for saved microphone data (default from FROSHINE_AUDIO_FORMAT)",
    )
    parser.add_argument(
        "--audio-bitrate-kbps",
        type=int,
        default=DEFAULT_AUDIO_BITRATE_KBPS,
        help="Target bitrate (kbps) when --audio-format=opus (default from FROSHINE_AUDIO_BITRATE_KBPS)",
    )
    return parser.parse_args()


def log_transcription(transcription, is_command=False, confidence=None):
    if output_manager and not is_command:
        output_manager.write_transcript(transcription)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "command" if is_command else "transcription",
        "content": transcription,
    }
    if confidence is not None:
        log_entry["confidence"] = f"{confidence:.2f}"
    logging.info(log_entry)


def execute_command(command):
    global running, is_paused, command_mode
    logging.info("Command recognized: %s", command)
    if command == "quit":
        print("\nQuitting voice commander...")
        logging.info("User initiated graceful shutdown")
        running = False
        if loop_ref and shutdown_event:
            loop_ref.call_soon_threadsafe(shutdown_event.set)
        if loop_ref and queue_ref:
            def _signal_queue():
                try:
                    queue_ref.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            loop_ref.call_soon_threadsafe(_signal_queue)
        return
    if command == "enter command mode":
        command_mode = True
        logging.info("Command mode enabled")
        print("Command mode enabled.")
        return
    if command == "exit command mode":
        command_mode = False
        logging.info("Command mode disabled")
        print("Command mode disabled.")
        return
    if command == "enter":
        subprocess.run(["xdotool", "key", "Return"])
    elif command == "switch to browser":
        subprocess.run(["xdotool", "search", "--name", "Browser", "windowactivate"])
    elif command == "save file":
        subprocess.run(["xdotool", "key", "ctrl+s"])
    elif command == "pause":
        is_paused = True
        print("Transcription paused.")
    elif command == "unpause":
        is_paused = False
        print("Transcription resumed.")
    else:
        print(f"Unknown command: {command}")


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def match_phrase(tokens, start, phrase_list, phrase_map):
    for phrase in phrase_list:
        length = len(phrase)
        if start + length > len(tokens):
            continue
        if tokens[start : start + length] == list(phrase):
            return phrase_map[phrase], length
    return None, 0


def match_control_command(tokens, start):
    return match_phrase(tokens, start, CONTROL_PHRASES_BY_LEN, CONTROL_PHRASES)


def match_flow_command(tokens, start):
    return match_phrase(tokens, start, FLOW_COMMANDS_BY_LEN, FLOW_COMMANDS)


def match_macro(tokens, start):
    return match_phrase(tokens, start, MACRO_PHRASES_BY_LEN, MACROS)


def type_text_raw(text: str):
    global typed_history
    if not text:
        return
    subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text])
    typed_history += text


def execute_flow_action(action):
    action_type, value = action
    if action_type == "text":
        type_text_raw(value)
    elif action_type == "key":
        subprocess.run(["xdotool", "key", value])
    elif action_type == "click":
        subprocess.run(["xdotool", "click", str(value)])


def execute_macro(steps):
    for action_type, value in steps:
        if action_type == "sleep":
            time.sleep(value)
        else:
            execute_flow_action((action_type, value))


def is_wake_word(token: str) -> bool:
    norm = normalize_word(token)
    if not norm:
        return False
    if norm in COMMAND_WORD_ALIASES:
        return True
    if abs(len(norm) - len(COMMAND_WORD)) <= 1:
        ratio = SequenceMatcher(None, norm, COMMAND_WORD).ratio()
        if ratio >= WAKE_WORD_RATIO_THRESHOLD:
            return True
    return False


def process_transcription_text(text: str, confidence: float):
    global typed_history, is_paused, pending_command_word, command_mode
    global pending_command_tokens, pending_command_words
    global pending_mode_tokens, pending_mode_words

    if not text:
        return

    words = text.split()
    if not words:
        return

    tokens = [normalize_word(word) for word in words]
    def print_wake_pair(wake_word: str, following_word: str | None):
        if following_word:
            print(f"WAKE: {wake_word} {following_word} ({confidence:.2f})")
        else:
            print(f"WAKE: {wake_word} (pending) ({confidence:.2f})")

    def apply_direct_commands():
        nonlocal words, tokens
        if pending_mode_tokens:
            words = pending_mode_words + words
            tokens = pending_mode_tokens + tokens
            pending_mode_tokens.clear()
            pending_mode_words.clear()
        idx = 0
        typed_words = []
        while idx < len(tokens):
            cmd, cmd_len = match_control_command(tokens, idx)
            if cmd:
                snippet = " ".join(words[idx : idx + cmd_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {cmd} ({confidence:.2f})")
                execute_command(cmd)
                idx += cmd_len
                continue
            macro, macro_len = match_macro(tokens, idx)
            if macro:
                snippet = " ".join(words[idx : idx + macro_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {snippet} ({confidence:.2f})")
                execute_macro(macro)
                idx += macro_len
                continue
            action, action_len = match_flow_command(tokens, idx)
            if action:
                snippet = " ".join(words[idx : idx + action_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {snippet} ({confidence:.2f})")
                execute_flow_action(action)
                idx += action_len
                continue
            remaining = tuple(tokens[idx:])
            if remaining in PHRASE_PREFIXES:
                pending_mode_tokens[:] = list(remaining)
                pending_mode_words[:] = words[idx:]
                break
            typed_words.append(words[idx])
            idx += 1
        if not is_paused and typed_words:
            joined = " ".join(typed_words)
            log_transcription(joined, confidence=confidence)
            print(f"{joined} ({confidence:.2f})")
            type_text(joined, add_space=True)

    if command_mode or not USE_WAKE_WORD:
        pending_command_word = None
        pending_command_tokens.clear()
        pending_command_words.clear()
        if tokens and tokens[0] in COMMAND_WORD_ALIASES:
            next_word = words[1] if len(words) > 1 else None
            print_wake_pair(words[0], next_word)
            words = words[1:]
            tokens = tokens[1:]
        if not tokens:
            return
        apply_direct_commands()
        return

    in_command_scope = False
    skip_next_wake_word = False
    typed_words = []

    if pending_command_word:
        if tokens and is_wake_word(tokens[0]):
            if not is_paused:
                log_transcription(COMMAND_WORD, confidence=confidence)
                print(f"{COMMAND_WORD} ({confidence:.2f})")
                type_text(COMMAND_WORD, add_space=True)
            pending_command_word = None
        else:
            in_command_scope = True
            print_wake_pair(pending_command_word, words[0] if words else None)
            pending_command_word = None
    if pending_command_tokens:
        in_command_scope = True
        words = pending_command_words + words
        tokens = pending_command_tokens + tokens
        pending_command_tokens.clear()
        pending_command_words.clear()

    i = 0
    while i < len(words):
        norm = tokens[i]
        if skip_next_wake_word:
            skip_next_wake_word = False
            i += 1
            continue
        if is_wake_word(norm):
            if i < len(words) - 1:
                next_norm = tokens[i + 1]
                print_wake_pair(words[i], words[i + 1])
                if is_wake_word(next_norm):
                    typed_words.append(words[i])
                    skip_next_wake_word = True
                    i += 1
                    continue
                in_command_scope = True
                i += 1
                continue
            pending_command_word = COMMAND_WORD
            print_wake_pair(words[i], None)
            i += 1
            continue
        if in_command_scope:
            cmd, cmd_len = match_control_command(tokens, i)
            if cmd:
                snippet = " ".join(words[i : i + cmd_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {cmd} ({confidence:.2f})")
                execute_command(cmd)
                i += cmd_len
                in_command_scope = False
                continue
            macro, macro_len = match_macro(tokens, i)
            if macro:
                snippet = " ".join(words[i : i + macro_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {snippet} ({confidence:.2f})")
                execute_macro(macro)
                i += macro_len
                in_command_scope = False
                continue
            action, action_len = match_flow_command(tokens, i)
            if action:
                snippet = " ".join(words[i : i + action_len])
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {snippet} ({confidence:.2f})")
                execute_flow_action(action)
                i += action_len
                in_command_scope = False
                continue
            remaining = tuple(tokens[i:])
            if remaining and remaining[0] in COMMAND_WORD_ALIASES:
                typed_words.append(words[i])
                in_command_scope = False
                i += 1
                continue
            if remaining in PHRASE_PREFIXES:
                pending_command_tokens[:] = list(remaining)
                pending_command_words[:] = words[i:]
                break
            typed_words.append(words[i])
            in_command_scope = False
            i += 1
            continue
        typed_words.append(words[i])
        i += 1

    if not is_paused and typed_words:
        joined = " ".join(typed_words)
        log_transcription(joined, confidence=confidence)
        print(f"{joined} ({confidence:.2f})")
        type_text(joined, add_space=True)


def type_text(text: str, add_space: bool = False):
    global typed_history
    prefix = ""
    if typed_history and typed_history[-1] in {".", "!", "?"}:
        prefix = " "
    payload = prefix + text
    if add_space and payload and payload[-1] not in {".", "!", "?", " "}:
        payload += " "
    subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", payload])
    typed_history += payload


def get_input_device_info(preferred: str | None, list_only: bool = False):
    """Inspect audio devices and optionally return the preferred record device."""
    audio = pyaudio.PyAudio()

    def device_dict(index):
        info = audio.get_device_info_by_index(index)
        info["index"] = index
        return info

    if list_only or os.environ.get("FROSHINE_LIST_DEVICES") == "1":
        logging.info("\nAvailable audio devices:")
        for i in range(audio.get_device_count()):
            try:
                info = device_dict(i)
                logging.info(
                    "Device %s: %s (inputs=%s, rate=%s)",
                    i,
                    info["name"],
                    info["maxInputChannels"],
                    info["defaultSampleRate"],
                )
            except Exception:
                continue
        audio.terminate()
        return None

    selected = None
    if preferred:
        try:
            if preferred.isdigit():
                info = device_dict(int(preferred))
            else:
                for i in range(audio.get_device_count()):
                    info = device_dict(i)
                    if preferred.lower() in info["name"].lower():
                        break
                else:
                    info = None
            if info and info.get("maxInputChannels", 0) > 0:
                selected = info
                logging.info("Using configured input device: %s", info["name"])
        except Exception as exc:
            logging.warning("Error using configured device '%s': %s", preferred, exc)

    if not selected:
        try:
            info = audio.get_default_input_device_info()
            info["index"] = int(info.get("index", 0))
            if info.get("maxInputChannels", 0) > 0:
                selected = info
                logging.info("Using system default input device: %s", info["name"])
        except Exception as exc:
            logging.warning("Could not get system default input device: %s", exc)

    if not selected:
        for i in range(audio.get_device_count()):
            try:
                info = device_dict(i)
                if info.get("maxInputChannels", 0) > 0:
                    selected = info
                    logging.info("Using first available input device: %s", info["name"])
                    break
            except Exception:
                continue

    audio.terminate()
    return selected


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def is_local_ws_url(ws_url: str) -> bool:
    parsed = urlparse(ws_url)
    host = parsed.hostname or ""
    return host in LOCAL_HOSTS


def parse_moshi_server_target(ws_url: str) -> tuple[str | None, int | None]:
    parsed = urlparse(ws_url)
    host = parsed.hostname
    if not host:
        return None, None
    if host not in LOCAL_HOSTS:
        return None, None
    if parsed.port is None:
        logging.warning(
            "Websocket URL missing port; set FROSHINE_KYUTAI_WS_URL with an explicit port "
            "to auto-start moshi-server."
        )
        return host, None
    return host, parsed.port


def has_arg(arg_list: list[str], names: set[str]) -> bool:
    for idx, arg in enumerate(arg_list):
        if arg in names:
            return True
        for name in names:
            if arg.startswith(f"{name}="):
                return True
        if idx > 0 and arg_list[idx - 1] in names:
            return True
    return False


def detect_free_vram_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def select_config_path() -> Path:
    configured = os.getenv("FROSHINE_KYUTAI_CONFIG")
    if configured:
        return Path(configured).expanduser()
    if os.getenv("FROSHINE_KYUTAI_AUTO_CONFIG", "1").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return DEFAULT_EN_CONFIG
    free_vram_mb = detect_free_vram_mb()
    threshold = int(
        os.getenv("FROSHINE_MIN_FREE_VRAM_EN_MB", str(DEFAULT_MIN_FREE_VRAM_EN_MB))
    )
    if free_vram_mb is None or free_vram_mb < threshold:
        logging.info(
            "Auto-config selected en_fr (free VRAM %s MB, threshold %s MB).",
            free_vram_mb,
            threshold,
        )
        return DEFAULT_EN_FR_CONFIG
    logging.info(
        "Auto-config selected en (free VRAM %s MB, threshold %s MB).",
        free_vram_mb,
        threshold,
    )
    return DEFAULT_EN_CONFIG


def expand_config_log_dir(config_path: Path) -> Path:
    text = config_path.read_text()
    match = re.search(r'^(log_dir\s*=\s*")([^"]*)(")', text, flags=re.MULTILINE)
    if not match:
        return config_path
    original = match.group(2)
    expanded = os.path.expandvars(os.path.expanduser(original))
    if expanded == original:
        return config_path
    updated = (
        text[: match.start(2)] + expanded + text[match.end(2) :]
    )
    expanded_path = config_path.with_name(
        f"{config_path.stem}.expanded{config_path.suffix}"
    )
    expanded_path.write_text(updated)
    return expanded_path


def resolve_output_dir(raw_dir: str | None) -> Path:
    if raw_dir:
        return Path(raw_dir).expanduser()
    return Path(DEFAULT_OUTPUT_DIR)


def next_output_path(output_dir: Path, date_prefix: str, label: str, ext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        rf"^{re.escape(date_prefix)}-{re.escape(label)}-(\d+)\.{re.escape(ext)}$"
    )
    max_index = 0
    for entry in output_dir.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if match:
            try:
                max_index = max(max_index, int(match.group(1)))
            except ValueError:
                continue
    index = max_index + 1
    path = output_dir / f"{date_prefix}-{label}-{index}.{ext}"
    while path.exists():
        index += 1
        path = output_dir / f"{date_prefix}-{label}-{index}.{ext}"
    return path


class OutputManager:
    def __init__(
        self,
        output_dir: Path,
        save_audio: bool,
        save_transcript: bool,
        sample_rate: int,
        channels: int,
        sample_width: int,
        capture_format: int,
        capture_sample_width: int,
        audio_rotate_hours: float,
        audio_format: str,
        audio_bitrate_kbps: int,
    ):
        self.output_dir = output_dir
        self.save_audio = save_audio
        self.save_transcript = save_transcript
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.capture_format = capture_format
        self.capture_sample_width = capture_sample_width
        self.audio_rotate_hours = audio_rotate_hours
        self.audio_format = audio_format
        self.audio_bitrate_kbps = max(6, audio_bitrate_kbps)
        self.audio_path = None
        self.audio_tmp_path = None
        self.text_path = None
        self.audio_file = None
        self.text_file = None
        self.lock = threading.Lock()
        self.audio_opened_at = None

    def _open_audio_segment(self, date_prefix: str):
        if self.audio_format == "opus":
            self.audio_path = next_output_path(
                self.output_dir, date_prefix, "audio", "ogg"
            )
            self.audio_tmp_path = self.audio_path.with_suffix(".tmp.wav")
            wav_target = self.audio_tmp_path
        else:
            self.audio_path = next_output_path(
                self.output_dir, date_prefix, "audio", "wav"
            )
            self.audio_tmp_path = None
            wav_target = self.audio_path
        self.audio_file = wave.open(str(wav_target), "wb")
        self.audio_file.setnchannels(self.channels)
        self.audio_file.setsampwidth(self.sample_width)
        self.audio_file.setframerate(self.sample_rate)
        self.audio_opened_at = datetime.now()
        if self.audio_format == "opus":
            logging.info(
                "Saving audio to %s (opus %sk; temp wav %s)",
                self.audio_path,
                self.audio_bitrate_kbps,
                wav_target,
            )
        else:
            logging.info("Saving audio to %s", self.audio_path)

    def _finalize_audio_segment(self):
        if not self.audio_file:
            return
        self.audio_file.close()
        self.audio_file = None
        self.audio_opened_at = None
        if (
            self.audio_format != "opus"
            or not self.audio_tmp_path
            or not self.audio_path
        ):
            return
        temp_wav = self.audio_tmp_path
        target = self.audio_path
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temp_wav),
            "-c:a",
            "libopus",
            "-b:a",
            f"{self.audio_bitrate_kbps}k",
            "-vbr",
            "on",
            "-compression_level",
            "10",
            str(target),
        ]
        try:
            subprocess.run(cmd, check=True)
            temp_wav.unlink(missing_ok=True)
            logging.info(
                "Saved audio to %s (opus %sk)",
                target,
                self.audio_bitrate_kbps,
            )
        except Exception as exc:
            fallback = target.with_suffix(".wav")
            try:
                temp_wav.replace(fallback)
                logging.warning(
                    "Opus conversion failed (%s). Kept WAV fallback at %s.",
                    exc,
                    fallback,
                )
                self.audio_path = fallback
            except Exception as fallback_exc:
                logging.error(
                    "Opus conversion failed (%s) and WAV fallback failed (%s).",
                    exc,
                    fallback_exc,
                )
        finally:
            self.audio_tmp_path = None

    def open(self):
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        if self.save_audio:
            self._open_audio_segment(date_prefix)
        if self.save_transcript:
            self.text_path = next_output_path(
                self.output_dir, date_prefix, "text", "txt"
            )
            self.text_file = self.text_path.open("a", encoding="utf-8")
            logging.info("Saving transcripts to %s", self.text_path)

    def _rotate_audio_if_needed(self):
        if not self.audio_file or self.audio_rotate_hours <= 0:
            return
        if not self.audio_opened_at:
            return
        elapsed = datetime.now() - self.audio_opened_at
        if elapsed.total_seconds() < self.audio_rotate_hours * 3600:
            return
        self._finalize_audio_segment()
        self.audio_path = None
        self.audio_tmp_path = None
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        self._open_audio_segment(date_prefix)
        logging.info("Rotated audio; now saving to %s", self.audio_path)

    def write_audio(self, data: bytes):
        if not self.audio_file:
            return
        with self.lock:
            self._rotate_audio_if_needed()
            if (
                self.capture_format == pyaudio.paFloat32
                and self.sample_width == 2
                and self.capture_sample_width == 4
            ):
                float_data = np.frombuffer(data, dtype=np.float32)
                int_data = np.clip(float_data, -1.0, 1.0)
                pcm16 = (int_data * 32767.0).astype(np.int16)
                self.audio_file.writeframes(pcm16.tobytes())
            else:
                self.audio_file.writeframes(data)

    def write_transcript(self, text: str):
        if not self.text_file:
            return
        with self.lock:
            self.text_file.write(text + "\n")
            self.text_file.flush()

    def close(self):
        with self.lock:
            if self.audio_file:
                self._finalize_audio_segment()
            if self.text_file:
                self.text_file.close()
                self.text_file = None


async def wait_for_moshi_server(
    ws_url: str, process: subprocess.Popen | None
) -> bool:
    addr, port = parse_moshi_server_target(ws_url)
    if not addr or not port:
        return False
    deadline = time.monotonic() + AUTO_SERVER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            logging.error(
                "Auto-started moshi-server exited with code %s", process.returncode
            )
            return False
        try:
            reader, writer = await asyncio.open_connection(addr, port)
            writer.close()
            if hasattr(writer, "wait_closed"):
                await writer.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(AUTO_SERVER_READY_INTERVAL)
    logging.warning(
        "Timed out waiting for moshi-server to listen on %s:%s", addr, port
    )
    return False


def start_kyutai_server(ws_url: str) -> subprocess.Popen:
    config_path = select_config_path()
    subprocess.run(
        [
            "python3",
            str(Path(__file__).resolve().parent / "scripts" / "tune_kyutai_batch.py"),
            "--config",
            str(config_path),
        ],
        check=True,
    )
    config_path = expand_config_log_dir(config_path)
    extra_args = os.getenv("FROSHINE_MOSHI_SERVER_ARGS", "").split()
    addr, port = parse_moshi_server_target(ws_url)
    cmd = ["moshi-server", "worker", "--config", str(config_path)]
    if addr and not has_arg(extra_args, {"--addr", "-a"}):
        cmd.extend(["--addr", addr])
    if port and not has_arg(extra_args, {"--port", "-p"}):
        cmd.extend(["--port", str(port)])
    default_log = os.getenv("FROSHINE_MOSHI_LOG_LEVEL", "warn")
    if default_log and not has_arg(extra_args, {"--log", "-l"}):
        cmd.extend(["--log", default_log])
    cmd.extend(extra_args)
    logging.info("Auto-starting moshi-server: %s", " ".join(cmd))
    return subprocess.Popen(cmd)


class AudioRecorder:
    def __init__(self, loop, queue, device_index):
        self.loop = loop
        self.queue = queue
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None
        self.running = threading.Event()

    def start(self):
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=FRAME_SIZE,
            )
            logging.info("Opened audio stream with device index %s", self.device_index)
        except Exception as exc:
            logging.error("Failed to open audio stream: %s", exc)
            raise

        self.running.set()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def _record_loop(self):
        while self.running.is_set():
            try:
                data = self.stream.read(FRAME_SIZE, exception_on_overflow=False)
            except Exception as exc:
                logging.error("Audio capture error: %s", exc)
                break
            if output_manager:
                output_manager.write_audio(data)
            frame = np.frombuffer(data, dtype=np.float32).copy()

            def enqueue():
                if not running:
                    return
                try:
                    self.queue.put_nowait(frame)
                except asyncio.QueueFull:
                    logging.warning("Audio queue full; dropping frame")

            self.loop.call_soon_threadsafe(enqueue)

    def stop(self):
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


def handle_semantic_vad(step_msg):
    try:
        probs = step_msg.get("prs", [])
        if probs:
            pause_prob = probs[SEMANTIC_VAD_HEAD]
            if pause_prob > SEMANTIC_VAD_THRESHOLD:
                logging.debug(
                    "Semantic VAD pause detected (head=%s prob=%.2f)",
                    SEMANTIC_VAD_HEAD,
                    pause_prob,
                )
    except Exception:
        pass


async def send_audio_data(websocket, queue):
    while running:
        frame = await queue.get()
        if frame is None:
            break
        chunk = {"type": "Audio", "pcm": frame.tolist()}
        payload = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
        try:
            await websocket.send(payload)
        except Exception as exc:
            logging.error("Error sending audio: %s", exc)
            break


async def receive_server_messages(websocket, loop):
    try:
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)
            msg_type = data.get("type")
            if msg_type == "Word":
                text = data.get("text", "").strip()
                if text:
                    await loop.run_in_executor(
                        None, process_transcription_text, text, 1.0
                    )
            elif msg_type == "Step":
                handle_semantic_vad(data)
    except websockets.ConnectionClosed:
        logging.info("Kyutai websocket closed")
    except Exception as exc:
        logging.error("Error receiving Kyutai messages: %s", exc)


async def kyutai_stream_loop(args, queue, recorder):
    global server_process
    headers = {}
    if args.api_key:
        headers["kyutai-api-key"] = args.api_key
    backoff = 1
    loop = asyncio.get_running_loop()
    auto_server_enabled = AUTO_SERVER and not args.no_auto_server
    attempted_auto_start = False

    while running:
        try:
            logging.info("Connecting to Kyutai server at %s", args.ws_url)
            connect_params = inspect.signature(websockets.connect).parameters
            if "additional_headers" in connect_params:
                websocket_ctx = websockets.connect(
                    args.ws_url, additional_headers=headers
                )
            elif "extra_headers" in connect_params:
                websocket_ctx = websockets.connect(
                    args.ws_url, extra_headers=headers
                )
            else:
                if headers:
                    logging.warning(
                        "Websocket client does not support headers; upgrade websockets."
                    )
                websocket_ctx = websockets.connect(args.ws_url)
            async with websocket_ctx as websocket:
                if not recorder.running.is_set():
                    logging.info("Kyutai connected; starting audio capture.")
                    recorder.start()
                backoff = 1
                async def wait_for_shutdown():
                    if shutdown_event:
                        await shutdown_event.wait()
                        await websocket.close()

                sender = asyncio.create_task(send_audio_data(websocket, queue))
                receiver = asyncio.create_task(receive_server_messages(websocket, loop))
                stopper = asyncio.create_task(wait_for_shutdown())
                done, pending = await asyncio.wait(
                    [sender, receiver, stopper], return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    exc = task.exception()
                    if exc:
                        raise exc
        except asyncio.CancelledError:
            break
        except Exception as exc:
            if not running:
                break
            if (
                auto_server_enabled
                and not attempted_auto_start
                and is_local_ws_url(args.ws_url)
            ):
                attempted_auto_start = True
                global server_process
                if server_process is None or server_process.poll() is not None:
                    try:
                        server_process = start_kyutai_server(args.ws_url)
                        logging.info(
                            "Waiting for moshi-server to be ready (timeout %.1fs).",
                            AUTO_SERVER_READY_TIMEOUT,
                        )
                        ready = await wait_for_moshi_server(
                            args.ws_url, server_process
                        )
                        if not ready:
                            logging.warning(
                                "moshi-server did not become ready; will keep retrying."
                            )
                        continue
                    except Exception as start_exc:
                        logging.error("Auto-start failed: %s", start_exc)
            logging.error("Kyutai streaming error: %s", exc)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)


async def run_async(args):
    global running
    global shutdown_event, queue_ref, loop_ref
    global output_manager
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=200)
    shutdown_event = asyncio.Event()
    queue_ref = queue
    loop_ref = loop

    if args.list_devices:
        get_input_device_info(args.device, list_only=True)
        return

    if args.audio_bitrate_kbps < 6:
        logging.warning(
            "audio_bitrate_kbps=%s is too low for Opus; using 6 kbps.",
            args.audio_bitrate_kbps,
        )
        args.audio_bitrate_kbps = 6

    device_info = get_input_device_info(args.device)
    if not device_info:
        raise RuntimeError("No audio input devices available")

    if args.save_audio or args.save_transcript:
        audio = pyaudio.PyAudio()
        try:
            capture_sample_width = audio.get_sample_size(FORMAT)
        finally:
            audio.terminate()
        file_sample_width = capture_sample_width
        if FORMAT == pyaudio.paFloat32:
            file_sample_width = 2
            logging.info(
                "Converting float32 capture to 16-bit PCM for saved audio files."
            )
        output_manager = OutputManager(
            resolve_output_dir(args.output_dir),
            args.save_audio,
            args.save_transcript,
            SAMPLE_RATE,
            CHANNELS,
            file_sample_width,
            FORMAT,
            capture_sample_width,
            args.audio_rotate_hours,
            args.audio_format,
            args.audio_bitrate_kbps,
        )
        output_manager.open()

    recorder = AudioRecorder(loop, queue, device_info["index"])

    try:
        await kyutai_stream_loop(args, queue, recorder)
    finally:
        running = False
        await queue.put(None)
        if recorder.running.is_set():
            recorder.stop()
        if server_process and AUTO_SERVER_STOP and server_process.poll() is None:
            server_process.terminate()
        if output_manager:
            output_manager.close()
            output_manager = None


def main():
    global USE_WAKE_WORD
    global COMMAND_WORD, COMMAND_WORD_ALIASES
    args = parse_args()
    if args.word:
        COMMAND_WORD = args.word.lower()
        COMMAND_WORD_ALIASES = {COMMAND_WORD, *COMMAND_WORD_ALIASES}
    USE_WAKE_WORD = ENV_REQUIRE_WAKE_WORD and not args.no_wake_word
    print("\nFroshine Kyutai Voice Monitor")
    print(f"Streaming to: {args.ws_url}")
    print(f"Wake word: '{COMMAND_WORD}'")
    if USE_WAKE_WORD:
        print(
            f"Say:\n  '{COMMAND_WORD} pause' or '{COMMAND_WORD} unpause' to control transcription."
        )
        print(
            f"  '{COMMAND_WORD} quit' or '{COMMAND_WORD} stop' to stop the program;\n"
            f"  '{COMMAND_WORD} enter' to press Enter;\n"
            f"  '{COMMAND_WORD} mode command' to enter command mode;\n"
            f"  '{COMMAND_WORD} mode stop' to exit command mode\n"
        )
    else:
        print("Wake word disabled—commands execute as soon as they are recognized.\n")

    try:
        asyncio.run(run_async(args))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
