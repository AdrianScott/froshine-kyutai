import argparse
import asyncio
import inspect
import logging
import os
import re
import subprocess
import threading
import time
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
    "unpause": ["unpause", "onpause", "on pause", "un pause"],
    "enter": ["enter", "inner"],
    "quit": ["quit", "quick"],
    "switch to browser": ["switch to browser", "open browser"],
    "save file": ["save file", "save document"],
}
COMMANDS = {
    "enter": ["enter"],
    "switch to browser": ["switch to browser"],
    "save file": ["save file"],
    "pause": ["pause"],
    "unpause": ["unpause"],
    "quit": ["quit"],
}

running = True
typed_history = ""
is_paused = False
pending_command_word = None
server_process = None
shutdown_event = None
queue_ref = None
loop_ref = None


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
    return parser.parse_args()


def log_transcription(transcription, is_command=False, confidence=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "command" if is_command else "transcription",
        "content": transcription,
    }
    if confidence is not None:
        log_entry["confidence"] = f"{confidence:.2f}"
    logging.info(log_entry)


def execute_command(command):
    global running, is_paused
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


def find_command_for_synonym(syn: str) -> str:
    for cmd, synonyms_list in COMMAND_SYNONYMS.items():
        if syn in synonyms_list and cmd in COMMANDS:
            return cmd
    return ""


def interpret_potential_command(norm_word: str):
    cmd_key = find_command_for_synonym(norm_word)
    if cmd_key:
        return True, cmd_key
    return False, ""


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
    global typed_history, is_paused, pending_command_word

    if not text:
        return

    words = text.split()
    in_command_scope = False
    skip_next_wake_word = False

    if USE_WAKE_WORD and pending_command_word:
        if words:
            first_norm = normalize_word(words[0])
            if is_wake_word(first_norm):
                if not is_paused:
                    print(f"{pending_command_word} ({confidence:.2f})")
                    type_text(pending_command_word, add_space=True)
                pending_command_word = None
                words.pop(0)
                if not words:
                    return
            is_cmd, recognized = interpret_potential_command(first_norm)
            if not is_cmd:
                combined = pending_command_word + first_norm
                is_cmd, recognized = interpret_potential_command(combined)
            if is_cmd:
                snippet = f"{pending_command_word} {words[0]}"
                log_transcription(snippet, is_command=True, confidence=confidence)
                print(f"COMMAND: {recognized} ({confidence:.2f})")
                execute_command(recognized)
                words.pop(0)
                in_command_scope = False
            else:
                if not is_paused:
                    print(f"{pending_command_word} ({confidence:.2f})")
                    type_text(pending_command_word, add_space=True)
        else:
            if not is_paused:
                print(f"{pending_command_word} ({confidence:.2f})")
                type_text(pending_command_word, add_space=True)
        pending_command_word = None
    elif not USE_WAKE_WORD:
        pending_command_word = None

    typed_words = []
    for i, w in enumerate(words):
        norm = normalize_word(w)
        if USE_WAKE_WORD:
            if skip_next_wake_word:
                skip_next_wake_word = False
                continue
            if is_wake_word(norm) and i < len(words) - 1:
                next_norm = normalize_word(words[i + 1])
                if is_wake_word(next_norm):
                    typed_words.append(w)
                    skip_next_wake_word = True
                    continue
                in_command_scope = True
                continue
            if is_wake_word(norm) and i == len(words) - 1:
                pending_command_word = COMMAND_WORD
                continue
            if in_command_scope:
                is_cmd, recognized = interpret_potential_command(norm)
                if is_cmd:
                    log_transcription(w, is_command=True, confidence=confidence)
                    print(f"COMMAND: {recognized} ({confidence:.2f})")
                    execute_command(recognized)
                    in_command_scope = False
                else:
                    typed_words.append(w)
                    in_command_scope = False
            else:
                typed_words.append(w)
        else:
            is_cmd, recognized = interpret_potential_command(norm)
            if is_cmd:
                log_transcription(w, is_command=True, confidence=confidence)
                print(f"COMMAND: {recognized} ({confidence:.2f})")
                execute_command(recognized)
            else:
                typed_words.append(w)

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
    subprocess.run(["xdotool", "type", "--delay", "0", payload])
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
                logging.info(
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
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=200)
    shutdown_event = asyncio.Event()
    queue_ref = queue
    loop_ref = loop

    if args.list_devices:
        get_input_device_info(args.device, list_only=True)
        return

    device_info = get_input_device_info(args.device)
    if not device_info:
        raise RuntimeError("No audio input devices available")

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
            f"  '{COMMAND_WORD} quit' to stop the program;\n  '{COMMAND_WORD} enter' to press Enter\n"
        )
    else:
        print("Wake word disabled—commands execute as soon as they are recognized.\n")

    try:
        asyncio.run(run_async(args))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
