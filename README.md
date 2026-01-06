# Froshine VoiceCommander: Offline Voice-to-Text IDE Integration

A privacy-focused voice command system for developers that works entirely offline. It streams microphone audio to a local Kyutai STT server for transcription and injects the resulting text/commands directly into your editor.

Copyright 2025 Adrian Scott

## Quickstart (Kyutai)

Run the client; it will auto-start `moshi-server` locally (with batch tuning) if it is not already running. The auto-start uses the host/port from `FROSHINE_KYUTAI_WS_URL`.

```bash
python3 voice_monitor_command_word.py
```

## Features 

- **100% Offline** - No audio data leaves your machine (Kyutai server runs locally)
- **Real-time Monitoring** - Continuous voice input detection with semantic VAD from Kyutai
- **IDE Integration** - Direct text insertion into code editors
- **Voice Commands** - Custom commands for common actions
- **Command Word Support** - "Flow" prefix for commands (configurable)
- **Pause/Unpause Transcription** - Say "Flow pause" or "Flow unpause"
- **Streaming Output** - Words appear instantly without waiting for local silence detection

## Requirements 

- Ubuntu 20.04+ (other Linux distros may work)
- Python 3.8+
- Working microphone or audio input device
- xdotool (`sudo apt install xdotool` on Ubuntu/Debian)
- PortAudio libraries (`sudo apt install portaudio19-dev`)
- Rust toolchain for installing `moshi-server`

## Installation 

1. Clone the repository
2. Install system + Python dependencies:
   ```bash
   sudo apt install portaudio19-dev python3-dev xdotool ffmpeg
   pip install -r requirements.txt
   ```
3. Install the Kyutai streaming server (requires cargo):
   ```bash
   cargo install --features cuda moshi-server   # omit --features cuda for CPU-only
   ```

## Configuration 

Froshine can be configured using either:
1. Command-line arguments
2. Environment variables in a `.env` file
3. System environment variables

Command-line arguments take precedence over environment variables.

### Environment File

Copy the example configuration file to create your own:
```bash
cp .env.example .env
```

Then edit `.env` to customize your settings. See `.env.example` for available options.

### Audio Input Configuration

By default, Froshine uses your system's default audio input device. You can configure the audio input using environment variables:

- `FROSHINE_AUDIO_DEVICE`: Specify a preferred audio device by name or index
- `FROSHINE_LIST_DEVICES`: Set to "1" to list all available audio devices

Examples:
```bash
# List all available audio devices
FROSHINE_LIST_DEVICES=1 python voice_monitor_command_word.py

# Use a specific device by name (partial match)
FROSHINE_AUDIO_DEVICE="USB" python voice_monitor_command_word.py

# Use a specific device by index
FROSHINE_AUDIO_DEVICE="2" python voice_monitor_command_word.py
```

### Kyutai Streaming Setup

1. Copy or customize the provided TOML config at `configs/config1-stt-en-hf.toml` (English-only, default). A multilingual `configs/config1-stt-en_fr-hf.toml` is also included if you need it. The client can auto-select a smaller config when free VRAM is low.
2. Auto-tune the server batch size based on your free GPU memory (defaults: 16 for < 23 GB free VRAM, 32 otherwise):
   ```bash
   python3 scripts/tune_kyutai_batch.py
   ```
   You can override the choice with `FROSHINE_FORCE_BATCH_SIZE` or target a config via `--config` / `FROSHINE_KYUTAI_CONFIG`.
3. Start the Kyutai server locally:
   ```bash
   moshi-server worker --config configs/config1-stt-en-hf.toml
   ```
   Or use the one-shot helper (tunes, then starts the server):
   ```bash
   python3 scripts/run_kyutai_server.py
   ```
4. By default the client expects `ws://127.0.0.1:8908/api/asr-streaming` and the `kyutai-api-key: public_token` header. Override these with `FROSHINE_KYUTAI_WS_URL` and `FROSHINE_KYUTAI_API_KEY` if you are connecting to a remote host or a protected deployment. If your server is listening on a different port, update the websocket URL (auto-start will match it).

## Usage with voice_monitor_command_word.py

This script continuously streams microphone audio (24 kHz float frames) to the Kyutai STT server over websockets and types the transcribed text directly into your active window. Semantic pauses are detected via Kyutai Step messages rather than local silence detection, so words appear as soon as the server decodes them.

Start the script:

```bash
python3 voice_monitor_command_word.py
```

Begin speaking: The system will detect speech and automatically type the transcribed text into your currently focused application.

**Issue commands:**

- Say "Flow enter" to press Enter (use `--word` or `FROSHINE_COMMAND_WORD` to change it).
- Say "Flow save file" to simulate Ctrl+S.
- Say "Flow pause" to stop typing text (commands still work).
- Say "Flow unpause" to resume typing text.
- Stop the script: Say "Flow quit", or press Ctrl+C in the terminal to exit.

Environment / CLI parameters:

- `FROSHINE_AUDIO_DEVICE` or `--device`: select a microphone (name or index).
- `FROSHINE_LIST_DEVICES=1` or `--list-devices`: enumerate inputs.
- `FROSHINE_KYUTAI_WS_URL` or `--ws-url`: websocket endpoint.
- `FROSHINE_KYUTAI_API_KEY` or `--api-key`: API key header for `moshi-server`.
- `FROSHINE_KYUTAI_VAD_HEAD` and `FROSHINE_KYUTAI_VAD_THRESHOLD`: tweak semantic pause detection.
- `FROSHINE_COMMAND_WORD`, `FROSHINE_COMMAND_WORD_ALIASES`, `FROSHINE_REQUIRE_WAKE_WORD`, `--word`, or `--no-wake-word`: configure or disable the wake word requirement. When disabled, commands such as “pause” execute as soon as they are recognized, no prefix needed.
- `FROSHINE_KYUTAI_CONFIG` (or `--config` in the tuning/server helpers): choose which Kyutai TOML config to tune/run.
- `FROSHINE_AUTO_SERVER=1` (or `--no-auto-server` to disable): auto-start a local `moshi-server` when the websocket is unavailable.
- `FROSHINE_MOSHI_SERVER_ARGS`: extra args forwarded to `moshi-server` when auto-starting (overrides auto-selected `--addr`/`--port` if provided).
- `FROSHINE_AUTO_SERVER_STOP=1`: stop the auto-started server when the client exits.
- `FROSHINE_MOSHI_LOG_LEVEL` (default: `warn`): log level for auto-started `moshi-server` unless overridden in `FROSHINE_MOSHI_SERVER_ARGS`.
- `FROSHINE_KYUTAI_AUTO_CONFIG=1`: auto-select a smaller config when free VRAM is below the threshold.
- `FROSHINE_MIN_FREE_VRAM_EN_MB` (default: 12000): minimum free VRAM to use the 2.6B English model; otherwise the 1B en_fr config is selected.

## Troubleshooting 

**Common Issues:**

- **ALSA/JACK warnings**: Normal and safe to ignore
- **No audio input**:
  ```bash
  # Check recording devices
  arecord -l
  ```
- **Permission issues**:
  ```bash
  sudo usermod -a -G audio $USER
  # Reboot after running
  ```

## Privacy & Security 

- All audio processing happens locally
- No internet connection required
- No tracking or data collection

## Copyright

Copyright 2025 Adrian Scott

---

**Acknowledgements**:

- Kyutai delayed-stream STT models and `moshi-server`
- PyAudio for audio capture
- websockets/msgpack communities for solid streaming primitives

````
early work in progress

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

sudo apt install xdotool
````

```
WINDOW_ID=$(xdotool search --name "\(Workspace\) \- Windsurf")
echo $WINDOW_ID
xdotool windowactivate --sync $WINDOW_ID; xdotool type --window $WINDOW_ID --delay 0 "windsurf test froshine"

```

Current mechanism is to start voice recorder, voice_to_ide.sh, then click in the field of Windsurf I want it to go into.

Next step: voice detection to automatically fire up the recorder.

After that: voice commands to choose window, and especially use Freepoprompt and o1-xml-parser to update files.
