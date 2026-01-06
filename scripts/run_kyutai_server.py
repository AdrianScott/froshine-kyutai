#!/usr/bin/env python3
"""Tune and launch the Kyutai moshi-server worker."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EN_CONFIG = ROOT / "configs" / "config1-stt-en-hf.toml"
DEFAULT_EN_FR_CONFIG = ROOT / "configs" / "config1-stt-en_fr-hf.toml"
DEFAULT_CONFIG_PATH = DEFAULT_EN_CONFIG
DEFAULT_MIN_FREE_VRAM_EN_MB = 12000
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune batch size and start moshi-server with the chosen config."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Kyutai TOML config (default: FROSHINE_KYUTAI_CONFIG or repo default).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip auto-tuning batch size before launch.",
    )
    parser.add_argument(
        "moshi_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to moshi-server (prefix with --).",
    )
    return parser.parse_args()


def parse_moshi_target_from_ws_url(ws_url: str) -> tuple[str | None, int | None]:
    parsed = urlparse(ws_url)
    host = parsed.hostname
    if not host or host not in LOCAL_HOSTS:
        return None, None
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


def auto_select_config() -> Path:
    free_vram_mb = detect_free_vram_mb()
    threshold = int(
        os.getenv("FROSHINE_MIN_FREE_VRAM_EN_MB", str(DEFAULT_MIN_FREE_VRAM_EN_MB))
    )
    if free_vram_mb is None or free_vram_mb < threshold:
        return DEFAULT_EN_FR_CONFIG
    return DEFAULT_EN_CONFIG


def main() -> None:
    args = parse_args()
    env_config = os.getenv("FROSHINE_KYUTAI_CONFIG")
    config_value = args.config or env_config or str(DEFAULT_CONFIG_PATH)
    config_path = Path(config_value).expanduser()
    auto_config_enabled = os.getenv("FROSHINE_KYUTAI_AUTO_CONFIG", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    if auto_config_enabled and not args.config and not env_config:
        config_path = auto_select_config()

    if not args.no_tune:
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "tune_kyutai_batch.py"),
                "--config",
                str(config_path),
            ],
            check=True,
        )

    extra_args = os.getenv("FROSHINE_MOSHI_SERVER_ARGS", "").split()
    ws_url = os.getenv("FROSHINE_KYUTAI_WS_URL", "")
    addr = port = None
    if ws_url:
        addr, port = parse_moshi_target_from_ws_url(ws_url)
    moshi_cmd = ["moshi-server", "worker", "--config", str(config_path)]
    if addr and not has_arg(extra_args, {"--addr", "-a"}):
        moshi_cmd.extend(["--addr", addr])
    if port and not has_arg(extra_args, {"--port", "-p"}):
        moshi_cmd.extend(["--port", str(port)])
    default_log = os.getenv("FROSHINE_MOSHI_LOG_LEVEL", "warn")
    if default_log and not has_arg(extra_args, {"--log", "-l"}):
        moshi_cmd.extend(["--log", default_log])
    moshi_cmd.extend(extra_args)
    if args.moshi_args:
        moshi_cmd.extend(args.moshi_args)
    subprocess.run(moshi_cmd, check=True)


if __name__ == "__main__":
    main()
