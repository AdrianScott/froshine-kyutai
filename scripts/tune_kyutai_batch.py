#!/usr/bin/env python3
"""Adjust the Kyutai server batch size based on available GPU VRAM."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EN_CONFIG = ROOT / "configs" / "config1-stt-en-hf.toml"
DEFAULT_EN_FR_CONFIG = ROOT / "configs" / "config1-stt-en_fr-hf.toml"
DEFAULT_CONFIG_PATH = DEFAULT_EN_CONFIG
DEFAULT_MIN_FREE_VRAM_EN_MB = 12000
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.free,memory.total",
    "--format=csv,noheader,nounits",
]


def detect_vram_mb() -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(GPU_QUERY, text=True).strip()
    except Exception:
        return None
    values: list[tuple[int, int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = [p.strip() for p in line.split(",")]
            free = int(float(parts[0]))
            total = int(float(parts[1])) if len(parts) > 1 else free
            values.append((free, total))
        except ValueError:
            continue
    if not values:
        return None
    return max(values, key=lambda pair: pair[0])


def determine_batch_size(vram_mb: int | None) -> int:
    forced = os.getenv("FROSHINE_FORCE_BATCH_SIZE")
    if forced:
        try:
            return int(forced)
        except ValueError:
            print("Invalid FROSHINE_FORCE_BATCH_SIZE value; ignoring.", file=sys.stderr)
    if vram_mb is None:
        return 16
    return 32 if vram_mb >= 23000 else 16


def update_config(path: Path, batch_size: int) -> None:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    text = path.read_text()
    new_text, count = re.subn(
        r"(batch_size\s*=\s*)\d+", r"\g<1>{}".format(batch_size), text, count=1
    )
    if count == 0:
        raise SystemExit("Could not locate batch_size entry in config.")
    path.write_text(new_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Kyutai batch size based on available GPU VRAM."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Kyutai TOML config (default: FROSHINE_KYUTAI_CONFIG or repo default).",
    )
    parser.add_argument(
        "--auto-config",
        action="store_true",
        help="Auto-select config based on free VRAM if FROSHINE_KYUTAI_CONFIG is unset.",
    )
    return parser.parse_args()


def auto_select_config(free_vram_mb: int | None) -> Path:
    threshold = int(
        os.getenv("FROSHINE_MIN_FREE_VRAM_EN_MB", str(DEFAULT_MIN_FREE_VRAM_EN_MB))
    )
    if free_vram_mb is None:
        return DEFAULT_EN_FR_CONFIG
    if free_vram_mb < threshold:
        return DEFAULT_EN_FR_CONFIG
    return DEFAULT_EN_CONFIG


def main() -> None:
    args = parse_args()
    env_config = os.getenv("FROSHINE_KYUTAI_CONFIG")
    config_value = args.config or env_config or str(DEFAULT_CONFIG_PATH)
    config_path = Path(config_value).expanduser()
    vram = detect_vram_mb()
    free_vram_mb = vram[0] if vram else None
    total_vram_mb = vram[1] if vram else None
    auto_config_enabled = (
        args.auto_config
        or os.getenv("FROSHINE_KYUTAI_AUTO_CONFIG", "1").lower() in ("1", "true", "yes")
    )
    if auto_config_enabled and not args.config and not env_config:
        config_path = auto_select_config(free_vram_mb)
    batch_size = determine_batch_size(free_vram_mb)
    update_config(config_path, batch_size)
    if free_vram_mb is None:
        print(f"Set batch_size={batch_size} (VRAM unknown, default applied).")
    else:
        total_hint = f" (total {total_vram_mb} MB)" if total_vram_mb else ""
        print(
            f"Detected free GPU VRAM: {free_vram_mb} MB{total_hint} -> configured "
            f"batch_size={batch_size} in {config_path}."
        )


if __name__ == "__main__":
    main()
