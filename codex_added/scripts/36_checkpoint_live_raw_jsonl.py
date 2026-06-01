#!/usr/bin/env python3
# Added by Codex: JSONL-aware live checkpointing for raw inference shards.

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint complete JSONL rows from live raw shard files.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def fsync_path(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_complete_jsonl_rows(path: Path) -> list[str]:
    rows: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    break
                rows.append(line)
            os.fsync(handle.fileno())
    except FileNotFoundError:
        return []
    return rows


def write_atomic_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.writelines(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    fsync_dir(path.parent)


def checkpoint_once(run_dir: Path) -> dict[str, int]:
    raw_dir = run_dir / "raw"
    snap_dir = run_dir / "snapshots" / "live_latest"
    counts: dict[str, int] = {}
    for shard_path in sorted(raw_dir.glob("shard_*.jsonl")):
        rows = read_complete_jsonl_rows(shard_path)
        if not rows:
            continue
        dest = snap_dir / shard_path.name
        write_atomic_jsonl(dest, rows)
        counts[shard_path.name] = len(rows)
    return counts


def append_log(run_dir: Path, counts: dict[str, int]) -> None:
    log_path = run_dir / "snapshots" / "live_checkpoint.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        for name, count in sorted(counts.items()):
            handle.write(f"{name} lines={count}\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    while True:
        counts = checkpoint_once(run_dir)
        append_log(run_dir, counts)
        if (run_dir / "RUN_INFO.txt").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
