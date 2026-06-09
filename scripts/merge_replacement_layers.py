#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("No layers were selected.")
    return sorted(set(layers))


def layer_dir(root: Path, layer: int) -> Path:
    return root / f"layer_{layer:02d}"


def validate_source(source_root: Path, layers: list[int]) -> None:
    missing: list[str] = []
    for layer in layers:
        state_path = layer_dir(source_root, layer) / "custom_mole_layer.pt"
        if not state_path.exists():
            missing.append(str(state_path))
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing source checkpoint(s):\n  {joined}")


def copy_replacement_layers(
    source_root: Path,
    target_root: Path,
    layers: list[int],
    backup_root: Path | None,
    dry_run: bool,
) -> dict[str, object]:
    if not dry_run:
        validate_source(source_root, layers)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_stage_root = backup_root / timestamp if backup_root is not None else None
    operations: list[dict[str, str]] = []

    for layer in layers:
        src = layer_dir(source_root, layer)
        dst = layer_dir(target_root, layer)
        backup_dst = backup_stage_root / f"layer_{layer:02d}" if backup_stage_root is not None else None
        operations.append(
            {
                "layer": f"{layer:02d}",
                "source": str(src),
                "target": str(dst),
                "backup": str(backup_dst) if backup_dst is not None and dst.exists() else "",
            }
        )
        if dry_run:
            continue
        target_root.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if backup_dst is not None:
                backup_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dst, backup_dst)
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    manifest = {
        "timestamp_utc": timestamp,
        "source_root": str(source_root),
        "target_root": str(target_root),
        "layers": layers,
        "backup_root": str(backup_stage_root) if backup_stage_root is not None else "",
        "dry_run": dry_run,
        "operations": operations,
    }

    if not dry_run:
        history_path = target_root / "_merge_history.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely merge replacement layer checkpoints into a current-set directory.")
    parser.add_argument("--source", required=True, help="Source root, usually RUN_DIR/best_current.")
    parser.add_argument("--target", required=True, help="Target current-set root containing layer_XX directories.")
    parser.add_argument("--layers", required=True, help="Layers to copy, for example 0-5 or 0-3,8,11.")
    parser.add_argument("--backup_dir", default="", help="Where existing target layers are backed up before replacement.")
    parser.add_argument("--no_backup", action="store_true", help="Replace target layers without making a backup.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned operations without copying anything.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.source)
    target = Path(args.target)
    layers = parse_layers(args.layers)
    backup_root = None if args.no_backup else Path(args.backup_dir or target / "_backups")
    manifest = copy_replacement_layers(source, target, layers, backup_root, args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
