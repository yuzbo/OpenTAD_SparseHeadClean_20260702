#!/usr/bin/env python3
"""Collect OpenTAD experiment artifacts into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


AVERAGE_MAP_RE = re.compile(r"Average-mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)")
TIOU_MAP_RE = re.compile(r"mAP at tIoU\s+([0-9]+(?:\.[0-9]+)?)\s+is\s+([0-9]+(?:\.[0-9]+)?)%")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _newest(path: Path, pattern: str):
    candidates = sorted(path.glob(pattern), key=lambda item: (item.stat().st_mtime_ns, str(item)))
    return candidates[-1] if candidates else None


def parse_log(log_path: Path) -> dict:
    text = _read_text(log_path)
    average_maps = [float(match.group(1)) for match in AVERAGE_MAP_RE.finditer(text)]
    tiou_maps = {f"{float(match.group(1)):.2f}": float(match.group(2)) for match in TIOU_MAP_RE.finditer(text)}
    row = {
        "log_exists": log_path.exists(),
        "training_over": "Training Over" in text,
        "average_mAP": average_maps[-1] if average_maps else None,
    }
    for key, value in tiou_maps.items():
        row[f"tiou_mAP@{key}"] = value
    return row


def load_quality(exp_dir: Path) -> dict:
    quality_path = _newest(exp_dir, "detection_quality_summary*.json")
    if quality_path is None:
        return {}
    payload = _load_json(quality_path)
    if payload is None:
        return {}
    summary = payload.get("summary", payload) if isinstance(payload, dict) else {}
    row = {}
    for key in ("num_predictions", "recall@0.30", "recall@0.50", "recall@0.70", "best_iou_mean"):
        if key in summary:
            safe_key = key.replace("num_predictions", "num_predictions")
            row[f"quality_{safe_key}"] = summary[key]
    return row


def collect_one(label: str, exp_dir: Path) -> dict:
    exp_dir = Path(exp_dir)
    row = {
        "label": label,
        "exp_dir": str(exp_dir),
        "result_detection_exists": (exp_dir / "result_detection.json").exists(),
    }
    row.update(parse_log(exp_dir / "log.json"))
    row.update(load_quality(exp_dir))
    return row


def collect_experiments(items):
    return [collect_one(label, Path(exp_dir)) for label, exp_dir in items]


def _fieldnames(rows):
    preferred = [
        "label",
        "exp_dir",
        "log_exists",
        "result_detection_exists",
        "training_over",
        "average_mAP",
        "tiou_mAP@0.30",
        "tiou_mAP@0.40",
        "tiou_mAP@0.50",
        "tiou_mAP@0.60",
        "tiou_mAP@0.70",
        "quality_num_predictions",
        "quality_recall@0.30",
        "quality_recall@0.50",
        "quality_recall@0.70",
        "quality_best_iou_mean",
    ]
    keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fieldnames(rows)
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(field) is None else str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_item(value: str):
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name, path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment dirs. Use label=path to force a table label.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_experiments([parse_item(item) for item in args.experiments])
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_out:
        write_csv(rows, args.csv_out)
    if args.md_out:
        write_markdown(rows, args.md_out)


if __name__ == "__main__":
    main()
