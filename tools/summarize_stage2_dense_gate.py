#!/usr/bin/env python3
"""Summarize Stage-2 dense selected-axis sanity gates.

This tool is intentionally conservative: it does not train or evaluate models.
It only inspects already-produced OpenTAD logs, result JSON files, and Stage-4
quality summaries, then writes a machine-readable go/no-go report.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


AVERAGE_MAP_RE = re.compile(r"Average-mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)")
TIOU_MAP_RE = re.compile(r"mAP at tIoU\s+([0-9]+(?:\.[0-9]+)?)\s+is\s+([0-9]+(?:\.[0-9]+)?)%")
HARD_ERROR_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)|\bTraceback\b"),
    "oom": re.compile(r"CUDA out of memory|out of memory|\bOOM\b", re.IGNORECASE),
    "nonfinite_loss": re.compile(r"non[-_ ]finite (?:loss|cost)|loss=nan|cost=nan", re.IGNORECASE),
}
GRAD_SKIP_PATTERNS = (
    re.compile(r"non[-_ ]finite parameter gradient", re.IGNORECASE),
    re.compile(r"nonfinite_grad_skip_count", re.IGNORECASE),
)


@dataclass(frozen=True)
class Stage2Target:
    label: str
    split: str
    exp_dir: str
    min_average_map: Optional[float] = None
    require_average_map: bool = True
    min_quality_num_predictions: int = 1
    min_quality_recall_030: float = 1e-6


TARGETS = (
    Stage2Target(
        label="near63_random_short",
        split="short",
        exp_dir="exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_shortgate_n16r4/gpu1_id1",
    ),
    Stage2Target(
        label="near65_uniform_short",
        split="short",
        exp_dir="exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_shortgate_n16r4/gpu1_id1",
    ),
    Stage2Target(
        label="near63_random_long",
        split="long",
        exp_dir="exps/thumos/adatad/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4/gpu1_id0",
        min_average_map=60.0,
    ),
    Stage2Target(
        label="near65_uniform_long",
        split="long",
        exp_dir="exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4/gpu1_id0",
        min_average_map=62.0,
    ),
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_training_log(log_path: Path) -> dict:
    text = _read_text(log_path)
    average_maps = [float(match.group(1)) for match in AVERAGE_MAP_RE.finditer(text)]
    tiou_maps: dict[str, float] = {}
    for match in TIOU_MAP_RE.finditer(text):
        tiou_maps[f"{float(match.group(1)):.2f}"] = float(match.group(2))

    hard_errors = {name: bool(pattern.search(text)) for name, pattern in HARD_ERROR_PATTERNS.items()}
    grad_skip_count = sum(len(pattern.findall(text)) for pattern in GRAD_SKIP_PATTERNS)
    return {
        "path": str(log_path),
        "exists": log_path.exists(),
        "training_over": "Training Over" in text,
        "average_mAP": average_maps[-1] if average_maps else None,
        "tiou_mAP": tiou_maps,
        "hard_errors": hard_errors,
        "hard_error": any(hard_errors.values()),
        "nonfinite_grad_skip_count": grad_skip_count,
    }


def newest_quality_summary(exp_dir: Path) -> Optional[Path]:
    candidates = sorted(
        exp_dir.glob("detection_quality_summary_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    return candidates[-1] if candidates else None


def load_quality_summary(path: Optional[Path]) -> dict:
    if path is None:
        return {"exists": False, "path": None, "summary": None}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "exists": True,
        "path": str(path),
        "summary": payload.get("summary", payload),
    }


def summarize_target(root: Path, target: Stage2Target) -> dict:
    exp_dir = root / target.exp_dir
    log_path = exp_dir / "log.json"
    result_path = exp_dir / "result_detection.json"
    log_summary = parse_training_log(log_path)
    quality = load_quality_summary(newest_quality_summary(exp_dir))

    missing = []
    if not exp_dir.exists():
        missing.append("exp_dir")
    if not log_summary["exists"]:
        missing.append("log.json")
    if not result_path.exists():
        missing.append("result_detection.json")
    if not quality["exists"]:
        missing.append("detection_quality_summary")

    average_map = log_summary["average_mAP"]
    threshold_pass = None
    if target.min_average_map is not None and average_map is not None:
        threshold_pass = average_map >= target.min_average_map

    quality_summary = quality.get("summary") if isinstance(quality.get("summary"), dict) else None
    quality_num_predictions = None
    quality_recall_030 = None
    if quality_summary is not None:
        quality_num_predictions = quality_summary.get("num_predictions")
        quality_recall_030 = quality_summary.get("recall@0.30")

    blocked_reasons = []
    if missing:
        blocked_reasons.append("missing_artifacts")
    if log_summary["hard_error"]:
        blocked_reasons.append("hard_error")
    if log_summary["exists"] and not log_summary["training_over"]:
        blocked_reasons.append("training_not_over")
    if target.require_average_map and average_map is None:
        blocked_reasons.append("missing_average_mAP")
    if threshold_pass is False:
        blocked_reasons.append("average_mAP_below_threshold")
    if quality["exists"] and quality_summary is None:
        blocked_reasons.append("missing_quality_summary")
    if quality_summary is not None:
        if quality_num_predictions is None:
            blocked_reasons.append("missing_quality_num_predictions")
        elif int(quality_num_predictions) < target.min_quality_num_predictions:
            blocked_reasons.append("quality_num_predictions_below_threshold")
        if quality_recall_030 is None:
            blocked_reasons.append("missing_quality_recall@0.30")
        elif float(quality_recall_030) < target.min_quality_recall_030:
            blocked_reasons.append("quality_recall@0.30_below_threshold")

    return {
        "label": target.label,
        "split": target.split,
        "exp_dir": str(exp_dir),
        "result_detection": {"exists": result_path.exists(), "path": str(result_path)},
        "log": log_summary,
        "quality": quality,
        "min_average_mAP": target.min_average_map,
        "min_quality_num_predictions": target.min_quality_num_predictions,
        "min_quality_recall@0.30": target.min_quality_recall_030,
        "threshold_pass": threshold_pass,
        "missing": missing,
        "blocked_reasons": blocked_reasons,
        "ok": not blocked_reasons,
    }


def summarize_stage2(root: Path) -> dict:
    targets = [summarize_target(root, target) for target in TARGETS]
    short_targets = [item for item in targets if item["split"] == "short"]
    long_targets = [item for item in targets if item["split"] == "long"]
    can_submit_long = bool(short_targets) and all(item["ok"] for item in short_targets)
    long_dense_sanity_pass = bool(long_targets) and all(item["ok"] for item in long_targets)
    return {
        "root": str(root),
        "targets": targets,
        "overall": {
            "can_submit_long_after_short": can_submit_long,
            "long_dense_sanity_pass": long_dense_sanity_pass,
            "all_available_targets_ok": all(item["ok"] for item in targets),
            "blocked_labels": [item["label"] for item in targets if not item["ok"]],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing exps/")
    parser.add_argument("--json-out", default=None, help="Optional path to write the gate summary JSON")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit nonzero when any target is blocked. The default is report-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = summarize_stage2(Path(args.root).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if args.fail_on_blocked and not report["overall"]["all_available_targets_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
