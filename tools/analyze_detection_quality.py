#!/usr/bin/env python3
"""Offline localization-quality diagnostics for OpenTAD result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


DEFAULT_LENGTH_BINS = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf)


def _load_json(path_or_obj):
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return path_or_obj


def segment_iou(target, candidate):
    start = max(float(target[0]), float(candidate[0]))
    end = min(float(target[1]), float(candidate[1]))
    inter = max(end - start, 0.0)
    target_len = max(float(target[1]) - float(target[0]), 0.0)
    candidate_len = max(float(candidate[1]) - float(candidate[0]), 0.0)
    union = target_len + candidate_len - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def _format_threshold(value):
    return f"{float(value):.2f}"


def _length_bucket(length, bins=DEFAULT_LENGTH_BINS):
    length = float(length)
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= length < right:
            if math.isinf(right):
                return f"[{left:g},inf)"
            return f"[{left:g},{right:g})"
    return f"[{bins[-2]:g},inf)"


def _percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * float(q)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def _iter_ground_truth(gt_data, subset):
    rows = []
    for video_id, video in gt_data["database"].items():
        if subset is not None and video.get("subset") != subset:
            continue
        for gt_index, ann in enumerate(video.get("annotations", [])):
            start, end = ann["segment"]
            rows.append(
                {
                    "video_id": video_id,
                    "gt_index": gt_index,
                    "label": ann.get("label"),
                    "gt_start": float(start),
                    "gt_end": float(end),
                }
            )
    return rows


def _prediction_rows(pred_data):
    rows_by_video = defaultdict(list)
    for video_id, preds in pred_data["results"].items():
        for pred_index, pred in enumerate(preds):
            start, end = pred["segment"]
            rows_by_video[video_id].append(
                {
                    "video_id": video_id,
                    "pred_index": pred_index,
                    "label": pred.get("label"),
                    "score": float(pred.get("score", 0.0)),
                    "start": float(start),
                    "end": float(end),
                }
            )
    for video_id in rows_by_video:
        rows_by_video[video_id].sort(key=lambda item: item["score"], reverse=True)
    return rows_by_video


def _limited_predictions(preds_by_video, topk_per_video):
    if topk_per_video is None:
        return preds_by_video
    limited = {}
    for video_id, preds in preds_by_video.items():
        limited[video_id] = preds[: int(topk_per_video)]
    return limited


def _best_prediction_for_gt(gt_row, preds, label_aware):
    best = None
    best_iou = 0.0
    gt_segment = (gt_row["gt_start"], gt_row["gt_end"])
    for pred in preds:
        if label_aware and pred.get("label") != gt_row.get("label"):
            continue
        iou = segment_iou(gt_segment, (pred["start"], pred["end"]))
        if best is None or iou > best_iou or (iou == best_iou and pred["score"] > best["score"]):
            best = pred
            best_iou = iou
    return best, best_iou


def summarize_detection_quality(
    ground_truth,
    prediction,
    subset="validation",
    tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
    topk_per_video=None,
    label_aware=True,
):
    gt_data = _load_json(ground_truth)
    pred_data = _load_json(prediction)
    gt_rows = _iter_ground_truth(gt_data, subset)
    preds_by_video = _limited_predictions(_prediction_rows(pred_data), topk_per_video)
    num_predictions = sum(len(preds) for preds in _prediction_rows(pred_data).values())
    thresholds = tuple(float(t) for t in tiou_thresholds)

    rows = []
    for gt_row in gt_rows:
        preds = preds_by_video.get(gt_row["video_id"], [])
        best_pred, best_iou = _best_prediction_for_gt(gt_row, preds, label_aware)
        gt_length = max(gt_row["gt_end"] - gt_row["gt_start"], 1e-12)
        if best_pred is None:
            start_error = 0.0
            end_error = 0.0
            boundary_error_mean = 0.0
            best_score = 0.0
            best_start = None
            best_end = None
        else:
            start_error = abs(best_pred["start"] - gt_row["gt_start"]) / gt_length
            end_error = abs(best_pred["end"] - gt_row["gt_end"]) / gt_length
            boundary_error_mean = 0.5 * (start_error + end_error)
            best_score = best_pred["score"]
            best_start = best_pred["start"]
            best_end = best_pred["end"]

        row = {
            **gt_row,
            "gt_length": gt_length,
            "length_bucket": _length_bucket(gt_length),
            "best_iou": best_iou,
            "best_score": best_score,
            "best_start": best_start,
            "best_end": best_end,
            "start_error_norm": start_error,
            "end_error_norm": end_error,
            "boundary_error_mean": boundary_error_mean,
        }
        for threshold in thresholds:
            row[f"hit@{_format_threshold(threshold)}"] = best_iou >= threshold
        rows.append(row)

    best_ious = [row["best_iou"] for row in rows]
    boundary_errors = [row["boundary_error_mean"] for row in rows]
    summary = {
        "subset": subset,
        "num_gt": len(rows),
        "num_predictions": num_predictions,
        "topk_per_video": topk_per_video,
        "label_aware": bool(label_aware),
        "best_iou_mean": sum(best_ious) / len(best_ious) if best_ious else 0.0,
        "best_iou_p50": median(best_ious) if best_ious else 0.0,
        "best_iou_p90": _percentile(best_ious, 0.9),
        "boundary_error_mean_mean": sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0,
        "boundary_error_mean_p50": median(boundary_errors) if boundary_errors else 0.0,
    }
    for threshold in thresholds:
        key = _format_threshold(threshold)
        summary[f"recall@{key}"] = (
            sum(1 for row in rows if row[f"hit@{key}"]) / len(rows) if rows else 0.0
        )

    rows_by_bucket = defaultdict(list)
    for row in rows:
        rows_by_bucket[row["length_bucket"]].append(row)
    for bucket in sorted(rows_by_bucket):
        bucket_rows = rows_by_bucket[bucket]
        summary[f"bucket:{bucket}:num_gt"] = len(bucket_rows)
        summary[f"bucket:{bucket}:best_iou_mean"] = sum(row["best_iou"] for row in bucket_rows) / len(bucket_rows)
        for threshold in thresholds:
            key = _format_threshold(threshold)
            summary[f"bucket:{bucket}:recall@{key}"] = (
                sum(1 for row in bucket_rows if row[f"hit@{key}"]) / len(bucket_rows)
            )
    return summary, rows


def write_rows_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, help="OpenTAD annotation JSON with a database field")
    parser.add_argument("--prediction", required=True, help="OpenTAD result JSON with a results field")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--tiou-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--topk-per-video", type=int, default=None)
    parser.add_argument("--label-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = tuple(float(item) for item in args.tiou_thresholds.split(",") if item)
    summary, rows = summarize_detection_quality(
        args.ground_truth,
        args.prediction,
        subset=args.subset,
        tiou_thresholds=thresholds,
        topk_per_video=args.topk_per_video,
        label_aware=args.label_aware,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2, sort_keys=True)
    if args.output_csv:
        write_rows_csv(rows, args.output_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
