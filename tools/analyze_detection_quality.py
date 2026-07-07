#!/usr/bin/env python3
"""Offline localization-quality diagnostics for OpenTAD result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import median


DEFAULT_LENGTH_BINS = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf)


def _load_json(path_or_obj):
    if isinstance(path_or_obj, (str, Path)):
        with open(path_or_obj, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return path_or_obj


def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_config(config_path):
    config_path = Path(config_path)
    try:
        from mmengine.config import Config

        return Config.fromfile(str(config_path))
    except Exception:
        # Fallback for tiny standalone configs used by local tests or ad-hoc audits.
        return runpy.run_path(str(config_path))


def _resolve_path(value, base_dir):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def _iter_ground_truth_fallbacks(fallback_paths=None):
    if fallback_paths:
        for path in fallback_paths:
            yield path
    env_annotation = os.environ.get("THUMOS_ANNOTATION")
    if env_annotation:
        yield env_annotation
    thumos_root = os.environ.get("THUMOS_ROOT")
    if thumos_root:
        yield Path(thumos_root) / "annotations" / "thumos_14_anno.json"


def resolve_ground_truth_from_config(config_path, split="val", fallback_paths=None):
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    dataset = _cfg_get(cfg, "dataset")
    split_cfg = _cfg_get(dataset, split)
    ann_file = _cfg_get(split_cfg, "ann_file")
    if not ann_file:
        raise ValueError(f"Could not resolve dataset.{split}.ann_file from {config_path}")
    resolved = _resolve_path(ann_file, config_path.parent)
    if resolved.exists():
        return resolved
    for fallback in _iter_ground_truth_fallbacks(fallback_paths):
        fallback_path = _resolve_path(fallback, config_path.parent)
        if fallback_path.exists():
            return fallback_path
    return resolved


def resolve_prediction_path(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {path}")
    candidates = [candidate for candidate in path.rglob("result_detection.json") if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No result_detection.json found under {path}")
    return max(candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)))


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


def _row_video_id(row):
    return row.get("video_id", row.get("video_name", ""))


def _row_label(row):
    return row.get("label", row.get("class_id"))


def _row_score(row):
    return float(row.get("score", row.get("score_after_rescore", 0.0)))


def _row_segment_seconds(row):
    if "segment_seconds" in row:
        return [float(item) for item in row["segment_seconds"]]
    if "segment" in row:
        return [float(item) for item in row["segment"]]
    if "start" in row and "end" in row:
        return [float(row["start"]), float(row["end"])]
    raise KeyError(f"proposal row lacks segment_seconds/segment/start+end: {row}")


def _proposal_identity(row):
    segment = tuple(round(float(item), 4) for item in _row_segment_seconds(row))
    return (_row_video_id(row), str(_row_label(row)), round(_row_score(row), 6), segment)


def annotate_proposal_rows(
    ground_truth,
    proposal_rows,
    subset="validation",
    label_aware=True,
    tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
):
    """Attach best-GT IoU and boundary errors to proposal lifecycle rows.

    Proposal rows are intentionally lightweight JSONL-friendly dicts emitted by
    detector debug dumps. They may use either `video_id` or `video_name`, and
    either `segment_seconds`, `segment`, or `start`/`end`.
    """

    gt_data = _load_json(ground_truth)
    thresholds = tuple(float(t) for t in tiou_thresholds)
    gt_by_video = defaultdict(list)
    for row in _iter_ground_truth(gt_data, subset):
        gt_by_video[row["video_id"]].append(row)

    annotated = []
    for index, row in enumerate(proposal_rows):
        video_id = _row_video_id(row)
        label = _row_label(row)
        pred_segment = _row_segment_seconds(row)
        best_gt = None
        best_iou = 0.0
        for gt_row in gt_by_video.get(video_id, []):
            if label_aware and gt_row.get("label") != label:
                continue
            iou = segment_iou((gt_row["gt_start"], gt_row["gt_end"]), pred_segment)
            if best_gt is None or iou > best_iou:
                best_gt = gt_row
                best_iou = iou

        out = dict(row)
        out.setdefault("proposal_index", index)
        out["video_id"] = video_id
        out["label"] = label
        out["score"] = _row_score(row)
        out["segment_seconds"] = pred_segment
        out["best_iou"] = best_iou
        out["best_gt_index"] = None if best_gt is None else int(best_gt["gt_index"])
        out["best_gt_label"] = None if best_gt is None else best_gt.get("label")
        if best_gt is None:
            out["start_error"] = None
            out["end_error"] = None
            out["center_error"] = None
            out["duration_ratio_error"] = None
        else:
            pred_start, pred_end = pred_segment
            gt_start, gt_end = best_gt["gt_start"], best_gt["gt_end"]
            gt_length = max(gt_end - gt_start, 1e-12)
            out["start_error"] = pred_start - gt_start
            out["end_error"] = pred_end - gt_end
            out["center_error"] = 0.5 * (pred_start + pred_end) - 0.5 * (gt_start + gt_end)
            out["duration_ratio_error"] = ((pred_end - pred_start) / gt_length) - 1.0
        for threshold in thresholds:
            out[f"hit@{_format_threshold(threshold)}"] = best_iou >= threshold
        annotated.append(out)
    return annotated


def infer_nms_drop_reasons(pre_nms_rows, post_nms_rows, nms_iou_threshold=0.6):
    """Infer approximate NMS keep/drop reasons from pre/post proposal dumps."""

    kept_identities = {_proposal_identity(row) for row in post_nms_rows}
    post_by_video_label = defaultdict(list)
    for row in post_nms_rows:
        post_by_video_label[(_row_video_id(row), str(_row_label(row)))].append(row)

    rows = []
    for row in pre_nms_rows:
        out = dict(row)
        identity = _proposal_identity(row)
        kept = identity in kept_identities
        out["was_kept_by_nms"] = kept
        out["nms_drop_reason"] = None
        out["suppressor_score"] = None
        out["suppressor_iou"] = None
        out["suppressor_segment_seconds"] = None
        if not kept:
            suppressors = []
            row_segment = _row_segment_seconds(row)
            for candidate in post_by_video_label.get((_row_video_id(row), str(_row_label(row))), []):
                candidate_iou = segment_iou(row_segment, _row_segment_seconds(candidate))
                if candidate_iou >= float(nms_iou_threshold) and _row_score(candidate) >= _row_score(row):
                    suppressors.append((candidate_iou, _row_score(candidate), candidate))
            if suppressors:
                suppressors.sort(key=lambda item: (item[1], item[0]), reverse=True)
                iou, score, suppressor = suppressors[0]
                out["nms_drop_reason"] = "suppressed_by_higher_score_overlap"
                out["suppressor_score"] = score
                out["suppressor_iou"] = iou
                out["suppressor_segment_seconds"] = _row_segment_seconds(suppressor)
            else:
                out["nms_drop_reason"] = "missing_after_nms_or_voting_or_topk"
        rows.append(out)
    return rows


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {path}:{line_no}: {exc}") from exc
    return rows


def summarize_proposal_rows(rows, tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    thresholds = tuple(float(t) for t in tiou_thresholds)
    best_ious = [float(row.get("best_iou", 0.0)) for row in rows]
    summary = {
        "num_proposals": len(rows),
        "best_iou_mean": sum(best_ious) / len(best_ious) if best_ious else 0.0,
        "best_iou_p50": median(best_ious) if best_ious else 0.0,
        "best_iou_p90": _percentile(best_ious, 0.9),
    }
    for threshold in thresholds:
        key = _format_threshold(threshold)
        summary[f"proposal_hit_rate@{key}"] = (
            sum(1 for row in rows if float(row.get("best_iou", 0.0)) >= threshold) / len(rows) if rows else 0.0
        )
    if rows and "was_kept_by_nms" in rows[0]:
        dropped_good = {
            _format_threshold(threshold): sum(
                1
                for row in rows
                if not bool(row.get("was_kept_by_nms")) and float(row.get("best_iou", 0.0)) >= threshold
            )
            for threshold in thresholds
        }
        summary["nms_dropped_good_proposals"] = dropped_good
    return summary


def write_proposal_lifecycle_outputs(
    ground_truth,
    pre_nms_jsonl,
    post_nms_jsonl,
    output_prefix,
    subset="validation",
    label_aware=True,
    tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
    nms_iou_threshold=0.6,
):
    pre_rows = annotate_proposal_rows(
        ground_truth,
        read_jsonl(pre_nms_jsonl),
        subset=subset,
        label_aware=label_aware,
        tiou_thresholds=tiou_thresholds,
    )
    post_rows = annotate_proposal_rows(
        ground_truth,
        read_jsonl(post_nms_jsonl),
        subset=subset,
        label_aware=label_aware,
        tiou_thresholds=tiou_thresholds,
    )
    nms_rows = infer_nms_drop_reasons(pre_rows, post_rows, nms_iou_threshold=nms_iou_threshold)
    summary = {
        "pre_nms": summarize_proposal_rows(pre_rows, tiou_thresholds=tiou_thresholds),
        "post_nms": summarize_proposal_rows(post_rows, tiou_thresholds=tiou_thresholds),
        "nms": summarize_proposal_rows(nms_rows, tiou_thresholds=tiou_thresholds),
    }

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(pre_rows, f"{output_prefix}_pre_nms.csv")
    write_rows_csv(post_rows, f"{output_prefix}_post_nms.csv")
    write_rows_csv(nms_rows, f"{output_prefix}_nms.csv")
    summary_path = Path(f"{output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


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
    parser.add_argument("--ground-truth", default=None, help="OpenTAD annotation JSON with a database field")
    parser.add_argument("--prediction", default=None, help="OpenTAD result JSON with a results field")
    parser.add_argument("--config", default=None, help="OpenTAD config used to resolve dataset.<split>.ann_file")
    parser.add_argument(
        "--ground-truth-fallback",
        action="append",
        default=[],
        help="Fallback annotation path used when the config ann_file is stale or unavailable.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Experiment work_dir or run directory; the newest result_detection.json below it is used.",
    )
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--dataset-split", default="val", help="Config dataset split used with --config")
    parser.add_argument("--tiou-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--topk-per-video", type=int, default=None)
    parser.add_argument("--label-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-nms-jsonl", default=None, help="Detector proposal debug dump before NMS")
    parser.add_argument("--post-nms-jsonl", default=None, help="Detector proposal debug dump after NMS")
    parser.add_argument(
        "--proposal-output-prefix",
        default=None,
        help="Prefix for proposal lifecycle outputs: *_pre_nms.csv, *_post_nms.csv, *_nms.csv, *_summary.json",
    )
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ground_truth = args.ground_truth
    prediction = args.prediction

    if ground_truth is None and args.config is not None:
        ground_truth = resolve_ground_truth_from_config(
            args.config,
            args.dataset_split,
            fallback_paths=args.ground_truth_fallback,
        )
    if prediction is None and args.experiment_dir is not None:
        prediction = resolve_prediction_path(args.experiment_dir)
    if ground_truth is None:
        raise SystemExit("Missing --ground-truth or --config")

    has_proposal_lifecycle = args.pre_nms_jsonl is not None or args.post_nms_jsonl is not None
    if has_proposal_lifecycle:
        if args.pre_nms_jsonl is None or args.post_nms_jsonl is None:
            raise SystemExit("Both --pre-nms-jsonl and --post-nms-jsonl are required for proposal lifecycle analysis")
        if args.proposal_output_prefix is None:
            raise SystemExit("Missing --proposal-output-prefix for proposal lifecycle analysis")
        thresholds = tuple(float(item) for item in args.tiou_thresholds.split(",") if item)
        summary = write_proposal_lifecycle_outputs(
            ground_truth,
            args.pre_nms_jsonl,
            args.post_nms_jsonl,
            args.proposal_output_prefix,
            subset=args.subset,
            label_aware=args.label_aware,
            tiou_thresholds=thresholds,
            nms_iou_threshold=args.nms_iou_threshold,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if prediction is None and args.experiment_dir is None:
            return

    if prediction is None:
        raise SystemExit("Missing --prediction or --experiment-dir")

    output_json = args.output_json
    if output_json is None:
        if args.experiment_dir is None:
            raise SystemExit("Missing --output-json when --experiment-dir is not provided")
        output_json = str(Path(args.experiment_dir) / "detection_quality_summary.json")

    thresholds = tuple(float(item) for item in args.tiou_thresholds.split(",") if item)
    summary, rows = summarize_detection_quality(
        ground_truth,
        prediction,
        subset=args.subset,
        tiou_thresholds=thresholds,
        topk_per_video=args.topk_per_video,
        label_aware=args.label_aware,
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2, sort_keys=True)
    if args.output_csv:
        write_rows_csv(rows, args.output_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
