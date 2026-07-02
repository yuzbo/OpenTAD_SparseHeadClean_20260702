import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TORCH = None


def to_numpy(value):
    if value is None:
        return None
    if TORCH is not None and TORCH.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def as_float_list(value):
    array = to_numpy(value)
    if array is None:
        return []
    return [float(item) for item in array.reshape(-1).tolist()]


def infer_temporal_carrier_len(sample):
    masks = sample.get("masks")
    if masks is not None:
        return int(to_numpy(masks).reshape(-1).shape[0]), "masks"

    inputs = sample.get("inputs")
    if inputs is None:
        return None, "missing"

    shape = list(inputs.shape) if hasattr(inputs, "shape") else []
    if len(shape) == 0:
        return None, "unknown_inputs"
    return int(shape[-1]), "inputs_last_dim"


def summarize_array(name, value):
    array = to_numpy(value)
    if array is None:
        return {"name": name, "present": False}
    flat = array.reshape(-1)
    summary = {
        "name": name,
        "present": True,
        "shape": list(array.shape),
        "numel": int(flat.size),
    }
    if flat.size > 0:
        finite = np.isfinite(flat.astype(np.float64, copy=False))
        summary.update(
            {
                "min": float(np.nanmin(flat)),
                "max": float(np.nanmax(flat)),
                "finite": bool(finite.all()),
            }
        )
    return summary


def build_report(sample, index):
    metas = sample.get("metas", {})
    masks = to_numpy(sample.get("masks"))
    positions = to_numpy(metas.get("irregular_selected_positions"))
    native_valid_len = metas.get("irregular_selected_valid_len")
    gt_segments = to_numpy(sample.get("gt_segments"))
    gt_labels = to_numpy(sample.get("gt_labels"))

    carrier_len, carrier_source = infer_temporal_carrier_len(sample)
    input_shape = list(sample["inputs"].shape) if "inputs" in sample and hasattr(sample["inputs"], "shape") else None

    if positions is None:
        positions = np.zeros((0,), dtype=np.float32)
    positions = positions.astype(np.float64).reshape(-1)

    native_valid_float = None if native_valid_len is None else float(native_valid_len)
    native_valid_int = None if native_valid_float is None else int(round(native_valid_float))
    detector_effective_valid_len = None
    if carrier_len is not None and native_valid_int is not None:
        detector_effective_valid_len = max(min(native_valid_int, carrier_len), 1)

    gt_range = None
    if gt_segments is not None and gt_segments.size > 0:
        gt_segments = gt_segments.astype(np.float64).reshape(-1, 2)
        gt_range = [float(np.min(gt_segments)), float(np.max(gt_segments))]

    position_diffs = np.diff(positions) if positions.size > 1 else np.zeros((0,), dtype=np.float64)
    selected_position_range = None
    if positions.size > 0:
        selected_position_range = [float(np.min(positions)), float(np.max(positions))]

    mask_true = None
    mask_len = None
    if masks is not None:
        masks_flat = masks.astype(bool).reshape(-1)
        mask_len = int(masks_flat.shape[0])
        mask_true = int(masks_flat.sum())

    inferred_physical_span = None
    if native_valid_float is not None:
        inferred_physical_span = [0.0, native_valid_float]

    coordinate_scale_ratio = None
    if carrier_len not in (None, 0) and native_valid_float not in (None, 0.0):
        coordinate_scale_ratio = float(native_valid_float / carrier_len)

    checks = {}
    checks["has_irregular_meta"] = bool(
        "irregular_selected_positions" in metas and "irregular_selected_valid_len" in metas
    )
    checks["positions_monotonic_non_decreasing"] = bool((position_diffs >= -1e-6).all())
    checks["positions_unique"] = bool(np.unique(positions).size == positions.size)
    checks["mask_len_equals_carrier_len"] = bool(mask_len is None or carrier_len is None or mask_len == carrier_len)
    checks["mask_true_matches_position_count"] = bool(mask_true is None or mask_true == positions.size)
    checks["position_count_not_greater_than_carrier_len"] = bool(carrier_len is None or positions.size <= carrier_len)
    checks["native_valid_len_ge_position_count"] = bool(native_valid_float is None or native_valid_float + 1e-6 >= positions.size)
    checks["positions_within_native_span"] = bool(
        native_valid_float is None
        or positions.size == 0
        or (positions.min() >= -1e-6 and positions.max() < native_valid_float + 1e-6)
    )
    checks["gt_within_native_span"] = bool(
        native_valid_float is None
        or gt_segments is None
        or gt_segments.size == 0
        or (gt_segments.min() >= -1e-6 and gt_segments.max() <= native_valid_float + 1e-6)
    )
    checks["candidate_a_coordinate_mismatch_present"] = bool(
        carrier_len is not None
        and native_valid_float is not None
        and not math.isclose(float(carrier_len), native_valid_float, rel_tol=0.0, abs_tol=1e-6)
    )
    checks["detector_would_clamp_native_valid_len"] = bool(
        detector_effective_valid_len is not None and native_valid_int is not None and detector_effective_valid_len != native_valid_int
    )

    diagnostic_false_is_ok = {
        "candidate_a_coordinate_mismatch_present",
        "detector_would_clamp_native_valid_len",
    }
    failed_checks = [key for key, value in checks.items() if not value and key not in diagnostic_false_is_ok]
    warnings = []
    if checks["candidate_a_coordinate_mismatch_present"]:
        warnings.append(
            "feature carrier length and native coordinate span differ; Candidate A must use positions as physical centers, not feature indices"
        )
    if carrier_len is not None and positions.size != carrier_len:
        warnings.append(
            "selected positions cover only valid sparse observations; tensor carrier has padded slots controlled by masks"
        )
    if checks["detector_would_clamp_native_valid_len"]:
        warnings.append(
            "IrregularActionFormer._temporal_grid_from_metas would clamp irregular_selected_valid_len to mask length"
        )

    return {
        "sample_index": int(index),
        "video_name": metas.get("video_name"),
        "input_shape": input_shape,
        "carrier_len": carrier_len,
        "carrier_source": carrier_source,
        "mask_len": mask_len,
        "mask_true": mask_true,
        "selected_positions": summarize_array("irregular_selected_positions", positions),
        "selected_positions_head": as_float_list(positions[:8]),
        "selected_positions_tail": as_float_list(positions[-8:]),
        "native_valid_len": native_valid_float,
        "detector_effective_valid_len": detector_effective_valid_len,
        "inferred_physical_span": inferred_physical_span,
        "coordinate_scale_ratio_native_over_carrier": coordinate_scale_ratio,
        "gt_segments": summarize_array("gt_segments", gt_segments),
        "gt_labels": summarize_array("gt_labels", gt_labels),
        "gt_range": gt_range,
        "selected_position_range": selected_position_range,
        "irregular_native_axis": metas.get("irregular_native_axis"),
        "checks": checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "candidate_a_contract": {
            "carrier_axis": f"feature index [0, {carrier_len})" if carrier_len is not None else "unknown",
            "physical_axis": f"native coordinate [0, {native_valid_float})" if native_valid_float is not None else "unknown",
            "recommended_point_center": "irregular_selected_positions[i]",
            "do_not_use_as_point_center": "feature index i when native_valid_len != carrier_len",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Check P0426-R05A native irregular data contract.")
    parser.add_argument(
        "--config",
        default="configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_true_overfit_single_video.py",
        help="Config path relative to OpenTAD_Back or absolute.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to inspect.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to inspect.")
    parser.add_argument("--video", default=None, help="Optional allow-list video name.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    global TORCH
    args = parse_args()

    import torch  # noqa: WPS433
    from mmengine.config import Config  # noqa: WPS433
    from opentad.datasets import build_dataset  # noqa: WPS433

    TORCH = torch

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    cfg = Config.fromfile(config_path)
    dataset_cfg = getattr(cfg.dataset, args.split)
    if args.video is not None:
        dataset_cfg.allow_list = [args.video]

    dataset = build_dataset(dataset_cfg)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range for dataset length {len(dataset)}")

    sample = dataset[args.index]
    report = build_report(sample, args.index)
    report["config"] = str(config_path)
    report["split"] = args.split
    report["dataset_length"] = len(dataset)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if report["failed_checks"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
