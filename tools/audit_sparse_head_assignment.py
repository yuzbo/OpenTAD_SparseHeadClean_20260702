import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from mmengine.config import Config

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector


LENGTH_BUCKETS = [0, 4, 8, 16, 32, 64, 128, float("inf")]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit sparse-head assignment on the same train batches.")
    parser.add_argument("--configs", nargs="+", required=True, help="Config files to compare on the same batches.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to audit.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to audit.")
    parser.add_argument("--seed", type=int, default=20260705, help="Random seed for deterministic batch selection.")
    parser.add_argument("--device", default=None, help="Torch device, defaults to cuda if available else cpu.")
    parser.add_argument("--out", required=True, help="Output directory for JSON/CSV audit artifacts.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(value, device) for value in obj)
    return obj


def get_split_cfg(cfg, split):
    return getattr(cfg.dataset, split), getattr(cfg.solver, split)


def build_loader(cfg, split):
    dataset_cfg, solver_cfg = get_split_cfg(cfg, split)
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=None))
    return build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **solver_cfg,
    )


def length_bucket(length):
    for start, end in zip(LENGTH_BUCKETS[:-1], LENGTH_BUCKETS[1:]):
        if start <= length < end:
            right = "inf" if end == float("inf") else str(int(end))
            return f"[{int(start)},{right})"
    return "unknown"


def quantile_or_none(values, q):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q).item())


def max_or_none(values):
    if values.numel() == 0:
        return None
    return float(values.float().max().item())


def scalar_list(values):
    return [int(v) for v in values]


def level_offsets(points):
    offsets = []
    start = 0
    for level in points:
        length = int(level.shape[1] if level.dim() == 3 else level.shape[0])
        offsets.append((start, start + length))
        start += length
    return offsets


def split_points_per_sample(head, points, batch_size):
    if hasattr(head, "_points_per_sample"):
        return head._points_per_sample(points, batch_size)
    concat = torch.cat(points, dim=1)
    return list(concat)


def point_fields(head, point):
    if hasattr(head, "_point_fields_extended"):
        return head._point_fields_extended(point)
    if hasattr(head, "_point_fields"):
        fields = head._point_fields(point)
        center, reg_min, reg_max, left_scale, right_scale, point_scale = fields
        return center, reg_min, reg_max, left_scale, right_scale, point_scale, point_scale, point_scale
    center = point[..., 0]
    reg_min = point[..., 1]
    reg_max = point[..., 2]
    left_scale = point[..., 3]
    right_scale = point[..., 4] if point.shape[-1] >= 5 else left_scale
    point_scale = (left_scale + right_scale).clamp_min(1e-6)
    if point.shape[-1] >= 7:
        range_scale = point[..., 5].clamp_min(1e-6)
        radius_scale = point[..., 6].clamp_min(1e-6)
    else:
        range_scale = point_scale
        radius_scale = point_scale
    return center, reg_min, reg_max, left_scale, right_scale, point_scale, range_scale, radius_scale


def scale_base(head, left_scale, right_scale, point_scale, mode, range_scale=None, radius_scale=None):
    if hasattr(head, "_scale_base"):
        return head._scale_base(
            left_scale,
            right_scale,
            point_scale,
            mode,
            range_scale=range_scale,
            radius_scale=radius_scale,
        )
    if mode == "half_cell_span":
        return 0.5 * point_scale
    if mode == "min_side":
        return torch.minimum(left_scale, right_scale)
    if mode == "left_right_mean":
        return 0.5 * (left_scale + right_scale)
    if mode == "point_range" and range_scale is not None:
        return range_scale
    if mode == "point_radius" and radius_scale is not None:
        return radius_scale
    return point_scale


def encode_targets(head, left, right, left_scale, right_scale, point_scale, range_scale=None, radius_scale=None):
    if hasattr(head, "_encode_regression_targets"):
        return head._encode_regression_targets(
            left,
            right,
            left_scale,
            right_scale,
            point_scale,
            range_scale=range_scale,
            radius_scale=radius_scale,
        )
    return torch.stack([left / point_scale, right / point_scale], dim=-1).clamp_min(0.0)


def decode_encoded_targets(head, encoded, left_scale, right_scale, point_scale, range_scale=None, radius_scale=None):
    regression_mode = getattr(head, "regression_mode", "symmetric_linear")
    if regression_mode == "symmetric_linear":
        mode = getattr(head, "reg_denom_mode", "full_cell_span")
        denom = scale_base(
            head,
            left_scale,
            right_scale,
            point_scale,
            mode,
            range_scale=range_scale,
            radius_scale=radius_scale,
        )
        return encoded[:, 0] * denom, encoded[:, 1] * denom
    return torch.expm1(encoded[:, 0].clamp_min(0.0)) * left_scale, torch.expm1(encoded[:, 1].clamp_min(0.0)) * right_scale


@torch.no_grad()
def build_hard_diagnostics(head, point, gt_segment, offsets):
    center_t, reg_min, reg_max, left_scale, right_scale, point_scale, range_scale, radius_scale = point_fields(
        head, point
    )
    num_pts = int(point.shape[0])
    num_gts = int(gt_segment.shape[0])

    empty_counts = [0 for _ in offsets]
    result = {
        "assigned_gt": torch.full((num_pts,), -1, device=point.device, dtype=torch.long),
        "inside_gt_count_by_level": empty_counts[:],
        "center_pass_count_by_level": empty_counts[:],
        "per_level_candidate_count_before_range": empty_counts[:],
        "per_level_candidate_count_after_range": empty_counts[:],
        "range_fail_count_by_level": empty_counts[:],
        "center_fail_count_by_level": empty_counts[:],
        "multi_gt_conflict_count_before_shortest": 0,
        "shortest_gt_resolved_count": 0,
        "radius_base_p50": [None for _ in offsets],
        "radius_base_p90": [None for _ in offsets],
        "reg_target_left_p50": None,
        "reg_target_left_p90": None,
        "reg_target_left_max": None,
        "reg_target_right_p50": None,
        "reg_target_right_p90": None,
        "reg_target_right_max": None,
        "encoded_reg_p50": None,
        "encoded_reg_p90": None,
        "encoded_reg_max": None,
        "decode_reconstruction_max_error": None,
    }
    if num_gts == 0:
        return result

    gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
    left = center_t[:, None] - gt_segs[:, :, 0]
    right = gt_segs[:, :, 1] - center_t[:, None]
    reg_targets = torch.stack((left, right), dim=-1)
    inside_gt = reg_targets.min(dim=-1).values > 0

    if getattr(head, "center_sample", "radius") == "radius":
        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        radius_mode = getattr(head, "center_radius_scale", "full_cell_span")
        radius_base = scale_base(
            head,
            left_scale,
            right_scale,
            point_scale,
            radius_mode,
            range_scale=range_scale,
            radius_scale=radius_scale,
        )
        radius = radius_base[:, None] * float(getattr(head, "center_sample_radius", 1.5))
        t_mins = center_pts - radius
        t_maxs = center_pts + radius
        cb_left = center_t[:, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
        cb_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - center_t[:, None]
        center_mask = torch.stack((cb_left, cb_right), dim=-1).min(dim=-1).values > 0
    else:
        radius_base = point_scale
        center_mask = inside_gt

    max_regress_distance = reg_targets.max(dim=-1).values
    range_mask = torch.logical_and(max_regress_distance >= reg_min[:, None], max_regress_distance <= reg_max[:, None])
    before_range = torch.logical_and(inside_gt, center_mask)
    after_range = torch.logical_and(before_range, range_mask)
    gt_len = (gt_segment[:, 1] - gt_segment[:, 0])[None, :].repeat(num_pts, 1)
    gt_len = gt_len.masked_fill(~after_range, float("inf"))
    min_len, min_idx = gt_len.min(dim=1)
    assigned = torch.where(torch.isfinite(min_len), min_idx, torch.full_like(min_idx, -1))

    conflicts = after_range.sum(dim=1) > 1
    result["assigned_gt"] = assigned
    result["multi_gt_conflict_count_before_shortest"] = int(conflicts.sum().item())
    result["shortest_gt_resolved_count"] = int(torch.isfinite(min_len).sum().item())

    for level_idx, (lo, hi) in enumerate(offsets):
        result["inside_gt_count_by_level"][level_idx] = int(inside_gt[lo:hi].any(dim=1).sum().item())
        result["center_pass_count_by_level"][level_idx] = int(center_mask[lo:hi].any(dim=1).sum().item())
        result["per_level_candidate_count_before_range"][level_idx] = int(before_range[lo:hi].sum().item())
        result["per_level_candidate_count_after_range"][level_idx] = int(after_range[lo:hi].sum().item())
        result["range_fail_count_by_level"][level_idx] = int(torch.logical_and(before_range[lo:hi], ~range_mask[lo:hi]).sum().item())
        result["center_fail_count_by_level"][level_idx] = int(torch.logical_and(inside_gt[lo:hi], ~center_mask[lo:hi]).sum().item())
        result["radius_base_p50"][level_idx] = quantile_or_none(radius_base[lo:hi], 0.5)
        result["radius_base_p90"][level_idx] = quantile_or_none(radius_base[lo:hi], 0.9)

    pos = assigned >= 0
    if pos.any():
        rows = torch.arange(num_pts, device=point.device)[pos]
        cols = assigned[pos]
        raw_left = left[rows, cols]
        raw_right = right[rows, cols]
        enc = encode_targets(
            head,
            raw_left,
            raw_right,
            left_scale[pos],
            right_scale[pos],
            point_scale[pos],
            range_scale=range_scale[pos],
            radius_scale=radius_scale[pos],
        )
        dec_left, dec_right = decode_encoded_targets(
            head,
            enc,
            left_scale[pos],
            right_scale[pos],
            point_scale[pos],
            range_scale=range_scale[pos],
            radius_scale=radius_scale[pos],
        )
        result.update(
            {
                "reg_target_left_p50": quantile_or_none(raw_left, 0.5),
                "reg_target_left_p90": quantile_or_none(raw_left, 0.9),
                "reg_target_left_max": max_or_none(raw_left),
                "reg_target_right_p50": quantile_or_none(raw_right, 0.5),
                "reg_target_right_p90": quantile_or_none(raw_right, 0.9),
                "reg_target_right_max": max_or_none(raw_right),
                "encoded_reg_p50": quantile_or_none(enc.flatten(), 0.5),
                "encoded_reg_p90": quantile_or_none(enc.flatten(), 0.9),
                "encoded_reg_max": max_or_none(enc.flatten()),
                "decode_reconstruction_max_error": max_or_none(
                    torch.maximum((dec_left - raw_left).abs(), (dec_right - raw_right).abs())
                ),
            }
        )
    return result


def meta_value(meta, key, default=None):
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def make_sample_id(batch_idx, sample_idx, meta):
    video_name = meta_value(meta, "video_name", "unknown")
    start = meta_value(meta, "window_start", meta_value(meta, "snippet_start", "na"))
    return f"batch{batch_idx:04d}_sample{sample_idx:02d}_{video_name}_{start}"


@torch.no_grad()
def audit_config(config_path, batches, split, device):
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model).to(device)
    model.eval()
    head = model.rpn_head
    rows = []

    for batch_idx, data_dict in batches:
        data = to_device(data_dict, device)
        inputs = data["inputs"]
        masks = data["masks"]
        metas = data.get("metas", None)
        gt_segments = data.get("gt_segments", [])
        gt_labels = data.get("gt_labels", [])
        temporal_grids = data.get("temporal_grids", None)

        x = model.backbone(inputs, metas=metas) if model.with_backbone else inputs
        if temporal_grids is None:
            temporal_grids = model._temporal_grid_from_metas(metas, masks)
        x, masks, temporal_grid = model.pad_data(x, masks, temporal_grids)
        feat_list, mask_list, temporal_grid_list = model._project_features(x, masks, temporal_grid)
        points = head.prior_generator(feat_list, temporal_grid_list)
        point_list = split_points_per_sample(head, points, len(gt_segments))
        offsets = level_offsets(points)
        level_count = len(offsets)

        target = head.prepare_targets(points, gt_segments, gt_labels)
        if len(target) == 4:
            _, _, reg_weight_list, _ = target
        else:
            reg_weight_list = [None for _ in gt_segments]

        for sample_idx, (point, gt_segment) in enumerate(zip(point_list, gt_segments)):
            meta = metas[sample_idx] if metas is not None else {}
            sample_id = make_sample_id(batch_idx, sample_idx, meta)
            diag = build_hard_diagnostics(head, point, gt_segment, offsets)
            assigned = diag.pop("assigned_gt")
            reg_weight = reg_weight_list[sample_idx]
            per_level_pos_count = []
            valid_mask_true_count = []
            gt_coverage = []

            for lo, hi in offsets:
                if reg_weight is not None:
                    pos_mask = reg_weight[lo:hi] > 0
                else:
                    pos_mask = assigned[lo:hi] >= 0
                per_level_pos_count.append(int(pos_mask.sum().item()))
            for level_mask in mask_list:
                valid_mask_true_count.append(int(level_mask[sample_idx].sum().item()))

            for gt_idx, gt in enumerate(gt_segment):
                level_bitmap = []
                for lo, hi in offsets:
                    level_bitmap.append(bool((assigned[lo:hi] == gt_idx).any().item()))
                gt_len = float((gt[1] - gt[0]).item())
                gt_coverage.append(
                    {
                        "gt_idx": int(gt_idx),
                        "gt_length": gt_len,
                        "gt_length_bucket": length_bucket(gt_len),
                        "gt_covered_any": bool(any(level_bitmap)),
                        "gt_covered_level_bitmap": level_bitmap,
                        "gt_num_assigned_points": int((assigned == gt_idx).sum().item()),
                    }
                )

            rows.append(
                {
                    "config": str(config_path),
                    "config_name": Path(config_path).stem,
                    "split": split,
                    "batch_idx": int(batch_idx),
                    "sample_idx": int(sample_idx),
                    "sample_id": sample_id,
                    "video_name": meta_value(meta, "video_name", "unknown"),
                    "num_gt": int(gt_segment.shape[0]),
                    "assignment_mode": getattr(head, "assignment_mode", "unknown"),
                    "regression_mode": getattr(head, "regression_mode", "unknown"),
                    "center_radius_scale": getattr(head, "center_radius_scale", "legacy"),
                    "reg_denom_mode": getattr(head, "reg_denom_mode", "legacy"),
                    "range_mode": getattr(head.prior_generator, "range_mode", "unknown"),
                    "level_count": level_count,
                    "per_level_pos_count": per_level_pos_count,
                    "per_level_candidate_count_before_range": diag["per_level_candidate_count_before_range"],
                    "per_level_candidate_count_after_range": diag["per_level_candidate_count_after_range"],
                    "inside_gt_count_by_level": diag["inside_gt_count_by_level"],
                    "range_fail_count_by_level": diag["range_fail_count_by_level"],
                    "center_fail_count_by_level": diag["center_fail_count_by_level"],
                    "valid_mask_true_count": valid_mask_true_count,
                    "multi_gt_conflict_count_before_shortest": diag["multi_gt_conflict_count_before_shortest"],
                    "shortest_gt_resolved_count": diag["shortest_gt_resolved_count"],
                    "radius_base_p50": diag["radius_base_p50"],
                    "radius_base_p90": diag["radius_base_p90"],
                    "gt_coverage": gt_coverage,
                    "reg_target_left_p50": diag["reg_target_left_p50"],
                    "reg_target_left_p90": diag["reg_target_left_p90"],
                    "reg_target_left_max": diag["reg_target_left_max"],
                    "reg_target_right_p50": diag["reg_target_right_p50"],
                    "reg_target_right_p90": diag["reg_target_right_p90"],
                    "reg_target_right_max": diag["reg_target_right_max"],
                    "encoded_reg_p50": diag["encoded_reg_p50"],
                    "encoded_reg_p90": diag["encoded_reg_p90"],
                    "encoded_reg_max": diag["encoded_reg_max"],
                    "decode_reconstruction_max_error": diag["decode_reconstruction_max_error"],
                }
            )

    return rows


def json_default(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_outputs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "assignment_audit.json"
    csv_path = out_dir / "assignment_audit.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=json_default)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})
    return json_path, csv_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    first_cfg = Config.fromfile(args.configs[0])
    loader = build_loader(first_cfg, args.split)
    batches = []
    for batch_idx, data_dict in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        batches.append((batch_idx, data_dict))
    if not batches:
        raise RuntimeError("No batches were loaded for assignment audit.")

    all_rows = []
    for config_path in args.configs:
        all_rows.extend(audit_config(config_path, batches, args.split, device))

    json_path, csv_path = write_outputs(all_rows, Path(args.out))
    print(f"Wrote {len(all_rows)} audit rows")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
