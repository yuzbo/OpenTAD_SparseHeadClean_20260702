#!/usr/bin/env python
"""Verify bridge hard dense-equivalence against official dense semantics.

This is a small synthetic sanity verifier, not a training entrypoint. It
constructs dense/uniform point grids and compares the real
IrregularActionFormerBridgeHead hard target/decode path against an independent
implementation of the official dense ActionFormer target contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = ("stride1_dense_open_range", "multi_level_range_gate")


def _import_torch():
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - exercised only in missing envs.
        raise RuntimeError(f"torch is required to run bridge dense-equivalence verification: {exc}") from exc
    return torch, F


def _import_bridge_head():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from opentad.models.dense_heads.irregular_actionformer_bridge_head import IrregularActionFormerBridgeHead
    except Exception as exc:  # pragma: no cover - dependency failures are environment-specific.
        raise RuntimeError(f"failed to import IrregularActionFormerBridgeHead: {exc}") from exc
    return IrregularActionFormerBridgeHead


def _new_bridge_head(num_classes):
    BridgeHead = _import_bridge_head()
    head = object.__new__(BridgeHead)
    head.num_classes = int(num_classes)
    head.center_sample = "radius"
    head.center_sample_radius = 1.5
    head.assignment_mode = "hard"
    head.regression_mode = "symmetric_linear"
    head.reg_denom_floor = 0.5
    head.center_radius_scale = "point_radius"
    head.reg_denom_mode = "left_right_mean"
    head.filter_similar_gt = True
    head.debug_enabled = False
    return head


def _tensor(values, dtype=None):
    torch, _ = _import_torch()
    dtype = torch.float32 if dtype is None else dtype
    return torch.tensor(values, dtype=dtype)


def _make_level(center_values, stride, reg_range):
    torch, _ = _import_torch()
    centers = _tensor(center_values)
    strides = torch.full_like(centers, float(stride))
    reg_min = torch.full_like(centers, float(reg_range[0]))
    reg_max = torch.full_like(centers, float(reg_range[1]))
    official = torch.stack([centers, reg_min, reg_max, strides], dim=-1)

    # Bridge field layout:
    # [center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale].
    # For dense equivalence, decode/radius/linear denominator all equal level stride.
    bridge = torch.stack([centers, reg_min, reg_max, strides, strides, strides, strides], dim=-1)
    return official, bridge


def official_dense_targets(
    points,
    gt_segments,
    gt_labels,
    num_classes,
    center_sample_radius=1.5,
    filter_similar_gt=True,
    use_regress_range=True,
):
    """Independent implementation of AnchorFreeHead.prepare_targets semantics."""

    torch, F = _import_torch()
    concat_points = torch.cat(points, dim=0)
    num_pts = concat_points.shape[0]
    num_gts = gt_segments.shape[0]
    if num_gts == 0:
        return {
            "cls": gt_segments.new_zeros((num_pts, num_classes)),
            "reg": gt_segments.new_zeros((num_pts, 2)),
            "reg_weight": gt_segments.new_zeros((num_pts,)),
            "positive": torch.zeros((num_pts,), dtype=torch.bool, device=concat_points.device),
            "assigned_gt": torch.full((num_pts,), -1, dtype=torch.long, device=concat_points.device),
            "decoded": gt_segments.new_zeros((num_pts, 2)),
        }

    lens = (gt_segments[:, 1] - gt_segments[:, 0])[None, :].repeat(num_pts, 1)
    gt_segs = gt_segments[None].expand(num_pts, num_gts, 2)
    left = concat_points[:, 0, None] - gt_segs[:, :, 0]
    right = gt_segs[:, :, 1] - concat_points[:, 0, None]
    reg_targets = torch.stack((left, right), dim=-1)

    center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
    t_mins = center_pts - concat_points[:, 3, None] * float(center_sample_radius)
    t_maxs = center_pts + concat_points[:, 3, None] * float(center_sample_radius)
    cb_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
    cb_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
    center_seg = torch.stack((cb_left, cb_right), dim=-1)
    inside_gt_seg_mask = center_seg.min(dim=-1).values > 0

    max_regress_distance = reg_targets.max(dim=-1).values
    if use_regress_range:
        inside_regress_range = torch.logical_and(
            max_regress_distance >= concat_points[:, 1, None],
            max_regress_distance <= concat_points[:, 2, None],
        )
    else:
        inside_regress_range = torch.ones_like(inside_gt_seg_mask)

    lens = lens.clone()
    lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
    lens.masked_fill_(inside_regress_range == 0, float("inf"))
    min_len, min_len_inds = lens.min(dim=1)

    if filter_similar_gt:
        min_len_mask = torch.logical_and(lens <= (min_len[:, None] + 1e-3), lens < float("inf"))
    else:
        min_len_mask = lens < float("inf")
    min_len_mask = min_len_mask.to(reg_targets.dtype)

    one_hot = F.one_hot(gt_labels.long(), num_classes).to(reg_targets.dtype)
    cls_targets = min_len_mask @ one_hot
    cls_targets.clamp_(min=0.0, max=1.0)

    reg_target = reg_targets[torch.arange(num_pts, device=concat_points.device), min_len_inds]
    reg_target = reg_target / concat_points[:, 3, None]
    positive = min_len < float("inf")
    assigned_gt = torch.where(
        positive,
        min_len_inds,
        torch.full_like(min_len_inds, -1),
    )
    decoded = torch.stack(
        (
            concat_points[:, 0] - reg_target[:, 0] * concat_points[:, 3],
            concat_points[:, 0] + reg_target[:, 1] * concat_points[:, 3],
        ),
        dim=-1,
    )
    decoded = torch.where(positive[:, None], decoded, decoded.new_zeros(decoded.shape))
    reg_target = torch.where(positive[:, None], reg_target, reg_target.new_zeros(reg_target.shape))

    return {
        "cls": cls_targets,
        "reg": reg_target,
        "reg_weight": positive.to(reg_targets.dtype),
        "positive": cls_targets.sum(dim=-1) > 0,
        "assigned_gt": assigned_gt,
        "decoded": decoded,
    }


def _bridge_targets_and_decode(bridge_points, gt_segments, gt_labels, num_classes):
    torch, _ = _import_torch()
    head = _new_bridge_head(num_classes)
    gt_cls, gt_reg, reg_weight, _ = head._prepare_targets_hard(bridge_points, [gt_segments], [gt_labels])
    cls_targets = gt_cls[0]
    reg_targets = gt_reg[0]
    weights = reg_weight[0]

    reg_pred = []
    start = 0
    for level_points in bridge_points:
        length = int(level_points.shape[0])
        reg_pred.append(reg_targets[start : start + length].transpose(0, 1).unsqueeze(0))
        start += length
    decoded = head.get_refined_proposals(bridge_points, reg_pred)[0]
    decoded = torch.where(weights[:, None] > 0, decoded, decoded.new_zeros(decoded.shape))

    assigned = torch.full((reg_targets.shape[0],), -1, dtype=torch.long, device=reg_targets.device)
    for point_idx in torch.nonzero(weights > 0, as_tuple=False).flatten().tolist():
        matches = torch.isclose(decoded[point_idx][None, :], gt_segments, atol=1e-5, rtol=1e-5).all(dim=1)
        match_idx = torch.nonzero(matches, as_tuple=False).flatten()
        if match_idx.numel() > 0:
            assigned[point_idx] = match_idx[0]
        else:
            assigned[point_idx] = -2

    return {
        "cls": cls_targets,
        "reg": reg_targets,
        "reg_weight": weights,
        "positive": cls_targets.sum(dim=-1) > 0,
        "assigned_gt": assigned,
        "decoded": decoded,
    }


def _point_records(official_points):
    torch, _ = _import_torch()
    records = []
    offset = 0
    for level_idx, level_points in enumerate(official_points):
        for local_idx in range(level_points.shape[0]):
            records.append(
                {
                    "point": int(offset + local_idx),
                    "level": int(level_idx),
                    "local": int(local_idx),
                    "center": float(level_points[local_idx, 0].item()),
                    "stride": float(level_points[local_idx, 3].item()),
                    "range": [
                        float(level_points[local_idx, 1].item()),
                        float(level_points[local_idx, 2].item()),
                    ],
                }
            )
        offset += int(level_points.shape[0])
    return records


def _as_list(tensor):
    return tensor.detach().cpu().tolist()


def _max_abs(tensor):
    return float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0


def _add_indexed_mismatch(mismatches, kind, mask, point_records, official_value, bridge_value):
    torch, _ = _import_torch()
    bad_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not bad_indices:
        return
    mismatches.append(
        {
            "kind": kind,
            "count": len(bad_indices),
            "examples": [
                {
                    **point_records[idx],
                    "official": _as_list(official_value[idx]),
                    "bridge": _as_list(bridge_value[idx]),
                }
                for idx in bad_indices[:8]
            ],
        }
    )


def compare_bridge_to_official(case, atol=1e-6):
    torch, _ = _import_torch()
    official = official_dense_targets(
        case["official_points"],
        case["gt_segments"],
        case["gt_labels"],
        case["num_classes"],
    )
    bridge = _bridge_targets_and_decode(
        case["bridge_points"],
        case["gt_segments"],
        case["gt_labels"],
        case["num_classes"],
    )

    records = _point_records(case["official_points"])
    mismatches = []

    _add_indexed_mismatch(
        mismatches,
        "positive_mask",
        official["positive"] != bridge["positive"],
        records,
        official["positive"].to(torch.int64),
        bridge["positive"].to(torch.int64),
    )
    _add_indexed_mismatch(
        mismatches,
        "assigned_gt",
        official["assigned_gt"] != bridge["assigned_gt"],
        records,
        official["assigned_gt"],
        bridge["assigned_gt"],
    )
    _add_indexed_mismatch(
        mismatches,
        "classification_target",
        (official["cls"] - bridge["cls"]).abs().max(dim=-1).values > atol,
        records,
        official["cls"],
        bridge["cls"],
    )

    pos = official["positive"] | bridge["positive"]
    reg_error = (official["reg"] - bridge["reg"]).abs().max(dim=-1).values
    decode_error = (official["decoded"] - bridge["decoded"]).abs().max(dim=-1).values
    _add_indexed_mismatch(
        mismatches,
        "encoded_target",
        pos & (reg_error > atol),
        records,
        official["reg"],
        bridge["reg"],
    )
    _add_indexed_mismatch(
        mismatches,
        "decoded_proposal",
        pos & (decode_error > atol),
        records,
        official["decoded"],
        bridge["decoded"],
    )

    positive_count = int(official["positive"].sum().item())
    return {
        "name": case["name"],
        "ok": len(mismatches) == 0,
        "positive_count": positive_count,
        "bridge_positive_count": int(bridge["positive"].sum().item()),
        "assigned_gt": _as_list(official["assigned_gt"]),
        "bridge_assigned_gt": _as_list(bridge["assigned_gt"]),
        "encoded_max_abs_error": _max_abs((official["reg"] - bridge["reg"])[pos]),
        "decoded_max_abs_error": _max_abs((official["decoded"] - bridge["decoded"])[pos]),
        "mismatches": mismatches,
    }


def _case_stride1_dense_open_range():
    official_level, bridge_level = _make_level(range(8), stride=1.0, reg_range=(0.0, 10000.0))
    return {
        "name": "stride1_dense_open_range",
        "num_classes": 4,
        "official_points": [official_level],
        "bridge_points": [bridge_level],
        "gt_segments": _tensor([[2.0, 5.0], [1.0, 7.0]]),
        "gt_labels": _tensor([1, 2], dtype=_import_torch()[0].long),
    }


def _case_multi_level_range_gate():
    official_l0, bridge_l0 = _make_level(range(16), stride=1.0, reg_range=(0.0, 4.0))
    official_l1, bridge_l1 = _make_level([0.0, 4.0, 8.0, 12.0, 16.0], stride=4.0, reg_range=(4.0, 10000.0))
    return {
        "name": "multi_level_range_gate",
        "num_classes": 4,
        "official_points": [official_l0, official_l1],
        "bridge_points": [bridge_l0, bridge_l1],
        "gt_segments": _tensor([[2.0, 4.0], [2.0, 14.0]]),
        "gt_labels": _tensor([0, 1], dtype=_import_torch()[0].long),
    }


def build_cases():
    return [_case_stride1_dense_open_range(), _case_multi_level_range_gate()]


def run_all_checks(case_name=None, atol=1e-6):
    cases = build_cases()
    if case_name is not None:
        cases = [case for case in cases if case["name"] == case_name]
        if not cases:
            raise ValueError(f"unknown case {case_name!r}")

    results = [compare_bridge_to_official(case, atol=atol) for case in cases]
    return {
        "ok": all(result["ok"] for result in results),
        "cases": results,
    }


def _format_text(summary):
    lines = []
    for case in summary["cases"]:
        status = "OK" if case["ok"] else "FAIL"
        lines.append(
            "{status} {name}: positives official={pos} bridge={bpos}, "
            "encoded_max_abs_error={reg:.3g}, decoded_max_abs_error={dec:.3g}".format(
                status=status,
                name=case["name"],
                pos=case["positive_count"],
                bpos=case["bridge_positive_count"],
                reg=case["encoded_max_abs_error"],
                dec=case["decoded_max_abs_error"],
            )
        )
        for mismatch in case["mismatches"]:
            lines.append(f"  mismatch {mismatch['kind']} count={mismatch['count']}")
            for example in mismatch["examples"]:
                lines.append(
                    "    point={point} level={level} local={local} center={center} "
                    "official={official} bridge={bridge}".format(**example)
                )
    lines.append("overall: " + ("OK" if summary["ok"] else "FAIL"))
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, default=None)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_all_checks(case_name=args.case, atol=args.atol)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_text(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
