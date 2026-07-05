import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np


ALLOCATOR_VERSION = "bca_v1_triplet_coverage"
AXIS_VERSION = "dense_window_v1"


def stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def safe_video_cache_name(video_name):
    return str(video_name).replace("/", "_").replace("\\", "_")


def selection_target_unit_count(target_frame_num, group_size=1):
    target_frame_num = int(max(target_frame_num, 0))
    group_size = int(max(group_size, 1))
    if target_frame_num == 0:
        return 0
    if group_size == 1:
        return target_frame_num
    return max(target_frame_num // group_size, 1)


def num_selection_units(valid_len, group_size=1):
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    return int(math.ceil(valid_len / float(group_size)))


def positions_to_selection_units(positions, valid_len, group_size=1):
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size == 0:
        return np.zeros((0,), dtype=np.int64)
    group_size = int(max(group_size, 1))
    if group_size == 1:
        units = positions
    else:
        units = positions // group_size
    unit_count = num_selection_units(valid_len, group_size)
    units = units[(units >= 0) & (units < unit_count)]
    return np.sort(np.unique(units.astype(np.int64)))


def expand_selection_units(selected_units, valid_len, group_size=1):
    selected_units = np.asarray(selected_units, dtype=np.int64).reshape(-1)
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    if selected_units.size == 0 or valid_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    unit_count = num_selection_units(valid_len, group_size)
    selected_units = selected_units[(selected_units >= 0) & (selected_units < unit_count)]
    selected_units = np.sort(np.unique(selected_units.astype(np.int64)))
    if selected_units.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if group_size == 1:
        return selected_units

    expanded = []
    for unit in selected_units:
        start = int(unit) * group_size
        end = min(valid_len, start + group_size)
        if start < end:
            expanded.append(np.arange(start, end, dtype=np.int64))
    if not expanded:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(expanded).astype(np.int64)


def dense_index_to_time_span(dense_idx, dense_window=None, fps=None, frame_stride=1):
    dense_idx = int(dense_idx)
    if dense_window is None:
        frame_start = dense_idx * int(max(frame_stride, 1))
        frame_end = frame_start + int(max(frame_stride, 1))
    else:
        dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
        frame_start = int(dense_window[int(np.clip(dense_idx, 0, max(dense_window.size - 1, 0)))])
        if dense_idx + 1 < dense_window.size:
            frame_end = int(dense_window[dense_idx + 1])
        else:
            frame_end = frame_start + int(max(frame_stride, 1))
    if fps is None or float(fps) <= 0:
        return frame_start, frame_end, None, None
    sec_start = float(frame_start) / float(fps)
    sec_end = float(frame_end) / float(fps)
    return frame_start, frame_end, sec_start, sec_end


def normalize_boundary_scores(boundary_scores, valid_len):
    valid_len = int(valid_len)
    scores = np.zeros(valid_len, dtype=np.float32)
    if boundary_scores is None or valid_len <= 0:
        return scores

    raw = np.asarray(boundary_scores, dtype=np.float32).reshape(-1)
    if raw.size >= valid_len:
        scores = raw[:valid_len].copy()
    else:
        scores[: raw.size] = raw
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if scores.size == 0:
        return scores
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores)
    if finite.min() < -1e-6 or finite.max() > 1.0 + 1e-6:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = float(finite.max())
            lo = float(finite.min())
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def aggregate_unit_scores(scores, valid_len, group_size=1):
    scores = normalize_boundary_scores(scores, valid_len)
    group_size = int(max(group_size, 1))
    unit_count = num_selection_units(valid_len, group_size)
    unit_scores = np.zeros(unit_count, dtype=np.float32)
    unit_peak_dense = np.zeros(unit_count, dtype=np.int64)
    for unit in range(unit_count):
        start = unit * group_size
        end = min(valid_len, start + group_size)
        if start >= end:
            continue
        local = scores[start:end]
        local_argmax = int(np.argmax(local))
        unit_scores[unit] = float(local[local_argmax])
        unit_peak_dense[unit] = start + local_argmax
    return unit_scores, unit_peak_dense


def slice_global_scores_for_window(scores, global_indices):
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return None
    indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
    window_scores = np.zeros(indices.shape[0], dtype=np.float32)
    valid = (indices >= 0) & (indices < scores.size)
    if np.any(valid):
        window_scores[valid] = scores[indices[valid]]
    return window_scores


def load_bata_boundary_scores(cache_dir, video_name, allow_diagnostic_gt_cache=False):
    if cache_dir in (None, ""):
        return None, {}

    manifest = {}
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"BATA score cache requires manifest.json: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    required = {
        "cache_version",
        "axis",
        "score_source",
        "uses_gt",
        "diagnostic_only",
        "snippet_stride",
        "axis_frame_stride",
        "scale_factor",
    }
    missing = sorted(required.difference(manifest.keys()))
    if missing:
        raise ValueError(f"BATA score cache manifest missing required fields {missing}: {manifest_path}")
    axis = manifest.get("axis", "global_snippet_index")
    if axis != "global_snippet_index":
        raise ValueError(f"unsupported BATA score cache axis: {axis}")
    uses_gt = bool(manifest.get("uses_gt", False))
    diagnostic_only = bool(manifest.get("diagnostic_only", False))
    if uses_gt and (not allow_diagnostic_gt_cache or not diagnostic_only):
        raise ValueError(
            "BATA GT-derived score cache is diagnostic-only; set "
            "bata_allow_diagnostic_gt_cache=True only for explicitly labeled oracle/noisy diagnostics"
        )

    npz_path = os.path.join(cache_dir, f"{safe_video_cache_name(video_name)}.npz")
    if not os.path.exists(npz_path):
        return None, manifest
    with np.load(npz_path) as data:
        if "boundary_score" not in data:
            return None, manifest
        scores = np.asarray(data["boundary_score"], dtype=np.float32).reshape(-1)
    return scores, manifest


def validate_bata_cache_manifest_for_loader(
    manifest,
    frame_stride,
    scale_factor,
    loader_diagnostic_only=False,
):
    manifest = {} if manifest is None else dict(manifest)
    manifest_uses_gt = bool(manifest.get("uses_gt", False))
    manifest_diagnostic_only = bool(manifest.get("diagnostic_only", False))
    if manifest_uses_gt and not bool(loader_diagnostic_only):
        raise ValueError("BATA GT-derived score cache requires bata_diagnostic_only=True in the LoadFrames config")

    expected_cache_stride = int(manifest.get("axis_frame_stride", manifest.get("snippet_stride", frame_stride)))
    if expected_cache_stride != int(frame_stride):
        raise ValueError(
            "BATA score cache axis stride mismatch: "
            f"manifest axis_frame_stride={expected_cache_stride}, loader frame_stride={int(frame_stride)}"
        )

    manifest_scale_factor = int(manifest.get("scale_factor", scale_factor))
    if manifest_scale_factor != int(scale_factor):
        raise ValueError(
            "BATA score cache scale_factor mismatch: "
            f"manifest scale_factor={manifest_scale_factor}, loader scale_factor={int(scale_factor)}"
        )
    return bool(loader_diagnostic_only or manifest_diagnostic_only)


@dataclass
class BcaConfig:
    boundary_cap_ratio: float = 0.60
    coverage_floor_ratio: float = 0.20
    peak_nms_radius: int = 8
    local_triplet_radius: int = 4
    fallback_radii: tuple = (2, 6)
    max_peaks: int = 8
    dispersion_bins: int = 8
    coverage_max_gap_target: int = 8
    context_low_score_search_radius: int = 2
    reclaim_boundary_support_for_coverage: bool = True
    flat_score_eps: float = 1e-6


@dataclass
class BcaSelectionResult:
    selected_dense_indices: np.ndarray
    selected_units: np.ndarray
    selection_ledger: list
    boundary_peak_ledger: list
    coverage_diagnostics: dict
    ordered_candidate_dense_indices: np.ndarray


def _to_unit_radius(radius, group_size):
    return max(int(math.ceil(float(max(radius, 0)) / float(max(group_size, 1)))), 1)


def _coverage_gap_stats(selected_dense_indices):
    selected = np.sort(np.asarray(selected_dense_indices, dtype=np.int64).reshape(-1))
    if selected.size <= 1:
        return dict(p50_gap=0.0, p95_gap=0.0, max_gap=0)
    gaps = np.diff(selected)
    return dict(
        p50_gap=float(np.percentile(gaps, 50)),
        p95_gap=float(np.percentile(gaps, 95)),
        max_gap=int(gaps.max()),
    )


def _largest_gap_midpoint(selected_units, unit_count):
    selected_units = np.sort(np.asarray(selected_units, dtype=np.int64).reshape(-1))
    if selected_units.size == 0:
        return unit_count // 2
    anchors = np.concatenate([np.array([-1], dtype=np.int64), selected_units, np.array([unit_count], dtype=np.int64)])
    gaps = anchors[1:] - anchors[:-1]
    gap_idx = int(np.argmax(gaps))
    left = int(anchors[gap_idx])
    right = int(anchors[gap_idx + 1])
    if right - left <= 1:
        return None
    return int((left + right) // 2)


def _largest_dense_gap_midpoint_unit(selected_units, valid_len, group_size):
    selected_dense = expand_selection_units(selected_units, valid_len, group_size)
    if selected_dense.size == 0:
        return num_selection_units(valid_len, group_size) // 2
    anchors = np.concatenate(
        [
            np.array([-1], dtype=np.int64),
            np.sort(selected_dense.astype(np.int64)),
            np.array([int(valid_len)], dtype=np.int64),
        ]
    )
    gaps = anchors[1:] - anchors[:-1]
    gap_idx = int(np.argmax(gaps))
    left = int(anchors[gap_idx])
    right = int(anchors[gap_idx + 1])
    if right - left <= 1:
        return None
    midpoint_dense = int((left + right) // 2)
    return int(np.clip(midpoint_dense // int(max(group_size, 1)), 0, max(num_selection_units(valid_len, group_size) - 1, 0)))


def _choose_low_score_near(midpoint, selected_set, unit_scores, search_radius, excluded_units=None):
    unit_count = int(unit_scores.size)
    midpoint = int(np.clip(midpoint, 0, max(unit_count - 1, 0)))
    radius = int(max(search_radius, 0))
    excluded_units = set() if excluded_units is None else {int(unit) for unit in excluded_units}
    candidates = []
    for unit in range(max(0, midpoint - radius), min(unit_count, midpoint + radius + 1)):
        if unit in selected_set or unit in excluded_units:
            continue
        candidates.append((float(unit_scores[unit]), abs(unit - midpoint), unit))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return int(candidates[0][2])

    available = [unit for unit in range(unit_count) if unit not in selected_set and unit not in excluded_units]
    if not available:
        return None
    available.sort(key=lambda unit: (abs(unit - midpoint), unit))
    return int(available[0])


def select_bata_boundary_acquisition_positions(
    valid_len,
    target_frame_num,
    boundary_scores=None,
    sample_key="unknown",
    group_size=1,
    config=None,
    run_id="local",
    dataset="unknown",
    split="unknown",
    video_id="unknown",
    dense_window=None,
    fps=None,
    frame_stride=1,
    score_source="unknown",
    diagnostic_only=False,
    config_hash="",
):
    valid_len = int(valid_len)
    target_frame_num = int(target_frame_num)
    group_size = int(max(group_size, 1))
    cfg = config or BcaConfig()

    if valid_len <= 0 or target_frame_num <= 0:
        return BcaSelectionResult(
            selected_dense_indices=np.zeros((0,), dtype=np.int64),
            selected_units=np.zeros((0,), dtype=np.int64),
            selection_ledger=[],
            boundary_peak_ledger=[],
            coverage_diagnostics={},
            ordered_candidate_dense_indices=np.zeros((0,), dtype=np.int64),
        )

    unit_count = num_selection_units(valid_len, group_size)
    target_units = min(selection_target_unit_count(target_frame_num, group_size), unit_count)
    unit_scores, unit_peak_dense = aggregate_unit_scores(boundary_scores, valid_len, group_size)
    normalized_scores = normalize_boundary_scores(boundary_scores, valid_len)
    is_flat = bool(float(unit_scores.max() - np.median(unit_scores)) < float(cfg.flat_score_eps))

    k_cov_min = int(math.ceil(float(cfg.coverage_floor_ratio) * target_units))
    k_boundary_cap = min(int(math.floor(float(cfg.boundary_cap_ratio) * target_units)), target_units - k_cov_min)
    p_max = min(int(cfg.max_peaks), int(k_boundary_cap // 3))
    nms_radius = _to_unit_radius(cfg.peak_nms_radius, group_size)
    triplet_radius = _to_unit_radius(cfg.local_triplet_radius, group_size)
    fallback_radii = [_to_unit_radius(radius, group_size) for radius in tuple(cfg.fallback_radii)]
    context_radius = _to_unit_radius(cfg.context_low_score_search_radius, group_size)

    selected = {}
    selected_order = []
    peak_rows = []
    packet_id = 0
    accepted_peaks = []
    candidate_units = np.argsort(-unit_scores, kind="mergesort")
    ordered_candidate_dense = unit_peak_dense[candidate_units]

    def add_unit(unit, role, peak_id=None, packet=None, fallback_used=False, priority=2):
        unit = int(unit)
        if unit < 0 or unit >= unit_count or unit in selected:
            return False
        selected[unit] = dict(
            primary_role=role,
            peak_id=peak_id,
            packet_id=packet,
            fallback_used=bool(fallback_used),
            reason_priority=int(priority),
        )
        selected_order.append(unit)
        return True

    if not is_flat and p_max > 0:
        for unit in candidate_units.tolist():
            unit = int(unit)
            if any(abs(unit - prev) <= nms_radius for prev in accepted_peaks):
                continue
            accepted_peaks.append(unit)
            if len(accepted_peaks) >= p_max:
                break

        for peak_id, peak_unit in enumerate(accepted_peaks):
            packet = packet_id
            packet_id += 1
            selected_before = None
            selected_center = None
            selected_after = None

            for offset, role in ((-triplet_radius, "boundary_before"), (0, "boundary_center"), (triplet_radius, "boundary_after")):
                candidate = int(peak_unit + offset)
                fallback_used = False
                if not add_unit(candidate, role, peak_id=peak_id, packet=packet, fallback_used=False, priority=0 if role == "boundary_center" else 1):
                    if role != "boundary_center":
                        sign = -1 if offset < 0 else 1
                        for radius in fallback_radii:
                            fallback_candidate = int(peak_unit + sign * radius)
                            if add_unit(
                                fallback_candidate,
                                role,
                                peak_id=peak_id,
                                packet=packet,
                                fallback_used=True,
                                priority=1,
                            ):
                                candidate = fallback_candidate
                                fallback_used = True
                                break
                if role == "boundary_before" and candidate in selected:
                    selected_before = candidate
                elif role == "boundary_center" and candidate in selected:
                    selected_center = candidate
                elif role == "boundary_after" and candidate in selected:
                    selected_after = candidate

            suppressed = sum(1 for unit in candidate_units.tolist() if abs(int(unit) - int(peak_unit)) <= nms_radius) - 1
            _, _, peak_sec_start, peak_sec_end = dense_index_to_time_span(
                int(unit_peak_dense[peak_unit]),
                dense_window=dense_window,
                fps=fps,
                frame_stride=frame_stride,
            )
            peak_rows.append(
                dict(
                    run_id=run_id,
                    video_id=video_id,
                    peak_id=peak_id,
                    peak_dense_idx=int(unit_peak_dense[peak_unit]),
                    peak_sec=peak_sec_start,
                    peak_sec_end=peak_sec_end,
                    peak_score=float(unit_scores[peak_unit]),
                    nms_rank=peak_id,
                    nms_radius=int(cfg.peak_nms_radius),
                    triplet_radius=int(cfg.local_triplet_radius),
                    selected_before_idx=None if selected_before is None else int(unit_peak_dense[selected_before]),
                    selected_center_idx=None if selected_center is None else int(unit_peak_dense[selected_center]),
                    selected_after_idx=None if selected_after is None else int(unit_peak_dense[selected_after]),
                    suppressed_peak_count=int(max(suppressed, 0)),
                    dispersion_bin_id=int(min(cfg.dispersion_bins - 1, math.floor(peak_unit / max(unit_count, 1) * cfg.dispersion_bins))),
                )
            )

    max_gap_target_dense = int(cfg.coverage_max_gap_target)
    p95_gap_target_dense = max(
        4,
        int(math.ceil(1.5 * valid_len / max(target_frame_num, 1))),
        int(group_size) + 1 if int(group_size) > 1 else 0,
    )

    while len(selected) < target_units:
        midpoint = _largest_dense_gap_midpoint_unit(
            np.asarray(list(selected.keys()), dtype=np.int64),
            valid_len,
            group_size,
        )
        if midpoint is None:
            break
        dense_stats = _coverage_gap_stats(
            expand_selection_units(np.asarray(list(selected.keys()), dtype=np.int64), valid_len, group_size)
        )
        coverage_radius = 0 if dense_stats["max_gap"] > max_gap_target_dense else context_radius
        chosen = _choose_low_score_near(midpoint, set(selected.keys()), unit_scores, coverage_radius)
        if chosen is None:
            break
        add_unit(chosen, "coverage_context", peak_id=None, packet=None, fallback_used=False, priority=2)

    reclaimed = 0
    if cfg.reclaim_boundary_support_for_coverage:
        while selected:
            dense_stats = _coverage_gap_stats(expand_selection_units(np.asarray(list(selected.keys()), dtype=np.int64), valid_len, group_size))
            if dense_stats["max_gap"] <= max_gap_target_dense and dense_stats["p95_gap"] <= p95_gap_target_dense:
                break

            removable = [
                (data["reason_priority"], float(unit_scores[unit]), unit)
                for unit, data in selected.items()
                if data["primary_role"] in ("boundary_before", "boundary_after")
            ]
            if not removable:
                break
            removable.sort(key=lambda item: (-item[0], item[1], item[2]))
            _, _, remove_unit = removable[0]
            del selected[remove_unit]
            reclaimed += 1

            midpoint = _largest_dense_gap_midpoint_unit(
                np.asarray(list(selected.keys()), dtype=np.int64),
                valid_len,
                group_size,
            )
            if midpoint is None:
                break
            dense_stats = _coverage_gap_stats(
                expand_selection_units(np.asarray(list(selected.keys()), dtype=np.int64), valid_len, group_size)
            )
            coverage_radius = 0 if dense_stats["max_gap"] > max_gap_target_dense else context_radius
            chosen = _choose_low_score_near(
                midpoint,
                set(selected.keys()),
                unit_scores,
                coverage_radius,
                excluded_units={remove_unit},
            )
            if chosen is None:
                break
            add_unit(chosen, "coverage_repair", peak_id=None, packet=None, fallback_used=False, priority=2)

    for _ in range(max(target_units * 2, 1)):
        dense_stats = _coverage_gap_stats(
            expand_selection_units(np.asarray(list(selected.keys()), dtype=np.int64), valid_len, group_size)
        )
        if dense_stats["max_gap"] <= max_gap_target_dense and dense_stats["p95_gap"] <= p95_gap_target_dense:
            break
        midpoint = _largest_dense_gap_midpoint_unit(
            np.asarray(list(selected.keys()), dtype=np.int64),
            valid_len,
            group_size,
        )
        if midpoint is None:
            break
        add_candidate = _choose_low_score_near(midpoint, set(selected.keys()), unit_scores, 0)
        if add_candidate is None:
            break

        best = None
        for remove_unit, remove_data in list(selected.items()):
            if remove_unit == add_candidate or remove_data.get("primary_role") == "boundary_center":
                continue
            trial_units = [unit for unit in selected.keys() if unit != remove_unit] + [add_candidate]
            trial_stats = _coverage_gap_stats(expand_selection_units(np.asarray(trial_units, dtype=np.int64), valid_len, group_size))
            candidate = (
                int(trial_stats["max_gap"]),
                float(trial_stats["p95_gap"]),
                int(remove_data.get("reason_priority", 9)),
                float(unit_scores[remove_unit]),
                int(remove_unit),
                trial_stats,
            )
            if best is None or candidate < best:
                best = candidate

        if best is None:
            break
        current_key = (int(dense_stats["max_gap"]), float(dense_stats["p95_gap"]))
        best_key = (best[0], best[1])
        if best_key >= current_key:
            break
        remove_unit = int(best[4])
        del selected[remove_unit]
        add_unit(add_candidate, "coverage_repair", peak_id=None, packet=None, fallback_used=False, priority=2)

    selected_units = np.sort(np.asarray(list(selected.keys()), dtype=np.int64))
    selected_dense = expand_selection_units(selected_units, valid_len, group_size)
    stats = _coverage_gap_stats(selected_dense)
    coverage_violation = bool(
        stats["p95_gap"] > max(
            4,
            math.ceil(1.5 * valid_len / max(target_frame_num, 1)),
            int(group_size) + 1 if int(group_size) > 1 else 0,
        )
        or stats["max_gap"] > int(cfg.coverage_max_gap_target)
    )

    dispersion_hist = np.zeros(int(cfg.dispersion_bins), dtype=np.int64)
    if selected_dense.size > 0:
        bin_ids = np.minimum((selected_dense.astype(np.float64) / max(valid_len, 1) * int(cfg.dispersion_bins)).astype(np.int64), int(cfg.dispersion_bins) - 1)
        for bin_id in bin_ids.tolist():
            dispersion_hist[int(bin_id)] += 1

    selection_rows = []
    unit_rank = {unit: rank for rank, unit in enumerate(selected_order)}
    for dense_idx in selected_dense.tolist():
        unit = int(dense_idx // group_size)
        data = selected.get(unit, {})
        frame_start, frame_end, sec_start, sec_end = dense_index_to_time_span(
            dense_idx,
            dense_window=dense_window,
            fps=fps,
            frame_stride=frame_stride,
        )
        row = dict(
            run_id=run_id,
            dataset=dataset,
            split=split,
            video_id=video_id,
            axis_version=AXIS_VERSION,
            allocator_version=ALLOCATOR_VERSION,
            config_hash=config_hash,
            score_source=score_source,
            diagnostic_only=bool(diagnostic_only),
            T_dense=int(valid_len),
            target_k=int(target_frame_num),
            target_units=int(target_units),
            group_size=int(group_size),
            dense_idx=int(dense_idx),
            selected_rank=int(unit_rank.get(unit, len(unit_rank))),
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            sec_start=sec_start,
            sec_end=sec_end,
            primary_role=data.get("primary_role", "unknown"),
            peak_id=data.get("peak_id"),
            packet_id=data.get("packet_id"),
            boundary_score=float(normalized_scores[int(dense_idx)]),
            reason_priority=int(data.get("reason_priority", 9)),
            fallback_used=bool(data.get("fallback_used", False)),
            seed=int(stable_string_seed(sample_key)),
        )
        selection_rows.append(row)

    coverage = dict(
        run_id=run_id,
        video_id=video_id,
        T_dense=int(valid_len),
        target_k=int(target_frame_num),
        target_units=int(target_units),
        group_size=int(group_size),
        num_selected=int(selected_dense.size),
        num_selected_units=int(selected_units.size),
        num_boundary_tokens=int(sum(1 for row in selection_rows if str(row["primary_role"]).startswith("boundary"))),
        num_coverage_tokens=int(sum(1 for row in selection_rows if str(row["primary_role"]).startswith("coverage"))),
        num_reclaimed_tokens=int(reclaimed),
        p50_gap=stats["p50_gap"],
        p95_gap=stats["p95_gap"],
        max_gap=stats["max_gap"],
        coverage_violation=coverage_violation,
        dispersion_hist_8bins=dispersion_hist.tolist(),
        flat_score_fallback=bool(is_flat),
    )

    return BcaSelectionResult(
        selected_dense_indices=selected_dense.astype(np.int64),
        selected_units=selected_units.astype(np.int64),
        selection_ledger=selection_rows,
        boundary_peak_ledger=peak_rows,
        coverage_diagnostics=coverage,
        ordered_candidate_dense_indices=ordered_candidate_dense.astype(np.int64),
    )


def build_oracle_boundary_scores(valid_len, gt_segments, label_radius=4):
    valid_len = int(valid_len)
    scores = np.zeros(valid_len, dtype=np.float32)
    if valid_len <= 0 or gt_segments is None:
        return scores
    gt_segments = np.asarray(gt_segments, dtype=np.float32).reshape(-1, 2)
    radius = int(max(label_radius, 0))
    for start, end in gt_segments.tolist():
        for boundary in (start, end):
            center = int(round(boundary))
            lo = max(0, center - radius)
            hi = min(valid_len, center + radius + 1)
            for idx in range(lo, hi):
                value = 1.0 if radius == 0 else max(0.0, 1.0 - abs(idx - center) / float(radius + 1))
                scores[idx] = max(scores[idx], value)
    return scores


def build_noisy_oracle_boundary_scores(
    valid_len,
    gt_segments,
    label_radius=4,
    jitter_std=2.0,
    miss_rate=0.20,
    false_peak_ratio=0.50,
    noise_std=0.10,
    seed_key="noisy_oracle",
):
    valid_len = int(valid_len)
    scores = np.zeros(valid_len, dtype=np.float32)
    if valid_len <= 0:
        return scores
    gt_segments = np.asarray(gt_segments, dtype=np.float32).reshape(-1, 2) if gt_segments is not None else np.zeros((0, 2), dtype=np.float32)
    rng = np.random.RandomState(stable_string_seed(seed_key))
    noisy_boundaries = []
    for start, end in gt_segments.tolist():
        for boundary in (start, end):
            if rng.rand() < float(miss_rate):
                continue
            noisy = int(round(float(boundary) + rng.normal(0.0, float(jitter_std))))
            noisy_boundaries.append(int(np.clip(noisy, 0, valid_len - 1)))

    false_count = int(round(float(false_peak_ratio) * max(gt_segments.shape[0] * 2, 1)))
    for _ in range(false_count):
        noisy_boundaries.append(int(rng.randint(0, valid_len)))

    for boundary in noisy_boundaries:
        lo = max(0, boundary - int(label_radius))
        hi = min(valid_len, boundary + int(label_radius) + 1)
        for idx in range(lo, hi):
            dist = abs(idx - boundary)
            value = math.exp(-(dist * dist) / (2.0 * max(float(jitter_std), 1e-6) ** 2))
            scores[idx] = max(scores[idx], float(value))
    if noise_std > 0:
        scores = scores + rng.normal(0.0, float(noise_std), size=valid_len).astype(np.float32)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def compute_eval_boundary_match(selection_rows, gt_segments, hit_radii=(1, 2, 4), run_id="local", video_id="unknown"):
    selected = np.asarray([row["dense_idx"] for row in selection_rows], dtype=np.int64)
    boundary_selected = np.asarray(
        [row["dense_idx"] for row in selection_rows if str(row.get("primary_role", "")).startswith("boundary")],
        dtype=np.int64,
    )
    gt_segments = np.asarray(gt_segments, dtype=np.float32).reshape(-1, 2) if gt_segments is not None else np.zeros((0, 2), dtype=np.float32)
    rows = []
    for gt_idx, (start, end) in enumerate(gt_segments.tolist()):
        for boundary_type, boundary in (("start", start), ("end", end)):
            gt_dense_idx = int(round(boundary))
            nearest_any = None
            nearest_boundary = None
            dist_any = None
            dist_boundary = None
            if selected.size > 0:
                dist = np.abs(selected - gt_dense_idx)
                pos = int(np.argmin(dist))
                nearest_any = int(selected[pos])
                dist_any = int(dist[pos])
            if boundary_selected.size > 0:
                dist = np.abs(boundary_selected - gt_dense_idx)
                pos = int(np.argmin(dist))
                nearest_boundary = int(boundary_selected[pos])
                dist_boundary = int(dist[pos])
            row = dict(
                run_id=run_id,
                video_id=video_id,
                gt_instance_id=int(gt_idx),
                gt_boundary_type=boundary_type,
                gt_dense_idx=gt_dense_idx,
                nearest_selected_dense_idx=nearest_any,
                nearest_boundary_role_dense_idx=nearest_boundary,
                dist_to_any_selected=dist_any,
                dist_to_boundary_evidence=dist_boundary,
            )
            for radius in hit_radii:
                row[f"hit_at_{int(radius)}"] = bool(dist_any is not None and dist_any <= int(radius))
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
