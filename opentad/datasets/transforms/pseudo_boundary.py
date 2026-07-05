import hashlib
import json
import os

import numpy as np


def stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _safe_video_cache_name(video_name):
    return str(video_name).replace("/", "_").replace("\\", "_")


def load_boundary_scores(cache_dir, video_name):
    if cache_dir in (None, ""):
        return None

    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("uses_gt", False):
            raise ValueError(f"pseudo-boundary cache must not use GT: {manifest_path}")
        axis = manifest.get("axis", "global_snippet_index")
        if axis != "global_snippet_index":
            raise ValueError(f"unsupported pseudo-boundary cache axis: {axis}")

    cache_name = _safe_video_cache_name(video_name)
    npz_path = os.path.join(cache_dir, f"{cache_name}.npz")
    if not os.path.exists(npz_path):
        return None

    with np.load(npz_path) as data:
        if "boundary_score" not in data:
            return None
        scores = np.asarray(data["boundary_score"], dtype=np.float32).reshape(-1)
    return scores


def slice_global_scores_for_window(scores, global_indices):
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return None

    indices = np.asarray(list(global_indices), dtype=np.int64).reshape(-1)
    window_scores = np.zeros(indices.shape[0], dtype=np.float32)
    valid = (indices >= 0) & (indices < scores.size)
    if np.any(valid):
        window_scores[valid] = scores[indices[valid]]
    return window_scores


def _target_unit_count(target_frame_num, group_size):
    target_frame_num = int(max(target_frame_num, 0))
    group_size = int(max(group_size, 1))
    if target_frame_num == 0:
        return 0
    if group_size == 1:
        return target_frame_num
    return max(target_frame_num // group_size, 1)


def _num_units(valid_len, group_size):
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    return int(np.ceil(valid_len / float(group_size)))


def expand_units(selected_units, valid_len, group_size=1):
    selected_units = np.asarray(selected_units, dtype=np.int64).reshape(-1)
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    if selected_units.size == 0 or valid_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    if group_size == 1:
        selected = selected_units[(selected_units >= 0) & (selected_units < valid_len)]
        return np.sort(np.unique(selected.astype(np.int64)))

    expanded = []
    unit_count = _num_units(valid_len, group_size)
    for unit in np.sort(np.unique(selected_units)):
        if unit < 0 or unit >= unit_count:
            continue
        start = int(unit) * group_size
        end = min(valid_len, start + group_size)
        if start < end:
            expanded.append(np.arange(start, end, dtype=np.int64))
    if len(expanded) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(expanded)


def select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size=1):
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    unit_count = _num_units(valid_len, group_size)
    target_count = _target_unit_count(target_frame_num, group_size)
    if target_count >= unit_count:
        return expand_units(np.arange(unit_count, dtype=np.int64), valid_len, group_size)

    rng = np.random.RandomState(stable_string_seed(sample_key))
    selected = np.sort(rng.choice(unit_count, size=int(target_count), replace=False))
    return expand_units(selected.astype(np.int64), valid_len, group_size)


def select_stratified_positions(valid_len, target_frame_num, sample_key, group_size=1):
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    unit_count = _num_units(valid_len, group_size)
    target_count = _target_unit_count(target_frame_num, group_size)
    if target_count >= unit_count:
        return expand_units(np.arange(unit_count, dtype=np.int64), valid_len, group_size)

    rng = np.random.RandomState(stable_string_seed(sample_key))
    bucket_edges = np.linspace(0, unit_count, num=target_count + 1)
    selected = []
    for bucket_idx in range(target_count):
        start = int(np.floor(bucket_edges[bucket_idx]))
        end = int(np.floor(bucket_edges[bucket_idx + 1]))
        end = max(end, start + 1)
        end = min(end, unit_count)
        bucket_units = np.arange(start, end, dtype=np.int64)
        if bucket_units.size > 0:
            selected.append(int(rng.choice(bucket_units)))

    selected = np.asarray(selected, dtype=np.int64)
    if selected.size < target_count:
        remaining = np.setdiff1d(np.arange(unit_count, dtype=np.int64), selected, assume_unique=False)
        if remaining.size > 0:
            fill = rng.choice(remaining, size=min(target_count - selected.size, remaining.size), replace=False)
            selected = np.concatenate([selected, fill.astype(np.int64)])
    return expand_units(np.sort(np.unique(selected.astype(np.int64))), valid_len, group_size)


def _select_fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size):
    if fallback == "stratified":
        return select_stratified_positions(valid_len, target_frame_num, sample_key, group_size)
    if fallback == "random_fixed":
        return select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size)
    raise ValueError(f"unsupported pseudo-boundary fallback: {fallback}")


def _positions_to_units(positions, valid_len, group_size):
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size == 0:
        return np.zeros((0,), dtype=np.int64)
    group_size = int(max(group_size, 1))
    if group_size == 1:
        units = positions
    else:
        units = positions // group_size
    unit_count = _num_units(valid_len, group_size)
    units = units[(units >= 0) & (units < unit_count)]
    return np.sort(np.unique(units.astype(np.int64)))


def _fill_units_excluding(selected_units, valid_len, target_frame_num, sample_key, fallback, group_size):
    selected_units = np.asarray(selected_units, dtype=np.int64).reshape(-1)
    selected_units = np.sort(np.unique(selected_units))
    target_units = _target_unit_count(target_frame_num, group_size)
    fill_count = max(target_units - selected_units.size, 0)
    if fill_count == 0:
        return selected_units

    unit_count = _num_units(valid_len, group_size)
    available = np.setdiff1d(np.arange(unit_count, dtype=np.int64), selected_units, assume_unique=False)
    if available.size == 0:
        return selected_units
    fill_count = min(fill_count, available.size)

    rng = np.random.RandomState(stable_string_seed(f"{sample_key}|pseudo_fill|{fallback}"))
    if fallback == "random_fixed":
        fill = rng.choice(available, size=fill_count, replace=False)
    elif fallback == "stratified":
        bucket_edges = np.linspace(0, available.size, num=fill_count + 1)
        fill = []
        for bucket_idx in range(fill_count):
            start = int(np.floor(bucket_edges[bucket_idx]))
            end = int(np.floor(bucket_edges[bucket_idx + 1]))
            end = max(end, start + 1)
            end = min(end, available.size)
            bucket_units = available[start:end]
            if bucket_units.size > 0:
                fill.append(int(rng.choice(bucket_units)))
        fill = np.asarray(fill, dtype=np.int64)
    else:
        raise ValueError(f"unsupported pseudo-boundary fallback: {fallback}")

    return np.sort(np.unique(np.concatenate([selected_units, fill.astype(np.int64)])))


def _rank_pseudo_boundary_units(boundary_scores, valid_len, group_size, min_score):
    scores = np.asarray(boundary_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    scores = scores[:valid_len]
    if scores.size < valid_len:
        scores = np.pad(scores, (0, valid_len - scores.size), constant_values=0.0)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    if group_size > 1:
        unit_scores = []
        for unit in range(_num_units(valid_len, group_size)):
            start = unit * group_size
            end = min(valid_len, start + group_size)
            unit_scores.append(np.max(scores[start:end]) if start < end else -np.inf)
        scores = np.asarray(unit_scores, dtype=np.float32)

    candidate_units = np.where(scores >= float(min_score))[0]
    if candidate_units.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(scores[candidate_units], kind="mergesort")[::-1]
    return candidate_units[order].astype(np.int64)


def select_pseudo_boundary_hybrid_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores=None,
    pseudo_quota=64,
    pseudo_radius=1,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    valid_len = int(valid_len)
    target_frame_num = int(target_frame_num)
    group_size = int(max(group_size, 1))
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    if boundary_scores is None:
        return _select_fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)

    target_units = _target_unit_count(target_frame_num, group_size)
    pseudo_units_budget = min(max(int(pseudo_quota), 0), target_units)
    if pseudo_units_budget == 0:
        return _select_fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)

    ranked_units = _rank_pseudo_boundary_units(boundary_scores, valid_len, group_size, pseudo_min_score)
    if ranked_units.size == 0:
        return _select_fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)

    unit_count = _num_units(valid_len, group_size)
    selected_units = []
    seen = set()
    radius = max(int(pseudo_radius), 0)
    offsets = [0]
    for delta in range(1, radius + 1):
        offsets.extend([-delta, delta])

    for unit in ranked_units:
        for offset in offsets:
            candidate = int(unit) + offset
            if candidate < 0 or candidate >= unit_count or candidate in seen:
                continue
            selected_units.append(candidate)
            seen.add(candidate)
            if len(selected_units) >= pseudo_units_budget:
                break
        if len(selected_units) >= pseudo_units_budget:
            break

    selected_positions = expand_units(np.asarray(selected_units, dtype=np.int64), valid_len, group_size)
    selected_units = _positions_to_units(selected_positions, valid_len, group_size)
    merged_units = _fill_units_excluding(
        selected_units,
        valid_len,
        target_frame_num,
        sample_key,
        fallback,
        group_size,
    )
    merged = expand_units(merged_units, valid_len, group_size)
    if merged.size < target_frame_num:
        all_positions = np.arange(valid_len, dtype=np.int64)
        remaining = np.setdiff1d(all_positions, merged, assume_unique=False)
        merged = np.sort(np.concatenate([merged, remaining[: target_frame_num - merged.size]]))
    if merged.size > target_frame_num:
        pseudo_set = set(selected_positions.tolist())
        keep = [pos for pos in merged.tolist() if pos in pseudo_set]
        for pos in merged.tolist():
            if len(keep) >= target_frame_num:
                break
            if pos not in pseudo_set:
                keep.append(pos)
        merged = np.asarray(sorted(keep[:target_frame_num]), dtype=np.int64)
    return merged.astype(np.int64)


def select_pseudo_boundary_snap_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores=None,
    pseudo_quota=64,
    pseudo_snap_distance=2,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    valid_len = int(valid_len)
    target_frame_num = int(target_frame_num)
    group_size = int(max(group_size, 1))
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    base_positions = _select_fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)
    base_units = _positions_to_units(base_positions, valid_len, group_size)
    if boundary_scores is None or base_units.size == 0:
        return base_positions

    snap_budget = min(max(int(pseudo_quota), 0), base_units.size)
    if snap_budget == 0:
        return base_positions

    ranked_units = _rank_pseudo_boundary_units(boundary_scores, valid_len, group_size, pseudo_min_score)
    if ranked_units.size == 0:
        return base_positions

    max_distance = max(int(pseudo_snap_distance), 0)
    selected_units = base_units.copy()
    selected_set = set(selected_units.tolist())
    locked_sources = set()
    snapped = 0

    for target_unit in ranked_units:
        if snapped >= snap_budget:
            break
        target_unit = int(target_unit)
        if target_unit in selected_set:
            snapped += 1
            continue

        best_index = None
        best_distance = None
        for idx, source_unit in enumerate(selected_units.tolist()):
            if idx in locked_sources:
                continue
            distance = abs(int(source_unit) - target_unit)
            if distance > max_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is None:
            continue

        selected_set.remove(int(selected_units[best_index]))
        selected_units[best_index] = target_unit
        selected_set.add(target_unit)
        locked_sources.add(best_index)
        snapped += 1

    return expand_units(np.sort(selected_units.astype(np.int64)), valid_len, group_size).astype(np.int64)
