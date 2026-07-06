import copy
import hashlib
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F
from .pseudo_boundary import (
    load_boundary_scores,
    select_pseudo_boundary_hybrid_positions,
    select_pseudo_boundary_snap_positions,
    slice_global_scores_for_window,
)
from .boundary_acquisition import (
    BcaConfig,
    load_bata_boundary_scores,
    select_bata_boundary_acquisition_positions,
    slice_global_scores_for_window as slice_bata_global_scores_for_window,
    validate_bata_cache_manifest_for_loader,
)


def _stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


_EVAL_SPLIT_NAMES = frozenset(("val", "valid", "validation", "test", "testing", "eval", "evaluation"))
_SPLIT_RESULT_KEYS = ("subset", "split", "data_split")
_EVAL_DIAGNOSTIC_SHORTCUT_FLAGS = (
    "bata_allow_diagnostic_gt_cache",
    "bata_diagnostic_only",
    "diagnostic_only",
    "teacher_cache",
    "use_teacher_cache",
    "allow_teacher_cache",
    "bata_teacher_cache",
    "bata_use_teacher_cache",
    "prediction_cache_shortcut",
    "raw_prediction_shortcut",
    "use_raw_prediction_shortcut",
    "use_prediction_cache",
    "allow_prediction_cache_shortcut",
    "cache_shortcut",
    "use_cache_shortcut",
)


def _normalise_split_name(value):
    if value is None:
        return None
    return str(value).strip().lower()


def _eval_split_marker(results):
    for key in _SPLIT_RESULT_KEYS:
        split_name = _normalise_split_name(results.get(key, None))
        if split_name in _EVAL_SPLIT_NAMES:
            return key, split_name
    return None, None


def _is_enabled_flag(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return bool(value)


@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        results["filename"] = os.path.join(
            results["data_path"],
            self.prefix + results["video_name"] + "." + self.format,
        )
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # todo: support to  change FPS
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == snippet_num, "snippet center number should be equal to snippet number"
        assert frame_idxs.shape[1] == self.clip_len, "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
        keep_ratio=0.5,
        method_base=None,
        target_len=None,
        source_len=None,
        selection_unit="frame",
        selection_tubelet_size=2,
        oracle_boundary_radius=2,
        sampling_boundary_weight=4.0,
        sampling_action_weight=2.0,
        sampling_background_weight=1.0,
        remap_gt_to_selected_axis=True,
        allow_drop_selected_axis_gt=False,
        store_dense_window=False,
        pseudo_boundary_cache_dir=None,
        pseudo_boundary_quota=64,
        pseudo_boundary_radius=1,
        pseudo_boundary_snap_distance=2,
        pseudo_boundary_min_score=0.0,
        pseudo_boundary_fallback="random_fixed",
        bata_boundary_cache_dir=None,
        bata_score_source="unknown",
        bata_allow_diagnostic_gt_cache=False,
        bata_diagnostic_only=False,
        bata_boundary_cap_ratio=0.60,
        bata_coverage_floor_ratio=0.20,
        bata_peak_nms_radius=8,
        bata_local_triplet_radius=4,
        bata_fallback_radii=(2, 6),
        bata_max_peaks=8,
        bata_coverage_max_gap_target=8,
        bata_context_low_score_search_radius=2,
        bata_reclaim_boundary_support_for_coverage=True,
        bata_config_hash="",
        fixed_trunc_start=None,
        fixed_trunc_gt_index=None,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize/padding/random_trunc/sliding_window/*_subsample
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        # input-side subsample settings
        self.keep_ratio = keep_ratio
        self.method_base = method_base
        self.target_len = target_len
        self.source_len = source_len
        self.selection_unit = selection_unit
        self.selection_tubelet_size = selection_tubelet_size
        self.oracle_boundary_radius = oracle_boundary_radius
        self.sampling_boundary_weight = sampling_boundary_weight
        self.sampling_action_weight = sampling_action_weight
        self.sampling_background_weight = sampling_background_weight
        self.remap_gt_to_selected_axis = remap_gt_to_selected_axis
        self.allow_drop_selected_axis_gt = bool(allow_drop_selected_axis_gt)
        self.store_dense_window = store_dense_window
        self.pseudo_boundary_cache_dir = pseudo_boundary_cache_dir
        self.pseudo_boundary_quota = pseudo_boundary_quota
        self.pseudo_boundary_radius = pseudo_boundary_radius
        self.pseudo_boundary_snap_distance = pseudo_boundary_snap_distance
        self.pseudo_boundary_min_score = pseudo_boundary_min_score
        self.pseudo_boundary_fallback = pseudo_boundary_fallback
        self.bata_boundary_cache_dir = bata_boundary_cache_dir
        self.bata_score_source = bata_score_source
        self.bata_allow_diagnostic_gt_cache = bool(bata_allow_diagnostic_gt_cache)
        self.bata_diagnostic_only = bool(bata_diagnostic_only)
        self.bata_config = BcaConfig(
            boundary_cap_ratio=float(bata_boundary_cap_ratio),
            coverage_floor_ratio=float(bata_coverage_floor_ratio),
            peak_nms_radius=int(bata_peak_nms_radius),
            local_triplet_radius=int(bata_local_triplet_radius),
            fallback_radii=tuple(bata_fallback_radii),
            max_peaks=int(bata_max_peaks),
            coverage_max_gap_target=int(bata_coverage_max_gap_target),
            context_low_score_search_radius=int(bata_context_low_score_search_radius),
            reclaim_boundary_support_for_coverage=bool(bata_reclaim_boundary_support_for_coverage),
        )
        self.bata_config_hash = bata_config_hash
        self.fixed_trunc_start = fixed_trunc_start
        self.fixed_trunc_gt_index = fixed_trunc_gt_index

    def _assert_no_eval_diagnostic_shortcuts(self, results):
        split_key, split_name = _eval_split_marker(results)
        if split_name is None:
            return

        for flag_name in _EVAL_DIAGNOSTIC_SHORTCUT_FLAGS:
            if _is_enabled_flag(getattr(self, flag_name, False)):
                raise ValueError(
                    f"{flag_name}=True is forbidden for validation/test LoadFrames splits "
                    f"({split_key}={split_name}); diagnostic GT/cache shortcuts must not affect mAP."
                )

    def _apply_trunc_window(self, feats, st, ed, gt_segments, gt_labels, offset=0):
        feats = feats[st:ed]
        num_segs = gt_segments.shape[0]

        if num_segs == 0:
            return feats, np.zeros((0, 2), dtype=np.float32), gt_labels[:0]

        window = np.repeat(np.array([[st, ed]], dtype=np.float32), num_segs, axis=0)
        left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
        right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
        inter = np.clip(right - left, a_min=0, a_max=None)
        area_segs = np.clip(np.abs(gt_segments[:, 1] - gt_segments[:, 0]), a_min=1e-6, a_max=None)
        inter_ratio = inter / area_segs

        if self.trunc_thresh is None:
            seg_idx = inter > 0
        else:
            seg_idx = inter_ratio >= self.trunc_thresh

        if seg_idx.sum().item() > 0:
            gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1).astype(np.float32)
            gt_segments = gt_segments - st
            gt_labels = gt_labels[seg_idx]
        else:
            gt_segments = np.zeros((0, 2), dtype=np.float32)
            gt_labels = gt_labels[:0]
        return feats, gt_segments, gt_labels

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None or self.fixed_trunc_start is not None or self.fixed_trunc_gt_index is not None:
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        if self.fixed_trunc_start is not None or self.fixed_trunc_gt_index is not None:
            max_start = max(feat_len - trunc_len, 0)
            if self.fixed_trunc_start is not None:
                st = int(np.clip(int(self.fixed_trunc_start), 0, max_start))
            else:
                if num_segs == 0:
                    raise ValueError("fixed_trunc_gt_index requires at least one gt segment")
                gt_idx = int(np.clip(int(self.fixed_trunc_gt_index), 0, num_segs - 1))
                center = 0.5 * (float(gt_segments[gt_idx, 0]) + float(gt_segments[gt_idx, 1]))
                st = int(round(center - trunc_len / 2.0))
                st = int(np.clip(st, 0, max_start))
            ed = st + trunc_len
            feats, gt_segments, gt_labels = self._apply_trunc_window(
                feats,
                st,
                ed,
                gt_segments,
                gt_labels,
                offset=offset,
            )
            if num_segs > 0 and gt_segments.shape[0] == 0:
                raise ValueError(
                    f"fixed truncation kept no gt segments: start={st}, trunc_len={trunc_len}, num_segs={num_segs}"
                )
            return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        last_result = None
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            last_result = self._apply_trunc_window(feats, st, ed, gt_segments, gt_labels, offset=offset)
            _, truncated_segments, _ = last_result
            if truncated_segments.shape[0] > 0:
                break

        if last_result is None:
            raise RuntimeError("random_trunc failed to produce a truncation result")
        return last_result

    def _uniform_pick_indices(self, indices, k):
        indices = np.asarray(indices, dtype=np.int64)
        if k <= 0 or indices.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if k >= indices.size:
            return indices.copy()

        sample_pos = np.linspace(0, indices.size - 1, num=k)
        picked = np.round(sample_pos).astype(np.int64)
        picked = np.unique(picked)
        if picked.size < k:
            remaining = np.setdiff1d(np.arange(indices.size, dtype=np.int64), picked, assume_unique=False)
            picked = np.concatenate([picked, remaining[: k - picked.size]])
        picked = np.sort(picked[:k])
        return indices[picked]

    def _selection_group_size(self):
        if self.selection_unit == "frame":
            return 1
        if self.selection_unit == "tubelet":
            group_size = int(self.selection_tubelet_size)
            if group_size <= 0:
                raise ValueError("selection_tubelet_size must be positive when selection_unit='tubelet'")
            return group_size
        raise ValueError(f"Unsupported selection_unit: {self.selection_unit}")

    def _selection_target_count(self, target_frame_num):
        target_frame_num = int(max(target_frame_num, 0))
        if target_frame_num == 0:
            return 0
        group_size = self._selection_group_size()
        if group_size == 1:
            return target_frame_num
        return max(target_frame_num // group_size, 1)

    def _num_selection_units(self, valid_len):
        group_size = self._selection_group_size()
        return int(np.ceil(max(valid_len, 0) / float(group_size)))

    def _positions_to_selection_units(self, positions, valid_len):
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size == 0:
            return np.zeros((0,), dtype=np.int64)

        group_size = self._selection_group_size()
        if group_size == 1:
            return np.sort(np.unique(positions))

        num_units = self._num_selection_units(valid_len)
        units = positions // group_size
        units = units[(units >= 0) & (units < num_units)]
        if units.size == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.sort(np.unique(units.astype(np.int64)))

    def _expand_selected_units(self, selected_units, valid_len):
        selected_units = np.asarray(selected_units, dtype=np.int64)
        if selected_units.size == 0 or valid_len <= 0:
            return np.zeros((0,), dtype=np.int64)

        group_size = self._selection_group_size()
        if group_size == 1:
            selected_units = selected_units[(selected_units >= 0) & (selected_units < valid_len)]
            if selected_units.size == 0:
                return np.zeros((0,), dtype=np.int64)
            return np.sort(np.unique(selected_units))

        expanded = []
        num_units = self._num_selection_units(valid_len)
        for unit in np.sort(np.unique(selected_units)):
            if unit < 0 or unit >= num_units:
                continue
            start = int(unit) * group_size
            end = min(valid_len, start + group_size)
            if start < end:
                expanded.append(np.arange(start, end, dtype=np.int64))
        if len(expanded) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(expanded)

    def _build_frame_oracle_groups(self, valid_len, gt_segments, profile):
        boundary_positions = set()
        action_positions = set()
        radius = int(self.oracle_boundary_radius)

        if gt_segments is None:
            gt_segments = np.zeros((0, 2), dtype=np.float32)

        for seg in gt_segments:
            start = float(seg[0])
            end = float(seg[1])
            if end <= start:
                continue

            left = max(int(np.floor(start)), 0)
            right = min(int(np.ceil(end)) - 1, valid_len - 1)
            if right < left:
                continue

            for center in (left, right):
                lo = max(0, center - radius)
                hi = min(valid_len, center + radius + 1)
                boundary_positions.update(range(lo, hi))

            if profile == "action_boundary_dense":
                for pos in range(left, right + 1):
                    action_positions.add(pos)

        boundary = np.array(sorted(boundary_positions), dtype=np.int64)
        action = np.array(sorted(action_positions.difference(boundary_positions)), dtype=np.int64)
        all_positions = np.arange(valid_len, dtype=np.int64)
        used = boundary if profile == "boundary_dense" else np.union1d(boundary, action)
        background = np.setdiff1d(all_positions, used, assume_unique=False)
        return boundary, action, background

    def _build_oracle_groups(self, valid_len, gt_segments, profile):
        boundary, action, background = self._build_frame_oracle_groups(valid_len, gt_segments, profile)
        if self._selection_group_size() == 1:
            return boundary, action, background

        boundary_units = self._positions_to_selection_units(boundary, valid_len)
        action_units = self._positions_to_selection_units(action, valid_len)
        action_units = np.setdiff1d(action_units, boundary_units, assume_unique=False)
        all_units = np.arange(self._num_selection_units(valid_len), dtype=np.int64)
        used_units = boundary_units if profile == "boundary_dense" else np.union1d(boundary_units, action_units)
        background_units = np.setdiff1d(all_units, used_units, assume_unique=False)
        return boundary_units, action_units, background_units

    def _select_oracle_positions(self, valid_len, gt_segments, target_frame_num, profile):
        boundary, action, background = self._build_oracle_groups(valid_len, gt_segments, profile)
        groups = [boundary]
        if profile == "action_boundary_dense":
            groups.append(action)
        groups.append(background)

        selected = []
        total_units = self._num_selection_units(valid_len)
        target_count = self._selection_target_count(target_frame_num)
        remaining = int(min(target_count, total_units))
        for group in groups:
            if remaining <= 0:
                break
            chosen = self._uniform_pick_indices(group, min(remaining, group.size))
            if chosen.size > 0:
                selected.append(chosen)
                remaining -= chosen.size

        if selected:
            selected = np.unique(np.concatenate(selected))
        else:
            selected = np.zeros((0,), dtype=np.int64)
        return self._expand_selected_units(np.sort(selected), valid_len)

    def _build_weighted_sampling_probs(self, valid_len, gt_segments, profile):
        boundary, action, background = self._build_oracle_groups(valid_len, gt_segments, profile)
        weights = np.zeros(self._num_selection_units(valid_len), dtype=np.float64)

        if background.size > 0:
            weights[background] = float(self.sampling_background_weight)
        if profile == "action_boundary_dense" and action.size > 0:
            weights[action] = float(self.sampling_action_weight)
        if boundary.size > 0:
            weights[boundary] = float(self.sampling_boundary_weight)

        weights = np.clip(weights, a_min=0.0, a_max=None)
        if (not np.isfinite(weights).all()) or weights.sum() <= 0:
            weights = np.ones(self._num_selection_units(valid_len), dtype=np.float64)
        return weights / weights.sum()

    def _select_weighted_random_positions(self, valid_len, gt_segments, target_frame_num, profile, sample_key):
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)

        total_units = self._num_selection_units(valid_len)
        target_count = self._selection_target_count(target_frame_num)
        if target_count >= total_units:
            return self._expand_selected_units(np.arange(total_units, dtype=np.int64), valid_len)

        probs = self._build_weighted_sampling_probs(valid_len, gt_segments, profile)
        rng = np.random.RandomState(_stable_string_seed(sample_key))
        selected = rng.choice(total_units, size=int(target_count), replace=False, p=probs)
        return self._expand_selected_units(np.sort(selected.astype(np.int64)), valid_len)

    def _select_random_fixed_positions(self, valid_len, target_frame_num, sample_key):
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)

        total_units = self._num_selection_units(valid_len)
        target_count = self._selection_target_count(target_frame_num)
        if target_count >= total_units:
            return self._expand_selected_units(np.arange(total_units, dtype=np.int64), valid_len)

        rng = np.random.RandomState(_stable_string_seed(sample_key))
        selected = np.sort(rng.choice(total_units, size=int(target_count), replace=False))
        return self._expand_selected_units(selected.astype(np.int64), valid_len)

    def _select_uniform_fixed_positions(self, valid_len, target_frame_num):
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)

        total_units = self._num_selection_units(valid_len)
        target_count = self._selection_target_count(target_frame_num)
        if target_count >= total_units:
            return self._expand_selected_units(np.arange(total_units, dtype=np.int64), valid_len)

        return self._expand_selected_units(
            self._uniform_pick_indices(np.arange(total_units), int(target_count)),
            valid_len,
        )

    def _select_stratified_random_fixed_positions(self, valid_len, target_frame_num, sample_key):
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)

        total_units = self._num_selection_units(valid_len)
        target_count = self._selection_target_count(target_frame_num)
        if target_count >= total_units:
            return self._expand_selected_units(np.arange(total_units, dtype=np.int64), valid_len)

        rng = np.random.RandomState(_stable_string_seed(sample_key))
        bucket_edges = np.linspace(0, total_units, num=target_count + 1)
        selected = []
        for bucket_idx in range(target_count):
            start = int(np.floor(bucket_edges[bucket_idx]))
            end = int(np.floor(bucket_edges[bucket_idx + 1]))
            end = max(end, start + 1)
            end = min(end, total_units)
            bucket_units = np.arange(start, end, dtype=np.int64)
            if bucket_units.size == 0:
                continue
            selected.append(int(rng.choice(bucket_units)))

        selected = np.asarray(selected, dtype=np.int64)
        if selected.size < target_count:
            remaining = np.setdiff1d(np.arange(total_units, dtype=np.int64), selected, assume_unique=False)
            if remaining.size > 0:
                fill = rng.choice(remaining, size=min(target_count - selected.size, remaining.size), replace=False)
                selected = np.concatenate([selected, fill.astype(np.int64)])

        return self._expand_selected_units(np.sort(np.unique(selected.astype(np.int64))), valid_len)

    def _map_coord_to_selected_axis(self, coord, kept_positions, valid_len):
        if kept_positions.size == 0:
            return 0.0

        xp = np.concatenate(
            [kept_positions.astype(np.float32), np.array([float(valid_len)], dtype=np.float32)]
        )
        fp = np.concatenate(
            [np.arange(kept_positions.size, dtype=np.float32), np.array([float(kept_positions.size)], dtype=np.float32)]
        )
        coord = float(np.clip(coord, 0.0, float(valid_len)))
        return float(np.interp(coord, xp, fp))

    def _clear_selected_axis_gt_drop_state(self):
        self._last_dropped_selected_axis_gt_segments = []
        self._last_selected_axis_gt_input_count = 0
        self._last_selected_axis_gt_keep_count = 0
        self._last_selected_axis_gt_drop_count = 0

    def _remap_gt_to_selected_axis(self, gt_segments, gt_labels, kept_positions, valid_len):
        self._clear_selected_axis_gt_drop_state()
        if gt_segments is None or gt_labels is None or len(gt_segments) == 0 or kept_positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        self._last_selected_axis_gt_input_count = int(len(gt_segments))
        remapped_segments = []
        remapped_labels = []
        max_coord = float(kept_positions.size)
        for idx, seg in enumerate(gt_segments):
            start = self._map_coord_to_selected_axis(seg[0], kept_positions, valid_len)
            end = self._map_coord_to_selected_axis(seg[1], kept_positions, valid_len)
            start = float(np.clip(start, 0.0, max_coord))
            end = float(np.clip(end, 0.0, max_coord))
            if end <= start:
                self._last_dropped_selected_axis_gt_segments.append(
                    dict(
                        index=int(idx),
                        label=int(gt_labels[idx]),
                        original_segment=[float(seg[0]), float(seg[1])],
                        mapped_segment=[start, end],
                    )
                )
                continue
            if end > start:
                remapped_segments.append([start, end])
                remapped_labels.append(int(gt_labels[idx]))

        self._last_selected_axis_gt_drop_count = int(len(self._last_dropped_selected_axis_gt_segments))
        self._last_selected_axis_gt_keep_count = int(len(remapped_segments))
        if self._last_selected_axis_gt_drop_count > 0 and not bool(
            getattr(self, "allow_drop_selected_axis_gt", False)
        ):
            video_name = getattr(self, "_current_video_name", "unknown")
            raise ValueError(
                "selected-axis GT remap dropped/collapsed ground truth for "
                f"video={video_name}: dropped={self._last_selected_axis_gt_drop_count}, "
                f"kept={self._last_selected_axis_gt_keep_count}, "
                f"input={self._last_selected_axis_gt_input_count}, "
                f"valid_len={valid_len}, kept_positions={int(kept_positions.size)}. "
                "Set allow_drop_selected_axis_gt=True only for legacy/diagnostic routes."
            )

        if len(remapped_segments) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.asarray(remapped_segments, dtype=np.float32), np.asarray(remapped_labels, dtype=np.int32)

    def _set_irregular_axis_meta(self, results, kept_positions, valid_len):
        scale = float(max(self.scale_factor, 1))
        results["irregular_selected_positions"] = np.asarray(kept_positions, dtype=np.float32) / scale
        results["irregular_selected_valid_len"] = float(valid_len) / scale
        results["dropped_selected_axis_gt_segments"] = list(
            getattr(self, "_last_dropped_selected_axis_gt_segments", [])
        )
        results["selected_axis_gt_input_count"] = int(getattr(self, "_last_selected_axis_gt_input_count", 0))
        results["selected_axis_gt_keep_count"] = int(getattr(self, "_last_selected_axis_gt_keep_count", 0))
        results["selected_axis_gt_drop_count"] = int(getattr(self, "_last_selected_axis_gt_drop_count", 0))
        results["allow_drop_selected_axis_gt"] = bool(getattr(self, "allow_drop_selected_axis_gt", False))
        results["irregular_native_axis"] = bool(not self.remap_gt_to_selected_axis)
        gt_axis = "selected" if self.remap_gt_to_selected_axis else "native"
        proposal_axis = gt_axis
        postprocess_axis = "native"
        results["irregular_gt_axis"] = gt_axis
        results["irregular_proposal_axis"] = proposal_axis
        results["irregular_postprocess_axis"] = postprocess_axis
        results["irregular_axis_contract"] = dict(
            gt_axis=gt_axis,
            proposal_axis=proposal_axis,
            postprocess_axis=postprocess_axis,
        )

    def _oracle_subsample_window(self, dense_frame_idxs, gt_segments, gt_labels, target_frame_num, profile):
        valid_len = int(len(dense_frame_idxs))
        if valid_len <= 0:
            raise RuntimeError("oracle subsample received an empty dense window")

        keep_positions = self._select_oracle_positions(
            valid_len=valid_len,
            gt_segments=gt_segments,
            target_frame_num=target_frame_num,
            profile=profile,
        )
        if keep_positions.size == 0:
            keep_positions = np.array([0], dtype=np.int64)

        kept_frame_idxs = dense_frame_idxs[keep_positions]
        if self.remap_gt_to_selected_axis:
            out_segments, out_labels = self._remap_gt_to_selected_axis(
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                kept_positions=keep_positions,
                valid_len=valid_len,
            )
        else:
            self._clear_selected_axis_gt_drop_state()
            out_segments, out_labels = gt_segments, gt_labels

        frame_num = int(target_frame_num)
        if kept_frame_idxs.shape[0] < frame_num:
            kept_frame_idxs = np.pad(kept_frame_idxs, (0, frame_num - kept_frame_idxs.shape[0]), mode="edge")

        valid_mask_len = min(
            int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
            int(np.ceil(frame_num / max(self.scale_factor, 1))),
        )
        target_mask_len = int(np.ceil(frame_num / max(self.scale_factor, 1)))
        if valid_mask_len < target_mask_len:
            masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
        else:
            masks = torch.ones(target_mask_len).bool()

        return kept_frame_idxs, out_segments, out_labels, masks, keep_positions.astype(np.float32), valid_len

    def _weighted_random_subsample_window(self, dense_frame_idxs, gt_segments, gt_labels, target_frame_num, profile, sample_key):
        valid_len = int(len(dense_frame_idxs))
        if valid_len <= 0:
            raise RuntimeError("weighted random subsample received an empty dense window")

        keep_positions = self._select_weighted_random_positions(
            valid_len=valid_len,
            gt_segments=gt_segments,
            target_frame_num=target_frame_num,
            profile=profile,
            sample_key=sample_key,
        )
        if keep_positions.size == 0:
            keep_positions = np.array([0], dtype=np.int64)

        kept_frame_idxs = dense_frame_idxs[keep_positions]
        if self.remap_gt_to_selected_axis:
            out_segments, out_labels = self._remap_gt_to_selected_axis(
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                kept_positions=keep_positions,
                valid_len=valid_len,
            )
        else:
            self._clear_selected_axis_gt_drop_state()
            out_segments, out_labels = gt_segments, gt_labels

        frame_num = int(target_frame_num)
        if kept_frame_idxs.shape[0] < frame_num:
            kept_frame_idxs = np.pad(kept_frame_idxs, (0, frame_num - kept_frame_idxs.shape[0]), mode="edge")

        valid_mask_len = min(
            int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
            int(np.ceil(frame_num / max(self.scale_factor, 1))),
        )
        target_mask_len = int(np.ceil(frame_num / max(self.scale_factor, 1)))
        if valid_mask_len < target_mask_len:
            masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
        else:
            masks = torch.ones(target_mask_len).bool()

        return kept_frame_idxs, out_segments, out_labels, masks, keep_positions.astype(np.float32), valid_len

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        self._current_video_name = results.get("video_name", "unknown")
        self._clear_selected_axis_gt_drop_state()
        self._assert_no_eval_diagnostic_shortcuts(results)
        total_frames = results["total_frames"]
        fps = results.get("avg_fps", results.get("fps", None))

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            # trunc the frame_idxs
            frame_idxs, gt_segments, gt_labels = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"] * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
            )
            results["gt_segments"] = gt_segments / self.scale_factor  # convert back to original scale
            results["gt_labels"] = gt_labels

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method in (
            "random_fixed_subsample",
            "uniform_fixed_subsample",
            "stratified_random_fixed_subsample",
            "pseudo_boundary_hybrid_subsample",
            "pseudo_boundary_snap_subsample",
            "bata_boundary_acquisition_subsample",
        ):
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            keep_ratio = float(self.keep_ratio)
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None

            if self.method_base == "random_trunc":
                if gt_segments is None or gt_labels is None:
                    raise ValueError("random_fixed_subsample with random_trunc requires gt_segments and gt_labels")
                if self.trunc_len is None and self.target_len is None:
                    raise ValueError("random_fixed_subsample requires trunc_len or target_len when method_base='random_trunc'")

                target_len = int(self.target_len) if self.target_len is not None else int(self.trunc_len)
                source_len = int(self.source_len) if self.source_len is not None else int(round(target_len / max(keep_ratio, 1e-6)))
                frame_num = target_len * self.scale_factor
                dense_frame_num = int(source_len * self.scale_factor)
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=int(source_len * self.scale_factor),
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )

            elif self.method_base == "sliding_window":
                if "window_size" not in results:
                    raise ValueError("random_fixed_subsample with sliding_window requires window_size in results")

                dense_window_len = int(results["window_size"])
                target_len = int(self.target_len) if self.target_len is not None else int(round(dense_window_len * keep_ratio))
                frame_num = target_len * self.scale_factor
                dense_frame_num = int(dense_window_len * self.scale_factor)
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError("random_fixed_subsample requires method_base='random_trunc' or 'sliding_window'")

            valid_len = int(len(dense_window))
            if valid_len <= 0:
                raise RuntimeError("random_fixed_subsample received an empty dense window")

            if self.store_dense_window:
                oracle_frame_inds = dense_window
                if oracle_frame_inds.shape[0] < dense_frame_num:
                    oracle_frame_inds = np.pad(
                        oracle_frame_inds,
                        (0, dense_frame_num - oracle_frame_inds.shape[0]),
                        mode="edge",
                    )
                else:
                    oracle_frame_inds = oracle_frame_inds[:dense_frame_num]
                results["oracle_dense_frame_inds"] = oracle_frame_inds.astype(int)

            sample_profile = "random_fixed"
            if self.method == "stratified_random_fixed_subsample":
                sample_profile = f"{self.method}|{self.selection_unit}|{self._selection_group_size()}"
            elif self.method == "pseudo_boundary_hybrid_subsample":
                sample_profile = (
                    f"{self.method}|{self.selection_unit}|{self._selection_group_size()}|"
                    f"q{int(self.pseudo_boundary_quota)}|r{int(self.pseudo_boundary_radius)}|"
                    f"{self.pseudo_boundary_fallback}"
                )
            elif self.method == "pseudo_boundary_snap_subsample":
                sample_profile = (
                    f"{self.method}|{self.selection_unit}|{self._selection_group_size()}|"
                    f"q{int(self.pseudo_boundary_quota)}|d{int(self.pseudo_boundary_snap_distance)}|"
                    f"{self.pseudo_boundary_fallback}"
                )
            elif self.method == "bata_boundary_acquisition_subsample":
                sample_profile = (
                    f"{self.method}|{self.selection_unit}|{self._selection_group_size()}|"
                    f"{self.bata_score_source}|diag{int(self.bata_diagnostic_only)}|"
                    f"nms{int(self.bata_config.peak_nms_radius)}|tri{int(self.bata_config.local_triplet_radius)}|"
                    f"cap{float(self.bata_config.boundary_cap_ratio):.2f}|cov{float(self.bata_config.coverage_floor_ratio):.2f}"
                )
            sample_key = (
                f"{results.get('video_name', 'unknown')}|{sample_profile}|"
                f"{int(dense_window[0]) if valid_len > 0 else -1}|"
                f"{int(dense_window[-1]) if valid_len > 0 else -1}|"
                f"{valid_len}|{frame_num}"
            )
            if self.method == "stratified_random_fixed_subsample":
                keep_positions = self._select_stratified_random_fixed_positions(valid_len, frame_num, sample_key)
            elif self.method == "uniform_fixed_subsample":
                keep_positions = self._select_uniform_fixed_positions(valid_len, frame_num)
            elif self.method == "pseudo_boundary_hybrid_subsample":
                boundary_scores = load_boundary_scores(
                    self.pseudo_boundary_cache_dir,
                    results.get("video_name", "unknown"),
                )
                global_indices = np.rint(dense_window / max(frame_stride, 1)).astype(np.int64)
                window_scores = slice_global_scores_for_window(boundary_scores, global_indices)
                keep_positions = select_pseudo_boundary_hybrid_positions(
                    valid_len=valid_len,
                    target_frame_num=frame_num,
                    sample_key=sample_key,
                    boundary_scores=window_scores,
                    pseudo_quota=self.pseudo_boundary_quota,
                    pseudo_radius=self.pseudo_boundary_radius,
                    pseudo_min_score=self.pseudo_boundary_min_score,
                    fallback=self.pseudo_boundary_fallback,
                    group_size=self._selection_group_size(),
                )
            elif self.method == "pseudo_boundary_snap_subsample":
                boundary_scores = load_boundary_scores(
                    self.pseudo_boundary_cache_dir,
                    results.get("video_name", "unknown"),
                )
                global_indices = np.rint(dense_window / max(frame_stride, 1)).astype(np.int64)
                window_scores = slice_global_scores_for_window(boundary_scores, global_indices)
                keep_positions = select_pseudo_boundary_snap_positions(
                    valid_len=valid_len,
                    target_frame_num=frame_num,
                    sample_key=sample_key,
                    boundary_scores=window_scores,
                    pseudo_quota=self.pseudo_boundary_quota,
                    pseudo_snap_distance=self.pseudo_boundary_snap_distance,
                    pseudo_min_score=self.pseudo_boundary_min_score,
                    fallback=self.pseudo_boundary_fallback,
                    group_size=self._selection_group_size(),
                )
            elif self.method == "bata_boundary_acquisition_subsample":
                boundary_scores, score_manifest = load_bata_boundary_scores(
                    self.bata_boundary_cache_dir,
                    results.get("video_name", "unknown"),
                    allow_diagnostic_gt_cache=self.bata_allow_diagnostic_gt_cache,
                )
                if boundary_scores is None:
                    raise FileNotFoundError(
                        f"BATA boundary score cache missing for {results.get('video_name', 'unknown')} "
                        f"in {self.bata_boundary_cache_dir}"
                    )
                effective_diagnostic_only = validate_bata_cache_manifest_for_loader(
                    score_manifest,
                    frame_stride=frame_stride,
                    scale_factor=self.scale_factor,
                    loader_diagnostic_only=self.bata_diagnostic_only,
                )
                global_indices = np.rint(dense_window / max(frame_stride, 1)).astype(np.int64)
                window_scores = slice_bata_global_scores_for_window(boundary_scores, global_indices)
                score_source = self.bata_score_source
                if score_source in (None, "", "unknown"):
                    score_source = score_manifest.get("score_source", "unknown")
                selection_result = select_bata_boundary_acquisition_positions(
                    valid_len=valid_len,
                    target_frame_num=frame_num,
                    boundary_scores=window_scores,
                    sample_key=sample_key,
                    group_size=self._selection_group_size(),
                    config=self.bata_config,
                    run_id=results.get("run_id", "dataloader"),
                    dataset="thumos14",
                    split=results.get("subset", "unknown"),
                    video_id=results.get("video_name", "unknown"),
                    dense_window=dense_window,
                    fps=fps,
                    frame_stride=frame_stride,
                    score_source=score_source,
                    diagnostic_only=effective_diagnostic_only,
                    config_hash=self.bata_config_hash,
                )
                keep_positions = selection_result.selected_dense_indices
                results["bata_selected_dense_indices"] = keep_positions.astype(np.int64)
                results["bata_selection_ledger"] = selection_result.selection_ledger
                results["bata_boundary_peak_ledger"] = selection_result.boundary_peak_ledger
                results["bata_coverage_diagnostics"] = selection_result.coverage_diagnostics
                results["bata_score_source"] = score_source
                results["bata_diagnostic_only"] = effective_diagnostic_only
            else:
                keep_positions = self._select_random_fixed_positions(valid_len, frame_num, sample_key)
            if keep_positions.size == 0:
                keep_positions = np.array([0], dtype=np.int64)

            frame_idxs = dense_window[keep_positions]
            if gt_segments is not None and gt_labels is not None:
                if self.remap_gt_to_selected_axis:
                    gt_segments, gt_labels = self._remap_gt_to_selected_axis(
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        kept_positions=keep_positions,
                        valid_len=valid_len,
                    )
                else:
                    self._clear_selected_axis_gt_drop_state()
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels
            else:
                self._clear_selected_axis_gt_drop_state()
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

            if len(frame_idxs) < frame_num:
                valid_mask_len = min(
                    int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
                    int(np.ceil(frame_num / max(self.scale_factor, 1))),
                )
                target_mask_len = int(np.ceil(frame_num / self.scale_factor))
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
            else:
                masks = torch.ones(int(np.ceil(frame_num / self.scale_factor))).bool()

        elif self.method in ("oracle_boundary_subsample", "oracle_action_boundary_subsample"):
            assert "gt_segments" in results.keys(), f"{self.method} requires gt_segments in the pipeline"
            assert "gt_labels" in results.keys(), f"{self.method} requires gt_labels in the pipeline"
            assert results["snippet_stride"] >= self.scale_factor
            assert results["snippet_stride"] % self.scale_factor == 0

            profile = "boundary_dense" if self.method == "oracle_boundary_subsample" else "action_boundary_dense"
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor
            gt_labels = results["gt_labels"]

            if self.target_len is None:
                raise ValueError(f"{self.method} requires target_len to be set")
            frame_num = int(self.target_len * self.scale_factor)

            if self.method_base == "random_trunc":
                source_len = self.source_len
                if source_len is None:
                    source_len = int(round(self.target_len / max(self.keep_ratio, 1e-6)))
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=int(source_len * self.scale_factor),
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )
            elif self.method_base == "sliding_window":
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError(
                    f"{self.method} requires method_base='random_trunc' or 'sliding_window', got {self.method_base}"
                )

            frame_idxs, gt_segments, gt_labels, masks, keep_positions, valid_len = self._oracle_subsample_window(
                dense_frame_idxs=dense_window,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                target_frame_num=frame_num,
                profile=profile,
            )
            results["gt_segments"] = gt_segments / self.scale_factor
            results["gt_labels"] = gt_labels
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

        elif self.method in ("weighted_random_boundary_subsample", "weighted_random_action_boundary_subsample"):
            assert "gt_segments" in results.keys(), f"{self.method} requires gt_segments in the pipeline"
            assert "gt_labels" in results.keys(), f"{self.method} requires gt_labels in the pipeline"
            assert results["snippet_stride"] >= self.scale_factor
            assert results["snippet_stride"] % self.scale_factor == 0

            profile = "boundary_dense" if self.method == "weighted_random_boundary_subsample" else "action_boundary_dense"
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor
            gt_labels = results["gt_labels"]

            if self.target_len is None:
                raise ValueError(f"{self.method} requires target_len to be set")
            frame_num = int(self.target_len * self.scale_factor)

            if self.method_base == "random_trunc":
                source_len = self.source_len
                if source_len is None:
                    source_len = int(round(self.target_len / max(self.keep_ratio, 1e-6)))
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=int(source_len * self.scale_factor),
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )
            elif self.method_base == "sliding_window":
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError(
                    f"{self.method} requires method_base='random_trunc' or 'sliding_window', got {self.method_base}"
                )

            sample_key = (
                f"{results.get('video_name', 'unknown')}|{profile}|"
                f"{int(dense_window[0]) if len(dense_window) > 0 else -1}|"
                f"{int(dense_window[-1]) if len(dense_window) > 0 else -1}|"
                f"{len(dense_window)}|{frame_num}|{self.selection_unit}|{self._selection_group_size()}"
            )
            frame_idxs, gt_segments, gt_labels, masks, keep_positions, valid_len = self._weighted_random_subsample_window(
                dense_frame_idxs=dense_window,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                target_frame_num=frame_num,
                profile=profile,
                sample_key=sample_key,
            )
            results["gt_segments"] = gt_segments / self.scale_factor
            results["gt_labels"] = gt_labels
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            if results[key].shape[2:] != self.size:
                results[key] = F.interpolate(
                    results[key],
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                )
        return results
