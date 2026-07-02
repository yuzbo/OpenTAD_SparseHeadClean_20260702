import os
import pickle
import torch
import torch.nn.functional as F


def boundary_choose(score):
    mask_high = score > score.max(dim=1, keepdim=True)[0] * 0.5
    mask_peak = score == F.max_pool1d(score, kernel_size=3, stride=1, padding=1)
    mask = mask_peak | mask_high
    return mask


def save_predictions(predictions, metas, folder):
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]

        file_path = os.path.join(folder, f"{video_name}.pkl")
        prediction = [data[idx] for data in predictions]
        with open(file_path, "wb") as outfile:
            pickle.dump(prediction, outfile, pickle.HIGHEST_PROTOCOL)


def load_single_prediction(metas, folder):
    """Should not be used for sliding window. Since we saved the files with video name, and sliding window will have multiple files with the same name."""
    predictions = []
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]
        file_path = os.path.join(folder, f"{video_name}.pkl")
        with open(file_path, "rb") as infile:
            prediction = pickle.load(infile)
        predictions.append(prediction)

    batched_predictions = []
    for i in range(len(predictions[0])):
        data = torch.stack([prediction[i] for prediction in predictions])
        batched_predictions.append(data)
    return batched_predictions


def load_predictions(metas, infer_cfg):
    if "fuse_list" in infer_cfg.keys():
        predictions = []
        predictions_list = [load_single_prediction(metas, folder) for folder in infer_cfg.fuse_list]
        for i in range(len(predictions_list[0])):
            predictions.append(torch.stack([pred[i] for pred in predictions_list]).mean(dim=0))
        return predictions
    else:
        return load_single_prediction(metas, infer_cfg.folder)


def selected_axis_to_dense_axis(coords, meta):
    positions = meta.get("irregular_selected_positions", None)
    valid_len = meta.get("irregular_selected_valid_len", None)
    if positions is None or valid_len is None or meta.get("irregular_native_axis", False):
        return coords

    positions = torch.as_tensor(positions, dtype=coords.dtype, device=coords.device).reshape(-1)
    if positions.numel() == 0:
        return coords

    xp = torch.arange(positions.numel(), dtype=coords.dtype, device=coords.device)
    xp = torch.cat([xp, xp.new_tensor([float(positions.numel())])], dim=0)
    fp = torch.cat([positions, positions.new_tensor([float(valid_len)])], dim=0)

    coord_shape = coords.shape
    coord_flat = coords.reshape(-1).clamp(min=0.0, max=float(positions.numel()))
    right_idx = torch.searchsorted(xp, coord_flat, right=True).clamp(min=1, max=xp.numel() - 1)
    left_idx = right_idx - 1
    x0 = xp[left_idx]
    x1 = xp[right_idx]
    y0 = fp[left_idx]
    y1 = fp[right_idx]
    weight = (coord_flat - x0) / (x1 - x0).clamp(min=1e-6)
    return (y0 + weight * (y1 - y0)).reshape(coord_shape)


def sparse_visibility_support(segments, meta, min_support=0.0):
    """Estimate class-agnostic support of selected-axis proposals.

    A support of 1 means the proposal lies in locally typical selected-frame
    density. Values below 1 indicate that one endpoint or the proposal span
    crosses a larger-than-average gap in the sparse selected positions.
    """

    support = torch.ones(segments.shape[0], dtype=segments.dtype, device=segments.device)
    positions = meta.get("irregular_selected_positions", None)
    valid_len = meta.get("irregular_selected_valid_len", None)
    if (
        positions is None
        or valid_len is None
        or meta.get("irregular_native_axis", False)
        or segments.numel() == 0
    ):
        return support

    positions = torch.as_tensor(positions, dtype=segments.dtype, device=segments.device).reshape(-1)
    if positions.numel() < 2:
        return support

    valid_len = float(valid_len)
    if valid_len <= 0:
        return support

    fp = torch.cat([positions, positions.new_tensor([valid_len])], dim=0)
    local_gap = (fp[1:] - fp[:-1]).clamp(min=1e-6)
    expected_gap = max(valid_len / float(positions.numel()), 1e-6)

    coords = segments.clamp(min=0.0, max=float(positions.numel()))
    endpoint_idx = torch.floor(coords).to(dtype=torch.long).clamp(min=0, max=local_gap.numel() - 1)
    endpoint_support = (expected_gap / local_gap[endpoint_idx]).clamp(max=1.0).min(dim=1).values

    dense_segments = selected_axis_to_dense_axis(coords, meta)
    selected_len = (coords[:, 1] - coords[:, 0]).clamp(min=1e-6)
    dense_len = (dense_segments[:, 1] - dense_segments[:, 0]).clamp(min=1e-6)
    span_support = (selected_len * expected_gap / dense_len).clamp(max=1.0)

    support = torch.minimum(endpoint_support, span_support)
    return support.clamp(min=float(min_support), max=1.0)


def apply_visibility_rescore(scores, segments, meta, cfg=None):
    if cfg is None or not bool(cfg.get("enabled", False)):
        return scores

    gamma = float(cfg.get("gamma", 0.0))
    if gamma <= 0:
        return scores

    support = sparse_visibility_support(
        segments,
        meta,
        min_support=float(cfg.get("min_support", 0.0)),
    )
    factor = support.pow(gamma).to(dtype=scores.dtype, device=scores.device)
    if scores.dim() == 2:
        factor = factor.unsqueeze(-1)
    return scores * factor


def convert_to_seconds(segments, meta):
    if meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0
        irregular_positions = meta.get("irregular_selected_positions", None)
        irregular_valid_len = meta.get("irregular_selected_valid_len", None)
        if irregular_positions is not None and irregular_valid_len is not None and not meta.get("irregular_native_axis", False):
            segments = selected_axis_to_dense_axis(segments, meta)
        segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration]
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        segments[segments >= meta["duration"]] = segments[segments >= meta["duration"]] * 0.0 + meta["duration"]
    return segments
