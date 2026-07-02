import torch
import torch.nn.functional as F


def _masked_mean(value, mask, dim=-1, keepdim=False, eps=1e-6):
    weight = mask.to(value.dtype)
    numer = (value * weight).sum(dim=dim, keepdim=keepdim)
    denom = weight.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return numer / denom


def clone_temporal_grid(grid):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in grid.items()}


def normalize_temporal_grid_input(temporal_grid, mask, device=None, dtype=torch.float32):
    cell_left = None
    cell_right = None
    if temporal_grid is None:
        batch, length = mask.shape
        center = torch.arange(length, device=mask.device, dtype=dtype)[None].repeat(batch, 1)
        fresh_mask = mask.clone()
    elif torch.is_tensor(temporal_grid):
        center = temporal_grid.to(device=mask.device if device is None else device, dtype=dtype)
        fresh_mask = mask.clone()
    else:
        center = temporal_grid["center"].to(device=mask.device if device is None else device, dtype=dtype)
        fresh_mask = temporal_grid.get("fresh_mask", mask).to(mask.device).bool()
        cell_left = temporal_grid.get("cell_left", None)
        cell_right = temporal_grid.get("cell_right", None)
        if cell_left is not None:
            cell_left = cell_left.to(device=mask.device if device is None else device, dtype=dtype)
        if cell_right is not None:
            cell_right = cell_right.to(device=mask.device if device is None else device, dtype=dtype)

    valid_mask = mask.bool()
    return build_temporal_grid(
        center,
        valid_mask=valid_mask,
        fresh_mask=fresh_mask,
        cell_left=cell_left,
        cell_right=cell_right,
    )


def build_temporal_grid(center, valid_mask, fresh_mask=None, cell_left=None, cell_right=None, min_scale=1e-4):
    center = center.to(dtype=torch.float32)
    valid_mask = valid_mask.bool()
    if fresh_mask is None:
        fresh_mask = valid_mask
    fresh_mask = fresh_mask.bool()

    batch, length = center.shape
    if cell_left is not None or cell_right is not None:
        if cell_left is None or cell_right is None:
            raise ValueError("cell_left and cell_right must be provided together.")
        left = cell_left.to(device=center.device, dtype=center.dtype)
        right = cell_right.to(device=center.device, dtype=center.dtype)
        if left.shape != center.shape or right.shape != center.shape:
            raise ValueError(
                "cell_left and cell_right must match center shape: "
                f"center={tuple(center.shape)}, left={tuple(left.shape)}, right={tuple(right.shape)}"
            )
    elif length == 1:
        gap = torch.ones_like(center)
        left = gap
        right = gap
    else:
        delta = center[:, 1:] - center[:, :-1]
        delta = delta.clamp_min(min_scale)

        left = torch.empty_like(center)
        right = torch.empty_like(center)
        left[:, 1:] = delta
        left[:, 0] = delta[:, 0]
        right[:, :-1] = delta
        right[:, -1] = delta[:, -1]

    level_scale = _masked_mean(0.5 * (left + right), valid_mask, dim=1)
    return {
        "center": center,
        "cell_left": left.clamp_min(min_scale),
        "cell_right": right.clamp_min(min_scale),
        "valid_mask": valid_mask,
        "fresh_mask": fresh_mask,
        "level_scale": level_scale,
    }


def downsample_temporal_grid(grid, min_scale=1e-4):
    center = grid["center"].to(dtype=torch.float32)
    valid_mask = grid["valid_mask"].bool()
    fresh_mask = grid["fresh_mask"].bool()
    cell_left = grid["cell_left"].to(dtype=center.dtype).clamp_min(min_scale)
    cell_right = grid["cell_right"].to(dtype=center.dtype).clamp_min(min_scale)

    def _split_and_pad(tensor, pad_value):
        even = tensor[:, 0::2]
        odd = tensor[:, 1::2]
        if odd.shape[1] < even.shape[1]:
            pad_len = even.shape[1] - odd.shape[1]
            pad_tensor = tensor.new_full((tensor.shape[0], pad_len), pad_value)
            odd = torch.cat([odd, pad_tensor], dim=1)
        return even, odd

    even_center, odd_center = _split_and_pad(center, 0.0)
    even_valid, odd_valid = _split_and_pad(valid_mask, False)
    even_fresh, odd_fresh = _split_and_pad(fresh_mask, False)
    even_left, odd_left = _split_and_pad(cell_left, 0.0)
    even_right, odd_right = _split_and_pad(cell_right, 0.0)

    pair_valid = even_valid | odd_valid

    even_width = 0.5 * (even_left + even_right) * even_valid.to(center.dtype)
    odd_width = 0.5 * (odd_left + odd_right) * odd_valid.to(center.dtype)
    pair_weight = even_width + odd_width
    fallback_center = torch.where(
        even_valid,
        even_center,
        torch.where(odd_valid, odd_center, even_center.new_zeros(even_center.shape)),
    )
    merged_center = torch.where(
        pair_valid,
        (even_center * even_width + odd_center * odd_width) / pair_weight.clamp_min(min_scale),
        fallback_center,
    )

    even_start = even_center - 0.5 * even_left
    even_end = even_center + 0.5 * even_right
    odd_start = odd_center - 0.5 * odd_left
    odd_end = odd_center + 0.5 * odd_right

    inf = center.new_full(even_center.shape, float("inf"))
    neg_inf = center.new_full(even_center.shape, float("-inf"))
    merged_start = torch.minimum(
        torch.where(even_valid, even_start, inf),
        torch.where(odd_valid, odd_start, inf),
    )
    merged_end = torch.maximum(
        torch.where(even_valid, even_end, neg_inf),
        torch.where(odd_valid, odd_end, neg_inf),
    )
    merged_start = torch.where(pair_valid, merged_start, merged_center)
    merged_end = torch.where(pair_valid, merged_end, merged_center)

    merged_left = (2.0 * (merged_center - merged_start)).clamp_min(min_scale)
    merged_right = (2.0 * (merged_end - merged_center)).clamp_min(min_scale)
    merged_fresh = (even_fresh & even_valid) | (odd_fresh & odd_valid)
    level_scale = _masked_mean(0.5 * (merged_left + merged_right), pair_valid, dim=1)

    return {
        "center": merged_center,
        "cell_left": merged_left,
        "cell_right": merged_right,
        "valid_mask": pair_valid,
        "fresh_mask": merged_fresh,
        "level_scale": level_scale,
    }


def linear_interpolate_features(source_feat, source_grid, target_grid):
    source_center = source_grid["center"]
    source_valid = source_grid["valid_mask"]
    target_center = target_grid["center"]
    target_valid = target_grid["valid_mask"]

    batch, channels, _ = source_feat.shape
    out = source_feat.new_zeros(batch, channels, target_center.shape[1])

    for b in range(batch):
        src_mask = source_valid[b]
        tgt_mask = target_valid[b]
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            continue

        src_x = source_center[b, src_mask]
        src_y = source_feat[b, :, src_mask]
        tgt_x = target_center[b, tgt_mask]

        if src_x.numel() == 1:
            out[b, :, tgt_mask] = src_y[:, :1].expand(-1, tgt_x.numel())
            continue

        right_idx = torch.searchsorted(src_x, tgt_x, right=True)
        right_idx = right_idx.clamp(max=src_x.numel() - 1)
        left_idx = (right_idx - 1).clamp(min=0)

        x0 = src_x[left_idx]
        x1 = src_x[right_idx]
        same = (right_idx == left_idx) | ((x1 - x0).abs() < 1e-6)
        alpha = torch.where(same, torch.zeros_like(tgt_x), (tgt_x - x0) / (x1 - x0).clamp_min(1e-6))

        y0 = src_y[:, left_idx]
        y1 = src_y[:, right_idx]
        interp = y0 * (1.0 - alpha.unsqueeze(0)) + y1 * alpha.unsqueeze(0)
        out[b, :, tgt_mask] = interp

    out = out * target_valid.unsqueeze(1).to(out.dtype)
    return out
