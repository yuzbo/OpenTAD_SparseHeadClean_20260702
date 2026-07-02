import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = "configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify DenseAdapter and PointGenerator coordinate alignment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path, absolute or relative to repo root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Dataset split to load.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the single diagnostic batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the diagnostic batch.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def resolve_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path


def move_tensor_list(values, device, dtype=None):
    moved = []
    for value in values:
        if torch.is_tensor(value):
            value = value.to(device=device)
            if dtype is not None:
                value = value.to(dtype=dtype)
        moved.append(value)
    return moved


def move_temporal_grids(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: item.to(device=device) if torch.is_tensor(item) else item for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_temporal_grids(item, device) for item in value)
    return value


def tensor_range(value):
    if value is None or value.numel() == 0:
        return None
    flat = value.detach().float().reshape(-1).cpu()
    return float(flat.min().item()), float(flat.max().item())


def format_range(value):
    if value is None:
        return "empty"
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def gt_segment_range(gt_segments):
    segments = []
    for item in gt_segments:
        if torch.is_tensor(item) and item.numel() > 0:
            segments.append(item.detach().float().reshape(-1, 2).cpu())
    if not segments:
        return None
    return tensor_range(torch.cat(segments, dim=0))


def build_batch(cfg, split, batch_size, num_workers):
    from opentad.datasets import build_dataset, build_dataloader

    dataset_cfg = getattr(cfg.dataset, split)
    dataset = build_dataset(dataset_cfg)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return next(iter(loader))


def run_component_forward(model, batch, device):
    inputs = batch["inputs"].to(device=device)
    masks = batch["masks"].to(device=device).bool()
    metas = batch.get("metas", None)
    temporal_grids = move_temporal_grids(batch.get("temporal_grids", None), device)

    gt_segments = move_tensor_list(batch.get("gt_segments", []), device, dtype=torch.float32)
    gt_labels = move_tensor_list(batch.get("gt_labels", []), device, dtype=torch.long)

    x = model.backbone(inputs, metas=metas) if model.with_backbone else inputs
    if temporal_grids is None and hasattr(model, "_temporal_grid_from_metas"):
        temporal_grids = model._temporal_grid_from_metas(metas, masks)
    x, masks, temporal_grid = model.pad_data(x, masks, temporal_grids)

    feat_list, mask_list, temporal_grid_list = model.projection(x, masks, temporal_grid)
    if not model.with_neck:
        raise RuntimeError("The selected config does not build a neck; DenseAdapter coordinates cannot be checked.")
    feat_list, mask_list, temporal_grid_list = model.neck(feat_list, mask_list, temporal_grid_list)

    if model.with_rpn_head:
        if gt_segments and gt_labels:
            model.rpn_head.forward_train(
                feat_list,
                mask_list,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                temporal_grid_list=temporal_grid_list,
            )
        else:
            model.rpn_head.forward_test(feat_list, mask_list, temporal_grid_list=temporal_grid_list)

    return feat_list, mask_list, temporal_grid_list, gt_segments


def build_points(prior_generator, feat_list, temporal_grid_list):
    try:
        return prior_generator(feat_list)
    except TypeError:
        return prior_generator(feat_list, temporal_grid_list)


def report_level(level_idx, feat, dense_grid, points, gt_range, atol):
    dense_center = dense_grid["center"].detach().float().cpu()
    point_center = points[:, 0].detach().float().cpu()
    dense_range = tensor_range(dense_center)
    point_range = tensor_range(point_center)

    if dense_center.shape[1] != point_center.numel():
        mismatch = True
        max_abs_diff = None
    else:
        expected = point_center[None].expand_as(dense_center)
        max_abs_diff = float((dense_center - expected).abs().max().item())
        mismatch = max_abs_diff > atol

    stride = float(points[0, 3].detach().cpu().item()) if points.numel() > 0 and points.shape[1] >= 4 else None
    valid_mask = dense_grid["valid_mask"].detach().cpu()
    valid_count = int(valid_mask.sum().item())
    total_points = int(valid_mask.numel())

    print(f"Level {level_idx}")
    print(f"  feature_shape: {tuple(feat.shape)}")
    print(f"  valid_points: {valid_count}/{total_points}")
    print(f"  stride: {stride:g}" if stride is not None else "  stride: unknown")
    print(f"  dense_center_range: {format_range(dense_range)}")
    print(f"  point_generator_center_range: {format_range(point_range)}")
    print(f"  gt_segment_range: {format_range(gt_range)}")
    if max_abs_diff is None:
        print("  coordinate_mismatch: True (length mismatch)")
    else:
        print(f"  max_abs_center_diff: {max_abs_diff:.6g}")
        print(f"  coordinate_mismatch: {mismatch}")


def main():
    args = parse_args()
    config_path = resolve_config(args.config)
    device = torch.device(args.device)

    from mmengine.config import Config
    from opentad.models import build_detector

    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model).to(device)
    model.eval()

    batch = build_batch(cfg, args.split, args.batch_size, args.num_workers)
    with torch.no_grad():
        feat_list, mask_list, dense_grid_list, gt_segments = run_component_forward(model, batch, device)
        points_list = build_points(model.rpn_head.prior_generator, feat_list, dense_grid_list)

    gt_range = gt_segment_range(gt_segments)
    print(f"config: {config_path}")
    print(f"split: {args.split}")
    print(f"device: {device}")
    print(f"num_levels: {len(feat_list)}")

    for level_idx, (feat, dense_grid, points) in enumerate(zip(feat_list, dense_grid_list, points_list)):
        report_level(level_idx, feat, dense_grid, points, gt_range, atol=1e-4)


if __name__ == "__main__":
    main()
