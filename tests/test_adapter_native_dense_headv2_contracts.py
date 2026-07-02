from pathlib import Path
import importlib.util
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_module(rel_path, name):
    module_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_torch_or_skip():
    if sys.platform.startswith("win"):
        pytest.skip("torch tensor tests run on the Linux training environment")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch import unavailable in this environment: {exc}")
    return torch


def load_mmengine_config_or_skip(rel_path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(ROOT / rel_path))


def test_adapter_native_dense_headv2_safe_config_and_launcher_contract():
    detector = read("opentad/models/detectors/irregular_actionformer.py")
    temporal_grid = read("opentad/models/utils/temporal_grid.py")
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_native_dense_headv2_safe.py")
    script = read("scripts/run_adapter_native_dense_headv2_safe.sh")

    assert "from ..utils import build_temporal_grid, normalize_temporal_grid_input" in detector
    assert 'if meta.get("irregular_native_axis", False):' in detector
    assert "native_end - float(pos[-1].item())" in detector
    assert "valid_len = max(int(round(float(valid_len))), 1)" in detector
    assert "valid_len = max(min(valid_len, target_len), 1)" not in detector

    assert 'cell_left = temporal_grid.get("cell_left", None)' in temporal_grid
    assert 'cell_right = temporal_grid.get("cell_right", None)' in temporal_grid
    assert "cell_left and cell_right must be provided together" in temporal_grid

    assert '_base_ = ["./input_random_fixed_50pct_adapter_irregular_actionformer_base.py"]' in config
    assert "use_irregular_time_embed=False" in config
    assert "add_irregular_time_embed=False" in config
    assert 'type="DensePassthroughConv1DTransformerProj"' in config
    assert 'type="DensePassthroughFPNIdentity"' in config
    assert 'type="IrregularActionFormerHeadV2"' in config
    assert 'type="IrregularPointGeneratorV2"' in config
    assert "input_pdrop=0.0" in config
    assert "input_pdrop=0.2" not in config

    assert "input_random_fixed_50pct_adapter_native_dense_headv2_safe.py" in script
    assert "adapter_native_dense_headv2_after_review.ok" in script
    assert "require_approval" in script
    assert 'cfg.model.type == "IrregularActionFormer"' in script
    assert 'cfg.model.backbone.backbone.type == "VisionTransformerAdapter"' in script
    assert "use_irregular_time_embed" in script
    assert 'cfg.model.projection.type == "DensePassthroughConv1DTransformerProj"' in script
    assert 'cfg.model.neck.type == "DensePassthroughFPNIdentity"' in script
    assert 'cfg.model.rpn_head.type == "IrregularActionFormerHeadV2"' in script
    assert 'cfg.model.rpn_head.prior_generator.type == "IrregularPointGeneratorV2"' in script
    assert 'train_load.method == "random_fixed_subsample"' in script
    assert "train_load.keep_ratio" in script
    assert "train_load.remap_gt_to_selected_axis" in script
    assert 'val_load.method == "random_fixed_subsample"' in script
    assert "val_load.keep_ratio" in script
    assert "val_load.remap_gt_to_selected_axis" in script
    assert 'test_load.method == "random_fixed_subsample"' in script
    assert "test_load.keep_ratio" in script
    assert "test_load.remap_gt_to_selected_axis" in script
    assert "tools/train.py" in script


def test_adapter_native_dense_headv2_config_loads_native_axis_contract_with_mmengine():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_native_dense_headv2_safe.py"
    )

    backbone = cfg.model.backbone.backbone
    head = cfg.model.rpn_head
    assert cfg.model.type == "IrregularActionFormer"
    assert backbone.type == "VisionTransformerAdapter"
    assert not bool(backbone.use_irregular_time_embed)
    assert not bool(backbone.add_irregular_time_embed)
    assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert abs(float(cfg.model.projection.get("input_pdrop", 0.0))) < 1e-12
    assert cfg.model.neck.type == "DensePassthroughFPNIdentity"
    assert head.type == "IrregularActionFormerHeadV2"
    assert head.prior_generator.type == "IrregularPointGeneratorV2"

    train_load = next(step for step in cfg.dataset.train.pipeline if step.get("type") == "LoadFrames")
    val_load = next(step for step in cfg.dataset.val.pipeline if step.get("type") == "LoadFrames")
    test_load = next(step for step in cfg.dataset.test.pipeline if step.get("type") == "LoadFrames")
    assert train_load.method == "random_fixed_subsample"
    assert abs(float(train_load.keep_ratio) - 0.5) < 1e-12
    assert train_load.method_base == "random_trunc"
    assert not bool(train_load.remap_gt_to_selected_axis)
    assert val_load.method == "random_fixed_subsample"
    assert abs(float(val_load.keep_ratio) - 0.5) < 1e-12
    assert val_load.method_base == "sliding_window"
    assert not bool(val_load.remap_gt_to_selected_axis)
    assert test_load.method == "random_fixed_subsample"
    assert abs(float(test_load.keep_ratio) - 0.5) < 1e-12
    assert test_load.method_base == "sliding_window"
    assert not bool(test_load.remap_gt_to_selected_axis)
    assert int(cfg.solver.train.batch_size) == 2
    assert int(cfg.solver.val.batch_size) == 2
    assert int(cfg.solver.test.batch_size) == 2
    assert int(cfg.workflow.checkpoint_interval) == 10
    assert not bool(cfg.workflow.get("disable_checkpoint", False))
    assert int(cfg.workflow.val_start_epoch) == 40
    assert int(cfg.workflow.val_eval_interval) == 2
    assert "input_random_fixed_50pct_adapter_native_dense_headv2_safe" in cfg.work_dir


def test_temporal_grid_explicit_cells_are_preserved_on_linux():
    torch = import_torch_or_skip()
    temporal_grid = load_module("opentad/models/utils/temporal_grid.py", "temporal_grid_explicit_cells")

    center = torch.tensor([[0.0, 2.0, 5.0, 100.0]], dtype=torch.float32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    left = torch.tensor([[2.0, 2.0, 3.0, 95.0]], dtype=torch.float32)
    right = torch.tensor([[2.0, 3.0, 95.0, 668.0]], dtype=torch.float32)

    grid = temporal_grid.normalize_temporal_grid_input(
        {"center": center, "fresh_mask": mask, "cell_left": left, "cell_right": right},
        mask,
    )

    assert torch.allclose(grid["center"], center)
    assert torch.allclose(grid["cell_left"], left)
    assert torch.allclose(grid["cell_right"], right)
    assert torch.allclose(grid["level_scale"], torch.tensor([108.75]))


def test_irregular_actionformer_native_grid_preserves_dense_right_boundary_on_linux():
    torch = import_torch_or_skip()
    irregular = pytest.importorskip("opentad.models.detectors.irregular_actionformer")

    detector = object.__new__(irregular.IrregularActionFormer)
    masks = torch.ones(1, 4, dtype=torch.bool)
    metas = [
        {
            "irregular_selected_positions": [0.0, 2.0, 5.0, 100.0],
            "irregular_selected_valid_len": 768.0,
            "irregular_native_axis": True,
        }
    ]

    grid = detector._temporal_grid_from_metas(metas, masks)

    assert torch.allclose(grid["center"], torch.tensor([[0.0, 2.0, 5.0, 100.0]]))
    assert torch.allclose(grid["cell_left"], torch.tensor([[2.0, 2.0, 3.0, 95.0]]))
    assert torch.allclose(grid["cell_right"], torch.tensor([[2.0, 3.0, 95.0, 668.0]]))
    assert torch.equal(grid["valid_mask"], masks)
    assert torch.equal(grid["fresh_mask"], masks)
