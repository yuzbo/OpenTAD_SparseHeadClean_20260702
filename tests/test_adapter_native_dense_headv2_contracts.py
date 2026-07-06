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


def load_frame_steps(cfg):
    return {
        split: next(step for step in getattr(cfg.dataset, split).pipeline if step.get("type") == "LoadFrames")
        for split in ("train", "val", "test")
    }


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


def test_clean_repo_gitignore_keeps_source_dataset_packages_visible():
    gitignore = read(".gitignore").splitlines()

    assert "/datasets/" in gitignore
    assert "datasets/" not in gitignore


def test_irregular_headv3_default_does_not_train_unused_boundary_auxiliary():
    head_impl = read("opentad/models/dense_heads/irregular_actionformer_head_v3.py")

    assert "boundary_loss_weight=0.0" in head_impl


def test_adapter_sparse_headv3_recommended_config_does_not_train_unused_boundary_auxiliary():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"
    )

    head = cfg.model.rpn_head
    boundary_weight = float(head.get("boundary_loss_weight", 0.0))
    boundary_inference = head.get("boundary_inference", {})

    assert boundary_weight == 0.0 or bool(boundary_inference.get("enabled", False))


def test_adapter_sparse_headv3_n16r4_ablation_configs_are_isolated():
    nogeometry = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_nogeometry_n16r4.py"
    )
    reggate = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_reggate_n16r4.py"
    )

    assert nogeometry.model.rpn_head.type == "IrregularActionFormerHeadV3"
    assert float(nogeometry.model.rpn_head.boundary_loss_weight) == 0.0
    assert float(nogeometry.model.rpn_head.geometry_scale) == 0.0
    assert not bool(nogeometry.model.rpn_head.get("use_regress_range", False))
    assert "nogeometry_n16r4" in nogeometry.work_dir

    assert reggate.model.rpn_head.type == "IrregularActionFormerHeadV3"
    assert float(reggate.model.rpn_head.boundary_loss_weight) == 0.0
    assert float(reggate.model.rpn_head.geometry_scale) == 0.25
    assert bool(reggate.model.rpn_head.use_regress_range)
    assert "reggate_n16r4" in reggate.work_dir


def test_irregular_headv2_soft_assignment_has_optional_regression_range_gate():
    head_impl = read("opentad/models/dense_heads/irregular_actionformer_head_v2.py")
    head_v3_impl = read("opentad/models/dense_heads/irregular_actionformer_head_v3.py")

    assert "use_regress_range=False" in head_impl
    assert "self.use_regress_range = use_regress_range" in head_impl
    assert "inside_regress_range" in head_impl
    assert "candidate_mask = torch.logical_and(candidate_mask, inside_regress_range)" in head_impl
    assert "use_regress_range=False" in head_v3_impl
    assert "use_regress_range=use_regress_range" in head_v3_impl


def test_adapter_sparse_bridge_dense_like_configs_cover_assignment_and_regression_axes():
    hard_linear = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"
    )
    hard_log = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_log_n16r4.py"
    )
    soft_topk1 = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_soft_topk1_binary_linear_n16r4.py"
    )
    openrange = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py"
    )
    absrange = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"
    )
    absrange_radiuslevel = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py"
    )
    levelstride = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py"
    )
    absrange_expanded = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py"
    )

    for cfg in (
        hard_linear,
        hard_log,
        soft_topk1,
        openrange,
        absrange,
        absrange_radiuslevel,
        levelstride,
        absrange_expanded,
    ):
        head = cfg.model.rpn_head
        assert cfg.model.projection.type == "GridAwareConv1DTransformerProj"
        assert cfg.model.neck.type == "GridAwareFPNIdentity"
        assert head.type == "IrregularActionFormerBridgeHead"
        assert head.prior_generator.type == "IrregularPointGeneratorV2"
        assert int(head.num_classes) == 20
        assert int(head.num_convs) == 2
        assert float(head.center_sample_radius) == 1.5
        assert "adapter_irregular_bridge" in cfg.work_dir
        assert "n16r4" in cfg.work_dir

    assert hard_linear.model.rpn_head.assignment_mode == "hard"
    assert hard_linear.model.rpn_head.regression_mode == "symmetric_linear"
    assert "hard_linear_n16r4" in hard_linear.work_dir

    assert hard_log.model.rpn_head.assignment_mode == "hard"
    assert hard_log.model.rpn_head.regression_mode == "asymmetric_log1p"
    assert "hard_log_n16r4" in hard_log.work_dir

    assert soft_topk1.model.rpn_head.assignment_mode == "soft"
    assert soft_topk1.model.rpn_head.regression_mode == "symmetric_linear"
    assert int(soft_topk1.model.rpn_head.soft_assign_topk) == 1
    assert soft_topk1.model.rpn_head.soft_reg_weight_mode == "binary"
    assert soft_topk1.model.rpn_head.soft_cls_target_mode == "binary"
    assert soft_topk1.model.rpn_head.soft_loss_normalizer_mode == "pos_count"
    assert "soft_topk1_binary_linear_n16r4" in soft_topk1.work_dir

    assert openrange.model.rpn_head.assignment_mode == "hard"
    assert openrange.model.rpn_head.regression_mode == "symmetric_linear"
    assert openrange.model.rpn_head.prior_generator.range_mode == "open"
    assert all(tuple(item) == (0, 10000) for item in openrange.model.rpn_head.prior_generator.regression_range)
    assert "hard_linear_openrange_n16r4" in openrange.work_dir

    assert absrange.model.rpn_head.assignment_mode == "hard"
    assert absrange.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange.model.rpn_head.prior_generator.range_mode == "absolute"
    assert tuple(absrange.model.rpn_head.prior_generator.regression_range[2]) == (8, 16)
    assert "hard_linear_absrange_n16r4" in absrange.work_dir

    assert absrange_radiuslevel.model.rpn_head.assignment_mode == "hard"
    assert absrange_radiuslevel.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange_radiuslevel.model.rpn_head.center_radius_scale == "point_radius"
    assert absrange_radiuslevel.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert absrange_radiuslevel.model.rpn_head.prior_generator.range_mode == "absolute"
    assert absrange_radiuslevel.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert absrange_radiuslevel.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert tuple(absrange_radiuslevel.model.rpn_head.prior_generator.regression_range[2]) == (8, 16)
    assert "hard_linear_absrange_radiuslevel_n16r4" in absrange_radiuslevel.work_dir

    assert levelstride.model.rpn_head.assignment_mode == "hard"
    assert levelstride.model.rpn_head.regression_mode == "symmetric_linear"
    assert levelstride.model.rpn_head.center_radius_scale == "point_radius"
    assert levelstride.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert levelstride.model.rpn_head.prior_generator.range_mode == "level_stride"
    assert levelstride.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert levelstride.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert "hard_linear_levelstride_n16r4" in levelstride.work_dir

    assert absrange_expanded.model.rpn_head.assignment_mode == "hard"
    assert absrange_expanded.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange_expanded.model.rpn_head.center_radius_scale == "point_radius"
    assert absrange_expanded.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert absrange_expanded.model.rpn_head.prior_generator.range_mode == "absolute"
    assert absrange_expanded.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert absrange_expanded.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert [tuple(item) for item in absrange_expanded.model.rpn_head.prior_generator.regression_range] == [
        (0, 8),
        (2, 16),
        (4, 32),
        (8, 64),
        (16, 128),
        (32, 10000),
    ]
    assert "hard_linear_absrange_expanded_n16r4" in absrange_expanded.work_dir


def test_bridge_head_exposes_explicit_radius_and_regression_scale_modes():
    bridge_impl = read("opentad/models/dense_heads/irregular_actionformer_bridge_head.py")

    assert "center_radius_scale=\"full_cell_span\"" in bridge_impl
    assert "reg_denom_mode=\"full_cell_span\"" in bridge_impl
    assert "def _scale_base(" in bridge_impl
    assert "def _point_fields_extended(" in bridge_impl
    assert 'mode == "full_cell_span"' in bridge_impl
    assert 'mode == "half_cell_span"' in bridge_impl
    assert 'mode == "min_side"' in bridge_impl
    assert 'mode == "left_right_mean"' in bridge_impl
    assert 'mode == "point_range"' in bridge_impl
    assert 'mode == "point_radius"' in bridge_impl


def test_sparse_head_assignment_audit_tool_contract():
    script = read("tools/audit_sparse_head_assignment.py")

    assert "sample_id" in script
    assert "video_name" in script
    assert "gt_axis" in script
    assert "proposal_axis" in script
    assert "postprocess_axis" in script
    assert "gt_length_bucket" in script
    assert "gt_covered_any" in script
    assert "gt_covered_level_bitmap" in script
    assert "per_level_pos_count" in script
    assert "per_level_candidate_count_after_range" in script
    assert "range_fail_count_by_level" in script
    assert "center_fail_count_by_level" in script
    assert "decode_reconstruction_max_error" in script
    assert "radius_base_p50" in script
    assert "valid_mask_true_count" in script
    assert "--configs" in script
    assert "--num-batches" in script
    assert "--seed" in script


def test_detection_quality_analyzer_reports_high_iou_recall_and_boundary_error(tmp_path):
    analyzer = load_module("tools/analyze_detection_quality.py", "detection_quality_analyzer")

    gt = {
        "database": {
            "video_1": {
                "subset": "validation",
                "annotations": [
                    {"segment": [0.0, 10.0], "label": "A"},
                    {"segment": [20.0, 30.0], "label": "A"},
                ],
            },
            "video_2": {
                "subset": "validation",
                "annotations": [
                    {"segment": [0.0, 4.0], "label": "A"},
                ],
            },
        }
    }
    pred = {
        "results": {
            "video_1": [
                {"segment": [0.0, 10.0], "score": 0.9, "label": "A"},
                {"segment": [19.0, 31.0], "score": 0.8, "label": "A"},
                {"segment": [40.0, 50.0], "score": 0.7, "label": "A"},
            ],
            "video_2": [
                {"segment": [0.0, 2.0], "score": 0.6, "label": "A"},
            ],
        }
    }

    summary, rows = analyzer.summarize_detection_quality(
        gt,
        pred,
        subset="validation",
        tiou_thresholds=(0.5, 0.7, 0.9),
        topk_per_video=None,
        label_aware=True,
    )

    assert summary["num_gt"] == 3
    assert summary["num_predictions"] == 4
    assert summary["recall@0.50"] == pytest.approx(1.0)
    assert summary["recall@0.70"] == pytest.approx(2.0 / 3.0)
    assert summary["recall@0.90"] == pytest.approx(1.0 / 3.0)
    assert summary["best_iou_mean"] == pytest.approx((1.0 + (10.0 / 12.0) + 0.5) / 3.0)
    assert summary["boundary_error_mean_p50"] == pytest.approx(0.1)

    assert len(rows) == 3
    assert rows[0]["video_id"] == "video_1"
    assert rows[0]["best_iou"] == pytest.approx(1.0)
    assert rows[1]["best_iou"] == pytest.approx(10.0 / 12.0)
    assert rows[2]["length_bucket"] == "[4,8)"
    assert summary["bucket:[4,8):recall@0.50"] == pytest.approx(1.0)


def test_bridge_head_half_cell_scale_mode_on_linux():
    torch = import_torch_or_skip()
    mmengine_config = pytest.importorskip("mmengine.config")
    bridge_head = pytest.importorskip("opentad.models.dense_heads.irregular_actionformer_bridge_head")
    config_dict = mmengine_config.ConfigDict

    head = bridge_head.IrregularActionFormerBridgeHead(
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=1,
        prior_generator=config_dict(
            type="IrregularPointGeneratorV2",
            strides=[1],
            regression_range=[(0, 4)],
            range_mode="absolute",
        ),
        loss=config_dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        center_radius_scale="half_cell_span",
        reg_denom_mode="half_cell_span",
    )
    left_scale = torch.tensor([2.0, 4.0])
    right_scale = torch.tensor([2.0, 8.0])
    point_scale = left_scale + right_scale

    assert torch.allclose(
        head._scale_base(left_scale, right_scale, point_scale, "half_cell_span"),
        torch.tensor([2.0, 6.0]),
    )

    encoded = head._encode_regression_targets(
        torch.tensor([2.0, 6.0]),
        torch.tensor([4.0, 12.0]),
        left_scale,
        right_scale,
        point_scale,
    )
    assert torch.allclose(encoded, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))


def test_bridge_hard_uniform_grid_matches_official_dense_target_contract_on_linux():
    torch = import_torch_or_skip()
    mmengine_config = pytest.importorskip("mmengine.config")
    bridge_head = pytest.importorskip("opentad.models.dense_heads.irregular_actionformer_bridge_head")
    config_dict = mmengine_config.ConfigDict

    head = bridge_head.IrregularActionFormerBridgeHead(
        num_classes=5,
        in_channels=512,
        feat_channels=512,
        num_convs=1,
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        prior_generator=config_dict(
            type="IrregularPointGeneratorV2",
            strides=[1],
            regression_range=[(0, 4)],
            range_mode="absolute",
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
        loss=config_dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        center_sample="radius",
        center_sample_radius=1.5,
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        debug_cfg=dict(enable=False),
    )

    point = torch.tensor(
        [
            [
                [0.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [5.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )
    gt_segments = [torch.tensor([[1.0, 5.0]])]
    gt_labels = [torch.tensor([2])]

    cls_targets, reg_targets, reg_weights, _ = head._prepare_targets_hard([point], gt_segments, gt_labels)

    assert reg_weights[0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert cls_targets[0][:, 2].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert torch.allclose(
        reg_targets[0],
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_adapter_sparse_cross_over_configs_isolate_projection_neck_from_head():
    dense_head_grid = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_gridaware_n16r4.py"
    )
    bridge_dense_pass = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_densepass_n16r4.py"
    )
    headv3_dense_pass = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_densepass_n16r4.py"
    )

    assert dense_head_grid.model.projection.type == "GridAwareConv1DTransformerProj"
    assert dense_head_grid.model.neck.type == "GridAwareFPNIdentity"
    assert dense_head_grid.model.rpn_head.type == "ActionFormerHead"
    assert dense_head_grid.model.rpn_head.prior_generator.type == "PointGenerator"
    assert "densehead_gridaware_n16r4" in dense_head_grid.work_dir

    assert bridge_dense_pass.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert bridge_dense_pass.model.neck.type == "DensePassthroughFPNIdentity"
    assert bridge_dense_pass.model.rpn_head.type == "IrregularActionFormerBridgeHead"
    assert bridge_dense_pass.model.rpn_head.assignment_mode == "hard"
    assert bridge_dense_pass.model.rpn_head.regression_mode == "symmetric_linear"
    assert "bridge_hard_linear_densepass_n16r4" in bridge_dense_pass.work_dir

    assert headv3_dense_pass.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert headv3_dense_pass.model.neck.type == "DensePassthroughFPNIdentity"
    assert headv3_dense_pass.model.rpn_head.type == "IrregularActionFormerHeadV3"
    assert float(headv3_dense_pass.model.rpn_head.boundary_loss_weight) == 0.0
    assert float(headv3_dense_pass.model.rpn_head.geometry_scale) == 0.25
    assert "headv3_x_pdrop0_densepass_n16r4" in headv3_dense_pass.work_dir


def test_adapter_sparse_configs_make_gt_axis_contract_explicit():
    headv3 = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"
    )
    bridge = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"
    )
    dense_control = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py"
    )
    dense_head_grid = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_gridaware_n16r4.py"
    )

    for cfg in (headv3, bridge):
        for step in load_frame_steps(cfg).values():
            assert not bool(step.remap_gt_to_selected_axis)

    for cfg in (dense_control, dense_head_grid):
        for step in load_frame_steps(cfg).values():
            assert bool(step.remap_gt_to_selected_axis)


def test_loadframes_supports_deterministic_uniform_fixed_subsample_for_equal_interval_control():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")

    assert '"uniform_fixed_subsample"' in load_frames_impl
    assert "keep_positions = self._select_uniform_fixed_positions" in load_frames_impl
    assert "def _select_uniform_fixed_positions(" in load_frames_impl
    assert "return self._expand_selected_units(" in load_frames_impl
    assert "self._uniform_pick_indices(np.arange(total_units), int(target_count))" in load_frames_impl


def test_uniform_fixed_50pct_controls_keep_axis_contracts_and_only_change_sampling():
    dense_uniform = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py"
    )
    bridge_uniform = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py"
    )

    assert dense_uniform.model.rpn_head.type == "ActionFormerHead"
    assert dense_uniform.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert dense_uniform.model.neck.type == "DensePassthroughFPNIdentity"
    assert "input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4" in dense_uniform.work_dir

    assert bridge_uniform.model.rpn_head.type == "IrregularActionFormerBridgeHead"
    assert bridge_uniform.model.rpn_head.assignment_mode == "hard"
    assert bridge_uniform.model.rpn_head.regression_mode == "symmetric_linear"
    assert bridge_uniform.model.rpn_head.prior_generator.range_mode == "absolute"
    assert bridge_uniform.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert bridge_uniform.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert "input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4" in bridge_uniform.work_dir

    for step in load_frame_steps(dense_uniform).values():
        assert step.method == "uniform_fixed_subsample"
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12
        assert bool(step.remap_gt_to_selected_axis)

    for step in load_frame_steps(bridge_uniform).values():
        assert step.method == "uniform_fixed_subsample"
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12
        assert not bool(step.remap_gt_to_selected_axis)


def test_uniform_fixed_50pct_remote_launchers_are_gpu1_slurm_preflighted():
    dense_runner = read("remote_runs/run_gpu1_uniform_fixed_dense_control_long_20260706.sh")
    dense_launcher = read("remote_runs/launch_gpu1_uniform_fixed_dense_control_long_20260706.sh")
    bridge_runner = read("remote_runs/run_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh")
    bridge_launcher = read("remote_runs/launch_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh")

    assert "input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py" in dense_runner
    assert "gpu1_uniform_fixed_dense_control_long" in dense_runner
    assert "head= ActionFormerHead" not in dense_runner
    assert "CUDA_VISIBLE_DEVICES=1" in dense_runner
    assert "config load preflight" in dense_runner
    assert "py_compile preflight" in dense_runner
    assert "torchrun" in dense_runner
    assert "--nproc_per_node=1" in dense_runner

    assert "input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py" in bridge_runner
    assert "gpu1_uniform_fixed_bridge_absrange_expanded_long" in bridge_runner
    assert "CUDA_VISIBLE_DEVICES=1" in bridge_runner
    assert "config load preflight" in bridge_runner
    assert "py_compile preflight" in bridge_runner
    assert "torchrun" in bridge_runner
    assert "--nproc_per_node=1" in bridge_runner

    for launcher in (dense_launcher, bridge_launcher):
        assert "srun --jobid=1118197" in launcher
        assert "--overlap -w g0030 -N1 -n1" in launcher
        assert "CUDA_VISIBLE_DEVICES=1" in launcher


def test_absrange_expanded_waiter_only_launches_after_old_gpu1_step_clears():
    waiter = read("remote_runs/watch_and_launch_gpu1_bridge_absrange_expanded_20260706.sh")
    launcher = read("remote_runs/launch_watch_gpu1_bridge_absrange_expanded_20260706.sh")

    assert 'OLD_STEP_ID="${OLD_STEP_ID:-1118197.621}"' in waiter
    assert 'SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-1118197}"' in waiter
    assert 'TARGET_WORK_DIR="$ROOT/exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4/gpu1_id1"' in waiter
    assert 'TARGET_LAUNCHER="$ROOT/remote_runs/launch_gpu1_bridge_absrange_expanded_long_20260706.sh"' in waiter
    assert 'LOG_DIR="$ROOT/logs/gpu1_bridge_absrange_expanded_waiter"' in waiter
    assert "squeue --steps -j \"$SLURM_JOB_ID_TARGET\"" in waiter
    assert "grep -Fxq \"$OLD_STEP_ID\"" in waiter
    assert "target_already_started" in waiter
    assert "MAX_WAIT_SECONDS" in waiter
    assert "POLL_SECONDS" in waiter
    assert "bash \"$TARGET_LAUNCHER\"" in waiter
    assert "run_gpu1_bridge_absrange_expanded_long_20260706.sh" not in waiter

    assert "nohup bash" in launcher
    assert "watch_and_launch_gpu1_bridge_absrange_expanded_20260706.sh" in launcher
    assert "logs/gpu1_bridge_absrange_expanded_waiter" in launcher


def test_loadframes_records_explicit_axis_contract_metadata():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")

    assert 'axis = "selected" if self.remap_gt_to_selected_axis else "native"' in load_frames_impl
    assert 'results["irregular_gt_axis"] = axis' in load_frames_impl
    assert 'results["irregular_proposal_axis"] = axis' in load_frames_impl
    assert 'results["irregular_postprocess_axis"] = axis' in load_frames_impl
    assert 'results["irregular_axis_contract"]' in load_frames_impl


def test_irregular_actionformer_validates_axis_contract_and_exposes_proposal_dump_helper():
    detector_impl = read("opentad/models/detectors/irregular_actionformer.py")

    assert "def _axis_contract_from_meta(" in detector_impl
    assert "def _assert_axis_contract(" in detector_impl
    assert "def _proposal_axis_debug_records(" in detector_impl
    assert "debug_dump_proposals" in detector_impl
    assert "proposal_axis" in detector_impl
    assert "postprocess_axis" in detector_impl


def test_irregular_actionformer_selected_axis_grid_uses_selected_indices_not_native_positions():
    detector_impl = read("opentad/models/detectors/irregular_actionformer.py")

    assert "selected_center = torch.arange" in detector_impl
    assert '"center": selected_center[:target_len][None]' in detector_impl
    assert "selected-axis proposals must be emitted on the selected index axis" in detector_impl


def test_irregular_actionformer_axis_contract_rejects_mismatched_native_route_on_linux():
    torch = import_torch_or_skip()
    detector = pytest.importorskip("opentad.models.detectors.irregular_actionformer")

    model = object.__new__(detector.IrregularActionFormer)
    good_meta = dict(
        irregular_native_axis=True,
        irregular_gt_axis="native",
        irregular_proposal_axis="native",
        irregular_postprocess_axis="native",
    )
    bad_meta = dict(
        irregular_native_axis=True,
        irregular_gt_axis="native",
        irregular_proposal_axis="selected",
        irregular_postprocess_axis="native",
    )

    assert model._axis_contract_from_meta(good_meta) == ("native", "native", "native")
    model._assert_axis_contract(good_meta, stage="test")
    with pytest.raises(ValueError, match="axis contract"):
        model._assert_axis_contract(bad_meta, stage="test")

    records = model._proposal_axis_debug_records(
        torch.tensor([[0.0, 4.0], [1.0, 5.0]]),
        torch.tensor([0.9, 0.7]),
        torch.tensor([3, 4]),
        dict(
            video_name="video_test_0001",
            fps=1.0,
            snippet_stride=1.0,
            offset_frames=0.0,
            window_start_frame=0.0,
            duration=10.0,
            irregular_native_axis=True,
            irregular_gt_axis="native",
            irregular_proposal_axis="native",
            irregular_postprocess_axis="native",
        ),
        topk=1,
    )
    assert records == [
        dict(
            video_name="video_test_0001",
            rank=0,
            proposal_axis="native",
            postprocess_axis="native",
            label=3,
            score=0.9,
            segment_axis=[0.0, 4.0],
            segment_seconds=[0.0, 4.0],
        )
    ]


def test_official_dense_reference_verifier_tracks_upstream_opentad_sources():
    script = read("scripts/verify_official_dense_reference.py")

    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py" in script
    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/prior_generator/point_generator.py" in script
    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/necks/fpn.py" in script
    assert "difflib.unified_diff" in script
    assert "--official-root" in script


def test_root_cause_notes_record_gt_axis_and_route_sanity_limits():
    notes = read("root-cause-notes.md")

    assert "selected-axis GT" in notes
    assert "native-axis GT" in notes
    assert "route sanity" in notes
    assert "not a strong projection/neck attribution" in notes


def test_irregular_point_generator_v2_full_cell_span_scale_on_linux():
    torch = import_torch_or_skip()
    point_generator = pytest.importorskip("opentad.models.dense_heads.prior_generator.irregular_point_generator")

    generator = point_generator.IrregularPointGeneratorV2(
        strides=[1],
        regression_range=[(1, 2)],
        range_mode="hard",
    )
    feat = torch.zeros(1, 1, 3)
    grid = {
        "center": torch.tensor([[0.0, 2.0, 4.0]]),
        "cell_left": torch.tensor([[2.0, 2.0, 2.0]]),
        "cell_right": torch.tensor([[2.0, 2.0, 2.0]]),
    }

    points = generator([feat], [grid])[0]

    assert torch.allclose(points[..., 0], grid["center"])
    assert torch.allclose(points[..., 1], torch.full((1, 3), 4.0))
    assert torch.allclose(points[..., 2], torch.full((1, 3), 8.0))
    assert torch.allclose(points[..., 3], grid["cell_left"])
    assert torch.allclose(points[..., 4], grid["cell_right"])


def test_irregular_point_generator_v2_level_stride_separates_range_decode_radius_on_linux():
    torch = import_torch_or_skip()
    point_generator = pytest.importorskip("opentad.models.dense_heads.prior_generator.irregular_point_generator")

    generator = point_generator.IrregularPointGeneratorV2(
        strides=[4],
        regression_range=[(1, 2)],
        range_mode="level_stride",
        decode_scale_mode="cell",
        radius_scale_mode="level_stride",
    )
    feat = torch.zeros(1, 1, 3)
    grid = {
        "center": torch.tensor([[0.0, 2.0, 4.0]]),
        "cell_left": torch.tensor([[2.0, 4.0, 8.0]]),
        "cell_right": torch.tensor([[3.0, 5.0, 9.0]]),
    }

    points = generator([feat], [grid])[0]

    assert points.shape[-1] == 7
    assert torch.allclose(points[..., 0], grid["center"])
    assert torch.allclose(points[..., 1], torch.full((1, 3), 4.0))
    assert torch.allclose(points[..., 2], torch.full((1, 3), 8.0))
    assert torch.allclose(points[..., 3], grid["cell_left"])
    assert torch.allclose(points[..., 4], grid["cell_right"])
    assert torch.allclose(points[..., 5], torch.full((1, 3), 4.0))
    assert torch.allclose(points[..., 6], torch.full((1, 3), 4.0))


def test_irregular_point_generator_v2_absolute_range_does_not_scale_on_linux():
    torch = import_torch_or_skip()
    point_generator = pytest.importorskip("opentad.models.dense_heads.prior_generator.irregular_point_generator")

    generator = point_generator.IrregularPointGeneratorV2(
        strides=[1],
        regression_range=[(8, 16)],
        range_mode="absolute",
    )
    feat = torch.zeros(1, 1, 3)
    grid = {
        "center": torch.tensor([[0.0, 2.0, 4.0]]),
        "cell_left": torch.tensor([[2.0, 4.0, 8.0]]),
        "cell_right": torch.tensor([[2.0, 4.0, 8.0]]),
    }

    points = generator([feat], [grid])[0]

    assert torch.allclose(points[..., 1], torch.full((1, 3), 8.0))
    assert torch.allclose(points[..., 2], torch.full((1, 3), 16.0))
    assert torch.allclose(points[..., 3], grid["cell_left"])
    assert torch.allclose(points[..., 4], grid["cell_right"])


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


def test_temporal_grid_downsample_merges_full_cell_support_intervals_on_linux():
    torch = import_torch_or_skip()
    temporal_grid = load_module("opentad/models/utils/temporal_grid.py", "temporal_grid_downsample_cells")

    grid = temporal_grid.build_temporal_grid(
        torch.tensor([[0.0, 2.0, 5.0, 100.0]], dtype=torch.float32),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        cell_left=torch.tensor([[2.0, 2.0, 3.0, 95.0]], dtype=torch.float32),
        cell_right=torch.tensor([[2.0, 3.0, 95.0, 668.0]], dtype=torch.float32),
    )

    downsampled = temporal_grid.downsample_temporal_grid(grid)

    assert torch.allclose(downsampled["center"], torch.tensor([[1.5, 385.0]]))
    assert torch.allclose(downsampled["cell_left"], torch.tensor([[3.5, 383.0]]))
    assert torch.allclose(downsampled["cell_right"], torch.tensor([[3.5, 383.0]]))
    assert torch.allclose(downsampled["level_scale"], torch.tensor([193.25]))


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
