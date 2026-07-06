from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

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
    assert hard_linear.model.rpn_head.center_radius_scale == "full_cell_span"
    assert hard_linear.model.rpn_head.reg_denom_mode == "full_cell_span"
    assert hard_linear.model.rpn_head.allow_legacy_full_cell_span is True
    assert "hard_linear_n16r4" in hard_linear.work_dir

    assert hard_log.model.rpn_head.assignment_mode == "hard"
    assert hard_log.model.rpn_head.regression_mode == "asymmetric_log1p"
    assert hard_log.model.rpn_head.allow_legacy_full_cell_span is True
    assert "hard_log_n16r4" in hard_log.work_dir

    assert soft_topk1.model.rpn_head.assignment_mode == "soft"
    assert soft_topk1.model.rpn_head.regression_mode == "symmetric_linear"
    assert soft_topk1.model.rpn_head.allow_legacy_full_cell_span is True
    assert int(soft_topk1.model.rpn_head.soft_assign_topk) == 1
    assert soft_topk1.model.rpn_head.soft_reg_weight_mode == "binary"
    assert soft_topk1.model.rpn_head.soft_cls_target_mode == "binary"
    assert soft_topk1.model.rpn_head.soft_loss_normalizer_mode == "pos_count"
    assert "soft_topk1_binary_linear_n16r4" in soft_topk1.work_dir

    assert openrange.model.rpn_head.assignment_mode == "hard"
    assert openrange.model.rpn_head.regression_mode == "symmetric_linear"
    assert openrange.model.rpn_head.center_radius_scale == "full_cell_span"
    assert openrange.model.rpn_head.reg_denom_mode == "full_cell_span"
    assert openrange.model.rpn_head.allow_legacy_full_cell_span is True
    assert openrange.model.rpn_head.prior_generator.range_mode == "open"
    assert all(tuple(item) == (0, 10000) for item in openrange.model.rpn_head.prior_generator.regression_range)
    assert "hard_linear_openrange_n16r4" in openrange.work_dir

    assert absrange.model.rpn_head.assignment_mode == "hard"
    assert absrange.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange.model.rpn_head.center_radius_scale == "full_cell_span"
    assert absrange.model.rpn_head.reg_denom_mode == "full_cell_span"
    assert absrange.model.rpn_head.allow_legacy_full_cell_span is True
    assert absrange.model.rpn_head.prior_generator.range_mode == "absolute"
    assert tuple(absrange.model.rpn_head.prior_generator.regression_range[2]) == (8, 16)
    assert "hard_linear_absrange_n16r4" in absrange.work_dir

    assert absrange_radiuslevel.model.rpn_head.assignment_mode == "hard"
    assert absrange_radiuslevel.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange_radiuslevel.model.rpn_head.center_radius_scale == "point_radius"
    assert absrange_radiuslevel.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert absrange_radiuslevel.model.rpn_head.allow_legacy_full_cell_span is False
    assert absrange_radiuslevel.model.rpn_head.prior_generator.range_mode == "absolute"
    assert absrange_radiuslevel.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert absrange_radiuslevel.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert tuple(absrange_radiuslevel.model.rpn_head.prior_generator.regression_range[2]) == (8, 16)
    assert "hard_linear_absrange_radiuslevel_n16r4" in absrange_radiuslevel.work_dir

    assert levelstride.model.rpn_head.assignment_mode == "hard"
    assert levelstride.model.rpn_head.regression_mode == "symmetric_linear"
    assert levelstride.model.rpn_head.center_radius_scale == "point_radius"
    assert levelstride.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert levelstride.model.rpn_head.allow_legacy_full_cell_span is False
    assert levelstride.model.rpn_head.prior_generator.range_mode == "level_stride"
    assert levelstride.model.rpn_head.prior_generator.decode_scale_mode == "level_stride"
    assert levelstride.model.rpn_head.prior_generator.radius_scale_mode == "level_stride"
    assert "hard_linear_levelstride_n16r4" in levelstride.work_dir

    assert absrange_expanded.model.rpn_head.assignment_mode == "hard"
    assert absrange_expanded.model.rpn_head.regression_mode == "symmetric_linear"
    assert absrange_expanded.model.rpn_head.center_radius_scale == "point_radius"
    assert absrange_expanded.model.rpn_head.reg_denom_mode == "left_right_mean"
    assert absrange_expanded.model.rpn_head.allow_legacy_full_cell_span is False
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

    assert "center_radius_scale=\"point_radius\"" in bridge_impl
    assert "reg_denom_mode=\"left_right_mean\"" in bridge_impl
    assert "allow_legacy_full_cell_span=False" in bridge_impl
    assert "Legacy full-cell-span bridge scales require allow_legacy_full_cell_span=True" in bridge_impl
    assert "def _scale_base(" in bridge_impl
    assert "def _point_fields_extended(" in bridge_impl
    assert 'mode == "full_cell_span"' in bridge_impl
    assert 'mode == "half_cell_span"' in bridge_impl
    assert 'mode == "min_side"' in bridge_impl
    assert 'mode == "left_right_mean"' in bridge_impl
    assert 'mode == "point_range"' in bridge_impl
    assert 'mode == "point_radius"' in bridge_impl


def test_bridge_corrected_derivative_configs_clear_legacy_scale_opt_in():
    shortgate = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"
    )
    selected_axis = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py"
    )

    for cfg in (shortgate, selected_axis):
        head = cfg.model.rpn_head
        assert head.type == "IrregularActionFormerBridgeHead"
        assert head.center_radius_scale == "point_radius"
        assert head.reg_denom_mode == "left_right_mean"
        assert head.allow_legacy_full_cell_span is False


def test_sparse_head_assignment_audit_tool_contract():
    script = read("tools/audit_sparse_head_assignment.py")

    assert "def segment_iou(" in script
    assert "def axis_segments_to_native(" in script
    assert "def axis_segments_to_seconds(" in script
    assert "assigned_positive_target_decode_iou" in script
    assert "oracle_assigned_recall@IoU" in script
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


def test_sparse_head_assignment_segment_iou_handles_pairwise_and_matrix_on_linux():
    torch = import_torch_or_skip()
    pytest.importorskip("mmengine.config")
    audit = load_module("tools/audit_sparse_head_assignment.py", "sparse_head_assignment_audit_iou")

    anchors = torch.tensor(
        [
            [0.0, 10.0],
            [0.0, 10.0],
            [0.0, 2.0],
            [5.0, 5.0],
        ]
    )
    targets = torch.tensor(
        [
            [0.0, 10.0],
            [5.0, 15.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )

    pairwise = audit.segment_iou(anchors, targets, pairwise=True)

    assert torch.allclose(pairwise, torch.tensor([1.0, 5.0 / 15.0, 0.0, 0.0]))

    matrix = audit.segment_iou(
        torch.tensor([[0.0, 10.0], [20.0, 30.0]]),
        torch.tensor([[5.0, 15.0], [20.0, 30.0], [8.0, 8.0]]),
    )

    assert matrix.shape == (2, 3)
    assert torch.allclose(matrix[0], torch.tensor([5.0 / 15.0, 0.0, 0.0]))
    assert torch.allclose(matrix[1], torch.tensor([0.0, 1.0, 0.0]))


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


def test_detection_quality_analyzer_resolves_config_gt_and_latest_result(tmp_path):
    analyzer = load_module("tools/analyze_detection_quality.py", "detection_quality_analyzer_resolve")

    cfg_path = tmp_path / "toy_config.py"
    ann_path = tmp_path / "annotations.json"
    cfg_path.write_text(
        "\n".join(
            [
                "dataset = dict(",
                "    train=dict(ann_file='train.json'),",
                f"    val=dict(ann_file=r'{ann_path}'),",
                "    test=dict(ann_file='test.json'),",
                ")",
                "work_dir = 'exps/toy_exp'",
            ]
        ),
        encoding="utf-8",
    )
    ann_path.write_text('{"database": {}}', encoding="utf-8")

    older = tmp_path / "exps" / "toy_exp" / "gpu1_id1" / "result_detection.json"
    newer = tmp_path / "exps" / "toy_exp" / "gpu1_id2" / "result_detection.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text('{"results": {"old": []}}', encoding="utf-8")
    newer.write_text('{"results": {"new": []}}', encoding="utf-8")
    newer.touch()

    assert analyzer.resolve_ground_truth_from_config(cfg_path, "val") == ann_path
    assert analyzer.resolve_prediction_path(tmp_path / "exps" / "toy_exp") == newer


def test_detection_quality_analyzer_uses_existing_gt_fallback_when_config_path_is_stale(tmp_path):
    analyzer = load_module("tools/analyze_detection_quality.py", "detection_quality_analyzer_gt_fallback")

    cfg_path = tmp_path / "toy_config.py"
    stale_ann_path = tmp_path / "missing" / "annotations.json"
    fallback_ann_path = tmp_path / "real" / "annotations.json"
    fallback_ann_path.parent.mkdir(parents=True)
    fallback_ann_path.write_text('{"database": {}}', encoding="utf-8")
    cfg_path.write_text(
        "\n".join(
            [
                "dataset = dict(",
                f"    val=dict(ann_file=r'{stale_ann_path}'),",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    resolved = analyzer.resolve_ground_truth_from_config(
        cfg_path,
        "val",
        fallback_paths=[fallback_ann_path],
    )

    assert resolved == fallback_ann_path


def test_detection_quality_candidate_configs_save_result_detection_json():
    candidates = [
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py",
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py",
        "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py",
        "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py",
        "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py",
    ]

    for rel_path in candidates:
        cfg = load_mmengine_config_or_skip(rel_path)
        assert bool(cfg.post_processing.save_dict), rel_path


def test_selected_axis_random_uniform_control_matrix_contracts():
    matrix = [
        (
            "uniform_dense_selected",
            "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
            "uniform_fixed_subsample",
            "ActionFormerHead",
        ),
        (
            "random_dense_selected",
            "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
            "random_fixed_subsample",
            "ActionFormerHead",
        ),
        (
            "random_bridge_selected_absrange_expanded",
            "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py",
            "random_fixed_subsample",
            "IrregularActionFormerBridgeHead",
        ),
    ]
    work_dirs = set()

    for name, rel_path, method, head_type in matrix:
        cfg = load_mmengine_config_or_skip(rel_path)
        steps = load_frame_steps(cfg)

        assert cfg.model.type == "IrregularActionFormer", name
        assert cfg.model.rpn_head.type == head_type, name
        assert bool(cfg.post_processing.save_dict), name
        assert "selected_axis_control" in cfg.work_dir, name
        assert cfg.work_dir not in work_dirs, name
        work_dirs.add(cfg.work_dir)

        assert cfg.dataset.train.ann_file == "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json", name
        assert cfg.dataset.val.data_path == "/data/run01/sczc063/yuzibo/thumos14/test", name
        assert cfg.dataset.test.data_path == "/data/run01/sczc063/yuzibo/thumos14/test", name

        for split, step in steps.items():
            assert step.method == method, (name, split)
            assert abs(float(step.keep_ratio) - 0.5) < 1e-12, (name, split)
            assert bool(step.remap_gt_to_selected_axis), (name, split)
            assert int(step.target_len) == 384, (name, split)

        assert steps["train"].method_base == "random_trunc", name
        assert steps["val"].method_base == "sliding_window", name
        assert steps["test"].method_base == "sliding_window", name

        if head_type == "ActionFormerHead":
            assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj", name
            assert cfg.model.neck.type == "DensePassthroughFPNIdentity", name
            assert cfg.model.rpn_head.prior_generator.type == "PointGenerator", name
        else:
            assert cfg.model.projection.type == "GridAwareConv1DTransformerProj", name
            assert cfg.model.neck.type == "GridAwareFPNIdentity", name
            assert cfg.model.rpn_head.assignment_mode == "hard", name
            assert cfg.model.rpn_head.regression_mode == "symmetric_linear", name
            assert cfg.model.rpn_head.prior_generator.type == "IrregularPointGeneratorV2", name
            assert cfg.model.rpn_head.prior_generator.range_mode == "absolute", name
            assert cfg.model.rpn_head.prior_generator.decode_scale_mode == "level_stride", name
            assert cfg.model.rpn_head.prior_generator.radius_scale_mode == "level_stride", name
            assert [tuple(item) for item in cfg.model.rpn_head.prior_generator.regression_range] == [
                (0, 8),
                (2, 16),
                (4, 32),
                (8, 64),
                (16, 128),
                (32, 10000),
            ]


def test_selected_axis_control_precheck_launcher_is_fail_closed():
    precheck = read("remote_runs/precheck_selected_axis_control_matrix_20260706.sh")
    launcher = read("remote_runs/launch_selected_axis_control_precheck_20260706.sh")

    for cfg_name in (
        "input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
        "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
        "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py",
    ):
        assert cfg_name in precheck

    assert "Config.fromfile" in precheck
    assert "remap_gt_to_selected_axis" in precheck
    assert "python -m py_compile" in precheck
    assert "python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q" in precheck
    assert "tools/train.py" not in precheck
    assert "torchrun" not in precheck
    assert "sbatch" not in precheck
    assert "srun" not in precheck

    assert "precheck_selected_axis_control_matrix_20260706.sh" in launcher
    assert "tools/train.py" not in launcher
    assert "torchrun" not in launcher
    assert "sbatch" not in launcher
    assert "srun" not in launcher


def test_absrange_expanded_shortgate_config_inherits_native_bounded_bridge_semantics():
    long_cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py"
    )
    short_cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"
    )

    long_head = long_cfg.model.rpn_head
    short_head = short_cfg.model.rpn_head
    long_prior = long_head.prior_generator
    short_prior = short_head.prior_generator

    assert short_cfg.model.projection.type == "GridAwareConv1DTransformerProj"
    assert short_cfg.model.neck.type == "GridAwareFPNIdentity"
    assert short_head.type == "IrregularActionFormerBridgeHead"
    assert short_head.assignment_mode == "hard"
    assert short_head.regression_mode == "symmetric_linear"
    assert short_head.center_radius_scale == "point_radius"
    assert short_head.reg_denom_mode == "left_right_mean"
    assert short_prior.type == "IrregularPointGeneratorV2"
    assert short_prior.range_mode == "absolute"
    assert short_prior.decode_scale_mode == "level_stride"
    assert short_prior.radius_scale_mode == "level_stride"
    assert [tuple(item) for item in short_prior.regression_range] == [
        (0, 8),
        (2, 16),
        (4, 32),
        (8, 64),
        (16, 128),
        (32, 10000),
    ]
    assert [tuple(item) for item in short_prior.regression_range] == [
        tuple(item) for item in long_prior.regression_range
    ]
    assert short_prior.range_mode == long_prior.range_mode
    assert short_prior.range_mode != "open"
    assert short_head.assignment_mode != "soft"
    assert short_head.type != "ActionFormerHead"

    for step in load_frame_steps(short_cfg).values():
        assert step.method == "random_fixed_subsample"
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12
        assert not bool(step.remap_gt_to_selected_axis)


def test_absrange_expanded_shortgate_config_is_short_early_and_isolated():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py"
    )

    assert int(cfg.scheduler.max_epoch) == 2
    assert int(cfg.workflow.end_epoch) == 2
    assert int(cfg.workflow.val_start_epoch) == 1
    assert int(cfg.workflow.val_eval_interval) == 1
    assert int(cfg.workflow.checkpoint_interval) == 1
    assert bool(cfg.workflow.disable_checkpoint)
    assert bool(cfg.post_processing.save_dict)
    assert "shortgate" in cfg.work_dir
    assert cfg.work_dir != "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4"


def test_absrange_expanded_shortgate_launcher_is_fail_closed_by_default():
    runner = read("remote_runs/run_bridge_absrange_expanded_shortgate_failclosed_20260706.sh")

    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py" in runner
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in runner
    assert 'RUN_TRAIN="${RUN_TRAIN:-0}"' in runner
    assert 'PYTHON_BIN="${PYTHON_BIN:-}"' in runner
    assert "_python_works()" in runner
    assert "command -v python.exe" in runner
    assert "command -v python3" in runner
    assert "config load preflight" in runner
    assert "py_compile preflight" in runner
    assert "pytest preflight" in runner
    assert "RUN_TRAIN=1" in runner
    assert "PRECHECK_ONLY=0" in runner
    assert "torchrun" in runner
    assert "DRY_RUN_TRAIN_CMD" in runner
    assert "ssh" not in runner
    assert "scp" not in runner
    assert "sbatch" not in runner
    assert "srun" not in runner


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


def test_bridge_head_default_and_legacy_scale_contract_on_linux():
    import_torch_or_skip()
    mmengine_config = pytest.importorskip("mmengine.config")
    bridge_head = pytest.importorskip("opentad.models.dense_heads.irregular_actionformer_bridge_head")
    config_dict = mmengine_config.ConfigDict

    base_kwargs = dict(
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
    )

    default_head = bridge_head.IrregularActionFormerBridgeHead(**base_kwargs)
    assert default_head.center_radius_scale == "point_radius"
    assert default_head.reg_denom_mode == "left_right_mean"
    assert default_head.allow_legacy_full_cell_span is False

    with pytest.raises(ValueError, match="allow_legacy_full_cell_span=True"):
        bridge_head.IrregularActionFormerBridgeHead(
            **base_kwargs,
            center_radius_scale="full_cell_span",
            reg_denom_mode="full_cell_span",
        )

    legacy_head = bridge_head.IrregularActionFormerBridgeHead(
        **base_kwargs,
        center_radius_scale="full_cell_span",
        reg_denom_mode="full_cell_span",
        allow_legacy_full_cell_span=True,
    )
    assert legacy_head.allow_legacy_full_cell_span is True


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
    assert "pgrep -f" in launcher
    assert "waiter already running" in launcher


def test_loadframes_records_explicit_axis_contract_metadata():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")

    assert 'gt_axis = "selected" if self.remap_gt_to_selected_axis else "native"' in load_frames_impl
    assert 'proposal_axis = gt_axis' in load_frames_impl
    assert 'postprocess_axis = "native"' in load_frames_impl
    assert 'results["irregular_gt_axis"] = gt_axis' in load_frames_impl
    assert 'results["irregular_proposal_axis"] = proposal_axis' in load_frames_impl
    assert 'results["irregular_postprocess_axis"] = postprocess_axis' in load_frames_impl
    assert 'results["irregular_axis_contract"]' in load_frames_impl


def test_loadframes_selected_axis_remap_does_not_create_tiny_collapsed_gt_targets():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")

    assert "start + 1e-3" not in load_frames_impl
    assert "dropped_selected_axis_gt_segments" in load_frames_impl


def test_loadframes_selected_axis_remap_drops_collapsed_segments_on_linux():
    import_torch_or_skip()
    import numpy as np

    end_to_end = pytest.importorskip("opentad.datasets.transforms.end_to_end")
    loader = object.__new__(end_to_end.LoadFrames)

    segments, labels = loader._remap_gt_to_selected_axis(
        gt_segments=np.asarray([[12.0, 13.0], [0.0, 5.0]], dtype=np.float32),
        gt_labels=np.asarray([7, 3], dtype=np.int32),
        kept_positions=np.asarray([0, 5], dtype=np.int64),
        valid_len=10,
    )

    assert segments.tolist() == [[0.0, 1.0]]
    assert labels.tolist() == [3]
    assert loader._last_dropped_selected_axis_gt_segments == [
        dict(index=0, label=7, original_segment=[12.0, 13.0], mapped_segment=[2.0, 2.0])
    ]


def test_random_fixed_path_records_dropped_selected_axis_gt_after_remap():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")
    start = load_frames_impl.index("frame_idxs = dense_window[keep_positions]")
    end = load_frames_impl.index("if len(frame_idxs) < frame_num:", start)
    branch = load_frames_impl[start:end]

    assert branch.index("_remap_gt_to_selected_axis(") < branch.index("_set_irregular_axis_meta(")


def test_selected_axis_helper_paths_clear_dropped_gt_state_for_native_axis():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")

    for func_name in ("_oracle_subsample_window", "_weighted_random_subsample_window"):
        start = load_frames_impl.index(f"def {func_name}(")
        end = load_frames_impl.index("        frame_num = int(target_frame_num)", start)
        helper = load_frames_impl[start:end]
        assert "else:\n            self._last_dropped_selected_axis_gt_segments = []" in helper


def test_irregular_actionformer_validates_axis_contract_and_exposes_proposal_dump_helper():
    detector_impl = read("opentad/models/detectors/irregular_actionformer.py")

    assert "def _axis_contract_from_meta(" in detector_impl
    assert "def _assert_axis_contract(" in detector_impl
    assert "def _proposal_axis_debug_records(" in detector_impl
    assert "debug_dump_proposals" in detector_impl
    assert "proposal_axis" in detector_impl
    assert "postprocess_axis" in detector_impl
    assert "def _segments_to_axis(" in detector_impl
    assert "def _segments_to_seconds(" in detector_impl


def test_irregular_actionformer_converts_selected_axis_before_nms():
    detector_impl = read("opentad/models/detectors/irregular_actionformer.py")

    assert "selected_axis_to_dense_axis" in detector_impl
    assert "segments = self._segments_to_axis(" in detector_impl
    assert detector_impl.index("segments = self._segments_to_axis(") < detector_impl.index("batched_nms(")


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
    selected_to_native_meta = dict(
        irregular_native_axis=False,
        irregular_gt_axis="selected",
        irregular_proposal_axis="selected",
        irregular_postprocess_axis="native",
    )

    assert model._axis_contract_from_meta(good_meta) == ("native", "native", "native")
    model._assert_axis_contract(good_meta, stage="test")
    assert model._axis_contract_from_meta(selected_to_native_meta) == ("selected", "selected", "native")
    model._assert_axis_contract(selected_to_native_meta, stage="test")
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


def test_irregular_actionformer_selected_axis_post_processing_nms_uses_native_axis_on_linux(monkeypatch):
    torch = import_torch_or_skip()
    detector = pytest.importorskip("opentad.models.detectors.irregular_actionformer")

    captured = {}

    def fake_batched_nms(segments, scores, labels, **kwargs):
        captured["segments"] = segments.clone()
        captured["scores"] = scores.clone()
        captured["labels"] = labels.clone()
        return segments, scores, labels

    monkeypatch.setattr(detector, "batched_nms", fake_batched_nms)
    model = object.__new__(detector.IrregularActionFormer)
    meta = dict(
        video_name="video_selected_axis",
        fps=1.0,
        snippet_stride=1.0,
        offset_frames=0.0,
        window_start_frame=0.0,
        duration=30.0,
        irregular_selected_positions=[0.0, 10.0, 20.0],
        irregular_selected_valid_len=30.0,
        irregular_native_axis=False,
        irregular_gt_axis="selected",
        irregular_proposal_axis="selected",
        irregular_postprocess_axis="native",
    )

    results = model.post_processing(
        predictions=(
            [torch.tensor([[0.5, 1.5]], dtype=torch.float32)],
            [torch.tensor([[0.9]], dtype=torch.float32)],
        ),
        metas=[meta],
        post_cfg=SimpleNamespace(
            pre_nms_thresh=0.001,
            pre_nms_topk=2000,
            sliding_window=False,
            nms={},
        ),
        ext_cls=["action"],
    )

    assert torch.allclose(captured["segments"], torch.tensor([[5.0, 15.0]]))
    assert results["video_selected_axis"] == [
        dict(segment=[5.0, 15.0], label="action", score=0.9)
    ]


def test_official_dense_reference_verifier_tracks_upstream_opentad_sources():
    script = read("scripts/verify_official_dense_reference.py")

    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py" in script
    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/prior_generator/point_generator.py" in script
    assert "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/necks/fpn.py" in script
    assert "difflib.unified_diff" in script
    assert "--official-root" in script


def test_official_dense_selected_axis_sanity_config_loads_contract_with_mmengine():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"
    )

    head = cfg.model.rpn_head
    load_steps = load_frame_steps(cfg)
    assert cfg.model.type == "IrregularActionFormer"
    assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert cfg.model.neck.type == "DensePassthroughFPNIdentity"
    assert head.type == "ActionFormerHead"
    assert head.prior_generator.type == "PointGenerator"
    assert head.type not in {"IrregularActionFormerHeadV2", "IrregularActionFormerBridgeHead"}
    assert head.prior_generator.type != "IrregularPointGeneratorV2"
    assert bool(cfg.post_processing.save_dict)
    assert "official_dense_selected_axis_sanity" in cfg.work_dir

    for split, step in load_steps.items():
        assert step.method == "uniform_fixed_subsample"
        assert abs(float(step.keep_ratio) - 0.5) < 1e-12
        assert bool(step.remap_gt_to_selected_axis), split
    assert load_steps["train"].method_base == "random_trunc"
    assert load_steps["val"].method_base == "sliding_window"
    assert load_steps["test"].method_base == "sliding_window"


def test_official_dense_selected_axis_sanity_uses_selected_proposals_native_postprocess():
    load_frames_impl = read("opentad/datasets/transforms/end_to_end.py")
    detector_impl = read("opentad/models/detectors/irregular_actionformer.py")

    assert 'gt_axis = "selected" if self.remap_gt_to_selected_axis else "native"' in load_frames_impl
    assert "proposal_axis = gt_axis" in load_frames_impl
    assert 'postprocess_axis = "native"' in load_frames_impl
    assert "selected_axis_to_dense_axis" in detector_impl
    assert "segments = self._segments_to_axis(" in detector_impl
    assert detector_impl.index("segments = self._segments_to_axis(") < detector_impl.index("batched_nms(")


def test_official_dense_selected_axis_precheck_is_fail_closed_and_non_training():
    precheck = read("remote_runs/precheck_official_dense_selected_axis_sanity_20260706.sh")

    assert "set -euo pipefail" in precheck
    assert "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py" in precheck
    assert "verify_official_dense_reference.py" in precheck
    assert "PYTHON_BIN" in precheck
    assert "for candidate in python python.exe python3" in precheck
    assert "import mmengine.config" in precheck
    assert "config load preflight" in precheck
    assert "py_compile preflight" in precheck
    assert "RUN_PYTEST" in precheck
    assert "-m pytest tests/test_adapter_native_dense_headv2_contracts.py -q" in precheck
    assert "torchrun" not in precheck
    assert "tools/train.py" not in precheck
    assert "srun" not in precheck


def test_official_dense_reference_verifier_checks_selected_axis_sanity_config():
    script = read("scripts/verify_official_dense_reference.py")

    assert "validate_official_dense_selected_axis_config" in script
    assert "--config" in script
    assert "--skip-reference-files" in script
    assert "ActionFormerHead" in script
    assert "PointGenerator" in script
    assert "DensePassthroughConv1DTransformerProj" in script
    assert "DensePassthroughFPNIdentity" in script
    assert "remap_gt_to_selected_axis" in script
    assert "save_dict" in script


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


def test_bridge_dense_equivalence_verifier_contract_exists():
    script = read("tools/verify_bridge_dense_equivalence.py")

    assert "def official_dense_targets(" in script
    assert "def compare_bridge_to_official(" in script
    assert "def run_all_checks(" in script
    assert "stride1_dense_open_range" in script
    assert "multi_level_range_gate" in script
    assert "--json" in script


def test_bridge_dense_equivalence_verifier_smoke_on_linux():
    import_torch_or_skip()
    verifier = load_module("tools/verify_bridge_dense_equivalence.py", "bridge_dense_equivalence_verifier")

    summary = verifier.run_all_checks()

    assert summary["ok"] is True
    assert [case["name"] for case in summary["cases"]] == [
        "stride1_dense_open_range",
        "multi_level_range_gate",
    ]
    for case in summary["cases"]:
        assert case["mismatches"] == []
        assert case["positive_count"] > 0
        assert case["decoded_max_abs_error"] <= 1e-6
