from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def load_cfg(rel_path):
    return Config.fromfile(str(ROOT / rel_path))


def load_frame_steps(cfg):
    return {
        split: next(item for item in getattr(cfg.dataset, split).pipeline if item.get("type") == "LoadFrames")
        for split in ("train", "val", "test")
    }


def test_exact0410_random_irregular_densepass_is_only_wrapper_bridge():
    cfg = load_cfg("configs/adatad/thumos/input_random_fixed_50pct_0410_exact_irregular_densepass_n16r4.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "IrregularActionFormer"
    assert cfg.model.backbone.backbone.type == "VisionTransformerCP"
    assert bool(cfg.model.backbone.custom.freeze_backbone)
    assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj"
    assert cfg.model.neck.type == "DensePassthroughFPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.prior_generator.type == "PointGenerator"
    assert int(cfg.solver.train.batch_size) == 2
    assert float(cfg.post_processing.nms.sigma) == 0.5
    assert float(cfg.post_processing.nms.min_score) == 0.001
    assert int(cfg.workflow.val_start_epoch) == 39
    assert int(cfg.workflow.val_eval_interval) == 5
    assert "0410_exact_irregular_densepass" in cfg.work_dir

    assert steps["train"].method == "random_fixed_subsample"
    assert steps["train"].method_base == "random_trunc"
    assert int(steps["train"].target_len) == 384
    assert int(steps["train"].source_len) == 768
    for split, step in steps.items():
        assert bool(step.remap_gt_to_selected_axis), split
        assert bool(step.allow_drop_selected_axis_gt), split
        assert bool(step.legacy_selected_axis_gt_drop_diagnostic), split


def test_current_random_actionformer_selected_axis_bs8_is_pure_actionformer():
    cfg = load_cfg("configs/adatad/thumos/input_random_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter"
    assert not bool(cfg.model.backbone.custom.freeze_backbone)
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.neck.type == "FPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert int(cfg.solver.train.batch_size) == 8
    assert float(cfg.post_processing.nms.sigma) == 0.7
    assert bool(cfg.post_processing.save_dict)
    assert "adapter_actionformer_selected_axis_bs8" in cfg.work_dir

    assert steps["train"].method == "random_fixed_subsample"
    assert steps["train"].method_base == "random_trunc"
    assert int(steps["train"].target_len) == 384
    assert int(steps["train"].source_len) == 768
    for split, step in steps.items():
        assert bool(step.remap_gt_to_selected_axis), split


def test_current_uniform_actionformer_selected_axis_bs8_is_pure_actionformer():
    cfg = load_cfg("configs/adatad/thumos/input_uniform_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.neck.type == "FPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert int(cfg.solver.train.batch_size) == 8
    assert bool(cfg.post_processing.save_dict)
    assert "uniform_fixed_50pct_adapter_actionformer_selected_axis_bs8" in cfg.work_dir

    assert steps["train"].method == "uniform_fixed_subsample"
    assert steps["train"].method_base == "random_trunc"
    assert int(steps["train"].source_len) == 768
    assert steps["val"].method == "uniform_fixed_subsample"
    assert steps["val"].method_base == "sliding_window"
    assert steps["test"].method == "uniform_fixed_subsample"
    assert steps["test"].method_base == "sliding_window"
    for split, step in steps.items():
        assert bool(step.remap_gt_to_selected_axis), split


def test_protocol_bridge_slurm_runner_tracks_only_bridge_controls():
    runner = (ROOT / "remote_runs/sbatch_protocol_bridge_train_20260708.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "remote_runs/submit_protocol_bridge_slurm_20260708.sh").read_text(encoding="utf-8")

    for config_name in (
        "input_random_fixed_50pct_0410_exact_irregular_densepass_n16r4.py",
        "input_random_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py",
        "input_uniform_fixed_50pct_adapter_actionformer_selected_axis_bs8_n16r4.py",
    ):
        assert config_name in runner
        assert config_name in submitter

    assert "--exclude=g0030" in runner
    assert 'SLURM_JOB_ID:-}" == "1118197"' in runner
    assert "tools/check_fail_closed_config.py" in runner
    assert "tools/analyze_detection_quality.py" in runner
    assert "ALLOW_OVERWRITE_PROTOCOL_BRIDGE_OUTPUT" in runner
    assert "slurm_protocol_bridge_20260708" in runner
    assert "bridge_exact0410_random_irregular_densepass" in submitter
    assert "bridge_current_random_actionformer_bs8" in submitter
    assert "bridge_current_uniform_actionformer_bs8" in submitter
