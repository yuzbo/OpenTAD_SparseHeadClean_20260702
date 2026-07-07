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


def test_0410_exact_random_fixed_dense_config_is_pure_actionformer():
    cfg = load_cfg("configs/adatad/thumos/input_random_fixed_50pct_0410_exact_n16r4.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.neck.type == "FPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.prior_generator.type == "PointGenerator"
    assert int(cfg.solver.train.batch_size) == 2
    assert int(cfg.workflow.val_start_epoch) == 39
    assert int(cfg.workflow.val_eval_interval) == 5
    assert float(cfg.post_processing.nms.sigma) == 0.5
    assert float(cfg.post_processing.nms.min_score) == 0.001
    assert bool(cfg.post_processing.save_dict)
    assert "0410_exact" in cfg.work_dir

    assert int(cfg.dataset.train.sample_stride) == 1
    assert int(cfg.dataset.val.sample_stride) == 1
    assert int(cfg.dataset.test.sample_stride) == 1
    assert int(cfg.dataset.val.window_size) == 768
    assert int(cfg.dataset.test.window_size) == 768
    assert steps["train"].method == "random_fixed_subsample"
    assert steps["train"].method_base == "random_trunc"
    assert int(steps["train"].target_len) == 384
    assert int(steps["train"].source_len) == 768
    assert bool(steps["train"].remap_gt_to_selected_axis)
    assert steps["val"].method == "random_fixed_subsample"
    assert steps["val"].method_base == "sliding_window"
    assert int(steps["val"].target_len) == 384
    assert bool(steps["val"].remap_gt_to_selected_axis)
    assert steps["test"].method == "random_fixed_subsample"
    assert steps["test"].method_base == "sliding_window"
    assert int(steps["test"].target_len) == 384
    assert bool(steps["test"].remap_gt_to_selected_axis)
    for split, step in steps.items():
        assert bool(step.allow_drop_selected_axis_gt), split
        assert bool(step.legacy_selected_axis_gt_drop_diagnostic), split


def test_0410_exact_stride2_uniform_dense_config_is_not_selected_axis_subsample():
    cfg = load_cfg("configs/adatad/thumos/input_stride2_uniform_0410_exact_n16r4.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.neck.type == "FPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.prior_generator.type == "PointGenerator"
    assert int(cfg.solver.train.batch_size) == 2
    assert int(cfg.workflow.val_start_epoch) == 39
    assert int(cfg.workflow.val_eval_interval) == 5
    assert float(cfg.post_processing.nms.sigma) == 0.5
    assert float(cfg.post_processing.nms.min_score) == 0.001
    assert bool(cfg.post_processing.save_dict)
    assert "0410_exact" in cfg.work_dir

    assert int(cfg.dataset.train.sample_stride) == 2
    assert int(cfg.dataset.val.sample_stride) == 2
    assert int(cfg.dataset.test.sample_stride) == 2
    assert int(cfg.dataset.val.window_size) == 384
    assert int(cfg.dataset.test.window_size) == 384
    assert steps["train"].method == "random_trunc"
    assert int(steps["train"].trunc_len) == 384
    assert steps["val"].method == "sliding_window"
    assert steps["test"].method == "sliding_window"
    for split, step in steps.items():
        assert step.method != "uniform_fixed_subsample", split
        assert "remap_gt_to_selected_axis" not in step, split


def test_current_near63_near65_controls_are_explicitly_not_0410_exact_dense():
    random_control = load_cfg(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py"
    )
    uniform_control = load_cfg(
        "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"
    )

    for cfg in (random_control, uniform_control):
        assert cfg.model.type == "IrregularActionFormer"
        assert cfg.model.projection.type == "DensePassthroughConv1DTransformerProj"
        assert cfg.model.neck.type == "DensePassthroughFPNIdentity"
        assert cfg.model.rpn_head.type == "ActionFormerHead"
        assert int(cfg.solver.train.batch_size) == 8
        assert float(cfg.post_processing.nms.sigma) == 0.7
        for step in load_frame_steps(cfg).values():
            assert bool(step.remap_gt_to_selected_axis)

    assert load_frame_steps(uniform_control)["train"].method == "uniform_fixed_subsample"
    assert int(uniform_control.dataset.train.sample_stride) == 1


def test_0410_exact_dense_slurm_scripts_are_separate_from_selected_axis_dense_runner():
    sbatch_body = (ROOT / "remote_runs/sbatch_0410_exact_dense_train_20260708.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "remote_runs/submit_0410_exact_dense_slurm_20260708.sh").read_text(encoding="utf-8")
    current_submitter = (ROOT / "remote_runs/submit_current_implemented_models_slurm_20260708.sh").read_text(
        encoding="utf-8"
    )

    assert "input_random_fixed_50pct_0410_exact_n16r4.py" in submitter
    assert "input_stride2_uniform_0410_exact_n16r4.py" in submitter
    assert "exact0410_random_fixed_dense63" in submitter
    assert "exact0410_stride2_uniform_dense65" in submitter
    assert "slurm_0410_exact_dense" in sbatch_body
    assert "slurm_0410_exact_dense" in submitter
    assert 'cfg.model.type == "ActionFormer"' in sbatch_body
    assert 'cfg.model.projection.type == "Conv1DTransformerProj"' in sbatch_body
    assert 'cfg.model.neck.type == "FPNIdentity"' in sbatch_body
    assert 'cfg.model.rpn_head.type == "ActionFormerHead"' in sbatch_body
    assert "sample_stride" in sbatch_body
    assert "historical_avg_map_reference" in sbatch_body
    assert "DensePassthroughConv1DTransformerProj" not in sbatch_body
    assert "uniform_fixed_subsample" not in sbatch_body
    assert "IrregularActionFormer" not in sbatch_body
    assert "srun --jobid=1118197" not in sbatch_body
    assert "CUDA_VISIBLE_DEVICES=1" not in sbatch_body

    assert "EXACT0410_BODY" in current_submitter
    assert "submit_one exact0410 \"exact0410_random_fixed_dense63\"" in current_submitter
    assert "submit_one exact0410 \"exact0410_stride2_uniform_dense65\"" in current_submitter
    assert "not exact 0410 baselines" in current_submitter
