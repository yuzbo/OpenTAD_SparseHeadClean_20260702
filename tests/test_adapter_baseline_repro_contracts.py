from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def load_cfg(rel_path):
    return Config.fromfile(str(ROOT / rel_path))


def load_frame_steps(cfg):
    return {
        split: next(step for step in getattr(cfg.dataset, split).pipeline if step.get("type") == "LoadFrames")
        for split in ("train", "val", "test")
    }


def test_random_fixed_adapter_baseline_repro_matches_historical_contract():
    cfg = load_cfg("configs/adatad/thumos/input_random_fixed_50pct_adapter_n16r4_retrain_20260604.py")
    steps = load_frame_steps(cfg)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.backbone.type == "mmaction.Recognizer3D"
    assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfg.model.backbone.backbone.adapter_index == list(range(12))
    assert bool(cfg.model.backbone.custom.freeze_backbone) is False
    assert bool(cfg.model.backbone.custom.norm_eval) is False
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.neck.type == "FPNIdentity"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.prior_generator.type == "PointGenerator"

    assert int(cfg.solver.train.batch_size) == 2
    assert int(cfg.solver.val.batch_size) == 2
    assert int(cfg.solver.test.batch_size) == 2
    assert float(cfg.post_processing.nms.sigma) == 0.7
    assert bool(cfg.post_processing.save_dict) is False
    assert int(cfg.workflow.val_start_epoch) == 40
    assert int(cfg.workflow.val_eval_interval) == 2
    assert int(cfg.workflow.end_epoch) == 60
    assert bool(cfg.workflow.disable_checkpoint) is False
    assert "input_random_fixed_50pct_adapter_n16r4_retrain_20260604" in cfg.work_dir

    assert "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json" == cfg.dataset.train.ann_file
    assert cfg.dataset.val.ann_file == cfg.dataset.train.ann_file
    assert cfg.dataset.test.ann_file == cfg.dataset.train.ann_file
    assert "/root/autodl-tmp" not in cfg.dataset.train.ann_file
    assert cfg.evaluation.ground_truth_filename == cfg.dataset.train.ann_file

    assert steps["train"].method == "random_fixed_subsample"
    assert steps["train"].method_base == "random_trunc"
    assert int(steps["train"].target_len) == 384
    assert int(steps["train"].source_len) == 768
    assert float(steps["train"].keep_ratio) == 0.5
    assert steps["val"].method == "random_fixed_subsample"
    assert steps["val"].method_base == "sliding_window"
    assert int(steps["val"].target_len) == 384
    assert steps["test"].method == "random_fixed_subsample"
    assert steps["test"].method_base == "sliding_window"
    assert int(steps["test"].target_len) == 384

    for split, step in steps.items():
        assert "remap_gt_to_selected_axis" not in step, split
        assert "allow_drop_selected_axis_gt" not in step, split
        assert "legacy_selected_axis_gt_drop_diagnostic" not in step, split


def test_adapter_baseline_slurm_submitter_is_separate_and_fail_closed():
    sbatch_body = (ROOT / "remote_runs/sbatch_adapter_baseline_train_20260708.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "remote_runs/submit_adapter_baseline_slurm_20260708.sh").read_text(encoding="utf-8")

    assert "input_random_fixed_50pct_adapter_n16r4_retrain_20260604.py" in sbatch_body
    assert "input_random_fixed_50pct_adapter_n16r4_retrain_20260604.py" in submitter
    assert "adapter_random_fixed_baseline63_repro" in submitter
    assert "VisionTransformerAdapter" in sbatch_body
    assert "freeze_backbone" in sbatch_body
    assert "norm_eval" in sbatch_body
    assert "historical_avg_map_reference" in sbatch_body
    assert "tools/check_fail_closed_config.py" in sbatch_body
    assert "Refusing adapter baseline training inside allocation 1118197" in sbatch_body
    assert "Refusing adapter baseline training on g0030" in sbatch_body
    assert "DensePassthroughConv1DTransformerProj" not in sbatch_body
    assert "IrregularActionFormer" not in sbatch_body
    assert "VisionTransformerCP" not in sbatch_body
    assert "srun --jobid=1118197" not in sbatch_body
    assert "CUDA_VISIBLE_DEVICES=1" not in sbatch_body
