_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint.py"]

workflow = dict(
    runtime_debug_interval=25,
    end_epoch=700,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint_resume_regwarm"
