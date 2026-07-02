_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_clsonly.py"]

workflow = dict(
    checkpoint_interval=500,
    disable_checkpoint=False,
    runtime_debug_interval=25,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_clsonly_ckpt"
