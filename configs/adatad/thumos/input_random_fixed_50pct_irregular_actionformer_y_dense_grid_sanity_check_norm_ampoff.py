_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm.py"]

solver = dict(
    amp=False,
    fp16_compress=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm_ampoff"
