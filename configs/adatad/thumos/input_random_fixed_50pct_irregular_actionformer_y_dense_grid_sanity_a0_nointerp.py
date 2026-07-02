_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_fix_alpha.py"]

model = dict(
    neck=dict(
        no_interp=True,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_a0_nointerp"
