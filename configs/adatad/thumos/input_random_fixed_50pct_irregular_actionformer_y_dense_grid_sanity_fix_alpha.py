_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py"]

model = dict(
    neck=dict(
        strides=[1, 2, 4, 8, 16, 32],
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_fix_alpha"
