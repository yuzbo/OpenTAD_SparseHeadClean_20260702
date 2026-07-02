_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_true_overfit_single_instance_regonly.py"]

model = dict(
    rpn_head=dict(
        soft_reg_weight_mode="binary",
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    runtime_debug_interval=25,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_binaryreg_true_overfit_single_instance_regonly"
