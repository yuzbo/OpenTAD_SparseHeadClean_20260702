_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_true_overfit_single_instance.py"]

model = dict(
    rpn_head=dict(
        assignment_mode="hard",
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    runtime_debug_interval=25,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_hardjoint_true_overfit_single_instance"
