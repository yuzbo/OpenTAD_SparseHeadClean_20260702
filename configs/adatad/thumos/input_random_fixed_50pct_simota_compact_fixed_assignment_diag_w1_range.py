_base_ = ["./input_random_fixed_50pct_simota_compact_fixed_assignment_diag.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=True,
        assigner=dict(
            confuse_weight=1.0,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact_fixed_assignment_diag_w1_range"
