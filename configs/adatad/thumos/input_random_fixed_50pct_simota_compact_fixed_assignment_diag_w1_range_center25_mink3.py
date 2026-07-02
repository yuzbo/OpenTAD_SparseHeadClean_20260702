_base_ = ["./input_random_fixed_50pct_simota_compact_fixed_assignment_diag.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=True,
        center_sample_radius=2.5,
        assigner=dict(
            center_radius=2.5,
            confuse_weight=1.0,
            min_k=3,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact_fixed_assignment_diag_w1_range_center25_mink3"
