_base_ = ["./input_random_fixed_50pct_simota_compact_fixed.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=True,
        center_sample_radius=2.5,
        assignment_debug=dict(enabled=True),
        assigner=dict(
            center_radius=2.5,
            confuse_weight=1.0,
            min_k=4,
        ),
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=39,
    end_epoch=100,
    runtime_debug_interval=1,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact_fixed_range_center25_mink4_w1"
