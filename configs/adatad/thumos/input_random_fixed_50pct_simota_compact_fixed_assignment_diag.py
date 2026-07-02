_base_ = ["./input_random_fixed_50pct_simota_compact_fixed.py"]

model = dict(
    rpn_head=dict(
        assignment_debug=dict(enabled=True),
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=999,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=5,
    disable_checkpoint=True,
    runtime_debug_interval=1,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact_fixed_assignment_diag"
