_base_ = ["./input_random_fixed_50pct_adapter_irregular_actionformer_base.py"]

model = dict(
    backbone=dict(
        backbone=dict(debug_time_grid=True),
    ),
)

workflow = dict(
    checkpoint_interval=10,
    logging_interval=10,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=1,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_contract_smoke"
