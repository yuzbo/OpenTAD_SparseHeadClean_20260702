_base_ = ["./input_random_fixed_50pct_simota_compact_fixed_range_center25_mink4_w1.py"]

workflow = dict(
    logging_interval=50,
    checkpoint_interval=999,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=39,
    end_epoch=100,
    runtime_debug_interval=1,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact_fixed_range_center25_mink4_w1"
