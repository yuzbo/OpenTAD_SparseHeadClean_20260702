_base_ = ["./input_random_fixed_50pct_irregular_actionformer_diag.py"]

workflow = dict(
    logging_interval=20,
    checkpoint_interval=-1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_diag_epoch1"
