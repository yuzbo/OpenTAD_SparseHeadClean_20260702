_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

model = dict(
    projection=dict(
        debug_cfg=dict(
            enable=True,
            target_embed_idx=0,
            target_layer_idx=0,
            max_reports=8,
        ),
    ),
    neck=dict(
        debug_cfg=dict(
            enable=True,
        ),
    ),
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=39,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_diag"
