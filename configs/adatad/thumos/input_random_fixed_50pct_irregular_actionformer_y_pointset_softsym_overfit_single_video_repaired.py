_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_overfit_single_video.py"]

model = dict(
    projection=dict(input_pdrop=0.0),
)

solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    amp=False,
    fp16_compress=False,
    ema=False,
)

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.0, paramwise=True)
scheduler = dict(_delete_=True, type="MultiStepLR", milestones=[400], gamma=0.1, max_epoch=500)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=50,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=500,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_overfit_single_video_repaired"
