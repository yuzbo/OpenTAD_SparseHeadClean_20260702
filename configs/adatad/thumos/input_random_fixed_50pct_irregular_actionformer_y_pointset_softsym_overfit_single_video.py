_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity_ampoff.py"]

overfit_video = "video_validation_0000190"

dataset = dict(
    train=dict(
        allow_list=[overfit_video],
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    amp=False,
    fp16_compress=False,
    ema=False,
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=20,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=200,
)

scheduler = dict(warmup_epoch=0, max_epoch=200)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_overfit_single_video"
