_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint.py"]

model = dict(
    rpn_head=dict(
        cls_loss_weight_schedule=dict(
            warmup_epochs=500,
            warmup_value=0.0,
            after_warmup_value=1.0,
        ),
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    runtime_debug_interval=25,
    end_epoch=700,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint_regwarm_schedule500"
