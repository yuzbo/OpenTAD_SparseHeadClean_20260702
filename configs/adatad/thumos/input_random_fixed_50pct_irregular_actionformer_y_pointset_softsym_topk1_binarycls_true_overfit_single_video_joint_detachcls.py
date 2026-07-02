_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint.py"]

model = dict(
    rpn_head=dict(
        detach_cls_input_from_backbone=True,
        detach_reg_input_from_backbone=False,
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    runtime_debug_interval=25,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint_detachcls"
