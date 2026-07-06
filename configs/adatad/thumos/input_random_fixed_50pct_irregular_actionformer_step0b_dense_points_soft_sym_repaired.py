_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

model = dict(
    rpn_head=dict(
        _delete_=True,
        type="IrregularActionFormerBridgeHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        predictor_kernel_size=3,
        assignment_mode="soft",
        regression_mode="symmetric_linear",
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        cls_prior_prob=0.01,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        soft_assign_topk=9,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        reg_denom_floor=0.5,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        debug_cfg=dict(enable=True),
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym_repaired"
