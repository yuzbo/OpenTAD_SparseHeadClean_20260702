_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"]

model = dict(
    rpn_head=dict(
        _delete_=True,
        type="IrregularActionFormerBridgeHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        predictor_kernel_size=3,
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        center_radius_scale="full_cell_span",
        reg_denom_mode="full_cell_span",
        allow_legacy_full_cell_span=True,
        allow_center_fallback_inside_gt=True,
        route_contract=dict(
            route_label="SPARSE_HEAD_WARN_REVIEW_FIXES_DIAGNOSTIC_ONLY",
            compatibility="legacy_ablation_only",
            dense_equivalent_claim_allowed=False,
            allow_legacy_full_cell_span=True,
            allow_center_fallback_inside_gt=True,
            gt_axis="native",
            proposal_axis="native",
            nms_axis="native",
            postprocess_axis="native",
            eval_axis="seconds",
            diagnostic_only=True,
            primary_result_allowed=False,
        ),
        cls_prior_prob=0.01,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        prior_generator=dict(
            type="IrregularPointGeneratorV2",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            range_mode="hard",
            overlap_factor=0.5,
        ),
        debug_cfg=dict(enable=True),
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4"
