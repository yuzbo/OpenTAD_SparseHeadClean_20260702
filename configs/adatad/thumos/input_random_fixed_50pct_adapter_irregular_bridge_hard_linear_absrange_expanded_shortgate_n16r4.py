_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        allow_center_fallback_inside_gt=False,
        route_contract=dict(
            route_label="SPARSE_HEAD_WARN_REVIEW_FIXES_DIAGNOSTIC_ONLY",
            compatibility="dense_compatible_diagnostic_candidate",
            dense_equivalent_claim_allowed=False,
            allow_legacy_full_cell_span=False,
            allow_center_fallback_inside_gt=False,
        ),
    )
)

scheduler = dict(warmup_epoch=1, max_epoch=2)

workflow = dict(
    logging_interval=10,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=2,
    disable_checkpoint=True,
)

post_processing = dict(save_dict=True)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4"
)
