_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"]

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
        prior_generator=dict(
            dense_compat_mode="official_actionformer",
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
    )
)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4"
)
