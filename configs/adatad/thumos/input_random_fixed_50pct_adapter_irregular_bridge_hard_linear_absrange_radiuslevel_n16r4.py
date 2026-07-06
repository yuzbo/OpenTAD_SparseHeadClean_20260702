_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="half_cell_span",
        reg_denom_mode="half_cell_span",
    )
)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4"
)
