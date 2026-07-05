_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(rpn_head=dict(prior_generator=dict(range_mode="absolute")))

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4"
