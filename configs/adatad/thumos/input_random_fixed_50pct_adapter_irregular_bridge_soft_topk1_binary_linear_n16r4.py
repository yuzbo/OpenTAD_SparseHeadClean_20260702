_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        assignment_mode="soft",
        regression_mode="symmetric_linear",
        soft_assign_topk=1,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        soft_loss_normalizer_mode="pos_count",
        soft_reg_weight_mode="binary",
        soft_cls_target_mode="binary",
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_soft_topk1_binary_linear_n16r4"
