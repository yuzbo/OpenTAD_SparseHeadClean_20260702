_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv2_x.py"]

model = dict(
    rpn_head=dict(
        soft_assign_topk=1,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv2_x_topk1"
