_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity.py"]

model = dict(
    rpn_head=dict(
        soft_assign_topk=1,
    ),
)

workflow = dict(
    runtime_debug_interval=5,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_softcls"
