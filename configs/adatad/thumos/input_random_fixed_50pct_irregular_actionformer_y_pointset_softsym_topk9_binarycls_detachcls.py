_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk9_binarycls.py"]

model = dict(
    rpn_head=dict(
        detach_cls_input_from_backbone=True,
        detach_reg_input_from_backbone=False,
    ),
)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk9_binarycls_detachcls"
)
