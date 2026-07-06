"""Stage-2 official dense selected-axis sanity config.

This is a thin, named entry point over the existing uniform-fixed dense control:
50% equal-interval input, GT/proposals on the selected axis, native-axis
post-processing, and the dense ActionFormer head/prior-generator contract.
"""

_base_ = ["./input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py"]

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4"
