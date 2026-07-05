---
updated: 2026-07-04
status: active
scope: HeadV3 performance-collapse root-cause hypotheses for the 50% fixed THUMOS adapter sparse-head route.
out-of-scope: Raw logs, checkpoints, generated result archives, and historical wiki material.
---

# HeadV3 Performance Root-Cause Notes

## Current Read

The current leading judgment is that HeadV3 did not suffer a training crash. The fixed rerun completed with `Average-mAP=40.20`, no `Traceback`, no OOM, no non-finite loss, and a stable prediction count of `422000`. The follow-up `nogeometry` and `reggate` runs also completed cleanly. The performance issue is therefore more likely caused by the supervision definition learned by the head, especially assignment, regression targets, and their interaction with irregular point scales, rather than by a broken training run.

Important comparison boundary: the dense control family uses selected-axis GT (`remap_gt_to_selected_axis=True`), while the HeadV3 and Bridge families use native-axis GT (`remap_gt_to_selected_axis=False`). Therefore the old `51.59 -> 40.20` gap is not a pure head-only causal comparison. It should be read as the gap between the dense selected-axis reference and the current native-axis sparse-head route.

Reference snapshot:

| Experiment | Status | Avg mAP | Notes |
|---|---:|---:|---|
| dense control old reference | verified | 51.59 | native dense ActionFormerHead |
| old headv3 fixed | verified | 42.41 | boundary auxiliary loss still affected the run |
| current fixed | complete | 40.20 | mAP@0.3/0.4/0.5/0.6/0.7 = 64.82/53.26/40.67/27.62/14.63 |
| nogeometry | complete | 39.32 | 64.40/52.85/39.58/26.47/13.28; geometry is not the first-order cause |
| reggate | complete | 38.32 | 61.98/53.32/40.37/25.14/10.83; range gate alone does not fix the route |

## Main Hypotheses

1. Primary suspect: the V2/V3 supervision contract is too far from dense ActionFormer.

   Dense `AnchorFreeHead` defaults to `use_regress_range=True`, then applies center sampling, regression range filtering, and shortest-GT conflict resolution for each point. Current V2/V3 uses per-GT top-k soft assignment, soft classification targets, soft regression weights, and asymmetric log regression. Early diagnostics showed `440` to `1016` positive points in the same batch range, `pos_mass/count ~= 0.71`, and many weak cross-GT positives. This likely diffuses the labels into many low-confidence positives instead of dense ActionFormer's smaller set of clear positives, hurting localization most at high tIoU.

2. Regression range gate is not a sufficient standalone fix.

   The `reggate` run adds `use_regress_range=True` to the V2/V3 soft assignment path and finishes at `38.32`, below the fixed baseline. This refutes the narrow hypothesis that missing regression range gate alone caused the collapse. It does not refute dense-like assignment, because the run still keeps soft per-GT top-k labels and asymmetric log regression.

3. Regression encoding remains a high-priority unisolated variable.

   Dense ActionFormer decodes with a linear `reg * stride` distance. V2/V3 use `log1p` / `expm1` with left/right cell scales. This may change boundary calibration or make high-IoU refinement too conservative. The completed `reggate` experiment only tested the range gate; it did not isolate linear-vs-log regression encoding.

4. Boundary auxiliary loss is a secondary factor, not the main cause.

   The gap from old headv3 fixed `42.41` to current fixed `40.20` is about `-2.21` mAP and is plausibly explained by disabling boundary auxiliary supervision. It does not explain the full gap to dense control, which is about `-11.39` mAP.

5. Geometry modulation is currently a lower-priority suspect.

   Final `nogeometry` is `39.32`, below fixed `40.20`, so geometry modulation should be treated as mostly exonerated for the current drop. It may help slightly, but it is not the first-order failure mode.

6. Projection/neck remains a mixed variable against the dense reference.

   The dense control and HeadV3 route differ not only in the head. They also differ in projection/neck, point generator choices, and GT axis. The current `GridAwareConv1DTransformerProj` / `GridAwareFPNIdentity` and `DensePassthroughConv1DTransformerProj` / `DensePassthroughFPNIdentity` variants mostly share the dense feature path and differ in temporal-grid passthrough/downsampling. The cross-over configs should therefore be treated as route sanity checks, not a strong projection/neck attribution.

7. Irregular point scale semantics are now explicit.

   `IrregularPointGeneratorV2` uses the full local cell span `cell_left + cell_right` as the regression-range point scale, while preserving `cell_left` and `cell_right` as separate decode scales in the point tensor. This is different from `temporal_grid["level_scale"]`, which stores `0.5 * (cell_left + cell_right)`. The bridge experiments should be interpreted with this scale convention in mind.

## Interpretation Plan

- Stop prioritizing geometry ablations until the supervision path is repaired.
- Stop treating missing regression range gate as a standalone primary cause; the completed `reggate` run refutes that narrow hypothesis.
- Run dense-like bridge ablations first:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_log_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_soft_topk1_binary_linear_n16r4.py`
- Run projection/neck cross-over configs as route sanity checks, not a strong projection/neck attribution:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_gridaware_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_densepass_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_densepass_n16r4.py`
- If `bridge_hard_linear` improves clearly over fixed, assignment/regression contract is the main lever.
- If `bridge_hard_log` trails `bridge_hard_linear`, regression encoding is a major cause.
- If `soft_topk1_binary_linear` trails hard assignment, soft labels/weights are the main cause.
- If dense head drops sharply under `GridAware`, inspect route compatibility and GT-axis effects before making a projection/neck claim.
- If HeadV3 or Bridge recovers under `DensePassthrough`, treat it as a route-compatibility signal first, then design a stricter projection/neck ablation.

## Code Anchors

- `opentad/models/dense_heads/anchor_free_head.py`: dense assignment defaults and target construction.
- `opentad/models/dense_heads/irregular_actionformer_head_v2.py`: V2/V3 soft assignment and regression encoding.
- `opentad/models/dense_heads/irregular_actionformer_head_v3.py`: geometry modulation and boundary auxiliary loss.
- `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`: existing hard/soft assignment knobs useful for the next dense-like ablation.
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_*_n16r4.py`: dense-like bridge and soft-label isolation configs.
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_*dense*_*n16r4.py`: route sanity cross-over configs.
