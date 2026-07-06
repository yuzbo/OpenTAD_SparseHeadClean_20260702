---
updated: 2026-07-05
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
| bridge hard linear openrange | complete | 42.44 | 65.56/56.44/44.29/30.73/15.20; restores all-level positives but high-tIoU remains weak |

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

8. Openrange confirms the range failure, but not the high-IoU failure.

   `bridge_hard_linear_openrange` finishes at `Average-mAP=42.44`, above the `HeadV3 fixed=40.20` gate and above the old `headv3 fixed=42.41` reference by a hair. However, the gain is uneven: compared with current fixed, tIoU 0.4/0.5/0.6 improve meaningfully, while tIoU 0.7 only moves from `14.63` to `15.20`. This means restoring positives across FPN levels mainly improves coarse recall / mid-IoU localization. The remaining high-IoU weakness is now more likely due to boundary calibration rather than mere lack of positive samples.

   Leading suspects for the remaining high-IoU gap:
   - Openrange removes level specialization entirely. It gives every level positives for every duration, which can improve recall but may create duplicate, poorly calibrated proposals across levels.
   - Symmetric-linear regression still uses irregular cell-scale denominators. With sparse/native-axis cells, small decode-scale mismatch directly becomes boundary error at high tIoU.
   - Native-axis sparse sampling can lose exact boundary evidence. Dense control remains selected-axis GT, so the 51.59 reference is still not a pure head-only comparison.
   - The bridge route lacks boundary auxiliary supervision and explicit boundary refinement, so it can recover proposal coverage without recovering sub-cell boundary precision.
   - Post-processing score / NMS calibration may still be tuned for dense ActionFormer distributions, not duplicated openrange all-level predictions.

## Interpretation Plan

- 2026-07-05 implementation update:
  - Added `center_radius_scale` and `reg_denom_mode` knobs to `IrregularActionFormerBridgeHead`.
  - Default values remain `full_cell_span`, preserving existing `bridge_hard_linear` behavior.
  - Added `bridge_hard_linear_absrange_radiuslevel_n16r4`, which combines `range_mode="absolute"` with `half_cell_span` center radius and symmetric-linear regression denominator.
  - Added `tools/audit_sparse_head_assignment.py` to compare current hard, openrange, absrange, and radius-level corrected configs on the same batch before any further long training.
  - Same-batch audit completed on remote `g0030` under Slurm allocation `1118197` with output at `/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/sparse_head_assignment_audit_20260705/same_batch_audit_20260705_20260705_205007/`.
  - Audit result: current `bridge_hard_linear` has `pos_by_level=[111, 21, 0, 0, 0, 0]` with `candidates_after_range=[111, 21, 0, 0, 0, 0]`. This confirms the current core failure: the regression range filters effectively crush positives into level0/1, leaving level2-5 with no positive samples.
  - `bridge_hard_linear_openrange` restores all levels with `pos_by_level=[195, 163, 104, 56, 34, 14]` and full `37/37` GT coverage. It is the strongest current long-training candidate and directly validates the regression-range failure hypothesis.
  - `bridge_hard_linear_absrange` also moves positives into mid/high levels with `pos_by_level=[9, 9, 23, 29, 11, 0]`, but it is much sparser and covers `36/37` GTs. Treat it as the second-priority corrected-range control, below openrange.
  - `bridge_hard_linear_absrange_radiuslevel` does not improve positives over absrange and increases center-filter failures (`center_fail_by_level=[347, 131, 29, 3, 0, 0]`), so it should not be long-trained now.
  - Current decision: run `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py` first on remote GPU1. Use `HeadV3 fixed=40.20` as the gate: if the first validation is clearly above `40.20`, continue the full run; if it remains poor, shift diagnosis from level-positive recovery to regression encode/decode, native-axis GT, fair dense comparisons, proposal recall, per-level recall, and high-IoU localization error.
  - `bridge_hard_linear_openrange` completed at `42.44`, so the range-collapse hypothesis is supported but incomplete. Next run `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py` as the corrected-range control.
  - If `absrange` trails openrange badly, openrange's broad recall is doing the work and absolute dense-like ranges may be too sparse under native-axis cells.
  - If `absrange` matches or beats openrange, the next target is not "more positives" but scale-calibrated, dense-like assignment / decode.

- Stop prioritizing geometry ablations until the supervision path is repaired.
- Stop treating missing regression range gate as a standalone primary cause; the completed `reggate` run refutes that narrow hypothesis.
- Audit before long-training the corrected bridge route:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py`
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
