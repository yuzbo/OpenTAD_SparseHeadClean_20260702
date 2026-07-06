---
updated: 2026-07-06
status: active
scope: HeadV3 performance-collapse root-cause hypotheses for the 50% fixed THUMOS adapter sparse-head route.
out-of-scope: Raw logs, checkpoints, generated result archives, and historical wiki material.
---

# HeadV3 Performance Root-Cause Notes

## Current Read

The current leading judgment is that HeadV3 did not suffer a training crash. The fixed rerun completed with `Average-mAP=40.20`, no `Traceback`, no OOM, no non-finite loss, and a stable prediction count of `422000`. The follow-up `nogeometry` and `reggate` runs also completed cleanly. The performance issue is therefore more likely caused by the supervision definition learned by the head, especially assignment, regression targets, and their interaction with irregular point scales, rather than by a broken training run.

Important comparison boundary: the dense control family uses selected-axis GT (`remap_gt_to_selected_axis=True`), while the HeadV3 and Bridge families use native-axis GT (`remap_gt_to_selected_axis=False`). Therefore the old `51.59 -> 40.20` gap is not a pure head-only causal comparison. It should be read as the gap between the dense selected-axis reference and the current native-axis sparse-head route.

Official-OpenTAD review boundary absorbed on 2026-07-06: the authority for dense AdaTAD / ActionFormer behavior must be the upstream OpenTAD implementation, not the dense code copied into this working repository. The external review in `pro-review-official-opentad-20260706.md` treats the current route as partially implemented but not contract-equivalent to official dense AdaTAD. It also reframes the largest performance gap: if the equal-interval 50% baseline is around `Average-mAP ~= 65`, then the current `40-42` results are a route-level collapse, not a small head regression. Equal-interval 50% remains close to a uniform dense grid, while random-fixed native-axis sparse detection changes the observation geometry and the supervision coordinate system at the same time.

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

   The legacy `range_mode="hard"` / `local_cell_span` behavior uses the full local cell span `cell_left + cell_right` as the regression-range scale while preserving `cell_left` and `cell_right` as decode scales. This is now treated as a reproducible ablation, not as the corrected dense-like contract. `IrregularPointGeneratorV2` now emits explicit point fields `[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]`, so bridge heads can separate regression range, regression decode denominator, and center-radius scale. The corrected dense-like candidate is `range_mode="level_stride"` with `decode_scale_mode="level_stride"`, `radius_scale_mode="level_stride"`, `center_radius_scale="point_radius"`, and `reg_denom_mode="left_right_mean"`.

8. Openrange confirms the range failure, but not the high-IoU failure.

   `bridge_hard_linear_openrange` finishes at `Average-mAP=42.44`, above the `HeadV3 fixed=40.20` gate and above the old `headv3 fixed=42.41` reference by a hair. However, the gain is uneven: compared with current fixed, tIoU 0.4/0.5/0.6 improve meaningfully, while tIoU 0.7 only moves from `14.63` to `15.20`. This means restoring positives across FPN levels mainly improves coarse recall / mid-IoU localization. The remaining high-IoU weakness is now more likely due to boundary calibration rather than mere lack of positive samples.

   Leading suspects for the remaining high-IoU gap:
   - Openrange removes level specialization entirely. It gives every level positives for every duration, which can improve recall but may create duplicate, poorly calibrated proposals across levels.
   - Symmetric-linear regression still uses irregular cell-scale denominators. With sparse/native-axis cells, small decode-scale mismatch directly becomes boundary error at high tIoU.
   - Native-axis sparse sampling can lose exact boundary evidence. Dense control remains selected-axis GT, so the 51.59 reference is still not a pure head-only comparison.
   - The bridge route lacks boundary auxiliary supervision and explicit boundary refinement, so it can recover proposal coverage without recovering sub-cell boundary precision.
   - Post-processing score / NMS calibration may still be tuned for dense ActionFormer distributions, not duplicated openrange all-level predictions.

9. Official dense parity is now a prerequisite, not an optional cleanup.

   The 2026-07-06 official OpenTAD review explicitly marks parity with official dense `ActionFormerHead` / `AnchorFreeHead` / `PointGenerator` as a fairness guard. Before making new architectural claims, the route must show which gap is caused by official-dense-to-current-repo drift, which gap is caused by selected-axis versus native-axis supervision, and which gap remains after bridge hard assignment matches official dense targets on a uniform grid.

10. The current highest-risk implementation contract is scale separation.

   The review identifies `IrregularPointGeneratorV2` and `IrregularActionFormerBridgeHead` as high-risk because range scale, decode scale, and center-radius scale have historically been mixed together through `cell_left + cell_right`. The current `range_mode="hard"` collapse is the visible symptom. This has now been fixed at the contract level by separating range, decode, and radius fields in V2 points and by teaching `IrregularActionFormerBridgeHead` to use explicit `point_range` / `point_radius` scale modes. The remaining question is empirical: whether the corrected dense-like contract recovers high-IoU localization under native-axis sparse sampling.

12. Temporal-grid downsampling had a real interval-semantics bug.

   `downsample_temporal_grid` previously used `center +/- 0.5 * cell_left/right` and then doubled the merged distances. This was inconsistent with `build_temporal_grid`, where `cell_left` and `cell_right` already represent the full distance to the left/right temporal support boundary. The implementation now merges full support intervals `center - cell_left` and `center + cell_right`, then stores the true merged left/right distances. This affects grid-aware projection/neck levels and should be included in any post-fix audit because it changes the temporal geometry delivered to the head.

11. Projection/backbone/neck geometry remains a possible first-order failure.

   Even when temporal grids are passed through the route, the main feature mixing may still be slot-index based. Official dense ConvTransformer/FPN assumes uniform token spacing. If current projection/backbone/neck do not actually model true temporal distances, the head receives features whose geometry disagrees with native-axis labels. This is now a leading explanation for why openrange restores positive coverage but does not restore high-IoU boundary precision.

## Interpretation Plan

- 2026-07-06 official OpenTAD review absorbed:
  - Full response recorded at `pro-review-official-opentad-20260706.md` with SHA256 `8e9fda2003be113d80e6d2c4886ef9261f54a3e40078e065bfed12f9e503dc41`.
  - Treat upstream OpenTAD as the dense authority and current-repo dense configs as objects to verify, not as trusted ground truth.
  - Promote official dense parity and selected-axis / native-axis sanity checks ahead of further long training.
  - Add or run controls for `equal_interval_50pct + current bridge/head`, `random_fixed + selected-axis/remap=True + official dense head`, and `random_fixed native + bridge hard + corrected scale-separated prior`.
  - Extend audits beyond positive counts to include proposal coordinate sanity, range-fail/center-fail decomposition, and high-IoU error distribution.

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

- 2026-07-06 implementation update after official review:
  - Confirmed and fixed two concrete implementation errors:
    - `IrregularPointGeneratorV2` / `IrregularActionFormerBridgeHead` previously conflated regression range scale, decode scale, and center-radius scale through `cell_left + cell_right`.
    - `downsample_temporal_grid` previously treated `cell_left/right` as half-widths during downsampling even though they are full support distances.
  - `IrregularPointGeneratorV2` now supports `range_mode="open"` and `range_mode="level_stride"`, plus explicit `decode_scale_mode` and `radius_scale_mode`.
  - `IrregularActionFormerBridgeHead` now keeps the old six-field `_point_fields()` contract for compatibility and adds `_point_fields_extended()` for scale-separated bridge assignment, regression encode/decode, proposal decode, and audit.
  - `tools/audit_sparse_head_assignment.py` now consumes the extended fields when present, so same-batch reports use the same radius/denominator semantics as the model.
  - Updated `bridge_hard_linear_openrange` to use semantic `range_mode="open"`.
  - Updated `bridge_hard_linear_absrange_radiuslevel` to use level-stride decode/radius fields instead of the older half-cell-span approximation.
  - Added `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py` as the corrected dense-like bridge candidate.
  - Added tests for V2 level-stride field separation and full-interval temporal-grid downsampling. Local Windows config tests pass; Linux tensor tests must be run on the remote environment.
  - Same-batch audit after the scale-separation patch compared `openrange`, `absrange`, and `levelstride` on the same batch:
    - `openrange`: GT coverage `38/38`, `pos_by_level=[200, 140, 90, 65, 34, 15]`. This confirms that opening the range restores all levels, but it also keeps the all-level generalization side effect.
    - `absrange`: GT coverage `38/38`, `pos_by_level=[9, 11, 20, 30, 14, 0]`. This is the best bounded compromise among the audited options.
    - `levelstride`: GT coverage `20/38`, `pos_by_level=[8, 18, 3, 0, 0, 0]`. This is too strict and should not be long-trained before redesign.
  - Added `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`, using expanded absolute ranges `[(0, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 10000)]` with level-stride decode/radius fields. This is the next audit candidate before any long training.
  - Remote same-batch audit for `absrange_expanded` completed at `/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/sparse_head_assignment_audit_20260706/same_batch_absrange_expanded_20260706_112548/assignment_audit.json`.
    - `absrange_expanded`: GT coverage `38/38`, `pos_by_level=[17, 30, 44, 45, 22, 4]`, axes `native/native/native`.
    - `absrange`: GT coverage `38/38`, `pos_by_level=[9, 11, 20, 30, 14, 0]`.
    - `levelstride`: GT coverage `20/38`, `pos_by_level=[8, 18, 3, 0, 0, 0]`.
    - `openrange`: GT coverage `38/38`, `pos_by_level=[200, 140, 90, 65, 34, 15]`.
    - Interpretation: `absrange_expanded` is now the strongest bounded native-axis candidate. It restores all GT coverage and level5 positives without openrange's all-level flood.
  - Added `tools/analyze_detection_quality.py` for post-training high-IoU localization diagnostics. It consumes standard OpenTAD annotation/result JSON files and reports per-GT best IoU, tIoU recall, GT-length bucket coverage, and normalized start/end boundary error. Use this after each completed candidate to separate three failure modes: no proposal near the GT, proposal near the GT but boundary-calibration error, and score/NMS ranking error.
  - The diagnostic script can now resolve GT from an OpenTAD config (`--config ... --dataset-split val`) and auto-select the newest `result_detection.json` below an experiment directory (`--experiment-dir ...`). This is the intended post-training command shape for `absrange_expanded` and the equal-interval controls, because it avoids manual path mistakes once a run finishes.
  - Because several legacy configs still contain stale `/root/autodl-tmp/...` annotation paths while the remote annotation actually lives under `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`, `tools/analyze_detection_quality.py` now supports `--ground-truth-fallback`, plus `THUMOS_ANNOTATION` and `THUMOS_ROOT/annotations/thumos_14_anno.json` fallbacks. This prevents post-training diagnostics from failing on a stale config path.
  - Added remote launch helpers for the next long run:
    - `remote_runs/run_gpu1_bridge_absrange_expanded_long_20260706.sh`
    - `remote_runs/launch_gpu1_bridge_absrange_expanded_long_20260706.sh`
    These helpers run config-load and `py_compile` preflight before training `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`.
  - Deployment status at 2026-07-06 11:45 CST: GPU1 is still occupied by the older `bridge_hard_linear_absrange_n16r4` long run (`Slurm step 1118197.621`, around epoch 25/60). That run was launched before the latest `absrange_expanded` candidate and should not be treated as post-fix evidence. Do not start `absrange_expanded` until GPU1 is free or the user explicitly asks to stop the older run.
  - Deployment status at 2026-07-06 12:09 CST: the older `bridge_hard_linear_absrange_n16r4` run is still active on GPU1 (`Slurm step 1118197.621`, GPU1 about `8855/24564 MiB`, no `Traceback`/OOM/non-finite loss and no validation result yet). `absrange_expanded` still has no `gpu1_id1/log.json`, so it has not started.
  - Added a conservative waiter for the next long run:
    - `remote_runs/watch_and_launch_gpu1_bridge_absrange_expanded_20260706.sh`
    - `remote_runs/launch_watch_gpu1_bridge_absrange_expanded_20260706.sh`
    The waiter only polls for the old step `1118197.621` to disappear, exits if the target already has output/logs, and then calls the existing `launch_gpu1_bridge_absrange_expanded_long_20260706.sh`. It does not start training directly and is intended to avoid racing the current GPU1 job. The launcher also refuses to start a duplicate waiter if one is already running.
  - Added equal-interval 50% sanity configs to separate the claimed `~65` equal-interval baseline from random-fixed native sparse failure:
    - `configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py`: dense `ActionFormerHead`, selected-axis GT, deterministic uniform 50% input. This is the fair current-repo counterpart for the equal-interval dense-control claim.
    - `configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`: current bridge hard + `absrange_expanded`, native-axis GT, deterministic uniform 50% input. This tests whether the corrected sparse bridge route still collapses when the sampling grid is regular.
    - Interpretation rule: if dense uniform is high but bridge uniform remains near 40, the bridge/head contract is still the main failure. If bridge uniform recovers substantially while random-fixed native remains poor, the dominant failure shifts toward irregular temporal geometry / feature mixing / boundary evidence rather than assignment alone.
  - Added but did not launch GPU1 Slurm helpers for the two equal-interval controls:
    - `remote_runs/run_gpu1_uniform_fixed_dense_control_long_20260706.sh` / `launch_gpu1_uniform_fixed_dense_control_long_20260706.sh`
    - `remote_runs/run_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh` / `launch_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh`
    These scripts are queued as later fairness controls after the random-fixed `absrange_expanded` candidate; they must not be run concurrently with the current GPU1 job.
  - Added explicit axis contract metadata from `LoadFrames`: `irregular_gt_axis`, `irregular_proposal_axis`, `irregular_postprocess_axis`, and `irregular_axis_contract`.
  - Added runtime axis-contract checks in `IrregularActionFormer` before train/test/post-processing, plus optional proposal-axis debug dumping through `post_cfg.debug_dump_proposals`, `debug_dump_path`, and `debug_dump_topk`.
  - Fixed a selected-axis route bug: when `remap_gt_to_selected_axis=True`, the bridge/head temporal grid must emit centers on the selected index axis `0..N`, while native selected positions are kept only for seconds conversion. Previously the selected-axis branch reused native selected positions as head centers, which could silently mismatch proposal coordinates.
  - Added deterministic `uniform_fixed_subsample` to support equal-interval 50% sanity controls without changing the existing `random_fixed_subsample` behavior.
  - Added `scripts/verify_official_dense_reference.py` to diff the local dense-reference files against upstream OpenTAD raw files, and added a Linux-only bridge hard uniform-grid target test that locks the official dense target/decode semantics in the stride-1 case.
  - Local dry run `python scripts/verify_official_dense_reference.py --no-fail-on-diff` succeeded and reported dense-reference drift, including major local changes in `anchor_free_head.py` and extra grid-aware FPN classes. Treat this as evidence that current-repo dense code is not an authoritative official baseline.

- Stop prioritizing geometry ablations until the supervision path is repaired.
- Stop treating missing regression range gate as a standalone primary cause; the completed `reggate` run refutes that narrow hypothesis.
- Audit before long-training the corrected bridge route:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py`
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
