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

Monitoring-collapse review boundary absorbed on 2026-07-06: the external response in `pro-review-monitoring-collapse-20260706.md` is stricter than the earlier review and explicitly rejects explaining `65 -> 40/42` as natural random 50% sparse degradation. It was based on GitHub source/raw review of commit `d7da39315f80405253bf705997d7c26e63c66ab0` plus upstream OpenTAD static review, not a local clone or training run, so every code claim must be rechecked against current HEAD. Its active checklist is: split supervision/decode/postprocess axes, verify whether NMS is performed on comparable native/seconds coordinates before suppression, revisit full-gap versus half-cell temporal-grid semantics, stop treating positive coverage alone as proof of high-IoU correctness, and gate `absrange_expanded` with stronger audit/short-run evidence before interpreting long-run results.

Strict FAIL review boundary absorbed on 2026-07-06: the external response in `pro-review-strict-fail-20260706.md` reviews GitHub commit `8689eff` and explicitly blocks long training and any `dense-equivalent` claim. It agrees that the five-stage diagnostic route is the correct order, but says the current evidence still cannot prove bridge / irregular-head equivalence to official dense ActionFormer/AdaTAD. Treat this as the active gate before further deployment: first strengthen the bridge equivalence verifier to cover real `IrregularPointGeneratorV2` output, add numeric selected/native/seconds roundtrip tests, add val/test fail-closed guards for diagnostic GT/cache paths, and verify official dense selected-axis performance before attributing the `65 -> 40/42` collapse to sparse-head design.

Implementation follow-up on 2026-07-06: the latest FAIL review was accepted as technically valid for local implementation / remote preflight only. Long training remains blocked. The code now makes the non-official BridgeHead missing-center fallback explicit through `allow_center_fallback_inside_gt=False` by default; historical legacy bridge routes opt in with `True`, while corrected/official-compatible routes explicitly keep it `False`. `IrregularPointGeneratorV2` now has `dense_compat_mode="official_actionformer"`, which locks absolute regression ranges plus level-stride decode/radius scales for dense-compatible candidates. `convert_to_seconds` now accepts explicit `source_axis` to prevent accidental selected-axis double conversion, and irregular post-processing passes the axis directly after selected-to-native conversion. Single-class irregular post-processing now applies `pre_nms_thresh` and `pre_nms_topk` before NMS. Added `tools/check_fail_closed_config.py` as a whole-config shortcut scanner for diagnostic GT, teacher/cache, raw-prediction, and load-prediction shortcuts; current THUMOS configs scan clean. `tools/audit_sparse_head_assignment.py` now emits `official_vs_current_assignment_diff` with positive mask, class, encoded target, decoded target, per-level positive, and GT coverage diffs against an official dense-like target builder. These are audit/preflight safeguards, not performance claims.

Placeholder-commit FAIL review absorbed on 2026-07-06: the external response in `pro-review-placeholder-commit-fail-20260706.md` is recorded verbatim with SHA256 `872BF8EC741B68C19389745AD866DF63D1C075087FF7A61950E14CBCEC89B6CA`. Its main technical/meta finding is valid: the submitted prompt still contained `<COMMIT_URL>`, so GPT could only inspect the public branch state visible on GitHub, not the local uncommitted fixes above. Therefore this review must not be treated as proof that the latest local fixes failed; it is proof that no fixed GitHub commit has yet been reviewed. Its code-level blocker list overlaps the just-implemented local fixes for Bridge fallback, point-generator dense compatibility, explicit seconds axis, single-class pre-NMS filtering, whole-config scanning, and audit diff output. The remaining active blockers are: commit and push the current local changes to a concrete SHA/URL, run Linux preflight and same-batch audit on that SHA, extend or replace the synthetic bridge verifier with batched/masked/full-dataloader evidence, and run the official dense selected-axis sanity route before any long training or dense-equivalent claim.

Fixed-SHA WARN review absorbed on 2026-07-06: the external response in `pro-review-fixed-sha-warn-20260706.md` is recorded verbatim with SHA256 `6C5D5690AD5F21E9D15EE54E26EAA75F0A2A558B2E0985DBFAF321C7FEE9C987`. It reviews the fixed public GitHub commit `2213388febace4cc90a6fb5c8159ea069fd47adf`, so it supersedes the earlier placeholder-URL meta failure. The verdict improves from FAIL to WARN, but the execution boundary remains strict: remote sync is allowed only for diagnostic Stage 0-2, Linux preflight is allowed, same-batch audit is allowed only as a hard gate, official dense selected-axis sanity is allowed, full training is denied until Stage 0/1/2 all pass, and any `dense-equivalent` claim remains denied.

New active blockers from the fixed-SHA WARN review:

- `IrregularActionFormerBridgeHead._point_fields()` still exposes the legacy `point_scale = cell_left + cell_right` behavior for 5+ field points. Corrected routes mostly avoid it through explicit range/radius/decode scales, but any fallback to `point_scale` can silently double stride-like scale. The review recommends separating an official-compatible mean scale from an explicitly named `legacy_cell_span`, and allowing full-cell-span semantics only behind an explicit legacy flag.
- The missing-center fallback fix is real but must be described narrowly. `_build_candidate_mask()` is fail-closed by `allow_center_fallback_inside_gt=False`, while hard assignment uses its own `_prepare_targets_hard()` path and does not rely on that helper. Audit output should record that hard assignment does not use missing-center fallback, instead of using the helper flag as proof of equivalence.
- Legacy bridge routes (`bridge_hard_linear`, `openrange`, `absrange`) still risk being misread as official-compatible. They need explicit metadata such as `compatibility="legacy_ablation_only"` and `dense_equivalent_claim_allowed=False`, and preferably work_dir names that cannot pollute corrected result tables.
- `dense_compat_mode="official_actionformer"` locks scale modes, but does not yet assert that the upstream temporal grid centers are truly `arange(T) * stride`. Without this, selected/native mixed coordinates could still pass through the dense-compatible point generator.
- NMS may still run on selected-axis coordinates if `postprocess_axis="selected"`. For irregular/selected inputs, the review requires proposals to be converted to native or seconds coordinates before NMS, because selected-axis IoU can destroy high-tIoU behavior under 50% or irregular sampling.
- `convert_to_seconds(source_axis="selected")` should fail closed when selected metadata is missing, and detector internals should avoid `source_axis="auto"` so future call sites cannot silently double-convert or native-convert selected proposals.
- The current official target builder in `tools/audit_sparse_head_assignment.py` is an independent approximation, not a direct upstream `AnchorFreeHead.prepare_targets()` call. It can serve as Stage 1 hard-gate evidence only if compared on real same-batch dataloader outputs with exact positive/class/encoded/decoded diffs.
- `tools/check_fail_closed_config.py` covers shortcut/cache keys but is not yet policy-aware for legacy versus dense-equivalent contract keys. It should reject or flag dense-equivalent claims on configs that opt into legacy full-cell-span or missing-center fallback behavior.
- `tools/verify_bridge_dense_equivalence.py` remains a synthetic sanity check, not a real dataloader/full-batch equivalence proof. The review asks for a `--config --split --num-batches` path that uses real dataloader batches, current head construction, masks/padding, and official target diffs.
- The official dense selected-axis sanity wrapper should spell out the axis contract directly (`gt_axis=selected`, `proposal_axis=selected`, `postprocess_axis=native`) rather than relying on a base config to preserve those semantics.

Stage-FIX-A bridge scale contract status on 2026-07-06: the BridgeHead scale contract now fails closed for legacy full-cell-span semantics. `IrregularActionFormerBridgeHead` defaults to the official-compatible pair `center_radius_scale="point_radius"` and `reg_denom_mode="left_right_mean"`. Any route that still asks for `full_cell_span` must also set `allow_legacy_full_cell_span=True`, which marks it as a reproducibility ablation rather than dense-equivalent evidence. The historical `bridge_hard_linear`, `openrange`, and `absrange` configs are kept as explicit legacy opt-in controls; corrected `radiuslevel`, `levelstride`, `absrange_expanded`, `shortgate`, and selected-axis `absrange_expanded` configs explicitly clear the legacy opt-in and keep level-stride decode/radius semantics. The earlier `step0b/step1-4/y_pointset` BridgeHead exploration configs also now state the official-compatible scale contract explicitly, so future config loads do not silently depend on class defaults.

Preflight/audit FAIL review boundary absorbed on 2026-07-06: the external response in `pro-review-fail-preflight-audit-20260706.md` reviews current GitHub commit `114f72c` against upstream OpenTAD and gives `CURRENT_TOTAL_VERDICT: FAIL`. The original response is recorded verbatim with SHA256 `8D9A7F200FFD783D179D76B44E3A5D92A1E36103699B1075FF761608AAC51D30`. Its gate is stricter than the code-fix status above: `SYNC_REMOTE_FOR_PREFLIGHT: YES`, `RUN_STAGE_0_1_2_ONLY: YES`, `START_LONG_TRAINING: NO`, and `CLAIM_SPARSE_HEAD_RESULT: NO`. Treat this as the active execution boundary until superseded by Linux evidence.

New blockers from the preflight/audit FAIL review:

- Official dense selected-axis sanity around the equal-interval 50% `Average-mAP ~= 65` reference remains unproved in the current repository; without it, the `65 -> 40/42` gap cannot be attributed to HeadV3 or sparse-head design.
- `IrregularActionFormerBridgeHead` still has a non-official missing-center fallback: when center sampling yields no candidate for a GT, it can fall back to all inside-GT points. This must become an explicit legacy/research option, not part of the official-compatible bridge route.
- `convert_to_seconds` still infers source axis from metadata. Since detector post-processing may already convert selected-axis proposals to native axis before NMS, utilities need an explicit `source_axis` to prevent accidental double conversion in tools or future call sites.
- Current eval/test leakage protection is transform-level only. A whole-config preflight scanner is still needed for raw prediction, teacher/cache, diagnostic GT, or post-processing shortcut fields outside `LoadFrames`.
- `tools/verify_bridge_dense_equivalence.py` now covers generated V2 points, but the review says it is still too narrow: B=1 synthetic only, no real masks/padding, no full dataloader batch, no post-processing/seconds conversion, no missing-center fallback counterexample, and no full `__init__` construction path.
- `tools/audit_sparse_head_assignment.py` must grow an official `AnchorFreeHead` target builder and same-batch exact diff before Stage 1 can be considered passed.

Actionable patch queue from this review, in priority order:

1. Add `allow_center_fallback_inside_gt=False` to `IrregularActionFormerBridgeHead`, disable fallback by default for official-compatible routes, and allow it only in explicit legacy/research ablations.
2. Add an `official_actionformer` / dense-compatible mode to `IrregularPointGeneratorV2` that locks `range_mode="absolute"`, `decode_scale_mode="level_stride"`, and `radius_scale_mode="level_stride"`.
3. Extend bridge dense-equivalence verification to batched, masked generated V2 points and add a missing-center fallback negative case.
4. Add whole-config fail-closed preflight scanning for diagnostic GT, teacher/cache, raw-prediction, and post-processing shortcut fields.
5. Make seconds conversion source axis explicit, and update detector/tools to pass `source_axis="native"` after selected-to-native conversion.
6. Make any single-class post-processing path obey the same threshold/top-k/NMS prefiltering contract as the multiclass path.

Updated experiment boundary from this review: run Stage 0 Linux preflight first, then Stage 1 same-batch official-vs-bridge assignment audit, then Stage 2 official dense selected-axis sanity. Do not start or continue new long training from these fixes until Stage 0-2 pass.

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

- 2026-07-06 monitoring-collapse review absorbed:
  - Full response recorded at `pro-review-monitoring-collapse-20260706.md` with source attachment SHA256 `9e43544dc36f49cf4d181c0b94f7fdffdc7e43239b4836b48086bfb5f79fca11`.
  - Treat `65 -> 40/42` as a likely contract failure until disproved, not as an acceptable cost of random 50% sampling.
  - Review claims to verify against current HEAD before further implementation claims:
    - `LoadFrames` / detector metadata must be able to express different GT, proposal, and postprocess axes. Selected-axis supervision should not force selected-axis NMS/eval.
    - Post-processing should run NMS on comparable native or seconds coordinates when proposals are decoded in selected coordinates.
    - `cell_left` / `cell_right` semantics need a clear decision: current full-neighbor-gap support versus a Voronoi half-cell support. This decision affects range scale, radius scale, regression denominator, downsampling, and high-IoU boundary calibration.
    - `_remap_gt_to_selected_axis` should not silently turn collapsed GTs into tiny `1e-3` segments without an ignore/visibility contract.
    - `random_fixed_subsample` should have an explicit branch and tests, so the central route cannot depend on hidden fallback behavior.
  - Audit extension required before treating any same-batch result as evidence of correctness:
    - record assigned-positive decode IoU against assigned GT;
    - record proposal-axis-to-native/seconds IoU;
    - record NMS-before/after high-IoU recall;
    - keep per-level positives and GT coverage as necessary but insufficient signals.
  - Experiment-order constraint: run official dense selected-axis sanity and bridge dense-equivalence sanity before making strong native-axis irregular-head claims. `absrange_expanded` is a short-run, gated candidate, not a proof of route correctness by itself.

- 2026-07-06 strict FAIL review absorbed:
  - Full response recorded at `pro-review-strict-fail-20260706.md` with source attachment SHA256 `7eaadb1bbe30546ba6034a4d22eafa1426835cd8cfddf865852cfe5294e1b719`.
  - Current verdict for commit `8689eff`: `FAIL` for release/long-train claims, not for the five-stage diagnostic direction.
  - Active blocking items:
    - `tools/verify_bridge_dense_equivalence.py` is too narrow because it uses hand-built stride-like points and does not cover real `IrregularPointGeneratorV2` generated grids.
    - Any bridge route that silently falls back to `center_radius_scale="full_cell_span"` or `reg_denom_mode="full_cell_span"` is not official dense stride semantics and must be either made explicit as a legacy ablation or fail-closed.
    - `selected_axis_to_dense_axis` / `convert_to_seconds` needs a numeric no-double-conversion regression test, not just metadata/string checks.
    - BATA / diagnostic GT cache paths need validation/test fail-closed guards even if current configs do not enable them.
    - The Stage 2 "official dense selected-axis sanity" config is a current-repo sanity route, not proof of official OpenTAD parity until upstream source diff/hash and near-65 reproduction are established.
  - Stage-FIX-D update: `LoadFrames` now fail-closes validation/test splits before reading diagnostic GT/cache shortcuts. The guard recognizes `subset`, `split`, and `data_split` values for `val`, `valid`, `validation`, `test`, and `testing`, and blocks BATA diagnostic GT cache / diagnostic-only aliases plus teacher or raw prediction cache shortcut flags if present. This protects future mAP credibility from eval-time leakage, but it is not evidence for the existing `65 -> 40/42` performance-collapse root cause because current configs did not enable these switches.
  - Updated experiment gate:
    - Run Linux config load, `py_compile`, pytest, and `python tools/verify_bridge_dense_equivalence.py --json` before any further training.
    - Run same-batch audits before long training, including assigned-positive target decode IoU and length-bucket GT coverage.
    - Train official dense selected-axis sanity first; if it cannot approach the claimed `~65` equal-interval baseline, do not blame sparse heads.
    - Only after the selected-axis dense/bridge controls pass should native `absrange_expanded` shortgate or long-run evidence be interpreted.

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
  - `post_processing.save_dict=True` is now enabled for the diagnostic candidates:
    - `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`
    - `input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py`
    - `input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`
    Without this, OpenTAD evaluation logs mAP but does not write `result_detection.json`, making the post-training high-IoU quality audit impossible.
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
  - Fixed a second selected-axis post-processing contract bug after the monitoring-collapse review:
    - `LoadFrames` now records `gt_axis=selected`, `proposal_axis=selected`, and `postprocess_axis=native` when GT is remapped to the selected axis.
    - `IrregularActionFormer` now allows this selected-to-native postprocess contract while still rejecting mismatched train/decode axes.
    - Non-sliding-window NMS now runs after proposals are converted from selected-axis coordinates to native coordinates, and seconds conversion is told which coordinate axis it is receiving to avoid double selected-to-native conversion.
  - Fixed selected-axis GT remap collapse handling: `_remap_gt_to_selected_axis` no longer creates synthetic `1e-3` GT segments when a segment collapses after remap/clipping. It drops those targets and records them in `dropped_selected_axis_gt_segments` for audit/debug. The random-fixed path now writes this metadata after remapping, so the dropped-target record belongs to the current sample rather than a previous loader call.
  - Added a Linux behavior regression test for selected-axis post-processing: it monkeypatches NMS and verifies selected-axis proposals are converted to native coordinates before NMS and not converted a second time during seconds conversion. Also hardened oracle/weighted helper paths to clear stale dropped-GT state on native-axis branches. This behavior test exposed and fixed a single-class post-processing label dtype bug (`torch.zeros` labels were float and could not index an `ext_cls` list).
  - Added deterministic `uniform_fixed_subsample` to support equal-interval 50% sanity controls without changing the existing `random_fixed_subsample` behavior.
  - Added `scripts/verify_official_dense_reference.py` to diff the local dense-reference files against upstream OpenTAD raw files, and added a Linux-only bridge hard uniform-grid target test that locks the official dense target/decode semantics in the stride-1 case.
  - Local dry run `python scripts/verify_official_dense_reference.py --no-fail-on-diff` succeeded and reported dense-reference drift, including major local changes in `anchor_free_head.py` and extra grid-aware FPN classes. Treat this as evidence that current-repo dense code is not an authoritative official baseline.
  - Stage 1 same-batch audit extension: `tools/audit_sparse_head_assignment.py` now reports `assigned_positive_target_decode_iou` and `oracle_assigned_recall@IoU` for hard-assigned positives on proposal, native, and seconds axes. This checks whether assigned positives still reconstruct high-IoU targets after target encode/decode, so positive coverage is no longer mistaken for localization correctness. It is diagnostic evidence only and is not an mAP conclusion.

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

## Stage 2: OFFICIAL_DENSE_SELECTED_AXIS_SANITY (2026-07-06)

- Added a named precheck-only sanity entry point:
  - `configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py`
  - This is a thin alias over the existing equal-interval dense control, preserving dense `ActionFormerHead`, official `PointGenerator`, `DensePassthroughConv1DTransformerProj`, `DensePassthroughFPNIdentity`, selected-axis GT/proposals, native post-processing, and `post_processing.save_dict=True`.
- Added `remote_runs/precheck_official_dense_selected_axis_sanity_20260706.sh` as a fail-closed local/remote precheck. It only performs config contract validation, `py_compile`, and optional pytest via `RUN_PYTEST=1`; it does not launch training, Slurm, sync, or torchrun.
- Enhanced `scripts/verify_official_dense_reference.py` with an explicit selected-axis dense sanity config validator while keeping the default upstream dense-reference diff behavior intact.
- Interpretation limit: this proves the repository can load and precheck an equal-interval 50% selected-axis dense-head sanity route without accidentally selecting the native sparse/bridge head. It does not prove mAP recovery or attribute any remaining gap to projection, neck, head, assignment, or post-processing until a controlled training/evaluation run is launched separately.

## Stage 3: Bridge Dense-Equivalence Sanity (2026-07-06)

- Added `tools/verify_bridge_dense_equivalence.py` as a synthetic fail-closed verifier for the bridge hard dense-equivalence contract. It constructs dense/uniform grids and compares the real `IrregularActionFormerBridgeHead._prepare_targets_hard` plus `get_refined_proposals` against an independent implementation of official dense `AnchorFreeHead.prepare_targets` semantics.
- Covered two small cases:
  - `stride1_dense_open_range`: one dense stride-1 level with open official-like range, overlapping GTs, center sampling, shortest-GT assignment, stride-normalized linear targets, and linear decode.
  - `multi_level_range_gate`: stride-1 plus stride-4 levels with explicit range gates, verifying short GT stays on the fine level while long GT is assigned on the coarse level.
- The verifier checks positive mask, assigned GT, classification target, encoded regression target, and decoded proposal. Mismatches include point index, level/local index, center, range, official value, and bridge value.
- Local verification on Windows:
  - `python -m py_compile tools/verify_bridge_dense_equivalence.py`: passed.
  - `python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q`: `30 passed, 12 skipped`.
  - Direct verifier execution could not run locally because Windows torch import fails while loading `c10.dll`; this is consistent with the existing Linux-only torch test policy. Linux must run `python tools/verify_bridge_dense_equivalence.py` for actual tensor equivalence evidence.
- No bridge/core head source changes were made in this stage. The verifier is intended to decide whether a core fix is needed before changing assignment/decode code.
- Stage-FIX-B update: `tools/verify_bridge_dense_equivalence.py` now extends the bridge dense-equivalence verifier beyond hand-built stride-like points. It adds `generated_v2_levelstride_equivalence`, which calls the real `IrregularPointGeneratorV2` on a uniform dense temporal grid with absolute regression ranges and level-stride decode/radius fields, then converts the generated `[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]` layout into official dense `[center, reg_min, reg_max, stride]` semantics for the same official-vs-bridge target/decode comparison.
- Interpretation limit: this is still a synthetic target/decode contract check, not mAP evidence. Passing it would show that bridge hard assignment and linear decode can match official dense semantics on generated V2 points in the covered uniform-grid case; it does not prove projection/neck behavior, training convergence, post-processing calibration, or THUMOS mAP recovery.

## Stage 4 Selected-Axis Random/Uniform Controls - 2026-07-06

- Added a minimal selected-axis control matrix without changing model/evaluator/post-processing source:
  - `input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py`: dense `ActionFormerHead`, deterministic equal-interval 50% input, selected-axis GT/proposals, `save_dict=True`.
  - `input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py`: dense `ActionFormerHead`, random-fixed 50% input, selected-axis GT/proposals, `save_dict=True`.
  - `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py`: bridge `absrange_expanded`, random-fixed 50% input, selected-axis GT/proposals, `save_dict=True`.
- All three configs keep the N16R4 THUMOS paths through the existing `*_n16r4.py` bases and use unique `work_dir` names containing `selected_axis_control`.
- Interpretation questions:
  - Uniform dense selected-axis asks whether equal-interval selected-axis can reproduce the claimed near-65 dense behavior in the current repo.
  - Random dense selected-axis asks whether random-fixed selected-axis is only a small drop from uniform selected-axis, rather than a collapse.
  - Random bridge selected-axis `absrange_expanded` asks whether the native-axis irregular contract is the dominant reason the native bridge route sits near 40/42.
- Added fail-closed precheck helpers only:
  - `remote_runs/precheck_selected_axis_control_matrix_20260706.sh`
  - `remote_runs/launch_selected_axis_control_precheck_20260706.sh`
  These scripts only load configs, run `py_compile`, and run the local pytest contract. They do not launch training or Slurm.

## Stage 5 Native Absrange-Expanded Short Gate - 2026-07-06

- Added `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py` as a short diagnostic gate over the native-axis random-fixed bridge hard linear `absrange_expanded` route. It preserves bounded expanded absolute ranges, level-stride decode/radius fields, `center_radius_scale="point_radius"`, `reg_denom_mode="left_right_mean"`, and `post_processing.save_dict=True`, while cutting the schedule to 2 epochs with validation from epoch 1 in a separate `shortgate` work directory.
- Added `remote_runs/run_bridge_absrange_expanded_shortgate_failclosed_20260706.sh` as a fail-closed helper. By default it performs config load, `py_compile`, and focused pytest checks only. It prints a train command only when `RUN_TRAIN=1`, and executes it only when both `RUN_TRAIN=1` and `PRECHECK_ONLY=0` are set.
- This gate must not be used as a metric claim. Any long-run decision still depends on the audit/decode-IoU checks, official dense sanity, selected/native-axis controls, postprocess/NMS axis sanity, and bridge-equivalence evidence.

## Stage-FIX-C Numeric Axis Guards - 2026-07-06

- Added focused numeric regression coverage for the selected/native/seconds post-processing contract. The guard uses fractional selected-axis coordinates, irregular selected positions, non-unit `fps`, `snippet_stride`, `window_start_frame`, and `offset_frames` to verify that `selected -> seconds` matches `selected -> native -> seconds`.
- Strengthened the selected-axis post-processing behavior test so monkeypatched NMS receives native-axis proposal coordinates before suppression, and final seconds conversion does not apply selected-to-native interpolation a second time.
- Evidence boundary: these tests guard coordinate conversion regressions only. They do not provide mAP evidence, do not validate training quality, and do not prove that the sparse-head route recovers the official dense baseline.

## Pro Review FAIL on Fixed SHA 06dce3d - 2026-07-06

- Full response recorded verbatim in `pro-review-fixed-sha-fail-20260706.md`.
- Source attachment SHA256: `A2BAC06990E0435A4EB8C8BDBB9FCD41282E0E57FD279E28EE6E9A65A5CFFDBF`.
- Reviewed commit: `06dce3d5955e9c2d0cf5232eace6b4224831900f` (`Harden sparse head route diagnostics`).
- Verdict absorbed as **FAIL for route correctness / dense-equivalence claim**, not as "no progress":
  - The commit made real hardening progress for bridge scale contracts, route metadata, selected/native axis checks, postprocess conversion, audit tooling, and tests.
  - It still does not prove that the sparse/irregular route is official-compatible, nor explain the `65/63 -> 40/42` collapse.
  - `HeadV3 fixed=40.20` and `bridge hard openrange=42.44` must remain diagnostic results only.
- Current allowed claim boundary:
  - Bridge hard official-compatible diagnostics are closer to auditable.
  - `official_actionformer` point generation rejects irregular centers and locks absolute range + level-stride scale.
  - Bridge hard no longer uses V2/V3 missing-center fallback.
  - The detector main path tends to convert selected proposals to native before NMS.
  - `check_fail_closed_config.py` can prevent some legacy routes from being mislabeled as dense-equivalent if it is part of mandatory preflight.
- Current forbidden claims:
  - Do not claim `40/42` is natural sparse/random sampling degradation.
  - Do not claim `HeadV3` is an official ActionFormer-equivalent irregular implementation.
  - Do not claim `openrange` is official-compatible sparse performance.
  - Do not claim selected-axis GT remap, utility conversion paths, and all scripts are fully fail-closed.
- Newly absorbed blockers before full training or dense-equivalent claims:
  - **S0:** `HeadV2/V3` still differ from official ActionFormer via soft assignment, log1p regression, geometry modulation, and V2/V3 missing-center fallback behavior.
  - **S0:** selected-axis GT remap can silently drop collapsed GT; default behavior must become fail-closed, with explicit legacy/diagnostic opt-in only.
  - **S1:** low-level postprocess utilities still allow fail-open selected-axis conversion in some paths; strict mode should become the irregular-route default, and `source_axis="auto"` must be forbidden for irregular metadata unless explicitly allowed.
  - **S1:** native irregular cell geometry and legacy full-cell-span/openrange routes are diagnostics only, not dense-equivalent evidence.
  - **S1:** config scanner protection is not enough unless enforced by every launch/preflight path.
- Required code-prep items from this review:
  - Add strict selected-axis conversion helpers in `opentad/models/utils/post_processing/utils.py`.
  - Require `metas` for `IrregularActionFormer` train/test/postprocess paths; forbid bypassing axis contracts.
  - Add `allow_center_fallback_inside_gt=False` as a first-class V2/V3 contract and block dense-equivalent claims when fallback or soft route is active.
  - Make selected-axis GT drop fail closed in `LoadFrames`, unless a config explicitly declares legacy/diagnostic allowance.
  - Force `check_fail_closed_config.py` into all launch/preflight scripts before training/eval/claim.
- Required experiment order remains:
  - Stage 0 Linux preflight: config load, `py_compile`, pytest, fail-closed scanner.
  - Stage 1 same-batch assignment/decode audit, with native and selected batches separated.
  - Stage 2 official/dense selected-axis sanity; random-fixed dense selected-axis should recover near historical `63.12`, uniform near `65.09`.
  - Stage 3 bridge selected/native controls only after Stage 0-2 pass.
  - Stage 4 HeadV2/V3 soft-assignment analysis only after bridge/dense contracts are clean.
- Updated root-cause ordering:
  - If dense selected-axis sanity also collapses, first suspect `end_to_end.py` GT remap, selected-axis metadata, seconds conversion, or NMS axis.
  - If dense selected-axis passes but bridge selected-axis collapses, first suspect bridge encode/decode scale, point layout, or target contract.
  - If bridge selected-axis passes but native bridge collapses, first suspect selected-to-native conversion, native temporal grid cell geometry, NMS axis, or duration clipping.
  - If bridge selected/native pass but HeadV3 remains near `40.20`, first suspect V2/V3 soft assignment, missing-center fallback, log1p regression, geometry modulation, and boundary auxiliary mixing.

## Pro Review WARN on Fail-Closed SHA 578fe0d - 2026-07-06

- Full response recorded verbatim in `pro-review-fail-closed-warn-578fe0d-20260706.md`.
- Source attachment SHA256: `53315A676A9937A16F3FBD325963FE119AEDEEBAB1EC2BCE0AD323BF743537B9`.
- Reviewed commit: `578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67` (`Harden sparse route fail-closed contracts`).
- Verdict absorbed as **WARN for fail-closed hardening**, not as dense-equivalent approval:
  - Patch A/B/C/D/E are mostly implemented and materially reduce fail-open routes.
  - The code hardening is close to sufficient as a guarded diagnostic baseline.
  - It still cannot support a dense-equivalent claim or final root-cause claim for the `65/63 -> 40/42` collapse.
- Remaining code-level concerns to verify/fix:
  - `selected_axis_to_dense_axis(..., strict=False)` remains backward-compatible; add lint/test coverage so sparse/irregular paths never call it without `strict=True`.
  - `check_fail_closed_config.py` should treat all `IrregularActionFormerHeadV2/V3` configs as soft/weighted by default, even when `soft_assign_topk` is omitted, unless a future audited hard-compatible flag exists.
  - `LoadFrames` should fail closed for non-empty GT with empty selected positions, rather than relying on normal sampling to avoid the case.
  - Add runner coverage tests: every train/eval/audit shell entry must call `tools/check_fail_closed_config.py` before execution.
  - Consider moving the fail-closed scanner into the common `tools/train.py` / eval runtime path, not only remote runner scripts.
- Required experiment gates remain unchanged:
  - Stage 0: Linux preflight on exact SHA `578fe0d`, including scanner, pytest, `bash -n`, and runner-gate coverage.
  - Stage 1: same-batch assignment audit with native/selected batches separated and per-sample hashes recorded.
  - Stage 2: official/dense selected-axis sanity must recover random-fixed near `63` and uniform near `65`.
  - Stage 3: bridge selected/native controls only after Stage 2 passes.
  - Stage 4: HeadV2/V3 soft/log1p/geometry decomposition only after bridge/dense contracts are clean.
- Claim boundary:
  - Allowed: fail-closed protection has been strengthened and review risks are now narrower.
  - Forbidden: claiming HeadV3/bridge openrange is official-compatible, claiming sparse selection naturally explains `40/42`, or writing paper-level mAP conclusions before Stage 0-4 evidence.

## Stage-2 Resource Boundary Update - 2026-07-06

- Commit `81c93cec133da0bb044e882de709ccfd98941599` adds a hard resource split: GPU1 on the existing Slurm allocation is reserved for short validation/audits only, while long training must be submitted through a fresh `sbatch` job that defaults to `--exclude=g0030`.
- Added `remote_runs/run_gpu1_stage2_dense_selected_axis_short_20260706.sh` and `remote_runs/launch_gpu1_stage2_dense_selected_axis_short_20260706.sh` for the two-epoch Stage-2 dense selected-axis smoke over:
  - `input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py`
  - `input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py`
- Added `remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh` and `remote_runs/submit_stage2_dense_selected_axis_long_slurm_20260706.sh` as the only current Stage-2 long-train entry. The Slurm body refuses allocation `1118197` and refuses `g0030` unless explicitly overridden.
- Disabled the old GPU1 long-train launchers and waiter scripts by replacing them with `DEPRECATED_GPU1_LONG_TRAINING_DISABLED` stubs. This prevents accidental reuse of `1118197/g0030/GPU1` for bridge/dense long runs.
- Verification:
  - Local: `python -m pytest tests/test_adapter_native_dense_headv2_contracts.py tests/test_audit_sparse_head_assignment_contracts.py tests/test_fail_closed_static_gates.py -q` passed with `67 passed, 21 skipped`; `bash -n remote_runs/*.sh` passed.
  - Remote synced marker: `.codex_synced_commit=81c93cec133da0bb044e882de709ccfd98941599`.
  - Remote: `bash -n remote_runs/*.sh`, `pytest tests/test_fail_closed_static_gates.py -q`, and Stage-2 short preflight all passed.
- Deployment status: Stage-2 short training has not been launched yet because `g0030/GPU1` is currently occupied by another C3/paction step (`1118197.669`, `CUDA_VISIBLE_DEVICES=1`). Do not launch the Stage-2 short smoke until GPU1 is free.

## Stage-2 Waiter and Stage-4 Quality Runner - 2026-07-06

- Added `remote_runs/watch_and_launch_gpu1_stage2_dense_selected_axis_short_20260706.sh` and `remote_runs/launch_watch_gpu1_stage2_dense_selected_axis_short_20260706.sh`.
  - The waiter does not train or reserve GPU by itself.
  - It launches the two-epoch Stage-2 dense selected-axis short smoke only when GPU1 memory is below `100 MiB` and the current allocation has no extra active step beyond the known allowed `1118197.660`, `.batch`, and `.extern` steps.
  - Latest deployed status at commit `86db984`: waiter is running, sees `g0030/GPU1` busy (`mem_mib=3959`, `compute_apps=1`, `other_steps=1` from step `1118197.669`), and is sleeping without launching training.
- Added `remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh`.
  - It is CPU-only and runs `tools/analyze_detection_quality.py` over Stage-2 dense short/long outputs once `result_detection.json` exists.
  - Remote preflight passed: fail-closed config scan ok, `py_compile` ok, and current no-result state safely reports `analyzed=0 missing=4`.
- Current next gate remains: wait for GPU1 to become free, let the Stage-2 short smoke run, then run the Stage-4 quality runner before deciding whether to submit Stage-2 dense long jobs through `sbatch`.

## Runtime Fail-Closed and Stage-3 Audit Fixes - 2026-07-06

- Commits `2af3344`, `9a3b080`, `fddd07d`, and `2962e8c` tighten the execution boundary after Linux/CPU audit exposed real contract bugs:
  - `tools/train.py` and `tools/test.py` now run `tools.check_fail_closed_config.scan_config_object()` immediately after config load and `--cfg-options` merge, before importing torch/OpenTAD runtime modules and before DDP. A bad runtime shortcut config now fails before `LOCAL_RANK`/DDP access.
  - `Collect` now preserves `irregular_gt_axis`, `irregular_proposal_axis`, `irregular_postprocess_axis`, `irregular_axis_contract`, and selected-axis GT drop counters in `metas`. Before this, `LoadFrames` wrote `postprocess_axis="native"` but the dataloader dropped it, so audits and runtime checks could fall back to `selected`.
  - `dense_compat_mode="official_actionformer"` is now fail-closed: it requires a dense reference grid neck (`DensePassthroughFPNIdentity`, `IrregularFPNDenseAdapter`, or `IrregularFPNDenseAdapterNorm`). Native/GridAware bridge candidates are labeled `irregular_geometry_diagnostic_candidate`, not `dense_compatible_diagnostic_candidate`.
  - `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py` now uses `IrregularFPNDenseAdapter` plus official dense prior semantics; the native shortgate remains GridAware and diagnostic-only.
- Remote verification on synced commit `2962e8c`:
  - Stage-2 dense selected-axis preflight passed with fail-closed scan, config load, `py_compile`, and runner-gate pytest; no training launched.
  - Stage-3 selected/native preflight passed with fail-closed scan, config load, and `py_compile`.
  - Stage-3 CPU one-batch assignment audit completed at `/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/stage3_bridge_selected_native_short_audit/stage3_bridge_selected_native_short_audit_20260706_225125/`.
    - selected-axis bridge control: `official_vs_current_assignment_diff.ok=True`, per-level positives `[2, 3, 6, 5, 2, 0]`, positive/class/encoded/decoded diffs all zero, target decode IoU exactly `1.0`.
    - native-axis shortgate diagnostic: `official_vs_current_assignment_diff.ok=True`, per-level positives `[0, 0, 3, 3, 2, 1]`, positive/class/encoded/decoded diffs all zero, target decode IoU exactly `1.0`.
- Interpretation:
  - The earlier `absrange_expanded` collapse risk was not only mAP noise. It revealed two implementation/contract mistakes: axis metadata was not carried into `metas`, and official dense prior semantics were being mixed with GridAware interval-center grids.
  - The selected-axis bridge control is now the cleaner bridge-vs-dense comparator. The native GridAware route remains a diagnostic irregular-geometry candidate and must not be used as dense-equivalent evidence.
  - Full/long training remains blocked until Stage-2 dense selected-axis short smoke runs and the Stage-4 quality runner has a real `result_detection.json` to analyze.
- Resource status:
  - The Stage-2 GPU1 waiter is still safe: it sees `g0030/GPU1` busy (`step 1118197.669`, about `3959 MiB`) and continues sleeping without launching training.

## Stage-2 Short to Stage-4 Quality Auto-Hook - 2026-07-06

- Added a post-short hook to `remote_runs/run_gpu1_stage2_dense_selected_axis_short_20260706.sh`:
  - Default `RUN_STAGE4_AFTER=1`.
  - The hook runs only after real short validation (`RUN_TRAIN=1`, `PRECHECK_ONLY=0`) and after both dense selected-axis short runs exit successfully.
  - The hook calls `remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh` with `REQUIRE_RESULTS=1`, so missing `result_detection.json` is a hard failure instead of a silent skip.
  - The hook unsets `CUDA_VISIBLE_DEVICES` before Stage-4, preserving the CPU-only quality-analysis boundary.
- This closes an operational gap: once the Stage-2 GPU1 waiter eventually launches the two-epoch dense sanity smoke, high-IoU detection-quality diagnostics should be produced automatically without waiting for a manual follow-up command.
- Remote preflight on synced commit `6ab8290` passed with `RUN_TRAIN=0 PRECHECK_ONLY=1 RUN_PYTEST=1`: fail-closed scan ok, both dense selected-axis configs load with the expected sampling contracts, `py_compile` ok, and the Stage-2 runner pytest gate passed (`2 passed, 8 deselected`). No training was launched by this preflight.

## Stage-2 Long Slurm to Stage-4 Quality Auto-Hook - 2026-07-06

- Added the same quality-analysis closure to `remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh`:
  - Default `RUN_STAGE4_AFTER=1`.
  - If training exits nonzero, the Slurm job returns the training error immediately.
  - If training succeeds, the job runs `remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh` with `REQUIRE_RESULTS=1`.
  - The Stage-4 child unsets `CUDA_VISIBLE_DEVICES`, preserving the CPU-only detection-quality boundary.
- This does not by itself authorize long training. The gate remains: Stage-2 short smoke must first finish and produce sane dense selected-axis metrics plus Stage-4 quality output; only then should the long Slurm submitter be used.

## Stage-2 Dense Gate Summary - 2026-07-06

- Added `tools/summarize_stage2_dense_gate.py`, a report-only gate summarizer for the Stage-2 dense selected-axis controls.
  - It inspects each target's `log.json`, `result_detection.json`, and latest `detection_quality_summary_*.json`.
  - It records `Average-mAP`, per-tIoU mAP, `Training Over`, Traceback/OOM/non-finite-loss flags, non-finite gradient skip count, missing artifacts, and latest quality summary path.
  - Short targets gate only on successful completion and artifacts by default.
  - Long targets additionally use conservative thresholds: random-fixed dense selected-axis `Average-mAP >= 60.0`, uniform/equal-interval official dense selected-axis `Average-mAP >= 62.0`. These are go/no-go guardrails, not final paper claims.
- `remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh` now writes `${RUN_TAG}_stage2_dense_gate.json` after quality analysis. This provides an immediate machine-readable answer for:
  - whether short smoke has enough evidence to submit long Slurm jobs;
  - whether long dense sanity is close enough to the expected near63/near65 band;
  - whether missing result/quality files or hard errors block interpretation.
- Local dry run without Stage-2 outputs correctly reports all four Stage-2 targets blocked by missing artifacts and `can_submit_long_after_short=false`.
- Remote preflight on synced commit `b6c14cd` passed: `py_compile`, `pytest tests/test_stage2_dense_gate_summary.py tests/test_fail_closed_static_gates.py -q`, and `RUN_TAG=stage4_gate_preflight_manual REQUIRE_RESULTS=0 bash remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh`. With no Stage-2 outputs yet, the remote gate correctly reports all four targets blocked by missing artifacts and writes `logs/stage4_detection_quality_stage2_dense/stage4_gate_preflight_manual_stage2_dense_gate.json`.

## Stage-2 Output Staleness Guard - 2026-07-06

- Added a fail-closed stale-output guard to both Stage-2 execution paths:
  - `remote_runs/run_gpu1_stage2_dense_selected_axis_short_20260706.sh`
  - `remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh`
- Before training starts, the scripts now compute the exact `gpu1_id*` run directory and refuse to proceed if it already exists.
- Controlled reruns must set `ALLOW_OVERWRITE_STAGE2_OUTPUT=1`; even then, deletion is restricted to the strict whitelist `$ROOT/exps/thumos/adatad/*/gpu1_id*`.
- Motivation: OpenTAD `log.json` is append-style and `result_detection.json` can remain from an older run. Without this guard, Stage-2 gate summaries could be polluted by stale errors or stale predictions.

## Stage-2 Short Gate Minimum Quality Signal - 2026-07-06

- Tightened `tools/summarize_stage2_dense_gate.py` so `can_submit_long_after_short` is not based on file existence alone.
- A Stage-2 target now also requires:
  - a parsed `Average-mAP` entry in `log.json`;
  - Stage-4 quality summary with `num_predictions >= 1`;
  - Stage-4 quality summary with `recall@0.30 >= 1e-6`.
- These are intentionally minimal sanity checks, not near63/near65 performance claims. They block empty-prediction or coordinate-broken short runs from authorizing long Slurm jobs while still allowing low early mAP to be inspected instead of overfitting the gate to two-epoch performance.
- Remote preflight on synced commit `1f0970a` passed: `py_compile`, `pytest tests/test_stage2_dense_gate_summary.py -q`, and `RUN_TAG=stage4_gate_min_quality_preflight REQUIRE_RESULTS=0 bash remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh`. With no Stage-2 outputs yet, the gate remains blocked and now records `missing_average_mAP` in addition to missing artifacts.
