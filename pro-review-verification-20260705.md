---
date: 2026-07-05
status: verified-against-local-tree
source_review: pro-review-20260705.md
scope: Verification of Pro review claims against the current local repository.
local_tree_note: This verification checks source/config/test files in the local working tree. Raw remote logs/checkpoints are intentionally not kept in this clean repository, so metric and per-level-positive claims are verified only against notes when no raw artifact exists locally.
---

# Pro Review Verification 2026-07-05

## Summary

Most implementation-level issues raised by Pro are real in the current local code/configs. The strongest confirmed problems are:

1. `IrregularPointGeneratorV2` uses full local cell span `cell_left + cell_right` and, in `range_mode="hard"`, scales `regression_range` by that span.
2. Bridge hard linear uses the same full-span `point_scale` for center sampling radius and symmetric-linear regression encode/decode.
3. V2/V3 soft assignment defaults to `use_regress_range=False` and uses per-GT top-k soft weights, not dense ActionFormer's hard shortest-GT assignment contract.
4. Dense control/grid-aware dense configs use selected-axis GT, while HeadV3/Bridge configs inherit native-axis GT.
5. GridAware/DensePassthrough cross-over configs are route sanity checks, not clean projection/neck attribution.

The main claims that are **not fully reproducible from this clean repo alone** are the exact per-level assignment audit counts (`[83, 22, 0, 0, 0, 0]`, `[184, 151, 104, 56, 27, 18]`) and final remote mAP values beyond what is recorded in notes. The repo deliberately excludes raw logs/results.

## Claim-by-Claim Verification

| Pro claim | Local status | Evidence |
|---|---|---|
| GitHub live review failed because URL returned 404 | Contextual, not a code issue | Pro recorded this boundary in `pro-review-20260705.md`. The repo was later created as private, so external Pro access can still fail without credentials. |
| HeadV3 fixed did not crash; low score is not a training failure | Supported by notes, raw logs absent locally | `root-cause-notes.md` records fixed `40.20`, no Traceback/OOM/non-finite loss, stable `422000` predictions. Raw remote logs are not in repo. |
| `IrregularPointGeneratorV2` uses `point_scale = cell_left + cell_right` | Confirmed | `opentad/models/dense_heads/prior_generator/irregular_point_generator.py:59-61`. |
| `range_mode="hard"` multiplies regression range by full cell span | Confirmed | `opentad/models/dense_heads/prior_generator/irregular_point_generator.py:67-69`. |
| V2 point tensor preserves left/right scales in slots 3/4 | Confirmed | `opentad/models/dense_heads/prior_generator/irregular_point_generator.py:82`; test asserts this at `tests/test_adapter_native_dense_headv2_contracts.py:333-337`. |
| `range_mode="absolute"` does not scale by cell span | Confirmed | Code: `opentad/models/dense_heads/prior_generator/irregular_point_generator.py:70-72`; test: `tests/test_adapter_native_dense_headv2_contracts.py:340-361`. |
| `temporal_grid["level_scale"]` is half-span while point scale is full-span | Confirmed | `opentad/models/utils/temporal_grid.py:79` and `149`; test preserves explicit cells and level-scale behavior at `tests/test_adapter_native_dense_headv2_contracts.py:364-381`. |
| Current bridge hard config uses dense-style `(0,4)...` ranges with `range_mode="hard"` | Confirmed | `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py:24-28`. |
| Openrange config sets all ranges to `(0,10000)` while keeping `range_mode="hard"` | Confirmed | Config and test: `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py`; `tests/test_adapter_native_dense_headv2_contracts.py:238-242`. |
| Absrange config switches to `range_mode="absolute"` | Confirmed | `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py:3`; test: `tests/test_adapter_native_dense_headv2_contracts.py:244-248`. |
| Current bridge hard positive counts are low-level-only and openrange restores high-level positives | Not independently reproducible from local repo | Exact counts appear only in Pro review text. No assignment audit JSON/CSV exists in this clean repo. This needs a reproducible audit artifact. |
| Openrange final mAP `42.44` proves range/scale issue but is not final fix | Supported by Pro/monitor notes, raw logs absent locally | The result is recorded in `pro-review-20260705.md`; no raw `log.json` is stored locally. Interpretation is technically plausible because openrange removes range specialization. |
| Bridge hard linear is not fully dense-like | Confirmed | It is hard/linear, but center radius uses full-span `point_scale` at `opentad/models/dense_heads/irregular_actionformer_bridge_head.py:484-487`; encode/decode use full-span `point_scale` at `242-267`. |
| Bridge hard uses shortest-duration conflict resolution | Confirmed present | `opentad/models/dense_heads/irregular_actionformer_bridge_head.py:502-504`; dense baseline has analogous logic at `opentad/models/dense_heads/anchor_free_head.py:1190-1195`. |
| Bridge hard has no obvious missing-GT fallback in hard mode | Confirmed | `_prepare_targets_hard` does not add a fallback after empty candidate sets; it zeros reg targets when `reg_weight=0` at `opentad/models/dense_heads/irregular_actionformer_bridge_head.py:523-525`. `oracle_point` has fallback behavior, but it is a separate assignment mode. |
| Dense `AnchorFreeHead` defaults to regression range gating | Confirmed | `opentad/models/dense_heads/anchor_free_head.py:28`; target logic at `1182-1188`. |
| Dense `AnchorFreeHead` uses center sampling and shortest-GT conflict | Confirmed | Center sampling: `opentad/models/dense_heads/anchor_free_head.py:1157-1177`; shortest duration: `1190-1195`. |
| V2/V3 soft assignment defaults to no regression range gate | Confirmed | V2: `opentad/models/dense_heads/irregular_actionformer_head_v2.py:33`; V3: `opentad/models/dense_heads/irregular_actionformer_head_v3.py:43`; test: `tests/test_adapter_native_dense_headv2_contracts.py:181-190`. |
| V2 soft assignment is per-GT top-k weighted assignment | Confirmed | Top-k/quality weights: `opentad/models/dense_heads/irregular_actionformer_head_v2.py:262-287`; soft class/reg targets: `320-334`. |
| V2/V3 actual positive counts `440-1016` and `pos_mass/count ~= 0.71` | Supported by notes only | Recorded in `root-cause-notes.md`, but no raw debug dump is stored locally. |
| `reggate=38.32` refutes “range gate alone fixes it” | Supported by notes only | `root-cause-notes.md` records fixed `40.20`, nogeometry `39.32`, reggate `38.32`. Raw logs are absent locally. |
| Dense control uses selected-axis GT while HeadV3/Bridge use native-axis GT | Confirmed | Base sparse route uses `remap_gt_to_selected_axis=False` at `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_actionformer_base.py:28,62,89`; dense control uses `True` at `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control.py:14,43,65`; tests assert both at `tests/test_adapter_native_dense_headv2_contracts.py:297-303`. |
| `51.59 -> 40.20` is not a pure head-only attribution | Confirmed as an attribution constraint | The GT-axis mismatch is real. Notes already record this boundary in `root-cause-notes.md:14`. |
| GridAware vs DensePassthrough cross-over is route sanity, not strong projection/neck attribution | Confirmed | Projection wrappers mainly keep dense feature path and pass/downsample grid: `opentad/models/projections/actionformer_proj.py:176-241`; FPN wrappers mainly pass temporal grids: `opentad/models/necks/fpn.py:130-148`; notes explicitly state route-sanity limit in `root-cause-notes.md:50`. |
| Proposed `center_radius_scale` / `reg_denom_mode` knobs exist | Implemented after verification | `opentad/models/dense_heads/irregular_actionformer_bridge_head.py` now exposes explicit scale-base knobs. Defaults preserve prior full-cell behavior. |
| Proposed `bridge_hard_linear_absrange_radiuslevel_n16r4` config exists | Implemented after verification | `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py` uses absolute range plus half-cell radius/denom. |
| Proposed assignment audit tool exists | Implemented after verification | `tools/audit_sparse_head_assignment.py` writes JSON/CSV same-batch assignment diagnostics. |

## Verified Root-Cause Status

### Confirmed Implementation Problems

- **Range/scale semantic mismatch is real.** The code really lets dense-like ranges be interpreted as local-cell-span-scaled ranges in `range_mode="hard"`.
- **Bridge hard linear is still scale-mismatched relative to dense ActionFormer.** It copies some dense target logic, but still uses full irregular cell span for center radius and linear regression denominator.
- **Soft assignment route is structurally different from dense.** The difference is not only a missing regression range gate; it is a different target construction contract.
- **GT-axis mismatch is real and invalidates head-only attribution.**

### Partially Verified / Needs Raw Artifact

- **Per-level positive collapse and recovery** are plausible and consistent with code, but exact counts require a saved audit output.
- **Openrange `42.44` and final metric comparisons** are recorded in notes/review, but should be tied to raw remote `log.json` or a copied concise metric file if they will be used in a paper/table.

### Implemented After This Verification

- `center_radius_scale` and `reg_denom_mode` knobs.
- `bridge_hard_linear_absrange_radiuslevel_n16r4.py`.
- Same-batch assignment audit tool with per-level positives, candidate counts, GT length-bucket coverage, radius-base stats, valid-mask counts, and encode/decode reconstruction error.

### Still Not Implemented

- Matched-axis diagnostic configs/results.

## Decision

Pro's main critique is valid: **do not continue long training based only on the current bridge/openrange configs.** The next engineering step should be:

1. Add a same-batch assignment audit tool.
2. Add corrected radius/denominator knobs to BridgeHead.
3. Add `bridge_hard_linear_absrange_radiuslevel_n16r4.py`.
4. Run audit before any new long training.

Long training should resume only if audit shows reasonable per-level positives, GT coverage by length bucket, and train/inference decode reconstruction consistency.
