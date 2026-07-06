## A. 总判决

**总判决：WARN（仅允许诊断推进，不允许 full training / dense-equivalent claim）。**

这个 commit 明显比前一轮更接近“official-compatible bridge”合同：`allow_center_fallback_inside_gt` 默认已改为 `False`，legacy/openrange/absrange 路线显式保留 `True`，部分 corrected 路线显式 `False`；`IrregularPointGeneratorV2` 也新增了 `dense_compat_mode="official_actionformer"`，并把 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"` 锁在一起。commit 页面显示本轮改动 24 个文件、约 +1620/-15 行，确实覆盖了你列出的核心文件族。([GitHub][1])

但它还不能支撑“dense-equivalent 已证明”或“可以长训归因”的 claim。官方 OpenTAD 的 ActionFormer target contract 是：PointGenerator 产生 `[center, reg_min, reg_max, stride]`，center sampling 用 `stride * center_sample_radius`，没有 missing-center fallback，regression target 用同一个 stride 归一化；post-processing seconds conversion 再按 `segments * snippet_stride + window_start_frame + offset_frames` 转秒。当前 commit 对这些点已有靠近，但 scanner/audit/verifier 仍偏窄，selected/native axis 与 NMS 顺序仍有可导致高 IoU 崩溃的风险。([GitHub][2]) ([GitHub][3]) ([GitHub][3]) ([GitHub][4])

| 门禁                                  | 判决                               |
| ----------------------------------- | -------------------------------- |
| remote sync                         | **ALLOW，仅限诊断分支 / Stage 0-2**     |
| Linux preflight                     | **ALLOW**                        |
| same-batch audit                    | **ALLOW，但 audit 输出必须 hard gate** |
| official dense selected-axis sanity | **ALLOW**                        |
| full training                       | **DENY，直到 Stage 0/1/2 全 PASS**   |
| dense-equivalent claim              | **DENY**                         |

---

## B. 逐文件问题

### 1. `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

**P1：`_point_fields()` 对 5+ field points 的 `point_scale = left + right` 仍保留 legacy 语义。**

* 位置：`_point_fields`, 约 L2367-L2389。
* 当前行为：当 point tensor 有 5 个以上字段时，`left_scale=point[...,3]`，`right_scale=point[...,4]`，然后 `point_scale = left_scale + right_scale`。([GitHub][5])
* 官方差异：官方 dense PointGenerator 只有一个 stride 字段，assignment radius 和 regression denominator 都直接用 stride。([GitHub][2])
* 风险：corrected V2 因为显式给了 `range_scale/radius_scale`，主路径大多能避开；但任何 fallback 到 `point_scale` 的路径都会把 stride 变成 2x，直接扩大 center radius / regression denominator / decode scale，能造成 positive 分布偏移和 proposal 宽度偏移，是 65 → 40/42 的典型根因之一。
* 修复：保留 legacy 但必须显式命名，不要让 corrected 路线隐式继承。

```python
def _point_fields(self, point_tensor):
    center = point_tensor[..., 0]
    reg_min = point_tensor[..., 1]
    reg_max = point_tensor[..., 2]

    if point_tensor.shape[-1] >= 5:
        left_scale = point_tensor[..., 3].clamp_min(self.reg_denom_floor)
        right_scale = point_tensor[..., 4].clamp_min(self.reg_denom_floor)

        # corrected/default: official-like stride scale
        point_scale = (0.5 * (left_scale + right_scale)).clamp_min(self.reg_denom_floor)

        # legacy full cell span should only be used by _scale_base("full_cell_span")
        legacy_cell_span = (left_scale + right_scale).clamp_min(self.reg_denom_floor)
    else:
        point_scale = point_tensor[..., 3].clamp_min(self.reg_denom_floor)
        left_scale = point_scale
        right_scale = point_scale
        legacy_cell_span = point_scale

    return center, reg_min, reg_max, left_scale, right_scale, point_scale, legacy_cell_span
```

然后 `_scale_base("full_cell_span")` 只在 `allow_legacy_full_cell_span=True` 时使用 `legacy_cell_span`。

---

**P1：missing-center fallback 已 fail-closed，但只覆盖 soft candidate mask，hard path 本身并不使用 `_build_candidate_mask()`。**

* 位置：`_build_candidate_mask`, `_prepare_targets_hard`。
* 当前行为：`_build_candidate_mask()` 只有 `missing_gt.any() and self.allow_center_fallback_inside_gt` 时才 fallback 到 inside-GT。([GitHub][5]) 但 hard assignment 的 `_prepare_targets_hard()` 是独立实现 center/range/shortest-GT，不调用 `_build_candidate_mask()`。([GitHub][5])
* 官方差异：官方 hard target 没有 fallback；center sampling 失败就是没有 positive。([GitHub][3])
* 风险：这不是新的 fatal bug，但“Bridge fallback 已 fail-closed”这个 claim 需要限定为 soft helper / common candidate path；hard route 的等价性要看 `_prepare_targets_hard()` 与官方 target 是否一致，不能靠 fallback flag 证明。
* 修复：在 debug/audit 输出中标明 `hard_uses_center_fallback=False`，避免误判。

---

**P1：legacy route 仍可能被误当 official-compatible。**

* 位置：configs 与 tests 中 legacy hard/openrange/absrange。
* 当前行为：测试明确要求 legacy hard/openrange/absrange `allow_legacy_full_cell_span=True` 且 `allow_center_fallback_inside_gt=True`。([GitHub][6])
* 官方差异：官方 ActionFormer 没有 full-cell-span 2x scale，也没有 missing-center fallback。([GitHub][3])
* 风险：如果结果表、work_dir 或 root-cause-notes 把这些 legacy 结果混入 corrected/official-compatible，就会把错误 contract 的 mAP 归因给“50% sparse sampling”。
* 修复：legacy configs 增加强制 metadata。

```python
route_contract = dict(
    compatibility="legacy_ablation_only",
    dense_equivalent_claim_allowed=False,
    allow_center_fallback_inside_gt=True,
    allow_legacy_full_cell_span=True,
)
```

---

### 2. `opentad/models/dense_heads/prior_generator/irregular_point_generator.py`

**P1：`dense_compat_mode="official_actionformer"` 锁住了 range/decode/radius，但没有验证 temporal grid 本身是 official dense-like。**

* 位置：`IrregularPointGeneratorV2.__init__`, `__call__`。
* 当前行为：`dense_compat_mode="official_actionformer"` 会强制 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"`。([GitHub][7])
* 官方差异：官方 PointGenerator 的 center 是 `arange(T) * stride`，可选 offset；不是任意 temporal grid。([GitHub][2])
* 风险：如果 upstream temporal grid 已经是 selected/native 混合坐标，即使 scale/range 正确，assignment center 也会错；这会直接改变 positive mask 和 encoded targets。
* 修复：在 dense compat 模式下加入 assert。

```python
def _assert_official_dense_grid(self, temporal_grid, stride, level_idx, atol=1e-5):
    center = temporal_grid["center"]
    if center.dim() == 2:
        ref = torch.arange(center.shape[1], device=center.device, dtype=center.dtype) * float(stride)
        if not torch.allclose(center[0], ref, atol=atol, rtol=0):
            raise ValueError(
                f"dense_compat_mode=official_actionformer requires dense centers "
                f"arange(T)*stride at level {level_idx}; got non-equidistant grid."
            )
```

---

### 3. `opentad/models/detectors/irregular_actionformer.py`

**P0/P1：NMS 仍可能在 selected axis 上执行。**

* 位置：`post_processing`, 约 L1937-L2044。
* 当前行为：单类别分支已执行 `pre_nms_thresh` / `pre_nms_topk`；之后先 `_segments_to_axis(... postprocess_axis)`，再 NMS，再 `convert_to_seconds(... source_axis=postprocess_axis)`。([GitHub][8])
* 官方差异：官方 post-processing 的 NMS 坐标与最终 seconds conversion 的线性时间轴一致；最终 seconds conversion 是 native frame/snippet 语义。([GitHub][4])
* 风险：如果 `postprocess_axis="selected"`，NMS 的 IoU 是 selected-index IoU，不是 native-time IoU。50% 等间隔时 selected axis 与 native axis 差一个比例；不规则采样时更严重。高 IoU mAP 尤其会崩。
* 修复：所有 irregular/selected 输入在 NMS 前统一转 native axis。

```python
# before batched_nms
nms_axis = "native"
if proposal_axis != "native":
    segments_for_nms = self._segments_to_axis(segments, metas[i], proposal_axis, nms_axis)
else:
    segments_for_nms = segments

if self.test_cfg.get("nms", None) is not None:
    segments_for_nms, scores, labels = batched_nms(
        segments_for_nms, scores, labels, **self.test_cfg.nms
    )

segments = self._segments_to_seconds(segments_for_nms, metas[i], "native")
```

---

### 4. `opentad/models/utils/post_processing/utils.py`

**P1：`convert_to_seconds(source_axis="selected")` 缺少 selected metadata 时会静默当 native/dense 转换。**

* 位置：`convert_to_seconds`, 约 L785-L832。
* 当前行为：新增 `source_axis` 是正确方向；`source_axis=="selected"` 时会尝试 selected→dense，但如果 meta 缺少 `selected_positions/valid_len`，当前逻辑仍可能静默落到普通 conversion。([GitHub][9])
* 官方差异：官方没有 selected/native 双轴概念，只有一个线性 segment 坐标到秒的公式。([GitHub][4])
* 风险：silent fallback 是最危险的 collapse 类型：训练看 selected GT，eval 却把 selected proposals 当 native 秒转，mAP 会大幅下降但日志不一定报错。
* 修复：selected 轴必须 fail-closed。

```python
def convert_to_seconds(segments, meta, source_axis="native"):
    if source_axis not in {"selected", "native"}:
        raise ValueError(f"source_axis must be explicit, got {source_axis!r}")

    if source_axis == "selected":
        if "selected_positions" not in meta or "valid_len" not in meta:
            raise ValueError("source_axis='selected' requires selected_positions and valid_len in meta")
        segments = selected_axis_to_dense_axis(segments, meta)

    # official-style native/frame conversion follows here
```

同时建议禁止 detector 内部继续使用 `source_axis="auto"`。

---

### 5. `tools/audit_sparse_head_assignment.py`

**P1：official target builder 是独立近似实现，不是直接调用官方 OpenTAD `AnchorFreeHead.prepare_targets()`。**

* 位置：`build_official_dense_targets`, `compare_current_targets_to_official_dense`。
* 当前行为：脚本确实新增了 `build_official_dense_targets()`，输出 `positive_mask`、`assigned_class`、encoded/decoded target、per-level positives、GT coverage diff；比较逻辑也有 `official_vs_current_assignment_diff`。([GitHub][10]) ([GitHub][10])
* 官方差异：它复刻了官方语义，但不是 import 官方代码；因此只能证明“与本脚本复刻语义一致”，不能单独证明“与官方 OpenTAD 一致”。
* 风险：如果复刻脚本和 current head 共享同一个错误假设，例如 left/right scale 解释、axis 解释、GT remap，audit 可能假阴性。
* 修复：增加可选 official upstream import 对照，或至少把官方公式逐字段校验。

```python
def assert_official_dense_point_contract(points):
    # point layout [center, reg_min, reg_max, stride]
    concat = torch.cat(points, dim=0)
    assert concat.shape[-1] == 4
    assert torch.all(concat[:, 3] > 0)

def compare_against_upstream_anchor_free_head(...):
    # optional: import official OpenTAD as external path
    # run AnchorFreeHead.prepare_targets on official points
    # compare with local official_dense_targets
```

---

### 6. `tools/check_fail_closed_config.py`

**P1：scanner 覆盖了 shortcut keys，但没有覆盖 dense-equivalent / legacy contract keys。**

* 位置：`BLOCKED_KEY_PATTERNS`, `scan_config_object`。
* 当前行为：扫描 `diagnostic_gt`、teacher/cache、raw prediction、prediction shortcut、`load_predictions`、`prediction_folder`、`fuse_list` 等启用字段，并递归遍历 mmengine config object。([GitHub][11])
* 缺口：没有扫描 `allow_center_fallback_inside_gt=True`、`allow_legacy_full_cell_span=True` 是否出现在 official-compatible config；也没有区分 train/eval/test shortcut。
* 风险：能防 GT/cache 泄漏，但不能防“legacy ablation 被命名成 official-compatible”。
* 修复：增加 policy-aware scanner。

```python
OFFICIAL_BLOCKED_TRUE_KEYS = {
    "allow_center_fallback_inside_gt",
    "allow_legacy_full_cell_span",
}

def scan_contract_object(obj, path="cfg", official_compatible=False):
    violations = []
    obj = _plain_value(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}"
            low = str(key).lower()
            if official_compatible and low in OFFICIAL_BLOCKED_TRUE_KEYS and is_enabled_value(value):
                violations.append({
                    "path": child,
                    "key": key,
                    "value": repr(_plain_value(value)),
                    "reason": "official-compatible config cannot enable legacy fallback/scale",
                })
            violations.extend(scan_contract_object(value, child, official_compatible))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            violations.extend(scan_contract_object(value, f"{path}[{i}]", official_compatible))
    return violations
```

---

### 7. `tools/verify_bridge_dense_equivalence.py`

**P1：verifier 是 synthetic sanity，不是 dataloader/full-batch equivalence proof。**

* 位置：file docstring, cases, `run_all_checks`。
* 当前行为：脚本自己说明是 “small synthetic sanity verifier”，构造三类 case：stride1 open range、多层 range gate、generated V2 levelstride，然后比较 positive、assigned_gt、cls target、encoded target、decoded proposal。([GitHub][12]) ([GitHub][12])
* 官方差异：官方训练中的 batch mask、真实 THUMOS GT、sliding window、padding、multi-GT conflict、empty GT video、short action、boundary cases 都没有被覆盖。
* 风险：synthetic PASS 不能推出真实 dataloader 下 50% selected-axis dense route 会回到 65。
* 修复：增加 `--config --split --num-batches`，跑真实 dataloader + current head + official target builder。

```python
parser.add_argument("--config")
parser.add_argument("--split", default="train")
parser.add_argument("--num-batches", type=int, default=8)
parser.add_argument("--fail-on-any-diff", action="store_true")
```

Gate 要求：
`positive_mask_diff_count == 0`，`assigned_class_diff_count == 0`，`encoded_target_max_abs_diff <= 1e-5`，`decoded_target_max_abs_diff <= 1e-5`，每 level positive count 完全一致。

---

### 8. `tests/test_adapter_native_dense_headv2_contracts.py`

**P2：大量测试是字符串/配置契约测试，不足以证明数值等价。**

* 当前行为：测试确实检查 corrected configs 的 `allow_center_fallback_inside_gt=False`、`dense_compat_mode="official_actionformer"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"`；也检查 legacy configs 显式 `True`。([GitHub][6]) ([GitHub][6])
* 缺口：没有覆盖 `source_axis`，没有覆盖单类别 pre-NMS filter，没有覆盖 NMS 前 selected→native。
* 修复：新增数值测试，而不是只读源码字符串。

---

### 9. `configs/adatad/thumos/*bridge*hard*linear*.py`

**P1：corrected/legacy 已部分分离，但命名仍可能污染结果表。**

* 当前行为：absrange expanded 等 corrected config 加了 `allow_center_fallback_inside_gt=False` 与 `dense_compat_mode="official_actionformer"`；legacy hard/openrange/absrange 仍显式 `allow_center_fallback_inside_gt=True`。([GitHub][1])
* 风险：`hard_linear_absrange_n16r4` 这类名字不够明显表达 legacy；建议全部重命名或加 `route_contract`.
* 修复：所有 legacy work_dir 加 `_legacy_ablation_do_not_claim_dense_equiv`。

---

### 10. `configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py`

**P1：这个 config 是 thin wrapper，sanity 的可信度取决于 base config。**

* 当前行为：它只继承 `input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py`，设置 `post_processing=save_dict=True` 和 work_dir；注释声称 50% equal-interval input、GT/proposals selected axis、native-axis post-processing、dense ActionFormer head。([GitHub][13])
* base config 的确使用 `uniform_fixed_subsample`、`remap_gt_to_selected_axis=True`，head 是 `ActionFormerHead` + official `PointGenerator` 风格。([GitHub][14]) ([GitHub][15])
* 风险：因为 wrapper 本身不显式 assert `postprocess_axis=native`、`proposal_axis=selected`、`gt_axis=selected`，base 变化会 silently 改变 sanity 语义。
* 修复：在 wrapper 显式写 axis contract。

```python
model = dict(
    detector_axis_contract=dict(
        gt_axis="selected",
        proposal_axis="selected",
        postprocess_axis="native",
        require_selected_positions=True,
        forbid_auto_seconds_conversion=True,
    )
)
```

---

## C. 当前修复是否真实完成

| 修复项                                                               | 核验结论                                                                                                                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bridge fallback 是否真正 fail-closed                                  | **部分完成。** 默认 `False`，soft candidate fallback 受 flag 控制；hard path本来不走该 helper。([GitHub][5]) ([GitHub][5])                                              |
| legacy route 是否仅作为 ablation                                       | **代码上显式 legacy，但命名/claim 仍需防污染。** tests 还要求 legacy fallback=True。([GitHub][6])                                                                        |
| dense_compat_mode 是否锁住 range/decode/radius                        | **基本完成。** 但未验证 temporal grid 本身 dense-like。([GitHub][7])                                                                                              |
| convert_to_seconds(source_axis=...) 是否仍有隐式二次转换风险                  | **仍有风险。** `source_axis` 是进步，但 `auto`/缺 metadata 静默 fallback 需要禁掉。([GitHub][9])                                                                        |
| selected→native 后 NMS 和 seconds conversion 是否坐标一致                 | **未完全证明。** 当前流程允许 NMS 在 `postprocess_axis` 上做，若是 selected axis，高 IoU 风险仍在。([GitHub][8])                                                               |
| 单类别 post-processing 是否执行 pre-NMS filter                           | **完成。** 单类别分支已 threshold/topk。([GitHub][8])                                                                                                           |
| check_fail_closed_config 是否覆盖 eval/test shortcut 风险               | **部分完成。** 覆盖 diagnostic/teacher/cache/raw prediction/load_predictions/fuse_list 等 key，但没有 policy-aware legacy/dense-equivalent scanner。([GitHub][11]) |
| audit 是否构造 official dense-like target                             | **部分完成。** 有 independent official target builder，但不是直接 official upstream implementation。([GitHub][10])                                                 |
| official_vs_current_assignment_diff 是否足以证明 assignment equivalence | **不足以单独证明。** 可作为 Stage 1 hard gate，但必须跑 same-batch real dataloader。                                                                                   |
| verify_bridge_dense_equivalence 是否仍过窄                             | **是。** 文件自己定位为 small synthetic sanity，不是 full dataloader equivalence。([GitHub][12])                                                                   |

---

## D. 65 → 40/42 Avg-mAP 崩溃根因排序

1. **assignment 监督定义偏移**：positive mask、shortest-GT conflict、center sampling、range gate 任一偏移都会让 head 学错监督合同。最高优先级。
2. **regression range / center sampling / point scale 偏移**：legacy full-cell-span、fallback inside-GT、radius scale 2x 是最像 65→40 的根因。
3. **regression encode/decode denominator 不一致**：训练 target 用一种 denominator，decode 用另一种 denominator，高 IoU 直接崩。
4. **selected-axis vs native-axis GT 混用**：GT remap、proposal axis、postprocess axis、seconds conversion 任何一个 silent fallback 都足以毁掉 mAP。
5. **post-processing / seconds conversion / NMS 坐标不一致**：尤其 NMS 在 selected axis、eval 在 native seconds。
6. **projection / neck / temporal grid alignment**：50% selected dense route必须保证 feature temporal length、mask、point stride、selected positions 一一对应。
7. **loss normalizer / positive count 分布**：positive 数过多/过少会改变 focal/reg loss balance。
8. **data pipeline / eval shortcut**：GT/cache/raw prediction shortcut 会制造假高或假低。
9. **optimizer / schedule**：通常不是 65→40 的首因，除非 batch/positive 分布大变后 schedule 没调。
10. **其他实现错误**：mask padding、valid length、sliding window offset、class label offset、proposal clipping。

---

## E. 下一步实验路线

### Stage 0：Linux preflight

命令：

```bash
python -m py_compile \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/prior_generator/irregular_point_generator.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/utils/post_processing/utils.py \
  tools/audit_sparse_head_assignment.py \
  tools/check_fail_closed_config.py \
  tools/verify_bridge_dense_equivalence.py

pytest -q tests/test_adapter_native_dense_headv2_contracts.py

python tools/check_fail_closed_config.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  --json-out work_dirs/preflight/fail_closed_scan.json

python tools/verify_bridge_dense_equivalence.py --json
```

PASS gate：

```text
py_compile PASS
pytest PASS
check_fail_closed_config ok=true
verify_bridge_dense_equivalence ok=true
```

失败解释：

* py_compile/pytest fail：禁止 Stage 1。
* scanner fail：说明 config 有 shortcut/legacy contamination。
* verifier fail：bridge hard target 不具备 synthetic dense equivalence，禁止 audit/训练 claim。

---

### Stage 1：same-batch official-vs-current assignment audit

configs：

```text
configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py
```

命令：

```bash
python tools/audit_sparse_head_assignment.py \
  --configs \
    configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
    configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  --split train \
  --num-batches 8 \
  --device cuda \
  --out-dir work_dirs/assignment_audit/stage1_same_batch
```

必须输出字段：

```text
official_vs_current_assignment_diff.ok
positive_mask_diff_count
assigned_class_diff_count
encoded_target_max_abs_diff
decoded_target_max_abs_diff
official_per_level_positive_count
current_per_level_positive_count
per_level_positive_count_diff
gt_coverage_diff
assigned_positive_target_decode_iou
oracle_assigned_recall@IoU
```

PASS gate：

```text
all rows official_vs_current_assignment_diff.ok == true
positive_mask_diff_count == 0
assigned_class_diff_count == 0
encoded_target_max_abs_diff <= 1e-5
decoded_target_max_abs_diff <= 1e-5
all per_level_positive_count_diff == 0
no gt_coverage_diff.differs
```

失败解释：

* positive diff：assignment contract 不等价。
* encoded diff：regression denominator 错。
* decoded diff：decode scale 错。
* GT coverage diff：center/range/axis 错。

---

### Stage 2：official dense selected-axis sanity

目标：验证 **50% 等间隔 dense route 本身是否能接近 65 Avg-mAP**。这一步不证明 irregular bridge，只证明“50% selected-axis dense + official head”不是天然崩溃。

命令示例：

```bash
bash tools/dist_train.sh \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  4 \
  --seed 42 \
  --deterministic
```

或本仓库实际入口：

```bash
python tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --seed 42 \
  --deterministic
```

PASS gate：

```text
Avg-mAP close to known 50% uniform dense baseline:
expected band: 64 ± 2, hard fail if < 60
mAP@0.6/@0.7 not catastrophically below prior uniform baseline
prediction count not explosively high/low
postprocess logs show native-axis seconds conversion
```

失败解释：

* 若 Stage 2 低于 60：先停，说明 data pipeline / selected-axis dense sanity / postprocess 仍错，不要跑 corrected bridge。
* 若 Stage 2 接近 65：50% sparse sampling 本身不是 collapse 主因，继续 Stage 3。

---

### Stage 3 以后，仅作为计划

1. corrected bridge short smoke：只跑 1-3 epoch，检查 loss、positive count、proposal recall。
2. gated long training：只有 Stage 0/1/2 全 PASS 后允许。
3. proposal recall / high-IoU localization diagnosis：分 tIoU@0.5/0.6/0.7 看 decoded proposal recall，而不是只看最终 mAP。

---

## F. 关键 patch 级代码

### 1. BridgeHead：禁止 corrected 路线隐式 legacy scale

```python
def _scale_base(self, left_scale, right_scale, point_scale, mode, range_scale=None, radius_scale=None):
    if mode == "full_cell_span":
        if not self.allow_legacy_full_cell_span:
            raise ValueError("full_cell_span requires allow_legacy_full_cell_span=True")
        return (left_scale + right_scale).clamp_min(self.reg_denom_floor)

    if mode == "left_right_mean":
        return (0.5 * (left_scale + right_scale)).clamp_min(self.reg_denom_floor)

    if mode == "point_radius":
        if radius_scale is None:
            raise ValueError("point_radius requires explicit radius_scale in corrected bridge")
        return radius_scale.clamp_min(self.reg_denom_floor)

    if mode == "point_range":
        if range_scale is None:
            raise ValueError("point_range requires explicit range_scale in corrected bridge")
        return range_scale.clamp_min(self.reg_denom_floor)

    raise ValueError(f"Unsupported scale mode: {mode}")
```

### 2. IrregularPointGeneratorV2：dense compat grid assert

```python
if self.dense_compat_mode == "official_actionformer":
    self.range_mode = "absolute"
    self.decode_scale_mode = "level_stride"
    self.radius_scale_mode = "level_stride"
    self.require_official_dense_grid = True
```

```python
if getattr(self, "require_official_dense_grid", False):
    expected = torch.arange(T, device=center.device, dtype=center.dtype) * float(stride)
    if not torch.allclose(center[0], expected, atol=1e-5, rtol=0):
        raise ValueError(
            f"official_actionformer dense_compat requires center=arange(T)*stride; "
            f"level={level_idx}, stride={stride}"
        )
```

### 3. post-processing：NMS 前统一 native axis

```python
segments_native = self._segments_to_axis(
    segments,
    metas[i],
    source_axis=proposal_axis,
    target_axis="native",
)

if self.test_cfg.get("nms", None) is not None:
    segments_native, scores, labels = batched_nms(
        segments_native, scores, labels, **self.test_cfg.nms
    )

segments_seconds = self._segments_to_seconds(segments_native, metas[i], "native")
```

### 4. fail-closed scanner：official policy

```python
def is_official_compatible_cfg(cfg):
    text = json.dumps(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg, default=str)
    return (
        "official_dense_selected_axis_sanity" in text
        or "dense_compat_mode" in text and "official_actionformer" in text
        or "official-compatible" in text
    )

def scan_official_contract(cfg):
    violations = []
    official = is_official_compatible_cfg(cfg)
    if official:
        violations += scan_for_true_keys(
            cfg,
            blocked_true_keys={
                "allow_center_fallback_inside_gt",
                "allow_legacy_full_cell_span",
            },
        )
    return violations
```

### 5. same-batch audit gate

```python
def gate_official_vs_current(row):
    diff = row["official_vs_current_assignment_diff"]
    assert diff["ok"], diff
    assert diff["positive_mask_diff_count"] == 0
    assert diff["assigned_class_diff_count"] == 0
    assert (diff["encoded_target_max_abs_diff"] or 0.0) <= 1e-5
    assert (diff["decoded_target_max_abs_diff"] or 0.0) <= 1e-5
    assert all(x == 0 for x in diff["per_level_positive_count_diff"])
    assert not any(item["differs"] for item in diff["gt_coverage_diff"])
```

### 6. verifier：扩展到真实 dataloader

```python
def verify_config_batches(cfg_path, split="train", num_batches=8, atol=1e-5):
    cfg = Config.fromfile(cfg_path)
    loader = build_loader(cfg, split)
    model = build_detector(cfg.model).eval().cuda()

    failures = []
    for batch_idx, data in enumerate(loader):
        if batch_idx >= num_batches:
            break
        with torch.no_grad():
            audit_rows = audit_one_batch(model, data)
        for row in audit_rows:
            diff = row.get("official_vs_current_assignment_diff")
            if diff is None or not diff["ok"]:
                failures.append((batch_idx, row["video_name"], diff))
    return {"ok": len(failures) == 0, "failures": failures}
```

---

## G. 最终目标要证明什么

最终不是先证明“irregular head 更强”，而是按三层因果拆开：

1. **50% sparse sampling 本身是否导致崩溃？**
   由 Stage 2 回答。若 official dense selected-axis sanity 接近 65，则 50% 等间隔采样本身不是 40/42 collapse 主因。

2. **irregular head 监督合同是否导致崩溃？**
   由 Stage 1 + corrected bridge smoke 回答。若 same-batch assignment/encoded/decoded 全等价但 mAP 仍崩，再查 projection/neck/postprocess；若 Stage 1 不等价，根因就是监督合同。

3. **能否构造 dense-equivalent bridge，在 50% 等间隔采样下接近 65 Avg-mAP？**
   只有当 Stage 0/1/2 PASS，corrected bridge short smoke 正常，proposal recall 正常，长训接近 65，才允许 claim。

4. **如果不能，下一条更优雅路线是什么？**
   放弃“把 irregular head 强行伪装成 dense-equivalent”的路线，转向：
   **official dense head selected-axis sanity 作为稳定检测器基线 + 明确 native-time proposal adapter / native-time NMS / seconds conversion + 单独训练 sparse sampler。**
   论文故事应变成：先证明检测器合同不崩，再证明 adaptive sampling 不是靠错误监督或 eval shortcut 获得收益。

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/2213388febace4cc90a6fb5c8159ea069fd47adf "Harden sparse-head preflight audit gates · yuzbo/OpenTAD_SparseHeadClean_20260702@2213388 · GitHub"
[2]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[3]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[4]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/utils/post_processing/utils.py "OpenTAD/opentad/models/utils/post_processing/utils.py at main · sming256/OpenTAD · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/tests/test_adapter_native_dense_headv2_contracts.py "OpenTAD_SparseHeadClean_20260702/tests/test_adapter_native_dense_headv2_contracts.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/opentad/models/detectors/irregular_actionformer.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/opentad/models/utils/post_processing/utils.py "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/post_processing/utils.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/tools/audit_sparse_head_assignment.py "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/tools/check_fail_closed_config.py "OpenTAD_SparseHeadClean_20260702/tools/check_fail_closed_config.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/tools/verify_bridge_dense_equivalence.py "OpenTAD_SparseHeadClean_20260702/tools/verify_bridge_dense_equivalence.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[13]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[14]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[15]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/2213388febace4cc90a6fb5c8159ea069fd47adf/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control.py at 2213388febace4cc90a6fb5c8159ea069fd47adf · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
