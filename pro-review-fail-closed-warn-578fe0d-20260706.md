## 结论先给

**最终 verdict：WARN。**

理由很明确：**578fe0d 作为“fail-closed hardening patch”基本方向正确，Patch A/B/C/D/E 大部分已经落地；但它还不能声明 dense-equivalent，不能声明 65/63 → 40/42 的根因已经被最终证明。** 当前最多能说：代码级防线明显加强，错误 claim 的入口被大量收窄；实验闭环还缺 Stage 0–4。

我审的是固定 commit `578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67`，不是浮动分支。GitHub 固定 commit 页显示它的 parent 是 `06dce3d...`，共改动 20 个文件、`+1674/-19`，文件列表覆盖你列出的 postprocess、detector、LoadFrames、HeadV2/V3、scanner、audit、tests、remote runners 和 notes。([GitHub][1])

---

## A. Patch A–E 是否完整实现

### Patch A：selected-axis postprocess strict / 禁止 irregular auto fail-open

**判定：基本 PASS，但仍有 utility-level 边缘 fail-open。**

证据：

`opentad/models/utils/post_processing/utils.py` 新增了 selected-axis metadata 检查：`_has_selected_axis_meta`、`_validate_selected_axis_meta`，并检查 metadata 缺失、native-axis 冲突、空 selected positions、非有限值、非单调、坐标越界等情况。([GitHub][2])

`convert_to_seconds(..., source_axis="auto")` 现在在 irregular selected-axis metadata 存在时默认拒绝 auto 轴推断；`source_axis="selected"` 时要求 selected metadata，并通过 strict selected-axis conversion 转 dense/native 轴再换秒。([GitHub][2])

**剩余问题：** `selected_axis_to_dense_axis(..., strict=False)` 仍保留默认非 strict。主 detector 路径已经传 `strict=True`，但 analysis/eval/dump 脚本如果直接调用这个 utility，仍可能绕过 fail-closed。建议保留 backward-compatible API，但新增 lint/test：禁止 sparse/irregular route 中直接调用不带 `strict=True` 的 `selected_axis_to_dense_axis`。

---

### Patch B：IrregularActionFormer 禁止 `metas=None` 绕过 axis contract

**判定：PASS。**

证据：

`opentad/models/detectors/irregular_actionformer.py` 的 `_assert_axis_contracts` 已从 `metas is None: return` 改成 `metas is None` 直接 raise；`_segments_to_axis` 和 `_segments_to_seconds` 对 selected-axis 路径都调用 strict conversion。commit diff 中可以看到 `_assert_axis_contracts`、`_segments_to_axis`、`_segments_to_seconds` 均被加严。([GitHub][3])

这解决了一个非常关键的问题：以前 `metas=None` 会让 detector 级 axis contract 完全失效；现在 train/test/postprocess 不能再用空 metadata 静默绕过。

---

### Patch C：HeadV2/V3 missing-center fallback 默认关闭，并阻止 dense-equivalent 误报

**判定：部分 PASS，scanner 仍有一个默认值盲点。**

证据：

`irregular_actionformer_head_v2.py` 的构造函数新增 `allow_center_fallback_inside_gt=False`，并保存为 `self.allow_center_fallback_inside_gt`；debug 计数也加入 missing / fallback 统计。([GitHub][4])

`_build_candidate_mask` 中只有在 `self.allow_center_fallback_inside_gt` 为真时，才会把 inside-GT fallback 合入候选 mask；默认关闭时，missing-center 不再自动补救。([GitHub][4])

`tools/check_fail_closed_config.py` 已经扫描 V2/V3 head，并在 dense-equivalent claim 与 soft assignment / center fallback 并存时给 violation。([GitHub][5])

**但 scanner 仍不够严：** 它用 `"soft_assign_topk" in obj` 或 `assignment_mode=="soft"` 判断 soft route。问题是 **HeadV2/V3 的实现本身就是 soft / weighted route**，如果某个 config 只写 `type="IrregularActionFormerHeadV2"` 但省略 `soft_assign_topk`，scanner 可能漏掉默认 soft 行为。这个地方应改成：只要 head type 是 V2/V3，就默认 `uses_soft_route=True`，除非显式标记某个未来 hard-compatible variant。

---

### Patch D：selected-axis GT remap drop 默认 fail-closed

**判定：基本 PASS，但有一个空 selected positions 边缘漏洞。**

证据：

`opentad/datasets/transforms/end_to_end.py` 新增 `allow_drop_selected_axis_gt=False`，并在 `_remap_gt_to_selected_axis` 中记录 input/keep/drop counts；当 remap 后发生 GT drop 且未显式允许时，raise `ValueError`。commit diff 里可以看到新增的 `selected_axis_gt_input_count / keep_count / drop_count / allow_drop_selected_axis_gt` metadata 和 fail-closed raise。([GitHub][3])

**边缘问题：** diff 显示早期 return 条件包含 `kept_positions.size == 0`。如果出现 “非空 GT + 空 selected positions”，当前实现可能直接返回空 GT，而不是先计数并 raise。正常采样下 selected positions 不应为空，但 fail-closed 代码不应依赖这种假设。这个需要补。

---

### Patch E：远端 runner/precheck/audit/train 入口加入 `check_fail_closed_config.py`

**判定：大体 PASS，但还应补一个 runner coverage test。**

证据：

例如 `remote_runs/run_gpu1_same_batch_audit_20260705.sh` 已经在运行 audit 前调用：

```bash
python tools/check_fail_closed_config.py "${CONFIGS[@]}" --json-out "$FAIL_CLOSED_JSON"
```

并把它标成 “fail-closed config scan”。([GitHub][6])

commit diff 也显示多个 `remote_runs/*.sh` 被修改，加入 preflight 逻辑。([GitHub][1])

**剩余问题：** 当前 scanner 是脚本入口 gate，不是 `tools/train.py` / config loader 的强制 runtime gate。也就是说，人仍然可以手工直接运行 train/eval 绕过 remote runner。建议补一个 pytest：遍历 `remote_runs/*.sh`，确认所有会调用 train/audit/eval 的脚本在此之前都调用了 `tools/check_fail_closed_config.py`。

---

## B. 仍可能存在的 fail-open 边缘路径

1. **utility 直接调用绕过 detector contract。**
   `selected_axis_to_dense_axis` 默认 `strict=False`，如果 eval/dump/analysis 脚本绕过 `IrregularActionFormer`，仍可能 silent pass。

2. **`source_axis="native"` 被调用方谎报。**
   `convert_to_seconds` 已经禁止 irregular selected metadata 下的 auto 推断，但如果外部错误地把 selected-axis segments 标成 native-axis，utility 本身无法完全识别。这必须靠 detector-level axis contract 和 metadata contract 拦截。

3. **prediction dump / cached result eval 路径。**
   scanner 屏蔽了很多 config 中的 cache/raw prediction/fuse pattern，但不能证明所有 ad-hoc analysis 脚本都不会读旧 prediction。需要 grep + CI test 覆盖 `tools/`、`remote_runs/`、`scripts/` 下所有 eval/dump 入口。

4. **GT remap 空 selected positions 边缘。**
   非空 GT + 空 selected positions 应该 raise，而不是返回空 GT。

5. **scanner 只能扫描 config object，不扫描运行时动态 mutation。**
   如果某 runner 在 Python 内部改 config，`check_fail_closed_config.py` 看不到。

---

## C. `check_fail_closed_config.py` 是否足够阻止误报 dense-equivalent

**结论：不完全足够。**

它已经能阻止三类明显误报：

* V2/V3 soft route 或 center fallback 冒充 dense-equivalent；
* legacy full-cell-span / openrange bridge 冒充 dense-equivalent；
* selected-axis GT drop route 未标 diagnostic 就通过。([GitHub][5])

但还有三个不足：

第一，V2/V3 soft route 的检测不应依赖 config 是否显式写了 `soft_assign_topk`。V2/V3 代码本身已经不是官方 dense ActionFormer 等价实现，因为它用 weighted / soft assignment、quality weighting、log1p regression 等机制；这与官方 hard shortest-GT assignment 和 linear stride-normalized target 不同。([GitHub][4])

第二，scanner 对 pipeline transform 的识别依赖 `type=="LoadFrames"`。如果未来换 wrapper 名字或 nested config，可能漏扫。

第三，它不能证明实验运行实际使用的 config 就是被扫描的 config。需要在 train/eval/audit runtime 里打印并保存 scanner JSON、git SHA、config hash。

---

## D. 当前 tests 是否覆盖真实行为

**现有 tests 有价值，但还不足以证明 integration 行为。**

你跑过的：

```text
tests/test_strict_fail_closed_contracts.py: 5 passed, 5 skipped
tests/test_adapter_native_dense_headv2_contracts.py: 57 passed, 21 skipped
```

这说明 unit/contract 层面已经有防线。但缺以下 Linux-only / integration tests：

1. **真实 DataLoader 样本测试**
   构造 non-empty GT + selected_positions 不覆盖 GT 的真实 `LoadFrames` pipeline，确认默认 raise，而不是只测 mock。

2. **detector forward_test postprocess 测试**
   输入 irregular selected-axis meta 缺失、`source_axis="auto"`、`metas=None`，确认从 `IrregularActionFormer.forward_test → post_processing → convert_to_seconds` 全链路 fail-closed。

3. **prediction dump / eval cache bypass 测试**
   对所有 `tools/*eval*`、`tools/*dump*`、`remote_runs/*.sh` 做 grep 或 AST 检查：禁止绕过 detector axis conversion 直接读 cached segments。

4. **remote runner coverage 测试**
   遍历 `remote_runs/*.sh`，凡是包含 train/audit/eval 命令的脚本，必须在前面包含 `tools/check_fail_closed_config.py`。

5. **same-batch assignment audit 真集成测试**
   必须保证 native/selected 不混 batch，不混 video，不混 window。这个不能只看字符串，需要实际 dump sample_id、video_name、window_start/end、selected_positions hash、gt hash。

---

## E. 当前 implementation 是否能声明 dense-equivalent

**不能。**

官方 OpenTAD 的 dense ActionFormer 语义很明确：

* `PointGenerator` 生成 point center、regression range、stride，格式为 `[point, min_reg_range, max_reg_range, stride]`。([GitHub][7])
* decode 是线性 target：`start = point - pred_left * stride`，`end = point + pred_right * stride`。([GitHub][8])
* assignment 是 hard center sampling + regression range mask + shortest-GT conflict resolution。([GitHub][8])
* regression target 是 left/right distance，并按 stride 线性归一化。([GitHub][8])

而当前 HeadV2/V3 soft/log1p/geometry route 仍然包含 soft assignment、weighted labels、log1p regression target、geometry-aware proposal refinement。这些设计可以作为新 sparse head，但不能叫 official dense-equivalent。([GitHub][4])

要声明 dense-equivalent，至少需要 Stage 2 证明：

* official dense selected-axis sanity 恢复 random-fixed 约 63；
* official dense selected-axis sanity 恢复 uniform 约 65；
* assignment audit 与官方 dense 在 same-batch 上一致；
* decode / seconds / NMS 输出坐标一致；
* 不使用 V2/V3 soft route、不使用 full-cell-span/openrange、不使用 center fallback、不 drop GT。

---

## F. 65/63 Average-mAP → 40/42 的最可能根因排序

我现在的排序如下：

### 1. 最高概率：HeadV2/V3 本身不是 official dense-equivalent

V2/V3 的 soft assignment、quality weighting、log1p target、geometry route 与官方 ActionFormer 的 hard shortest-GT + linear target 不同。这个差异足够导致 65/63 掉到 40/42，不能归因于 sparse selection 自然退化。

### 2. 高概率：axis / postprocess / seconds conversion 历史 fail-open

本 commit 修了 selected-axis postprocess strict 和 `metas=None` 绕过，说明之前确实存在 silent wrong-axis 的可能。若历史 40/42 来自 postprocess/NMS/seconds 轴错位，这是合理根因之一。

### 3. 高概率：selected-axis GT remap drop/collapse 改变监督合同

如果某些 GT 在 selected-axis remap 后被 drop，短动作和边界动作会被系统性削弱，尤其影响高 IoU。Patch D 正是针对这个问题。

### 4. 中高概率：legacy openrange / full-cell-span bridge 与 official-compatible 混淆

openrange/full-cell-span 不是官方 ActionFormer reg range / stride 合同。bridge hard official-compatible 必须和 legacy openrange/full-cell-span 分开解释。

### 5. 中概率：same-batch audit 未完成导致 native/selected 对照混批

如果 native/selected 不是同一 batch、同一 video、同一 window、同一 GT，assignment 差异无法解释。

### 6. 低优先级：真实 sparse information loss

只有当 Stage 2 official/dense selected-axis sanity 恢复 63/65，Stage 3 bridge hard official-compatible 也恢复或接近，才能讨论 sparse sampling 自然退化。现在还不能。

---

## G/H. 下一步实验路线、PASS/FAIL gate 与关键命令

### Stage 0：Linux preflight

目标：证明固定 SHA、fail-closed scanner、unit tests、runner gate 在 Linux 远端成立。

```bash
git checkout 578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67
git rev-parse HEAD

python -m py_compile \
  opentad/models/utils/post_processing/utils.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/dense_heads/irregular_actionformer_head_v2.py \
  opentad/models/dense_heads/irregular_actionformer_head_v3.py \
  tools/check_fail_closed_config.py \
  tools/audit_sparse_head_assignment.py

python tools/check_fail_closed_config.py configs/adatad/thumos/*.py \
  --json-out logs/stage0/fail_closed_config.json

pytest tests/test_strict_fail_closed_contracts.py \
       tests/test_adapter_native_dense_headv2_contracts.py -q

bash -n remote_runs/*.sh

python - <<'PY'
from pathlib import Path
bad = []
for p in Path("remote_runs").glob("*.sh"):
    s = p.read_text(errors="ignore")
    is_runner = any(x in s for x in ["tools/train.py", "tools/audit_", "test.py", "eval"])
    if is_runner and "tools/check_fail_closed_config.py" not in s:
        bad.append(str(p))
if bad:
    raise SystemExit("remote runners missing fail-closed gate:\n" + "\n".join(bad))
print("remote runner gate coverage ok")
PY
```

**PASS gate：**

* SHA 精确等于 `578fe0d...`;
* scanner `ok=true`;
* pytest 无新增 fail；
* 所有 runner bash syntax OK；
* 所有 train/audit/eval runner 都有 fail-closed gate。

**FAIL 含义：** 不进入 mAP。说明协议还没锁住，不允许继续声称 patch 已可远端闭环。

---

### Stage 1：same-batch assignment audit

目标：同一 batch 内比较 official dense / selected-axis / bridge，禁止 native/selected 混 batch。

建议直接跑已有 runner：

```bash
bash remote_runs/run_gpu1_same_batch_audit_20260705.sh
```

或直接跑 audit：

```bash
python tools/audit_sparse_head_assignment.py \
  --official-config <OFFICIAL_DENSE_CONFIG> \
  --selected-config <SELECTED_AXIS_CONFIG> \
  --bridge-config <BRIDGE_HARD_OFFICIAL_COMPAT_CONFIG> \
  --same-batch \
  --max-batches 8 \
  --out logs/stage1/same_batch_assignment_audit.json
```

**必须 dump 字段：**

```text
git_sha
config_hash
video_name
sample_id
window_start
window_end
gt_segments_hash
selected_positions_hash
point_centers
regression_ranges
center_sampling_mask_count
inside_gt_mask_count
assigned_positive_count
shortest_gt_index
reg_targets_left_right
missing_center_count
center_fallback_count
selected_axis_gt_drop_count
```

**PASS gate：**

* native/selected 同 video、同 window、同 GT hash；
* `center_fallback_count == 0`;
* `selected_axis_gt_drop_count == 0`;
* official-compatible route 的 point / range / assignment / target 与官方 dense 语义一致；
* 不出现 full-cell-span / openrange 被标 dense-equivalent。

**FAIL 解释：**

* point center 不一致：PointGenerator / selected-axis coordinate bug；
* range 不一致：regression range contract bug；
* positive assignment 不一致：center sampling / shortest-GT / GT remap bug；
* target 不一致：linear target / stride normalization bug；
* fallback/drop 非零：不能继续 dense-equivalent claim。

---

### Stage 2：official/dense selected-axis sanity

目标：先恢复 known-good 数字，而不是继续调 HeadV2/V3。

```bash
bash remote_runs/precheck_official_dense_selected_axis_sanity_20260706.sh
bash remote_runs/run_gpu1_uniform_fixed_dense_control_long_20260706.sh
```

如果没有现成 random-fixed runner，补：

```bash
python tools/train.py configs/adatad/thumos/<official_dense_random_fixed_50pct_selected_axis_sanity>.py \
  --seed 0 \
  --work-dir work_dirs/stage2_random_fixed_selected_axis_sanity

python tools/train.py configs/adatad/thumos/<official_dense_uniform_50pct_selected_axis_sanity>.py \
  --seed 0 \
  --work-dir work_dirs/stage2_uniform_selected_axis_sanity
```

**PASS gate：**

* random-fixed 50% selected-axis sanity：Average-mAP 约 63，允许小范围波动；
* uniform 50% selected-axis sanity：Average-mAP 约 65，允许小范围波动；
* GT drop = 0；
* missing center fallback = 0；
* postprocess axis contract 无 warning / no auto。

**FAIL 解释：**

* 若 random/uniform 都低：official dense selected-axis adapter 仍坏；
* 若 uniform 恢复、random 不恢复：selection ledger / random protocol 有问题；
* 若 train 高 eval 低：seconds conversion / NMS / eval coordinate 有问题。

---

### Stage 3：bridge selected-axis 与 native-axis controls

目标：隔离 bridge hard official-compatible 是否正确，不混 legacy openrange。

```bash
bash remote_runs/run_gpu1_bridge_absrange_expanded_shortgate_failclosed_20260706.sh
bash remote_runs/run_gpu1_bridge_absrange_expanded_long_20260706.sh
bash remote_runs/run_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh
```

**PASS gate：**

* bridge hard official-compatible selected-axis 接近 Stage 2 selected-axis sanity；
* native-axis control 与 selected-axis control 差异可由实际采样解释；
* legacy full-cell-span/openrange 只作为 diagnostic，不进入 dense-equivalent 表述。

**FAIL 解释：**

* Stage 2 恢复、Stage 3 不恢复：bridge assignment/decode/axis 有 bug；
* native ok、selected bad：selected-axis remap/conversion 有 bug；
* native/selected 都 bad：bridge head contract 或 point/range/target 有 bug。

---

### Stage 4：HeadV2/V3 soft/log1p/geometry 分解消融

目标：证明 HeadV2/V3 的每个非官方设计到底贡献还是破坏。

建议矩阵：

| Variant | Assignment    | Target        | Geometry    | Fallback | Claim                      |
| ------- | ------------- | ------------- | ----------- | -------- | -------------------------- |
| A       | official hard | linear stride | off         | off      | dense-equivalent candidate |
| B       | soft top-k    | linear stride | off         | off      | sparse-head ablation       |
| C       | official hard | log1p         | off         | off      | target ablation            |
| D       | official hard | linear stride | geometry on | off      | geometry ablation          |
| E       | soft top-k    | log1p         | geometry on | off      | V2/V3 route                |
| F       | any           | any           | any         | on       | legacy/diagnostic only     |

**PASS gate：**

* 每个 delta 有单独解释；
* V2/V3 不再和 official dense-equivalent 混称；
* 若 E 仍 40/42，而 A 恢复 63/65，则根因是 V2/V3 design，不是 sparse selection。

---

## I. 必要 patch 级修复建议

### 1. 修 `check_fail_closed_config.py` 的 V2/V3 soft 默认盲点

```python
# tools/check_fail_closed_config.py

def _scan_sparse_head_contract(obj, path, violations):
    head_type = str(obj.get("type", ""))
    is_v2_v3 = head_type in {
        "IrregularActionFormerHeadV2",
        "IrregularActionFormerHeadV3",
    }

    if not is_v2_v3:
        return

    dense_claim = bool(obj.get("dense_equivalent_claim_allowed", False))
    route_contract = obj.get("route_contract", {}) or {}
    dense_claim = dense_claim or bool(route_contract.get("dense_equivalent_claim_allowed", False))

    # V2/V3 implementation is soft/geometry route by design unless a future hard-compatible
    # variant explicitly opts out with a separately audited flag.
    hard_compatible_variant = bool(obj.get("official_hard_assignment_compatible", False))
    uses_soft_route = not hard_compatible_variant

    uses_center_fallback = bool(obj.get("allow_center_fallback_inside_gt", False))

    if dense_claim and (uses_soft_route or uses_center_fallback):
        violations.append({
            "path": path,
            "reason": (
                "dense-equivalent claim is forbidden for HeadV2/V3 unless "
                "official_hard_assignment_compatible=True and independently audited; "
                "center fallback must also be disabled"
            ),
        })
```

---

### 2. 修 selected-axis GT remap 空 selected positions fail-open

```python
# opentad/datasets/transforms/end_to_end.py

def _remap_gt_to_selected_axis(self, gt_segments, gt_labels, kept_positions, meta):
    self._clear_selected_axis_gt_drop_state()

    input_count = 0 if gt_segments is None else int(len(gt_segments))
    self._last_selected_axis_gt_input_count = input_count

    if gt_segments is None or gt_labels is None or input_count == 0:
        self._last_selected_axis_gt_keep_count = 0
        self._last_selected_axis_gt_drop_count = 0
        return gt_segments, gt_labels

    if kept_positions is None or len(kept_positions) == 0:
        self._last_selected_axis_gt_keep_count = 0
        self._last_selected_axis_gt_drop_count = input_count
        if not self.allow_drop_selected_axis_gt:
            video = getattr(self, "_current_video_name", "<unknown>")
            raise ValueError(
                f"selected-axis GT remap would drop all {input_count} GT segments "
                f"for video={video}; set allow_drop_selected_axis_gt=True only for "
                "legacy/diagnostic routes"
            )
        return gt_segments[:0], gt_labels[:0]

    # continue with existing remap logic...
```

---

### 3. 禁止 sparse route 中不带 strict 的 selected-axis conversion

```python
# tests/test_no_non_strict_selected_axis_conversion.py

from pathlib import Path
import re

BAD = re.compile(r"selected_axis_to_dense_axis\([^)]*\)(?!.*strict\s*=\s*True)")

def test_no_non_strict_selected_axis_conversion_in_sparse_paths():
    roots = [
        Path("opentad/models"),
        Path("opentad/datasets"),
        Path("tools"),
    ]
    offenders = []
    for root in roots:
        for p in root.rglob("*.py"):
            text = p.read_text(errors="ignore")
            if "selected_axis_to_dense_axis" not in text:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "def selected_axis_to_dense_axis" in line:
                    continue
                if "selected_axis_to_dense_axis(" in line and "strict=True" not in line:
                    offenders.append(f"{p}:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
```

---

### 4. 给 train/eval runtime 加 scanner，而不只依赖 remote runner

```python
# tools/train.py or common config entrypoint

def _run_fail_closed_config_gate(cfg, config_path=None):
    from tools.check_fail_closed_config import scan_config_object

    violations = scan_config_object(cfg)
    if violations:
        details = "\n".join(
            f"{v.get('path', '<unknown>')}: {v.get('reason', v)}"
            for v in violations
        )
        raise RuntimeError(
            "Fail-closed config gate rejected this run.\n"
            f"config={config_path}\n{details}"
        )
```

这一步不是为了替代 remote runner，而是防止手工直接运行 `tools/train.py` 绕过协议。

---

## 最终判定

**单一 verdict：WARN。**

* **代码硬化层面：WARN→接近 PASS。** Patch A/B/D/E 基本落地，Patch C 默认 fallback 修对了，但 scanner 对 V2/V3 soft 默认值还有盲点。
* **dense-equivalent 声明：FAIL。** 当前 HeadV2/V3 不能叫 dense-equivalent；必须等 Stage 1/2/3 证明。
* **65/63 → 40/42 根因证明：FAIL。** 当前只能给根因排序，不能给最终归因。
* **下一步是否继续：继续，但只能跑 Stage 0–4，禁止写 paper claim / dense-equivalent claim / sparse 自然退化 claim。**

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67 "Harden sparse route fail-closed contracts · yuzbo/OpenTAD_SparseHeadClean_20260702@578fe0d · GitHub"
[2]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67/opentad/models/utils/post_processing/utils.py "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/post_processing/utils.py at 578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[3]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67.patch "github.com"
[4]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67/opentad/models/dense_heads/irregular_actionformer_head_v2.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_head_v2.py at 578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67/tools/check_fail_closed_config.py "OpenTAD_SparseHeadClean_20260702/tools/check_fail_closed_config.py at 578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67/remote_runs/run_gpu1_same_batch_audit_20260705.sh "OpenTAD_SparseHeadClean_20260702/remote_runs/run_gpu1_same_batch_audit_20260705.sh at 578fe0dd827fd7fb3d2aabcfb5b9c922c4dfcd67 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[8]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
