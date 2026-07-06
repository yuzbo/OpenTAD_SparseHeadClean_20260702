## 0. 总判决

**总判决：FAIL，不能开始长训；可以同步到远端跑受限 preflight / audit。**

我能访问目标仓库、目标分支和固定提交。目标分支最新提交为 `114f72c Make early bridge configs state scale contract`，提交页显示只改了 9 个文件，主要把早期 bridge configs 的 scale contract 显式化。官方 OpenTAD 仓库也可访问，官方 ActionFormer 在 THUMOS14 I3D 配置下给出的参考 Average-mAP 是 **68.44**，这说明 dense ActionFormer/AdaTAD 语义必须作为硬参考，而不是相信当前改造仓库内复制的 dense 代码。([GitHub][1])

我的结论很直接：**当前 65 → 40/42 的崩溃最可能不是“50% sparse sampling 自然退化”，而是 sparse/irregular head 的 assignment、range/scale、encode/decode、score calibration 与 official dense ActionFormer 语义偏离叠加造成的实现级崩溃。** 最近的 BridgeHead scale contract 修复方向是对的，但还不够证明 bridge/head 路线正确，更不能直接放长训。

---

## 1. 官方 dense ActionFormer 的不可变合同

官方 OpenTAD 的 `PointGenerator` 产生的每个 point 是：

```text
[center, regression_range_min, regression_range_max, stride]
```

其中 `center = arange(T) * stride`，如果 `use_offset=True` 再加 `0.5 * stride`；`regression_range` 是配置里的原始区间，**不是再乘 cell span / point span**。([GitHub][2])

官方 `AnchorFreeHead.prepare_targets` 的关键语义是：

| 子系统                 | 官方语义                                                            |
| ------------------- | --------------------------------------------------------------- |
| center sampling     | 半径 = `stride * center_sample_radius`                            |
| range gate          | 用 raw left/right distance 的 max 与 `regression_range_min/max` 比较 |
| conflict resolution | 多个 GT 命中时选 **duration 最短的 GT**                                  |
| regression target   | `[left, right] / stride`                                        |
| decode              | `point_center ± pred * stride`                                  |
| loss normalizer     | hard positive count 的 EMA，而不是 soft mass                         |

这些不是风格差异，而是 ActionFormer dense baseline 能稳定工作的监督坐标合同。([GitHub][3])

---

## 2. 最近 5 项修复是否完整

| 检查项                                 |             结论 | 说明                                                                                                                                                                                                                           |
| ----------------------------------- | -------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BridgeHead scale contract           |    **部分 PASS** | `center_radius_scale="point_radius"`、`reg_denom_mode="left_right_mean"`、`allow_legacy_full_cell_span=False` 已成为 BridgeHead 默认，并且 `full_cell_span` 需要显式 legacy opt-in 才能用。这个修复是正确方向。([GitHub][4])                             |
| legacy / corrected configs 区分       |       **WARN** | 最新提交确实把若干 early bridge configs 显式设为 official-compatible，并新增测试检查这些字段。但这只证明被覆盖的 configs 显式化了，不证明全仓库所有 bridge configs 都不会静默依赖默认语义。([GitHub][5])                                                                                 |
| Bridge dense-equivalence verifier   |       **WARN** | `generated_v2_levelstride_equivalence` 确实覆盖了真实 `IrregularPointGeneratorV2` 生成点，并强制 `range_mode="absolute"`、decode/radius scale 为 `level_stride`，这比之前强很多。但它仍是 B=1、小合成、多数 mask/concat/postprocess 边界缺失的 verifier。([GitHub][6]) |
| 坐标轴合同                               |       **WARN** | `IrregularActionFormer.post_processing` 看起来已在 NMS 前做 selected-axis → native-axis，再转 seconds，这是正确方向。但底层 `convert_to_seconds` 仍会根据 meta 自动做 selected→dense，若调用者已转 native 但 meta 没更新，就有二次转换风险。([GitHub][7])                     |
| eval/test 泄漏防护                      |    **部分 PASS** | `LoadFrames.__call__` 已 fail-closed 禁止 validation/test split 上一组 diagnostic GT/cache/raw prediction shortcut flag，这是真修复。但它只覆盖 transform 实例上的若干 flag，不等价于全 config、inference、postprocess、tooling 级别的全局泄漏扫描。([GitHub][8])       |
| official dense selected-axis sanity | **FAIL / 未证明** | 仓库 README 和提交历史显示已有 dense-control / native dense headv2 / selected-axis sanity 相关入口，但我没有看到已经复现“等间隔 50% sparse selected-axis Avg-mAP≈65”的训练证据。没有这个，不能判断 HeadV3/bridge 崩溃是不是 head 本身。([GitHub][9])                             |

---

## 3. 当前代码是否可以同步 / preflight / 长训

### 可以同步到远端并跑 preflight 吗？

**可以，但只允许跑 Stage 0–2 的 preflight / audit / selected-axis sanity。**

允许同步的理由是：最新 commit 的 scale contract 修复方向明确，目标仓库已经有 verifier、assignment audit、diagnostic 工具和若干 contract tests；README 也明确当前 route 是 sparse/irregular 检测路线，并强调 validation/test 不能使用 GT/teacher/cache/raw prediction shortcut。([GitHub][9])

### 可以开始长训吗？

**不可以。阻断项如下：**

1. **official dense selected-axis 65 mAP sanity 尚未被当前 repo 证明。**
2. **V2/V3 的 assignment 与 decode 仍不是 official dense ActionFormer。**
3. **Bridge hard path 虽已修 scale，但 verifier 不覆盖 batch、mask、多层真实 concat、postprocess、seconds conversion、loss normalizer。**
4. **BridgeHead 仍有 center-sampling miss 后 fallback 到 inside-GT 的非官方逻辑。**
5. **`convert_to_seconds` 的 source-axis 不是显式参数，仍有工具链二次转换风险。**
6. **eval/test guard 是 transform-level，不是 whole-config fail-closed。**

---

## 4. 当前实现 vs 官方 dense ActionFormer 逐项差异表

| 模块                       | 官方 OpenTAD dense ActionFormer        | 当前 sparse/irregular 实现                                                                                                  | 风险                                   |
| ------------------------ | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Point layout             | `[center, reg_min, reg_max, stride]` | V2 irregular points 是 `[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]`；BridgeHead 做翻译 | 字段更丰富但也更容易 scale 语义错配                |
| regression range         | 配置原始 range，不乘 stride/cell            | `IrregularPointGenerator` legacy 会用 point/cell scale；V2 有 `absolute/open/cell_span/level_stride` 等模式                    | **高风险**：range gate 层级分配可能偏离 official |
| center sampling          | radius = stride × 1.5                | Bridge default 现在可等价；V2/V3/soft route 不是 hard official                                                                  | 高风险                                  |
| missing center candidate | official 不把整个 GT 内部强行 fallback 为候选   | BridgeHead 里有 `missing_gt` fallback 到 `inside_gt_seg`                                                                   | **高风险**：会改变 assignment 分布            |
| GT conflict              | shortest-GT                          | Bridge hard 试图复刻；V2/V3 soft assignment/top-k/权重混合                                                                       | 高风险                                  |
| regression encode        | `[left,right] / stride`              | Bridge symmetric_linear 可匹配；V2/V3 使用 log/expm1 族 decode                                                                 | **高风险**                              |
| regression decode        | `center ± pred * stride`             | Bridge official-compatible 可匹配；V2/V3 `expm1(reg_pred) * scale`                                                          | 高风险                                  |
| loss normalizer          | hard positive count EMA              | V2/V3 用 soft mass / reg_weight 参与                                                                                       | score calibration 风险                 |
| postprocess axis         | official dense 没 selected/native 轴问题 | 当前 detector 尝试 NMS 前 selected→native，再 seconds                                                                          | 方向正确，但工具调用仍有 double-conversion 风险    |
| projection/neck          | 官方 uniform temporal feature route    | 当前 temporal_grid 传入 projection/head，但 irregular geometry 是否被 backbone/neck 真正建模未证明                                      | 中风险                                  |
| validation/test shortcut | 官方 config 默认不从 raw predictions 加载    | 当前 LoadFrames guard 覆盖一批 diagnostic flags                                                                               | 部分安全，但需 whole-config scan            |

---

## 5. 逐文件 / 逐函数 bug 与 risk 清单

### 5.1 `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

**PASS：`__init__` scale contract 修复方向正确。**
默认值已是：

```python
center_radius_scale="point_radius"
reg_denom_mode="left_right_mean"
allow_legacy_full_cell_span=False
```

并且如果使用 `full_cell_span` 而未显式 `allow_legacy_full_cell_span=True`，会直接 `ValueError`。这是必要修复。([GitHub][4])

**FAIL 风险：`_build_candidate_mask` 的 missing-GT fallback 非官方。**
请定位：

```python
missing_gt = ~candidate_mask.any(dim=0)
if missing_gt.any():
    candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
```

这会在 center sampling 没有候选时，退化为 “GT 内所有点都可候选”。官方 ActionFormer 没有这个 fallback；官方是 center mask、inside mask、regression range mask、shortest-GT 共同决定。这个 fallback 会改变短动作、边界附近、稀疏点少时的正样本分布，是 65→40/42 的高优先级嫌疑。([GitHub][4])

**WARN：`_build_assignment_weights` 的 soft assignment 不能宣称 dense-equivalent。**
该函数引入 center/scale cost、top-k、soft weights，与 official hard shortest-GT assignment 是不同算法。它可以作为研究 head，但不能作为“修复版 official sparse dense baseline”。([GitHub][4])

---

### 5.2 `opentad/models/dense_heads/irregular_actionformer_head_v2.py`

**FAIL：`prepare_targets` 默认不是 official dense ActionFormer。**
V2 直接构造 soft cls targets、soft assignment weights、reg_weight，并且 `use_regress_range=False` 默认会放大 all-level flooding 风险。官方是 hard candidate mask + shortest GT + raw range gate。([GitHub][10])

**FAIL：`get_refined_proposals` decode 不是官方 linear stride decode。**
V2 decode 使用 `expm1(reg_pred.clamp_min(0)) * left/right denom`，而官方是 `pred * stride`。这不是小差异；它会改变回归尺度、梯度、proposal length 分布和高 IoU 定位。([GitHub][10])

---

### 5.3 `opentad/models/dense_heads/irregular_actionformer_head_v3.py`

**FAIL：V3 继承 V2 的核心非官方 assignment/decode。**
V3 增加 geometry encoder/modulation 和可选 boundary aux，但默认 forward_test 仍走继承的 proposal decode 路径；loss 也继承 soft mass / reg_weight 逻辑。它不是 official-compatible dense sanity head。([GitHub][11])

**研究判断：**
V3 可以做 irregular research head，但不能在 official dense selected-axis sanity 未过之前解释为“sparse sampling 自然下降”。

---

### 5.4 `opentad/models/dense_heads/prior_generator/irregular_point_generator.py`

**FAIL 风险：legacy `IrregularPointGenerator` 不是 official point 语义。**
legacy generator 用 `point_scale` / cell span 进入 point 字段与 range，天然偏离 official `[center, reg_min, reg_max, stride]`。([GitHub][12])

**WARN：`IrregularPointGeneratorV2` 只有特定模式才 official-compatible。**
当前 V2 支持 `range_mode`、`decode_scale_mode`、`radius_scale_mode`。要逼近官方 dense，必须是：

```python
range_mode="absolute"
decode_scale_mode="level_stride"
radius_scale_mode="level_stride"
```

如果用 `cell`、`geometric_mean`、`cell_span`、`level_stride range scaling` 等，就不是 official dense。([GitHub][12])

---

### 5.5 `opentad/models/detectors/irregular_actionformer.py`

**部分 PASS：post-processing 方向基本正确。**
当前 detector 在 NMS 前把 proposal 从 selected axis 转到 native axis，然后再做 seconds conversion。对于 irregular sampling，这是正确方向，因为 NMS 应该在真实时间轴上做，而不是在压缩 selected-index 轴上做。([GitHub][7])

**WARN：single-class branch 可能绕过常规 multiclass pre-NMS 过滤。**
如果 diagnostic 或单类实验使用单类 scores，应确认和 multiclass branch 一样执行 threshold / top-k / NMS 前筛选，否则容易出现 duplicate proposal flooding。THUMOS 20 类主路线未必直接触发，但这是后续诊断隐患。

**WARN：projection/neck 对 irregular temporal geometry 的建模未证明。**
即使 head 读 temporal_grid，如果 projection/neck 的 conv/stride 语义仍隐含 uniform grid，irregular bridge 也可能只是在 head 端补坐标，而 feature aggregation 端已经错位。

---

### 5.6 `opentad/models/utils/temporal_grid.py`

**WARN：`downsample_temporal_grid` 的中心定义需要 dense-equivalence 测试。**
它通过 interval union/merge 得到下采样中心。这个设计对 irregular 合理，但对 uniform dense 等价性必须证明：各 FPN level 的 center、left/right cell、mask 是否与 official dense level stride 完全一致。当前 verifier 主要看 head points，不足以覆盖 projection/neck 下采样后的真实 grid 行为。

---

### 5.7 `opentad/models/utils/post_processing/utils.py`

**部分 PASS：`selected_axis_to_dense_axis` 的 sentinel 插值思路正确。**
它把 selected-axis coordinate 线性插到 dense/native coordinate，末尾加 sentinel，适合处理非整数 proposal 边界。([GitHub][13])

**WARN：`convert_to_seconds` 的 source axis 是隐式的。**
它会根据 `meta["irregular_native_axis"]` 决定是否 selected→dense。问题是：如果 detector 已经转成 native，但 meta 仍写着 non-native，直接调用 utility 会二次转换。建议把 `source_axis` 改成显式参数。

---

### 5.8 `opentad/datasets/transforms/end_to_end.py`

**部分 PASS：validation/test diagnostic shortcut guard 是真修复。**
`LoadFrames.__call__` 会调用 `_assert_no_eval_diagnostic_shortcuts`，该函数在 eval split 检测到 diagnostic GT/cache/raw prediction shortcut flags 时会 `ValueError`。([GitHub][8])

**WARN：还需要 whole-config preflight。**
这个 guard 不等于扫描全部 config。仍需检查：

```text
inference.load_from_raw_predictions
post_processing.load_predictions
teacher/cache/prediction_cache/raw_prediction 字段
tool-level GT fallback
dataset ann_file / split / subset 错配
```

---

### 5.9 `tools/verify_bridge_dense_equivalence.py`

**部分 PASS：比之前强，确实覆盖 real `IrregularPointGeneratorV2` generated points。**
`generated_v2_levelstride_equivalence` 使用真实 V2 generator，并设置 official-compatible 的 absolute range、level_stride decode/radius scale。([GitHub][6])

**WARN：覆盖边界仍不够。**
当前 verifier 的主要缺口：

```text
B=1 only
没有真实 dataloader batch
没有 mask / fresh_mask / padding
没有 full multi-level concat 后的真实 loss normalizer 对比
没有 NMS / seconds conversion
没有 missing_gt fallback 反例
通过 object.__new__ 构造 BridgeHead，绕过 __init__ 全初始化路径
不覆盖 V2/V3
```

它可以作为 Stage 0 gate，但不能作为 long training gate。

---

### 5.10 `tools/audit_sparse_head_assignment.py`

**WARN：这是有用工具，但还不是 official same-batch diff。**
它能加载 config、构造 loader、拿 head/points、调用 `prepare_targets`，并计算 assignment、decode、axis 相关诊断。([GitHub][14])

阻断缺口是：它还需要内置 official `AnchorFreeHead` target builder，对同一 batch 输出：

```text
official positive mask
official assigned class
official encoded l/r target
official decoded target IoU
bridge/v2/v3 对应结果
exact diff / tolerance diff
```

没有这个，Stage 1 不能算过。

---

### 5.11 `tools/analyze_detection_quality.py`

**WARN：可用于 post-training 诊断，不可混入 mAP 结论。**
工具支持从 config 或 fallback path 找 GT，并加载 detection result 做 IoU/quality 诊断。这个适合 Stage 5，但必须把 fallback GT 标记为 diagnostic，不允许变成训练或正式 eval 的隐式依赖。([GitHub][15])

---

### 5.12 `tests/test_adapter_native_dense_headv2_contracts.py`

**部分 PASS：新增 config scale contract test 有意义。**
最新 commit 新增测试会检查 early bridge configs 使用 `IrregularActionFormerBridgeHead` 且显式设置：

```python
center_radius_scale == "point_radius"
reg_denom_mode == "left_right_mean"
allow_legacy_full_cell_span is False
```

这能防止早期 bridge config 静默回到 legacy full-cell semantics。([GitHub][5])

**WARN：这只是 config test，不是 numeric dense equivalence test。**

---

### 5.13 `root-cause-notes.md`

**WARN：notes 不是证据。**
该文件已经记录“上一轮 strict review 阻止长训，并要求强化 verifier、axis tests、eval guard、official dense selected-axis performance”。最新 commit 把 scale contract 修复状态写入 notes，这是好记录，但不能替代 Stage 1/2 实验证据。([GitHub][5])

---

## 6. 65 → 40/42 崩溃根因排序

### Rank 1：assignment 偏离 official dense ActionFormer

**证据强度：强。**

V2/V3 使用 soft assignment、top-k、weighted cls/reg targets、soft positive mass；official dense 是 hard center sampling + range gate + shortest-GT。Bridge hard 路线也仍有 missing-GT inside fallback。这个级别的监督定义变化完全足以把高 IoU mAP 拉崩。([GitHub][10])

### Rank 2：regression range / point scale / encode-decode 标尺错误

**证据强度：强。**

official 的 range 是 raw config，reg target 除以 stride；当前 irregular generator 和 V2/V3 引入 cell span、decode_left/right、radius_scale、log/expm1。最新 BridgeHead 修复只覆盖 bridge official-compatible 路线，不覆盖 V2/V3 默认路线。([GitHub][2])

### Rank 3：official dense selected-axis sanity 未闭环

**证据强度：中强。**

如果当前 repo 不能在同一 selected-axis / 同一 remap / 同一 postprocess 下复现等间隔 50% 的约 65 Avg-mAP，那么任何 HeadV3/bridge 的 40/42 都不能归因于 head。这个 sanity 是所有 sparse-head 研究结论的前置条件。

### Rank 4：score calibration / duplicate proposals / all-level flooding

**证据强度：中。**

soft label、soft positive mass、open range、log decode、single-class branch 筛选差异都会影响 proposal score 与 duplicate 分布。40/42 这种水平很可能含有 recall 与 precision 双重问题，而不是单纯定位误差。

### Rank 5：projection/neck/backbone 没有真正建模 irregular temporal geometry

**证据强度：中。**

当前 head 端知道 temporal_grid，不代表 projection/neck 的时序卷积已经正确处理非均匀时间。尤其多层下采样后的 center/cell 定义需要证明。这个更像后续 60→65 的瓶颈，不是我目前排第一的 65→40 根因。

### Rank 6：post-processing 坐标 / NMS 错误

**证据强度：中弱。**

detector 级 postprocess 方向看起来已经正确：NMS 前 selected→native，再 seconds。风险在 utility 直接调用、meta stale、二次转换、diagnostic tool 使用不一致。它可能造成高 IoU 崩，但目前证据弱于 assignment/scale。([GitHub][7])

### Rank 7：eval/test 泄漏或数据 pipeline shortcut

**证据强度：弱到中。**

LoadFrames guard 是真实修复；除非 config 还有别的 raw prediction/cache 路径，否则它更像合规风险，不像 40/42 主因。([GitHub][8])

---

## 7. 分阶段实验路线与 gate

### Stage 0：Linux preflight

**目标：** 确认代码、配置、guard、verifier 基础可运行。

建议命令：

```bash
python -m py_compile \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/irregular_actionformer_head_v2.py \
  opentad/models/dense_heads/irregular_actionformer_head_v3.py \
  opentad/models/dense_heads/prior_generator/irregular_point_generator.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/utils/temporal_grid.py \
  opentad/models/utils/post_processing/utils.py \
  opentad/datasets/transforms/end_to_end.py \
  tools/verify_bridge_dense_equivalence.py \
  tools/audit_sparse_head_assignment.py

pytest -q tests/test_adapter_native_dense_headv2_contracts.py

python tools/verify_bridge_dense_equivalence.py --json outputs/bridge_dense_equivalence.json
```

**PASS gate：**

```text
py_compile 全过
pytest 全过
verify_bridge_dense_equivalence 三个 case 全过
max_abs_target_error <= 1e-6
max_abs_decode_error <= 1e-6
无 eval/test diagnostic/cache/raw-prediction enabled config
```

**FAIL 后转向：** implementation hygiene / config hygiene，不准进入 Stage 1。

---

### Stage 1：same-batch assignment audit

**目标：** 在同一 batch、同一 GT、同一 points 上比较 official dense target 与 bridge hard target。

必须先增强 `tools/audit_sparse_head_assignment.py`，加入 official target builder。

**PASS gate：**

```text
bridge hard official-compatible:
positive mask 与 official 完全一致
assigned class 与 official 完全一致
encoded l/r target max_abs_diff <= 1e-6
decoded target IoU ≈ 1.0
per-level positive count 一致
GT coverage 一致
```

**FAIL 后转向：**

| 失败现象                 | 假设                                               |
| -------------------- | ------------------------------------------------ |
| positive mask 不一致    | center radius / range gate / missing_gt fallback |
| class assignment 不一致 | shortest-GT conflict resolution                  |
| encoded target 不一致   | reg_denom / point scale                          |
| decoded IoU 不一致      | decode scale / axis                              |

---

### Stage 2：official dense selected-axis sanity

**目标：** 先恢复等间隔 50% selected-axis dense baseline 的约 65 Avg-mAP。

**必须满足：**

```text
同一 THUMOS14 split
同一 selected positions
同一 selected-axis GT remap
同一 native-axis postprocess/NMS/seconds conversion
无 diagnostic/oracle/cache shortcut
```

**PASS gate：**

```text
Avg-mAP 接近既有等间隔 50% baseline
建议阈值：>= 63.5 或与历史 65.0/65.5 差距 <= 1.5~2.0 mAP
mAP@0.6/@0.7 不出现异常断崖
```

**FAIL 后转向：** 不再审 V2/V3；先查 data pipeline / axis conversion / postprocess / AdaTAD loader contract。

---

### Stage 3：bridge hard official-compatible shortgate

**目标：** 只测 bridge hard 是否在 official-compatible 语义下不崩。

配置必须硬锁：

```python
assignment_mode="hard"
regression_mode="symmetric_linear"
center_radius_scale="point_radius"
reg_denom_mode="left_right_mean"
allow_legacy_full_cell_span=False

prior_generator=dict(
    type="IrregularPointGeneratorV2",
    range_mode="absolute",
    decode_scale_mode="level_stride",
    radius_scale_mode="level_stride",
)
```

并且关闭 missing-GT fallback。

**PASS gate：**

```text
Stage 1 exact assignment audit 通过
短训 loss / pos count / proposal length 分布与 official dense sanity 接近
短 gate validation 不出现 40/42 级崩溃趋势
```

**FAIL 后转向：**

```text
如果 Stage 1 过但 Stage 3 崩：查 forward feature route、loss normalizer、postprocess、score calibration
如果 Stage 1 不过：回到 assignment/scale
```

---

### Stage 4：paired long training

**目标：** 成对长训，不做孤立实验。

最小 paired matrix：

```text
A: official dense selected-axis
B: bridge hard official-compatible
C: bridge soft only after B 通过
D: HeadV2 / HeadV3 only after B 通过
```

**PASS gate：**

```text
B 与 A 差距 <= 1~2 mAP
B 不出现 high-IoU 异常崩
C/D 若下降，能由 diagnostics 解释
```

**FAIL 后转向：**

| 失败                        | 解释方向                                           |
| ------------------------- | ---------------------------------------------- |
| A 低                       | pipeline/axis/data                             |
| A 高 B 低                   | bridge implementation                          |
| B 高 C/D 低                 | soft assignment / geometry head 设计问题           |
| 全部高但 irregular 真 sparse 低 | sparse evidence / feature loss / budget policy |

---

### Stage 5：post-training high-IoU localization diagnostics

**目标：** 解释 mAP@0.6 / mAP@0.7。

必须输出：

```text
per-tIoU AP curve
proposal recall by top-k
per-GT best IoU
boundary absolute error
proposal length distribution
duplicate proposal count after NMS
score calibration curve
per-level positive / prediction contribution
selected-axis vs native-axis segment diff
```

**PASS gate：**

```text
高 IoU recall 不塌
proposal length 不系统性过长/过短
NMS 前重复不过量
score 与 IoU 正相关
selected→native→seconds roundtrip 误差可控
```

**FAIL 后转向：**

| 失败现象            | 假设                                |
| --------------- | --------------------------------- |
| recall 低        | assignment/range                  |
| IoU 低但 recall 有 | decode/axis                       |
| duplicate 高     | NMS/score calibration             |
| 长度偏移            | reg scale                         |
| 高 IoU 单独塌       | boundary/projection/temporal grid |

---

### Stage 6：final route decision

**决策规则：**

```text
如果 Stage 2 失败：停止 sparse-head 讨论，修 dense selected-axis pipeline。
如果 Stage 2 过、Stage 3 失败：修 bridge official-compatible 实现。
如果 Stage 3 过、V2/V3 失败：V2/V3 的 soft/geometry 设计是主因。
如果 Bridge 与 V3 都恢复 65 附近，再讨论 irregular sparse-head 是否有研究价值。
```

---

## 8. 建议的关键 patch

### Patch 1：禁用 BridgeHead 非官方 missing-GT fallback

```python
# irregular_actionformer_bridge_head.py

class IrregularActionFormerBridgeHead(...):
    def __init__(
        self,
        ...,
        allow_center_fallback_inside_gt: bool = False,
        ...
    ):
        ...
        self.allow_center_fallback_inside_gt = bool(allow_center_fallback_inside_gt)
```

```python
def _build_candidate_mask(...):
    inside_gt_seg = torch.logical_and(
        reg_targets[..., 0] > 0,
        reg_targets[..., 1] > 0,
    )

    center_mask = center_seg.min(dim=-1).values > 0
    candidate_mask = inside_gt_seg & center_mask

    if self.allow_center_fallback_inside_gt:
        missing_gt = ~candidate_mask.any(dim=0)
        if missing_gt.any():
            candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]

    return candidate_mask
```

并在 official-compatible configs 中强制：

```python
allow_center_fallback_inside_gt=False
```

legacy ablation configs 才允许：

```python
allow_center_fallback_inside_gt=True
allow_legacy_full_cell_span=True
```

---

### Patch 2：给 `IrregularPointGeneratorV2` 加 official dense mode

```python
# irregular_point_generator.py

class IrregularPointGeneratorV2:
    def __init__(
        self,
        strides,
        regression_range,
        dense_compat_mode=None,
        range_mode="absolute",
        decode_scale_mode="level_stride",
        radius_scale_mode="level_stride",
        ...
    ):
        if dense_compat_mode == "official_actionformer":
            range_mode = "absolute"
            decode_scale_mode = "level_stride"
            radius_scale_mode = "level_stride"

        self.range_mode = range_mode
        self.decode_scale_mode = decode_scale_mode
        self.radius_scale_mode = radius_scale_mode

        if dense_compat_mode == "official_actionformer":
            assert self.range_mode == "absolute"
            assert self.decode_scale_mode == "level_stride"
            assert self.radius_scale_mode == "level_stride"
```

配置写法：

```python
prior_generator=dict(
    type="IrregularPointGeneratorV2",
    strides=[1, 2, 4, 8, 16, 32],
    regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
    dense_compat_mode="official_actionformer",
)
```

---

### Patch 3：增强 bridge verifier，覆盖 batch / mask / missing-GT

```python
def test_bridge_generated_v2_batched_masked_equivalence():
    generator = IrregularPointGeneratorV2(
        strides=[1.0, 4.0],
        regression_range=[(0, 4), (4, 10000)],
        dense_compat_mode="official_actionformer",
    )

    feats = [
        torch.zeros(2, 1, 16),
        torch.zeros(2, 1, 5),
    ]
    masks = [
        torch.tensor([[1] * 16, [1] * 12 + [0] * 4], dtype=torch.bool),
        torch.tensor([[1] * 5, [1] * 4 + [0]], dtype=torch.bool),
    ]

    points = generator(feats, temporal_grid=make_batched_uniform_grids(...))

    official_points = convert_irregular_v2_to_official(points)
    official_targets = official_anchorfree_targets(official_points, gt_segments, gt_labels)
    bridge_targets = bridge_head.prepare_targets(points, gt_segments, gt_labels)

    assert_same_positive_mask(official_targets, bridge_targets, masks)
    assert_same_labels(official_targets, bridge_targets, masks)
    assert_max_abs_reg_target_diff(official_targets, bridge_targets, tol=1e-6)
```

另加一个反例：

```python
def test_bridge_official_mode_does_not_fallback_inside_gt_when_center_empty():
    bridge = make_bridge_head(
        assignment_mode="hard",
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_center_fallback_inside_gt=False,
    )

    # 构造一个没有 center-radius candidate 的 GT
    official = official_anchorfree_targets(...)
    bridge_out = bridge.prepare_targets(...)

    assert bridge_out.positive_mask.sum() == official.positive_mask.sum()
```

---

### Patch 4：全 config fail-closed preflight

```python
FORBIDDEN_EVAL_KEYS = {
    "bata_allow_diagnostic_gt_cache",
    "bata_diagnostic_only",
    "diagnostic_only",
    "teacher_cache",
    "use_teacher_cache",
    "allow_teacher_cache",
    "prediction_cache_shortcut",
    "raw_prediction_shortcut",
    "use_raw_prediction_shortcut",
    "use_prediction_cache",
    "allow_prediction_cache_shortcut",
    "cache_shortcut",
    "use_cache_shortcut",
    "load_from_raw_predictions",
    "load_predictions",
}

def is_enabled(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() not in {"", "false", "0", "none", "null"}
    return v is not None

def scan_config_fail_closed(node, path="cfg"):
    if isinstance(node, dict):
        for k, v in node.items():
            key = str(k)
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_EVAL_KEYS and is_enabled(v):
                raise RuntimeError(
                    f"Forbidden eval/test shortcut enabled at {child_path}: {v}"
                )
            scan_config_fail_closed(v, child_path)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            scan_config_fail_closed(v, f"{path}[{i}]")
```

这个 preflight 必须在 Stage 0 跑，并输出 machine-readable JSON。

---

### Patch 5：让 seconds conversion 显式声明 source axis

```python
def convert_to_seconds(
    segments,
    meta,
    *,
    source_axis: str,
):
    if source_axis == "selected":
        segments = selected_axis_to_dense_axis(segments, meta)
    elif source_axis == "native":
        pass
    else:
        raise ValueError(f"Unknown source_axis={source_axis}")

    fps = meta.get("fps", -1)
    if fps == -1:
        return segments

    snippet_stride = meta["snippet_stride"]
    offset_frames = meta.get("offset_frames", 0)
    window_start_frame = meta.get("window_start_frame", 0)

    seconds = (segments * snippet_stride + offset_frames + window_start_frame) / fps
    duration = meta.get("duration", None)
    if duration is not None:
        seconds = seconds.clamp(min=0, max=float(duration))
    return seconds
```

detector 中调用：

```python
segments_native = self._segments_to_axis(
    segments,
    meta,
    proposal_axis=self.proposal_axis,
    target_axis="native",
)

segments_sec = convert_to_seconds(
    segments_native,
    meta,
    source_axis="native",
)
```

这样可以彻底避免 selected/native 轴隐式推断。

---

### Patch 6：single-class postprocess 也执行 threshold / top-k

```python
if scores.ndim == 2 and scores.shape[1] == 1:
    scores = scores[:, 0]
    labels = torch.zeros_like(scores, dtype=torch.long)

    keep = scores > pre_nms_thresh
    scores = scores[keep]
    segments = segments[keep]
    labels = labels[keep]

    if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
        topk_idx = torch.topk(scores, k=pre_nms_topk).indices
        scores = scores[topk_idx]
        segments = segments[topk_idx]
        labels = labels[topk_idx]
```

---

## 9. 最终目标定义

这条路线现在必须按下面顺序推进，不能跳步：

```text
目标 1：先恢复 official dense selected-axis 65 mAP sanity。
目标 2：再证明 bridge hard official-compatible 在同等坐标轴、同等监督定义下不崩。
目标 3：然后才允许比较 Bridge soft / HeadV2 / HeadV3。
目标 4：最后才讨论真正 sparse/irregular head 是否有研究价值。
```

当前最重要的研究结论是：

**40/42 不是足以说明 sparse/irregular sampling 没价值的结果；它更像一个尚未被 official dense contract 约束住的实现崩溃结果。**

所以我的最终门禁是：

```text
SYNC_REMOTE_FOR_PREFLIGHT: YES
RUN_STAGE_0_1_2_ONLY: YES
START_LONG_TRAINING: NO
CLAIM_SPARSE_HEAD_RESULT: NO
CURRENT_TOTAL_VERDICT: FAIL
```

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commits/codex/sparse-head-clean-20260702/ "Commits · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[2]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[3]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[4]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/114f72c "Make early bridge configs state scale contract · yuzbo/OpenTAD_SparseHeadClean_20260702@114f72c · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tools/verify_bridge_dense_equivalence.py "OpenTAD_SparseHeadClean_20260702/tools/verify_bridge_dense_equivalence.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/detectors/irregular_actionformer.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/datasets/transforms/end_to_end.py "OpenTAD_SparseHeadClean_20260702/opentad/datasets/transforms/end_to_end.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/tree/codex/sparse-head-clean-20260702 "GitHub - yuzbo/OpenTAD_SparseHeadClean_20260702 at codex/sparse-head-clean-20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/irregular_actionformer_head_v2.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_head_v2.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/irregular_actionformer_head_v3.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_head_v3.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[13]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/codex/sparse-head-clean-20260702/opentad/models/utils/post_processing/utils.py "raw.githubusercontent.com"
[14]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tools/audit_sparse_head_assignment.py "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[15]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tools/analyze_detection_quality.py "OpenTAD_SparseHeadClean_20260702/tools/analyze_detection_quality.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
