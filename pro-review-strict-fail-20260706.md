## 总判决：FAIL（禁止放行长训 / 禁止声明 dense-equivalent）

这不是说五阶段方向错。相反，你这五个 Stage 的诊断路线是对的：先把 **official dense selected-axis baseline**、**bridge dense-equivalence**、**same-batch target/decode IoU**、**selected/native control matrix**、**native shortgate** 拆开，是当前最正确的排查顺序。

但在 `8689eff` 上，我的判决是：**当前证据还不能证明 bridge / irregular head 与官方 dense ActionFormer/AdaTAD 语义等价，也不能排除 65→40/42 是坐标轴或 scale 合同错误导致。禁止直接长训、禁止写 paper claim。**

我需要先说明限制：我这里无法在容器里实际 clone/pytest，网络解析 GitHub 失败，所以没有声称本地跑过 Linux preflight。下面结论基于我在线阅读目标仓库 `8689eff` 与官方 OpenTAD 源码的代码级审查。

---

# Blocking bugs / 必须阻塞的问题

## B1. Bridge dense-equivalence verifier 过窄，不能证明真实训练 route 等价官方 dense

官方 `AnchorFreeHead` 的核心合同是：

* point layout 是 `[center, reg_min, reg_max, stride]`；
* assignment 用 center sampling radius × `stride`；
* regression target 是 `distance / stride`；
* decode 是 `center ± reg_pred × stride`；
* regression range 也是官方 point generator 上的 feature-grid/range 语义。([GitHub][1])

目标仓库的 `verify_bridge_dense_equivalence.py` 确实比较了 positive mask、assigned GT、class target、encoded regression target、decoded proposal，并且 mismatch 会 nonzero exit，这是好的。但它构造的是一个“受控等价场景”：bridge head 被强行设成 `center_radius_scale="point_radius"`、`reg_denom_mode="left_right_mean"`，points 也是手工构造为 stride-like 的 7-field layout。这个测试不能覆盖默认 bridge route，也不能覆盖真实 `IrregularPointGeneratorV2` 产出的 grid。([GitHub][2])

更危险的是，基础 hard-linear bridge config 没有覆盖 `center_radius_scale` / `reg_denom_mode`，会走 `IrregularActionFormerBridgeHead` 默认值 `full_cell_span`。在 V2 point layout 中，BridgeHead 会把 field 3/4 当作 left/right scale，再求和成 `point_scale`；如果 left/right 都接近 stride，那么 `full_cell_span = left + right ≈ 2 × stride`，这已经不是官方 dense 的 stride denominator/radius 语义。([GitHub][3])

**结论：当前 verifier 只能证明“某个人工 stride-like 子集”可等价，不能证明真实 bridge sparse/native route 等价官方 dense。**

---

## B2. `hard/local_cell_span` range 语义不是官方 dense range，可能直接压死或错分中高层 positives

`IrregularPointGeneratorV2` 的 7-field layout 是：

`[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]`

其中 `range_mode="hard"` / local-cell-span 类模式会把 regression range 乘以 local cell span / left+right 类 scale；而官方 dense 的 regression range 是 point generator 给出的固定 feature-grid/range 语义，不应该随着 irregular gap 的 cell span 被任意放大/缩小。([GitHub][4])

这能解释你现在看到的现象：
**openrange / absrange 可能恢复 positives，但 high-IoU 仍弱**。因为正样本覆盖只是第一层；只要 regression denominator、decode scale、assignment level 或 proposal axis 仍错，高 IoU 会继续崩。

`absrange_expanded` 是当前最像官方 dense 的修复方向：它显式设置 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"`、`center_radius_scale="point_radius"`、`reg_denom_mode="left_right_mean"`。这条 route 比 base hard-linear 安全得多。([GitHub][5])

---

## B3. selected/native postprocess 现在看起来已修，但缺少数值级防回归测试

目标仓库当前 `IrregularActionFormer` 已经有 axis contract：

* `gt_axis` / `proposal_axis` 必须一致；
* selected-axis proposal 会先通过 `selected_axis_to_dense_axis` 转回 native；
* NMS 在 native proposal 上做；
* seconds conversion 时若 `irregular_native_axis=True`，不会再次 selected→native 转换。([GitHub][6])

这说明 **selected/native 双重转换或 NMS 轴错误目前代码层面大概率已修正**。

但目前测试主要检查 metadata / config 字符串，缺少一个强制数值测试：

`selected segment -> native segment -> seconds`

必须等于：

`selected segment -> convert_to_seconds(selected_axis)`

并且 NMS 输入必须始终是 native dense coordinates。没有这个测试，后续改动很容易把老 bug 重新引入。

---

## B4. Stage 2 “official dense selected-axis sanity” 还不是强 official 证明

`input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py` 是一个薄 wrapper，它声明目标是 50% equal-interval selected-axis dense sanity route。这个设计没问题，但它本身仍然运行当前 repo 内的 dense code。([GitHub][7])

官方 OpenTAD 的 AdaTAD 官方实现/checkpoints 指向 OpenTAD；因此 Stage 2 不能只叫 “official”，还需要：

1. clone / pin `sming256/OpenTAD`；
2. 对 `AnchorFreeHead`、`PointGenerator`、postprocess、THUMOS AdaTAD config 做源码 diff 或 hash；
3. 在当前 repo 内 dense route 跑到接近 65 之前，不能把 sparse 40/42 归因给 sparse head。([GitHub][8])

---

## B5. BATA diagnostic GT cache 默认是关闭的，但必须加 val/test fail-closed

`LoadFrames` 里存在 BATA diagnostic cache / GT cache 相关参数，默认是 false；你列出的 random/uniform/bridge configs 看起来没有主动使用这条路径，所以我没有看到当前 selected-axis control 里有明确 test-time GT 泄漏。([GitHub][9])

但这类开关一旦被误开到 val/test，就是灾难级实验污染。需要在 loader 或 precheck 脚本里强制：

* val/test 禁止 `bata_allow_diagnostic_gt_cache=True`；
* val/test 禁止 diagnostic-only / teacher / cache shortcut；
* 保存的 result metadata 必须记录 `uses_gt=False, uses_teacher=False, uses_prediction_cache=False`。

---

# High-risk design mismatches

## H1. `full_cell_span` 和官方 stride 不是一回事

官方 dense decode 和 regression target 都用 stride。目标 bridge 默认 `full_cell_span` 在 7-field irregular point 上会变成 left+right。若 uniform grid 的 left=right=stride，则 denominator 是 `2×stride`。这会让 regression target 数值缩小、decode 放大或和 assignment radius 不一致，直接伤 high-IoU。

**我的判断：这是 65→40/42 的第一嫌疑。**

---

## H2. positive coverage 恢复 ≠ 高 IoU 恢复

Stage 1 加 `assigned_positive_target_decode_iou` 和 `oracle_assigned_recall@IoU` 是必要的。只看 positive 数量或 GT coverage 不够。

真正能排除 target/decode 崩溃的是：

* assigned positive decode 后是否能高 IoU 还原对应 GT；
* oracle-assigned recall@0.5/0.6/0.7 是否高；
* 按 GT 长度分桶后，短动作/中动作/长动作是否都能被 assigned positives decode 回来。

你的 audit tool 已经加入这些指标方向，这是正确的。([GitHub][10])

---

## H3. native irregular semantics 比 selected-axis dense semantics 难很多

如果只是要恢复接近 equal-interval 50% dense baseline，**selected-axis dense semantics 是最稳路线**：

* GT remap 到 selected axis；
* dense ActionFormerHead 仍认为时间轴是等间隔 selected tokens；
* postprocess 再 selected→native→seconds；
* detector 不直接处理 irregular cell width。

而 native irregular semantics 必须同时正确处理：

* cell width；
* local gap；
* assignment range；
* center sampling radius；
* regression denominator；
* proposal decode；
* NMS axis；
* seconds conversion；
* high-IoU calibration。

任何一个 scale 合同错，mAP@0.7 都会先死。

---

# 性能崩溃根因排序

## 1. 最可能：native irregular scale contract 错误

包括：

* `full_cell_span` 默认 denominator/radius；
* `range_mode="hard"` 把 range 乘 local cell span；
* V2 point layout 与 BridgeHead scale 解释不完全等价官方 dense；
* bridge verifier 没覆盖真实 generated grid。

这最能解释：positives 可能不少，但 high-IoU 仍弱。

## 2. selected/native/native-seconds 坐标合同历史错误或残留边界条件

当前 `IrregularActionFormer` 和 postprocess utils 看起来已经把 selected→native→seconds 的双重转换问题修掉了，但缺少数值测试。若 selected-axis control 仍明显低于 dense baseline，优先怀疑这里。

## 3. regression range 压死中高层 positives

`hard/local_cell_span` range 很可能导致 level assignment 错误。`openrange` / `absrange_expanded` 是合理修复，但只能证明 positive 覆盖，不自动证明 high-IoU decode 正确。

## 4. LoadFrames selected-axis remap / collapsed GT drop 影响短动作

`LoadFrames` 的 inverse interpolation 和 collapsed segment drop 设计总体合理，比强行制造 tiny GT 好；但必须按 GT 长度分桶看 coverage。短动作如果在 selected-axis 下大量 collapse，dense selected-axis baseline 也会下降。([GitHub][9])

## 5. postprocess / NMS 轴错误

当前代码层面不是第一嫌疑，因为 selected proposal 会先转 native，再 NMS，再 seconds；但由于缺数值测试，它仍是高风险回归点。

## 6. loss normalizer / valid mask / projection / backbone

官方 dense loss normalizer 是 EMA positive normalizer；目标 bridge 如果 valid mask 与 positive mask 对齐正确，这类问题通常不会单独造成 65→40。它们是二级嫌疑。([GitHub][1])

---

# 当前证据判断

| 命题                                              | 当前判断                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ |
| regression range 压死中高层 positives                | **有强嫌疑，但需要 per-level positives + length-bucket coverage 证明** |
| openrange 恢复 positives 但 high-IoU 弱             | **支持“不是单纯 positive 数量问题”，更像 decode/scale/axis**              |
| native-axis irregular 坐标合同导致定位误差                | **高度可疑，需要 selected-axis bridge vs native bridge 对照确认**       |
| regression denominator / decode scale 错误        | **最强嫌疑之一，尤其是 full_cell_span vs stride**                      |
| selected-axis postprocess/NMS 曾经错误但现在是否已修       | **代码看起来已修，但必须加 numeric roundtrip test**                      |
| bridge hard dense-equivalence 是否仍可能不成立          | **是。当前 verifier 不能覆盖真实 route**                               |
| 是否存在当前 random/uniform control 的 test-time GT 泄漏 | **未见明确泄漏，但 BATA diagnostic cache 必须 fail-closed**            |

---

# 严格实验路线

## Phase 0：Linux preflight，先证明代码可加载

必须先跑，不跑完禁止训练：

```bash
git checkout 8689eff2a0095db661a8b7d0132d24659e8cbaee
git status --short
```

检查 config load：

```bash
python - <<'PY'
try:
    from mmengine.config import Config
except Exception:
    from mmcv import Config

cfgs = [
    "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py",
    "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py",
]
for p in cfgs:
    cfg = Config.fromfile(p)
    print("[OK]", p, cfg.model.get("type", None), cfg.model.get("rpn_head", {}).get("type", None))
PY
```

编译核心文件：

```bash
python -m py_compile \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/irregular_actionformer_head_v2.py \
  opentad/models/dense_heads/irregular_actionformer_head_v3.py \
  opentad/models/dense_heads/prior_generator/irregular_point_generator.py \
  opentad/models/utils/post_processing/utils.py \
  tools/audit_sparse_head_assignment.py \
  tools/verify_bridge_dense_equivalence.py \
  scripts/verify_official_dense_reference.py
```

跑单测与 bridge equivalence：

```bash
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q
python tools/verify_bridge_dense_equivalence.py --json
```

**Gate：** 任一失败，停止。先修代码，不启动训练。

---

## Phase 1：same-batch audit matrix

目标不是 mAP，而是证明 target assignment / encode / decode 语义。

需要对以下 route 全部输出：

* per-level positives；
* GT coverage；
* assigned_positive_target_decode_iou；
* oracle_assigned_recall@IoU；
* 按 GT 长度分桶 coverage；
* per-level regression target percentile；
* decoded proposal 与 assigned GT 的 max IoU。

建议矩阵：

```bash
CFG_LIST=(
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py
)

for CFG in "${CFG_LIST[@]}"; do
  NAME=$(basename "$CFG" .py)
  python tools/audit_sparse_head_assignment.py "$CFG" \
    --split train \
    --batches 1 \
    --device cuda \
    --json "logs/audit_${NAME}.json" \
    | tee "logs/audit_${NAME}.log"
done
```

如果你的 CLI 不是 positional config，先用：

```bash
python tools/audit_sparse_head_assignment.py --help
```

**Gate：**

* 如果 openrange/absrange_expanded positives 恢复，但 `assigned_positive_target_decode_iou` 仍低，根因转向 denominator/decode/axis。
* 如果 selected-axis bridge audit 明显好于 native audit，根因指向 native irregular 坐标合同。
* 如果所有 audit 都好但 mAP 仍低，再看 postprocess/NMS/score calibration。

---

## Phase 2：official dense selected-axis sanity

先证明当前 repo 能复现接近 65 的 equal-interval 50% dense baseline：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --id 0
```

**Gate：**

* 目标：Average-mAP 接近用户已知 65 区间。
* 若低很多，例如掉到 40–50：停止，不要审 sparse head。说明当前 repo 的 dense/data/config 已经漂移。
* 若接近 65：可以继续 selected-axis control matrix。

---

## Phase 3：selected-axis control matrix

三条必须一起跑：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --id 0

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --id 0

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  --id 0
```

**解释：**

1. uniform dense selected-axis 回到约 65：说明 selected-axis dense semantics 成立。
2. random dense selected-axis 只小幅下降：说明 random 50% 本身不该导致 40。
3. random selected-axis bridge absrange_expanded 若接近 random dense：说明 hard bridge target/decode 基本成立。
4. 若 bridge selected-axis 仍 40：说明 BridgeHead 仍不等价官方 dense。

---

## Phase 4：native absrange-expanded shortgate

只跑 shortgate，不直接长训：

```bash
PRECHECK_ONLY=1 RUN_TRAIN=0 bash \
  remote_runs/run_bridge_absrange_expanded_shortgate_failclosed_20260706.sh
```

precheck 过后才允许：

```bash
PRECHECK_ONLY=0 RUN_TRAIN=1 CUDA_VISIBLE_DEVICES=0 bash \
  remote_runs/run_bridge_absrange_expanded_shortgate_failclosed_20260706.sh
```

**Gate：**

* 必须明显超过 `40.20 HeadV3 fixed gate`；
* 必须超过或至少接近超过 `42.44 openrange gate`；
* mAP@0.7 不能继续死亡；
* 若 selected-axis bridge 过、native shortgate 不过，根因几乎锁定为 native irregular coordinate/scale semantics。

---

## Phase 5：只在 gate 通过后长训

可长训顺序：

1. selected-axis dense official sanity；
2. selected-axis bridge absrange_expanded；
3. native bridge absrange_expanded；
4. native bridge dense-equivalence fixed route；
5. 最后才考虑 soft assignment sparse head。

禁止顺序：

* 不要先长训 HeadV3；
* 不要先长训 native openrange；
* 不要在 official dense sanity 未过前解释 sparse 崩溃；
* 不要在 bridge selected-axis 未过前引入 soft assignment。

---

# 关键修复 patch

## Patch 1：BridgeHead 默认 scale 改成 official-compatible，legacy full-cell 必须显式 opt-in

**文件：** `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`
**函数：** `IrregularActionFormerBridgeHead.__init__`

**修改前问题：**

默认：

```python
center_radius_scale="full_cell_span"
reg_denom_mode="full_cell_span"
```

这不是官方 dense stride 语义。V2 layout 下很容易变成 `left + right ≈ 2×stride`。

**修改后建议：**

```python
class IrregularActionFormerBridgeHead(...):
    def __init__(
        self,
        ...,
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        ...
    ):
        super().__init__(...)

        self.center_radius_scale = center_radius_scale
        self.reg_denom_mode = reg_denom_mode
        self.allow_legacy_full_cell_span = allow_legacy_full_cell_span

        legacy_modes = {"full_cell_span", "cell_span", "local_cell_span"}
        if (
            not allow_legacy_full_cell_span
            and (
                self.center_radius_scale in legacy_modes
                or self.reg_denom_mode in legacy_modes
            )
        ):
            raise ValueError(
                "IrregularActionFormerBridgeHead is in official-compatible mode by default. "
                "full_cell_span/local_cell_span are not official dense stride semantics. "
                "Set allow_legacy_full_cell_span=True only for explicitly named legacy ablations."
            )
```

**对应测试：**

```python
def test_bridge_head_rejects_legacy_full_cell_span_by_default():
    with pytest.raises(ValueError):
        IrregularActionFormerBridgeHead(
            num_classes=20,
            in_channels=256,
            feat_channels=256,
            center_radius_scale="full_cell_span",
            reg_denom_mode="full_cell_span",
        )
```

**为什么能验证根因：**

如果 65→40 是 denominator/radius scale 错误，这个 patch 会把所有无意的 full-cell route fail-closed，迫使实验只跑 official-compatible scale route。之后 selected-axis bridge 若恢复，就能直接确认 scale 合同是主因。

---

## Patch 2：Base hard-linear config 不得再默默使用 legacy scale

**文件：**
`configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py`

**函数/区域：** `model.rpn_head`

**修改前问题：**

base bridge route 只配置了 `assignment_mode`、`regression_mode`、`prior_generator` 等，但没有覆盖 `center_radius_scale` 和 `reg_denom_mode`，会走危险默认。

**修改后代码片段：**

```python
model = dict(
    rpn_head=dict(
        type="IrregularActionFormerBridgeHead",
        assignment_mode="hard",
        regression_mode="symmetric_linear",

        # Official dense-compatible scale contract.
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,

        prior_generator=dict(
            type="IrregularPointGeneratorV2",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            range_mode="absolute",
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
    )
)
```

如果你仍想保留旧路线，必须改名为：

```python
input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_legacy_fullcell_n16r4.py
```

并显式：

```python
allow_legacy_full_cell_span=True
```

**对应测试：**

```python
def test_base_bridge_config_uses_official_compatible_scale_contract():
    cfg = Config.fromfile(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"
    )
    head = cfg.model.rpn_head
    pg = head.prior_generator
    assert head.center_radius_scale == "point_radius"
    assert head.reg_denom_mode == "left_right_mean"
    assert pg.range_mode == "absolute"
    assert pg.decode_scale_mode == "level_stride"
    assert pg.radius_scale_mode == "level_stride"
```

**为什么能验证根因：**

如果这个 patch 后 same-batch `assigned_positive_target_decode_iou` 和 selected-axis bridge mAP 恢复，说明旧 route 的核心问题就是 scale/range 合同，不是 sparse 本身。

---

## Patch 3：Bridge dense-equivalence verifier 必须覆盖真实 PointGeneratorV2

**文件：** `tools/verify_bridge_dense_equivalence.py`
**函数：** 新增 `run_generated_v2_equivalence_case`

**修改前问题：**

现有 verifier 主要测人工构造 points，不测真实 `IrregularPointGeneratorV2` 输出。

**修改后代码片段：**

```python
def run_generated_v2_equivalence_case(device="cpu"):
    import torch
    from opentad.models.dense_heads.prior_generator.irregular_point_generator import (
        IrregularPointGeneratorV2,
    )

    stride = 4
    T = 8
    centers = torch.arange(T, device=device, dtype=torch.float32) * stride

    temporal_grid = dict(
        center=centers[None, :],
        cell_left=torch.full((1, T), float(stride), device=device),
        cell_right=torch.full((1, T), float(stride), device=device),
    )

    feat = torch.zeros(1, 256, T, device=device)

    prior = IrregularPointGeneratorV2(
        strides=[stride],
        regression_range=[(0, 10000)],
        range_mode="absolute",
        decode_scale_mode="level_stride",
        radius_scale_mode="level_stride",
    )

    bridge_points = prior([feat], [temporal_grid])[0]

    official_points = torch.stack(
        [
            centers,
            torch.zeros_like(centers),
            torch.full_like(centers, 10000.0),
            torch.full_like(centers, float(stride)),
        ],
        dim=-1,
    )

    # Required V2 layout:
    # center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale
    assert torch.allclose(bridge_points[:, 0], official_points[:, 0])
    assert torch.allclose(bridge_points[:, 1], official_points[:, 1])
    assert torch.allclose(bridge_points[:, 2], official_points[:, 2])
    assert torch.allclose(bridge_points[:, 3], official_points[:, 3])
    assert torch.allclose(bridge_points[:, 4], official_points[:, 3])
    assert torch.allclose(bridge_points[:, 5], torch.ones_like(centers))
    assert torch.allclose(bridge_points[:, 6], official_points[:, 3])
```

然后把这个 case 接入 `--json` 输出：

```python
results["generated_v2_levelstride_equivalence"] = run_generated_v2_equivalence_case(
    device=args.device
)
```

**对应测试：**

```bash
python tools/verify_bridge_dense_equivalence.py --json
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q
```

**为什么能验证根因：**

它直接检查真实 generator 输出是否满足 bridge dense-equivalence。旧 verifier 过不了这个 coverage gap。

---

## Patch 4：selected/native/seconds 禁止双重转换的数值测试

**文件：** `tests/test_adapter_native_dense_headv2_contracts.py`
**新增测试函数：**

```python
def test_selected_to_native_to_seconds_is_not_double_converted():
    import torch
    from opentad.models.utils.post_processing.utils import (
        selected_axis_to_dense_axis,
        convert_to_seconds,
    )

    meta = dict(
        fps=30.0,
        snippet_stride=16,
        offset_frames=0,
        window_start_frame=0,
        duration=100.0,
        irregular_selected_positions=[0.0, 10.0, 20.0],
        irregular_selected_valid_len=30.0,
    )

    selected_segments = torch.tensor([[0.0, 2.0], [0.5, 1.5]])

    native_segments = selected_axis_to_dense_axis(selected_segments.clone(), meta)

    seconds_from_selected = convert_to_seconds(
        selected_segments.clone(),
        {**meta, "irregular_native_axis": False},
    )

    seconds_from_native = convert_to_seconds(
        native_segments.clone(),
        {**meta, "irregular_native_axis": True},
    )

    assert torch.allclose(seconds_from_selected, seconds_from_native, atol=1e-5)
```

**为什么能验证根因：**

如果 selected-axis route NMS/seconds 曾经双重转换，这个测试会立刻抓住。它也是防止未来回归的最小测试。

---

## Patch 5：val/test 禁止 diagnostic GT cache

**文件：** `opentad/datasets/transforms/end_to_end.py`
**函数：** `LoadFrames.__call__` 或 BATA branch 前

**修改前问题：**

diagnostic cache 默认关闭，但缺少 val/test fail-closed。

**修改后代码片段：**

```python
def _subset_name(results):
    for key in ("subset", "split", "data_split"):
        if key in results and results[key] is not None:
            return str(results[key]).lower()
    return ""

def _assert_no_eval_diagnostic_gt_cache(self, results):
    subset = _subset_name(results)
    is_eval = subset in {"val", "valid", "validation", "test", "testing"}

    if not is_eval:
        return

    if getattr(self, "bata_allow_diagnostic_gt_cache", False):
        raise RuntimeError(
            "bata_allow_diagnostic_gt_cache=True is forbidden for validation/test."
        )

    if getattr(self, "bata_diagnostic_only", False):
        raise RuntimeError(
            "bata_diagnostic_only=True is forbidden for validation/test."
        )
```

在 `__call__` 开头加入：

```python
self._assert_no_eval_diagnostic_gt_cache(results)
```

**对应测试：**

```python
def test_loadframes_forbids_eval_diagnostic_gt_cache():
    loader = LoadFrames(
        method="bata_boundary_acquisition_subsample",
        bata_allow_diagnostic_gt_cache=True,
    )
    with pytest.raises(RuntimeError):
        loader({"subset": "validation"})
```

**为什么能验证根因：**

这不是性能崩溃主因，但它直接保护实验可信度。没有这个 gate，后续任何 val/test mAP 都可能被质疑。

---

# 最终路线建议

## 如果目标是恢复接近 equal-interval 50% dense baseline

**保留 selected-axis dense semantics。**

最稳结构：

```text
LoadFrames random/uniform fixed 50%
→ remap_gt_to_selected_axis=True
→ official dense ActionFormerHead / AnchorFreeHead semantics
→ proposals selected-axis
→ selected_axis_to_dense_axis before NMS
→ native dense axis NMS
→ convert_to_seconds
```

这条路线最优雅，因为它把“选了哪些帧”与“检测头是否理解 irregular gaps”解耦。先恢复 65，再谈 native irregular。

## 如果坚持 native irregular semantics

最小必要机制：

1. proposal / GT / postprocess axis 显式 contract；
2. regression denominator 必须 official-compatible，优先 `level_stride` 或 `left_right_mean`；
3. center radius 必须 official-compatible，优先 `point_radius`；
4. regression range 用 `absolute / absrange_expanded / levelstride`，不要默认 local cell span；
5. NMS 前统一 native dense axis；
6. seconds conversion 只能做一次 selected→native；
7. same-batch audit 必须每次输出 decode IoU 和 oracle recall；
8. 每个 config 都要 fail-closed 检查 `uses_gt/teacher/cache=False`。

## 最终 head 选择

我的推荐顺序：

1. **official dense ActionFormerHead + selected-axis remap**
   用来恢复 baseline 和建立可信对照。

2. **bridge hard dense-equivalent head**
   只在 selected-axis bridge 能接近 dense selected-axis 后，才作为 native irregular 的基础。

3. **soft assignment sparse head**
   只能在 hard bridge 等价且 mAP 恢复后做。否则 soft assignment 会掩盖根因。

4. **metric-aware projection/neck**
   这是后续优化，不是当前 65→40 崩溃的第一修复点。

---

# 下一步 24 小时内最应该跑什么

按这个顺序，不要跳步：

```bash
set -euo pipefail

cd /path/to/OpenTAD_SparseHeadClean_20260702
git fetch origin
git checkout 8689eff2a0095db661a8b7d0132d24659e8cbaee
git status --short

mkdir -p logs/strict_preflight_20260706
```

## 1. Config load

```bash
python - <<'PY' | tee logs/strict_preflight_20260706/config_load.log
try:
    from mmengine.config import Config
except Exception:
    from mmcv import Config

cfgs = [
    "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py",
    "configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py",
    "configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py",
]
for p in cfgs:
    cfg = Config.fromfile(p)
    print("[OK]", p)
    print("  model:", cfg.model.get("type", None))
    print("  head :", cfg.model.get("rpn_head", {}).get("type", None))
PY
```

## 2. Py compile

```bash
python -m py_compile \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/irregular_actionformer_head_v2.py \
  opentad/models/dense_heads/irregular_actionformer_head_v3.py \
  opentad/models/dense_heads/prior_generator/irregular_point_generator.py \
  opentad/models/utils/post_processing/utils.py \
  tools/audit_sparse_head_assignment.py \
  tools/verify_bridge_dense_equivalence.py \
  scripts/verify_official_dense_reference.py \
  | tee logs/strict_preflight_20260706/py_compile.log
```

## 3. Unit tests and bridge verifier

```bash
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q \
  | tee logs/strict_preflight_20260706/test_adapter_native_dense_headv2_contracts.log

python tools/verify_bridge_dense_equivalence.py --json \
  | tee logs/strict_preflight_20260706/verify_bridge_dense_equivalence.json
```

## 4. Fail-closed remote prechecks

```bash
bash remote_runs/precheck_official_dense_selected_axis_sanity_20260706.sh \
  | tee logs/strict_preflight_20260706/precheck_official_dense_selected_axis_sanity.log

bash remote_runs/precheck_selected_axis_control_matrix_20260706.sh \
  | tee logs/strict_preflight_20260706/precheck_selected_axis_control_matrix.log

PRECHECK_ONLY=1 RUN_TRAIN=0 bash \
  remote_runs/run_bridge_absrange_expanded_shortgate_failclosed_20260706.sh \
  | tee logs/strict_preflight_20260706/precheck_native_absrange_expanded_shortgate.log
```

## 5. Same-batch audit first，训练后置

```bash
python tools/audit_sparse_head_assignment.py --help

for CFG in \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py
do
  NAME=$(basename "$CFG" .py)
  python tools/audit_sparse_head_assignment.py "$CFG" \
    --split train \
    --batches 1 \
    --device cuda \
    --json "logs/strict_preflight_20260706/audit_${NAME}.json" \
    | tee "logs/strict_preflight_20260706/audit_${NAME}.log"
done
```

只有这些全部过，再启动第一条训练：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --id 0
```

**24 小时内唯一可以接受的成功标准：**

1. official dense selected-axis sanity 接近 65；
2. same-batch audit 证明 absrange_expanded / selected-axis bridge 的 assigned-positive decode IoU 正常；
3. selected-axis dense/random control 不塌到 40；
4. native shortgate 只作为最后 gate，不提前长训。

[1]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[2]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/tools/verify_bridge_dense_equivalence.py "OpenTAD_SparseHeadClean_20260702/tools/verify_bridge_dense_equivalence.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[3]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[4]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/opentad/models/detectors/irregular_actionformer.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/sming256/AdaTAD?utm_source=chatgpt.com "sming256/AdaTAD: [CVPR2024] The official ..."
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/opentad/datasets/transforms/end_to_end.py "OpenTAD_SparseHeadClean_20260702/opentad/datasets/transforms/end_to_end.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/8689eff2a0095db661a8b7d0132d24659e8cbaee/tools/audit_sparse_head_assignment.py "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at 8689eff2a0095db661a8b7d0132d24659e8cbaee · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
