# Goal86 P-A 残差 BC 契约阻塞

**日期**：2026-11-10<br>
**Goal**：Goal86<br>
**状态**：`P-A_BC_PRIOR_CONTRACT_MISMATCH / P-A_NOT_READY_FOR_SCREEN`<br>
**影响范围**：只阻塞 `P-A`；`T1/T2` 的 `PPO/SAC` 1M screen 可继续。

## 1. 预注册定义

Goal86 对 `P-A` 的定义是 **传统控制器 base + D3 教师残差 BC**：

```text
base_action    = ALOS(observation) -> PD/PI(observation)
teacher_action = D3_BC(observation)
label          = normalize(teacher_action - base_action)
```

`P-A` 的 BC 先验必须拟合这个残差标签；随后 PPO/SAC 可以在这个残差先验附近继续
学习。`T2` 是同一个 ALOS+PD/PI base 的零残差对照，不能把 `P-A` 的 base 偷换成
D3 绝对动作。

## 2. 当前实现的实际语义

现有 P-A PPO/SAC 初始化做的是：

```text
加载 D3 state/path encoder 和 actor 主干
-> 在 actor 输出端追加 command/current 输入
-> 将最终输出层（mu/head）置零
-> 从 ALOS 零残差附近开始 RL
```

因此它只实现了 **D3 表征先验 + 零残差初始化**，没有拟合
`D3_action - ALOS_action` 标签。manifest 中的
`path_encoder_loaded=true`、`bc_path_dim=8` 和
`initial_residual_abs_max=0.0` 不能证明残差 BC 已执行。

## 3. 证据

### 3.1 标签审计

远端 Goal85 训练分布 demos：

```text
/tmp/goal86_pa_demos
32 envs x 800 steps = 25,600 samples
training seeds 90000-90031, path12, command3
```

标签产物：

```text
/tmp/goal86_pa_residual_labels/pa_residual_labels.npz
/tmp/goal86_pa_residual_labels/pa_residual_label_audit.json
```

在 residual scale `0.25` 下，归一化残差统计为：

| 通道 | mean | std | abs_p95 | 超出 `[-1,1]` 比例 |
|---|---:|---:|---:|---:|
| rudder | `-0.9172` | `2.6719` | `3.4000` | `88.45%` |
| stern | `0.1174` | `2.4833` | `4.3308` | `73.86%` |
| rpm | `-0.2206` | `0.3317` | `0.8392` | `2.66%` |

物理残差：

| 通道 | abs_mean | abs_p95 |
|---|---:|---:|
| rudder | `0.2299 rad` | `0.2967 rad` |
| stern | `0.1853 rad` | `0.3779 rad` |
| rpm | `112.2` | `320.0` |

结论：D3 与 Goal85 的 ALOS+PD/PI base 在 actuator 空间差异很大，不是 `0.25`
包线内的小修正。若直接把标签裁剪到 `[-1,1]` 再训练，`P-A` 学到的将主要是
饱和方向，不能声称“执行了 D3 残差 BC”。

### 3.1.1 residual scale 只读网格

使用同一批 25,600 条样本，只改变 residual scale，不训练策略：

| scale | rudder 裁剪 | stern 裁剪 | rpm 裁剪 | rudder abs_p95 | stern abs_p95 |
|---:|---:|---:|---:|---:|---:|
| `0.25` | `88.45%` | `73.86%` | `2.66%` | `3.40` | `4.33` |
| `0.50` | `75.66%` | `52.95%` | `0.008%` | `1.70` | `2.17` |
| `1.00` | `0.77%` | `7.47%` | `0` | `0.85` | `1.08` |

这支持 **H86-PA-1**：`0.25` 的包线过小，标签被大面积裁剪；`1.00` 才接近覆盖
教师差异。`1.00` 不是自动选定的新协议，因为它会改变 T2/P-A 的动作包线；必须
在启动 P-A 前明确记录为协议修订，并让 T2 使用同一 scale，不能只放大 P-A。

### 3.2 D3 教师闭环

在 Goal85 诊断配置下（path8、关闭 endpoint context、`endpoint_stop`），D3 教师
在 dev seeds `20–23` 上 4/4 success，无 safety/timeout/plane-miss；material
saturation 约 `0.075–0.413`。这暂时不支持“D3 教师本身不可用”的解释，但不能替代
对残差包线和标签失真的审计。

### 3.3 架构和配对

- PPO/SAC 均已使用项目原生 `z_t` 编码器，未退回普通 flatten+MLP。
- PPO features：`[z_t(64), command(3), zero_dade(64)] = 131`。
- SAC features：`[z_t(64), zero_dade(64), command(3)] = 131`。
- 两者语义和宽度一致，但**不能声称张量列顺序完全相同**。
- zero-residual parity 已通过：PPO 112 项共同数值指标最大差 `0.0`，SAC 16 项
  最大差 `0.0`，残差指标最大绝对值 `0.0`。仅 dev seeds `20–23`，是工程配对，
  不是效果结论。

## 4. 可证伪假设

### H86-PA-1：残差包线过小导致标签失真

机制：`0.25` 的执行器残差包线远小于 D3 与 ALOS 的物理差异，标签被大量裁剪，
BC 无法表示教师修正。

成立时应看到：增大 residual scale 后，rudder/stern 的裁剪比例显著下降，BC 验证
loss 和闭环动作误差下降；在相同 base 和 seed 下，残差 BC 的动作不再长期贴边。

失败时应看到：即使放大包线，标签仍然双峰/饱和，或 BC 预测误差不降；这说明问题在
教师-base 不匹配或动作语义，而不是单纯 scale。

最便宜证伪：用现有 25,600 样本做 scale 网格 `0.25/0.50/1.00` 的只读标签统计；
不训练、不跑闭环。

### H86-PA-2：D3 教师不适配 Goal85 base

机制：D3 在旧观测/旧闭环条件下训练，与 Goal85 的 ALOS base、path12 endpoint
context 或噪声契约不匹配，导致 `D3 - ALOS` 不是可学习的小残差。

成立时应看到：D3 闭环在 Goal85 下出现 safety/timeout/plane-miss 或明显跟踪退化；
或者固定 base 后，教师动作与 base 的系统偏差无法由可部署观测解释。

失败时应看到：D3 闭环稳定、4/4 success，且偏差主要由 scale/动作包线解释；当前
4-seed 诊断更接近这一侧，故暂不支持 H86-PA-2 作为主因。

## 5. 当前裁决

```text
P-A_BC_PRIOR_CONTRACT_MISMATCH = HARD_BLOCK
P-A_NOT_READY_FOR_SCREEN       = true
T1_T2_SCREEN_CAN_PROCEED       = true
ALGORITHM_SWITCH               = false
HELDOUT_OPENED                 = false
```

不启动包含 P-A 的完整 screen；先运行 `T1/T2` 的 PPO/SAC 1M screen，保持预算、
seed、reward、terminal 和评价协议不变。

## 6. 修复选项

### 方案 A（推荐，保持预注册语义）

严格执行 `label = normalize(D3_actuator - ALOS_actuator)`：

1. 用只读审计确定不产生大面积裁剪的 residual scale（当前证据指向 `1.00`），或
   明确记录裁剪并把它作为 `BC_TEACHER_INVALID` 的一部分；
2. 用 D3 state/path encoder + `z_t` actor 主干初始化，最终残差头用真实残差标签
   做监督预训练；
3. BC-only 闭环通过 finite/safety/timeout 检查后，才能进入 P-A 的 PPO/SAC；
4. P-A 与 T2 必须使用相同 residual scale、reward、terminal 和 dev seeds。

### 方案 B（改变 Goal 语义，不能冒充 P-A）

保留“D3 绝对 actor 主干 + 零残差输出”，把它明确命名为新的 arm（例如
`P-B-D3-backbone-zero-residual`），并新写预注册 Goal。该 arm 不能与
`BC 残差先验` 共享结论，也不能用来回答 H86-A。

## 7. 下一步

1. 继续 `T1/T2` 的 1M screen，四组均使用同一 Goal85 环境、同一 reward/terminal、
   dev `20–59`，held-out `100–149` 保持关闭；
2. 对 P-A 做 scale 只读网格审计，决定 H86-PA-1 是否支持；
3. 在 P-A 契约修复前不运行 P-A screen/formal，不把现有零残差初始化写成 BC 残差；
4. 修复后补 BC-only 闭环和 zero-residual parity，再进入 P-A 1M screen；
5. 更新 `research/STATE.md`、`research/LEDGER.jsonl` 和 Goal86 执行计划。
