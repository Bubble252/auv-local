# Goal9 P 组优化复核报告

更新日期：2026-11-05

状态：**P1 优化复核完成。路径输入仍未通过因果归因门禁；actor 学习率被确认是
一个真实的优化瓶颈，但它没有让路径分支产生闭环收益。**

本报告只针对 P1 路径输入分支的优化复核，不覆盖、不重写历史 P0/P1/P2/P3
产物，也不把 DADE、ABC 或 V 阶段的结果混入本结论。

## 1. 问题

旧 P1 的 default success 提升 `+0.05`、3D RMSE 改善 `0.4308 m`，但 `path-zero`
只改变动作 L2 `4.70e-5`，闭环 success/RMSE 几乎完全不变。因此有两个竞争解释：

1. 路径输入本身没有价值；
2. 路径输入有价值，但编码器缩放、actor 更新预算、奖励尺度或优势归一化使它没有学出来。

本报告用单变量臂逐一排除第 2 组解释。

## 2. 实验矩阵

所有臂保持：

- 训练 seed：`91430`；
- 评估 seed：`1000–1019`，20 个固定 seed；
- 环境：C3 random-realistic，default + strict125；
- 路径消融：`none`、`path-zero`、`path-geom-zero`、`path-context-zero`；
- 基线：P0 `A0_stride1`；
- 原 P1 门禁：primary signal + guardrails + `path-zero` 因果归因。

| arm | 唯一变量 | iterations | 环境步数 |
|---|---|---:|---:|
| X1 `P1opt_alpha1_32` | `path_encoder_alpha: 0.01 -> 1.0` | 32 | 1,048,576 |
| X2 `P1opt_alpha1_128` | 训练预算 `32 -> 128` | 128 | 4,194,304 |
| X3 `P1opt_alpha1_rewardscale1_32` | `reward.pc_ppo_scale: 0.01 -> 1.0` | 32 | 1,048,576 |
| X4lr `P1opt_alpha1_lr1e-4_32` | `learning_rate_actor: 1e-5 -> 1e-4` | 32 | 1,048,576 |
| X4norm `P1opt_alpha1_mbadvnorm_32` | 逐 mini-batch advantage normalization | 32 | 1,048,576 |

## 3. 主结果

| arm | default success delta | strict125 success delta | 3D RMSE 改善 | path-zero 动作 L2 | path-zero RPM 差 | path-zero success 下降 | path-zero RMSE 恶化 | 路径归因 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| X1 | `+0.05` | `0.00` | `+0.4376 m` | `1.65e-3` | 约 `1.29 rpm` | `0.00` | `+0.0008 m` | 否 |
| X2 | `0.00` | `0.00` | `-0.0246 m` | `6.10e-3` | `7.12 rpm` | `0.00` | `-0.0085 m` | 否 |
| X3 | `0.00` | `0.00` | `+0.0308 m` | `1.95e-3` | `1.62 rpm` | `0.00` | `+0.0026 m` | 否 |
| X4lr | `+0.10` | `+0.25` | `+0.0297 m` | `2.27e-2` | `13.85 rpm` | `0.00` | `-0.0039 m` | 否 |
| X4norm | `+0.05` | `0.00` | `+0.4196 m` | `1.99e-3` | `0.86 rpm` | `0.00` | `+0.0008 m` | 否 |

说明：X4lr 的 default success 提升来自 seed 1001、1005 两次失败→成功；strict125
提升来自 seed 1003、1006、1009、1016、1019 五次失败→成功。相比 X1 的单 seed
翻转，这是更稳定的性能信号。

## 4. 机制证据

### 4.1 路径支路确实被打开

旧 P1 的路径末层有效输出尺度约 `1.20e-4`。X1 在 32 轮后达到约 `6.20e-3`，
约为旧 P1 的 52 倍；X2 在 128 轮后达到约 `1.27e-2`。路径零化动作 L2 也从旧 P1
的 `4.70e-5` 升到 X1 `1.65e-3`、X2 `6.10e-3`、X4lr `2.27e-2`。

因此，“路径编码器完全没有打开”已被排除。后续没有路径归因，不是零缩放造成的。

### 4.2 训练预算不是充分原因

X2 把预算提高到 4×，路径动作影响继续增加，但 performance 没有提升：
default/strict success delta 均为 `0.00`，3D RMSE 略退步 `0.0246 m`。

结论：继续无上限增加 P1 训练步数没有证据支持。

### 4.3 奖励尺度不是原因

X3 只把 `pc_ppo_scale` 从 `0.01` 改到 `1.0`，success delta 仍为 `0.00`。
这与 rsl-rl 在 `compute_returns` 阶段执行全局 advantage normalization 的实现一致：
单纯乘大奖励不会改变 policy gradient 的相对幅度。

结论：不要继续扫 `pc_ppo_scale`。

### 4.4 逐 mini-batch 优势归一化不是原因

X4norm 只打开 `normalize_advantage_per_mini_batch=true`。结果与 X1 几乎相同：
default success `+0.05`、strict `0.00`、路径零化 success drop `0.00`。

结论：归一化位置不是 P1 路径失效的主因。

### 4.5 actor 学习率是真瓶颈，但只解释了性能，不解释路径价值

X4lr 把 actor LR 从 `1e-5` 提高到 `1e-4`，得到本组最强的性能变化：

- default success `+0.10`；
- strict125 success `+0.25`；
- default 3D RMSE 基本不变，strict125 3D RMSE 改善约 `0.59 m`。

这说明 BC warmstart 后 actor 在旧 LR 下更新过慢，确实限制了 PPO 性能。但即使在
X4lr 中，`path-zero` 的 default success 下降仍是 `0.00`，3D RMSE 甚至略改善；
strict125 出现 success `+0.05` 下降但 RMSE 改善 `0.45 m` 的混合信号，不能支持
“路径输入带来稳定收益”。

## 5. 门禁是否过严

原门禁要求 `path-zero` 至少造成 `0.05` success 下降或 `0.10 m` 3D RMSE 恶化。
五个臂的 default `path-zero`：

- success 下降全部为 `0.00`；
- RMSE 变化在 `-0.0085` 到 `+0.0026 m` 之间，方向不稳定且远小于阈值。

这不是“差一点过门禁”，而是路径消融没有产生可检测的闭环影响。因此不能通过降低
门禁把 P1 写成成功。

## 6. 关于 saturation

用户已明确 saturation 不是本轮的重要指标，也不再作为任何门禁。按原始记录，X4lr 的
material saturation 最大增量为 `+0.0868`，只作为部署和执行器使用风险记录；它不参与
P1 的通过/否决，也不改变 P1 的负结论。后续汇总保留数值和 `exceeds_threshold` 标记，
但不再要求单独复核 saturation 或据此否决候选。

## 7. 最终结论

1. `alpha=1` 能显著打开路径支路，但仅凭这一点不能让路径输入产生闭环收益。
2. 4× 训练预算、100× 奖励尺度、逐 mini-batch advantage normalization 都没有解决
   路径归因问题。
3. actor LR `1e-4` 是一个真实有效的性能修复，可作为后续训练候选，但不能写成
   “路径分支有效”。
4. P1 在当前 ALOS command 输入结构下仍是 `no_path_attribution`。最可能的原因是
   `c_t` 已经把路径几何压缩成 heading/pitch/speed 指令，额外 8 维路径特征高度冗余；
   继续在 P1 上扫路径缩放、预算、奖励尺度或门禁不会产生新信息。
5. 若还要继续验证“路径信息本身是否有价值”，应进入 ALOS-free / 路径奖励阶段，
   删除或弱化 `c_t` 中的制导结果，让 actor 必须直接使用路径特征；这不是 P1 的
   单变量修补。

## 8. 权威产物

服务器根目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_S16_copy/artifacts/lowlevel/pc_p1opt_20261105/
```

各臂 summary：

```text
report/P1opt_alpha1_32/summary.{json,md}
report/P1opt_alpha1_128/summary.{json,md}
report/P1opt_alpha1_rewardscale1_32/summary.{json,md}
report/P1opt_alpha1_lr1e-4_32/summary.{json,md}
report/P1opt_alpha1_mbadvnorm_32/summary.{json,md}
```

历史 P 产物仍位于：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_abc_prework_20261105/
```

未覆盖、未重跑历史 P1/P2/P3。

## 9. 下一步

- 冻结本轮 P1 优化结论，不再扫 `pc_ppo_scale`、P1 训练预算或 P1 门禁。
- 将 actor LR `1e-4` 记入后续训练候选；只需复核多种子稳定性，saturation 继续作为
  信息性风险记录，不作为候选前置条件。
- 若继续研究路径价值，转到 ALOS-free / 路径奖励分支；只有在移除 `c_t` 冗余后仍无
  路径归因，才能把“路径信息本身无效”作为更强结论。
