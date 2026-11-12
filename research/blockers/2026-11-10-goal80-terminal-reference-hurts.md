# Goal80：terminal reference 前移导致 plane-miss 型退化

日期：2026-11-10<br>
关联：Goal80 / Goal71 / Goal67 / Goal64<br>
状态：`TERMINAL_REFERENCE_HURTS`<br>
决策：`pivot`，停止 goal-tangent 前移，转只读 termination/capture 审计

## 1. 现象

在 Goal71 limiter-on baseline 上，只把 terminal reference 沿 goal tangent
前移 `1.0 m`，其余配置不变，闭环结果显著变差：

| scenario | baseline true | candidate true | original lost | recovered | threshold | newly lost | true delta | plane miss delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 97 | 80 | 6 | 0 | 3 | 18 | -17 | +0.120 |
| strict125 | 44 | 38 | 7 | 0 | 3 | 10 | -6 | +0.060 |

timeout 和 safety delta 都是 `0`；pre-goal RMSE 只恶化 `1.23%/1.33%`。

## 2. 证据

- extension 不是空扰动：smoke 中 extension-on 改变 `37` 个 trace field、
  `29667` 个元素；original-lost 子集的视线角中位数为 `6.60°/7.62°`，
  terminal command changed seed fraction 为 `1.000/1.000`。
- 零扩展等价性通过：policy 与 zero-residual 的 episodes 全等，trace diff
  field/element/max abs diff 均为 `0`。
- baseline-true 转 candidate-non-true 的 18/10 个 default/strict125 episode 中，
  有 `17/8` 个被标成 `GOAL_PLANE_MISS`，只有 `1/2` 个变成 false success；
  没有 timeout/safety。
- plane-miss rate：default `0.36 -> 0.48`，strict125 `0.71 -> 0.77`。
- strict125 vertical offset paired median delta `+0.054 m`，越过预注册
  `+0.05 m` 不恶化门限。
- heading command rate P90 仍为 `0.524 rad/s`，exceed fraction 为 `0`；
  heading tracking error 反而改善 `0.186/0.207 rad`。因此不是 limiter 失效，
  也不是 tracking 没变化，而是 tracking 改善没有转化为 capture。

## 3. 已排除

| 解释 | 状态 | 理由 |
|---|---|---|
| 零扩展实现不等价 | 排除 | smoke episodes/trace 逐元素全等 |
| 扰动太弱、没有进入 terminal command | 排除 | 37 个 trace field 改变；几何角 `6.60°/7.62°`；command changed fraction 1.0 |
| NoisyNet 采样主导 | 排除 | deterministic mean 评估；zero-residual 仍有 default true `-1`、两个 scenario plane miss `+0.075` |
| residual scale 主导 | 排除 | policy 与 zero 两个 arm 都朝同方向变差；zero 没有 residual 也 plane miss 增加 |
| timeout 或 safety 恶化 | 排除 | 两个 scenario 的 timeout/safety delta 都为 `0` |
| command rate limiter 失效 | 排除 | heading P90 `0.524 rad/s`、exceed fraction `0` |
| 样本效率不足 | 无证据 | 这是冻结 checkpoint 的完全配对评估，不是训练曲线或预算问题 |

## 4. 可证伪 hypothesis

### H80-R1：新增丢失来自 plane-miss guard 先于 tight capture 触发，而不是目标不可达

预测：baseline-true 且 candidate-lost 的 episode 中，candidate 在 goal plane
上的 signed offset 先达到 `>=0`，而 terminal `p_nav` 与 goal 的最小距离仍大于
`r_tight`；如果用记录轨迹重新判定“不得在 tight radius 外触发 plane-miss”，
至少一部分 episode 会保留到 tight capture。

当前证据：支持前半部分。`17/18` 与 `8/10` 新增丢失已经是 plane miss；还需用
Goal80 原始 trace 做离线终止复盘，确认 guard 的具体触发时刻和最小距离。

### H80-R2：前移导致穿越点横向/纵向误差更大，进一步前移或缩小横向偏差会改善

预测：candidate 在 plane crossing 时的 perpendicular offset 比 baseline 大，
且 plane-miss episode 的 crossing offset 与 `r_tight` 的差距可解释 true/false
翻转；若把 terminal reference 前移量继续增大，效果应单调改善。

当前结果：第一部分的 `horizontal offset` 配对中位变化很小（`-0.006/+0.023 m`），
但 plane miss 大幅上升，说明不是简单“横向偏差变大”；“继续前移会改善”与
default/strict 同时退化矛盾。该 hypothesis 暂不支持，且禁止用扫 extension 值
来救 Gate。

### H80-R3：问题在 termination/capture 语义，而不是 terminal reference 方向

预测：即使完全 zero-residual，前移仍会增加 plane miss；反向或更保守的
termination 规则能在离线记录上恢复部分 true，而不需要改 residual/reward。

当前结果：支持 zero-residual 方向（default true `17->16`、strict 持平，
plane miss 都 `+0.075`）。下一步应验证离线 termination 规则，不得直接启动新
训练。

## 5. 最便宜的 falsification experiment

不重跑闭环，使用现有 10 个 Goal80 run 的 `trace.npz` 和 `episodes.json`：

1. 对 baseline-true / candidate-lost 的 episode，按 step 重放到首次
   `signed_offset >= 0` 的行；
2. 记录当时的 `p_nav/eta` 到 goal 的距离、horizontal/vertical offset、
   `guidance_terminal_phase`、`goal_plane_miss` 和 tight capture 状态；
3. 用同一份记录计算“延迟/禁止 plane-miss guard 或在 tight capture 时终止”的
   离线结果，报告可恢复 true 数、false success 变化和 safety 风险；
4. 只把这个只读审计写成新 goal，不改变 reward、gamma、checkpoint、seed 或
   held-out。

如果离线规则不能恢复 true，下一步才查动作参数化或控制器带宽；仍不换 SAC/
REDQ/DroQ。

## 6. 结论与 next

- Gate 结论：`TERMINAL_REFERENCE_HURTS`，H80-A 不支持；H80-B/C 支持；
- 这是终止几何/捕获语义失败，不是 NoisyNet 或 residual 幅度主导；
- 停止 extension 扫值、terminal window 扫值和任何 residual/reward 重开；
- 新开 Goal81 只读 termination/capture 审计，使用已有 trace，先不改主算法；
- 若 Goal81 证明确实是 guard 语义，再预注册单因素 termination arm；训练型 arm
  才做 `num_envs/batch` 吞吐基准和 8 卡 seed 并行。

## 7. 证据

- `doc/goals/goal80_result_20261110.md`
- `doc/goals/goal80_terminal_reference_extension_20261110.md`
- `research/paper_reading/2026-11-10-goal80-terminal-reference-plane-miss.md`
- `research/decisions/2026-11-10-goal80-terminal-reference-hurts.md`
- `artifacts/lowlevel/goal80_terminal_reference_20261110/analysis/goal80.json`
- `artifacts/lowlevel/goal80_terminal_reference_20261110/analysis/equivalence.json`
- 服务器原始输出：
  `/data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/artifacts/lowlevel/goal80_terminal_reference_20261110/`
