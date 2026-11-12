# Blocker：Goal117 末段失败机制不唯一，但 2 m entry geometry 与 reward credit 已分离

**日期**：2026-11-11<br>
**Goal**：Goal117 terminal closure + action/reward attribution<br>
**状态**：`MIXED_TERMINAL_MECHANISM`<br>
**触发条件**：预注册 Gate 未达到单一机制 `>= 8/11`；属于“现有 hypothesis 无法
解释全部 failure，且本地 ablation 无法区分”的 hard blocker。

## 1. 现象

Goal117 在 Goal116 R100 的严格同协议 trace 上完成 reward breakdown 重跑：

```text
success = 20/40
timeout = 0
safety  = 0
R100_RTRACE vs Goal116 R100 parity = true (atol=1e-10)
```

11 个预注册 primary failure（Goal111 PREENTRY_GEOMETRY_BAD 子集）的标签：

| label | count | seeds |
|---|---:|---|
| `BASE_AUTHORITY_LIMITED` | 1 | 27 |
| `RESIDUAL_DIRECTION_WRONG` | 3 | 25, 49, 50 |
| `UNCLASSIFIED` | 7 | 26, 29, 31, 33, 36, 51, 54 |

20 个 failure 的全集还多出 7 个 `APPROACH_DEFICIT`（closest >= 2 m）：

```text
32, 35, 39, 41, 47, 52, 58
```

没有单一标签达到 `8/11`，因此 Gate 判 `MIXED_TERMINAL_MECHANISM`。

## 2. 关键证据

### 2.1 2 m entry geometry 已经分离（最便宜 falsification）

| 指标 | success median | failure median | Cliff's delta |
|---|---:|---:|---:|
| `entry2_vertical_median5_m` | 0.412 | 1.339 | 0.854 |
| `entry2_pitch_median5_rad` | 0.198 | 0.515 | 0.769 |
| `entry2_heading_median5_rad` | 0.131 | 0.223 | 0.285 |
| `entry2_to_closest_s` | 0.51 | 0.68 | 0.412 |

计数：

```text
success 在 2 m entry geometry bad = 3/20
all failure 在 2 m entry geometry bad = 11/13
11-seed primary failure = 11/11
```

因此 11 个 primary failure 不是“进入 2 m 后 residual 方向突然坏掉”，而是
**进入 2 m 时 vertical/pitch 已经超过 Goal111 的冻结几何判据**。

### 2.2 terminal reward credit 被 dense 项压住

| 指标 | success median | failure median | Cliff's delta |
|---|---:|---:|---:|
| `terminal_share_last2s` | 0.594 | -0.202 | -1.000 |
| `terminal_share_episode` | 0.116 | -0.018 | -1.000 |
| `dense_terminal_ratio_last2s` | 0.684 | 3.885 | 1.000 |

failure 的最后 2 s terminal 项净和为负，dense path/command 项绝对值约为
terminal 的 3.9 倍；success 则 terminal 占优。这与 Goal107 的 credit 结论一致，
但 Goal117 首次在逐 step reward breakdown 上确认。

### 2.3 base authority 不是判别因素

```text
base_saturation_fraction: success 0.990, failure 1.000
```

两组末段都几乎一直饱和，因此 `BASE_AUTHORITY_LIMITED` 不能解释 failure 集合；
把它作为主标签会把 success 也一起标进去。

### 2.4 residual 方向不是单一主因

```text
alignment_heading: success 0.301, failure 0.369
alignment_pitch:   success 0.223, failure -0.136
```

heading 方向在 failure 中反而略好；pitch 方向在 failure 中更差，但 20 个
success control 里也有 7 个被同一判据标成 `RESIDUAL_DIRECTION_WRONG`。因此
方向指标有信号但不具备“只标 failure”的判别力。

## 3. 已排除项

- **不是 NoisyNet**：Goal108 扫描到 0 个 `rho/sigma/noise` 参数；当前是
  frozen-base deterministic eval。
- **不是 checkpoint 错配**：checkpoint SHA 与 Goal116 R100 一致，trace parity
  通过。
- **不是 guidance pitch rate limiter**：Goal116 四臂无因果效应。
- **不是 base authority 不足**：success/failure 都接近全饱和。
- **不是“残差幅度不够”**：failure 的 `residual_magnitude_ratio` 中位数
  0.877，success 1.121；failure 反而更小，但方向混合。
- **不是“reward 没有 terminal 信号”**：success 的 terminal share 0.594，
  信号存在；问题是 failure 的 dense 项把它压成负值。

## 4. 可证伪 hypotheses

### H117b-A：2 m entry 时已经不可达，后续 residual 无法补救

- 成立：对 primary failure 做 kinematic reachability bound，若在 2 m entry
  的 vertical/pitch 误差和剩余时间下，所需平均速度/角速度超过 actuator 包络的
  可承受范围，则 `UNREACHABLE_AT_2M`。
- 否定：多数 failure 在 2 m entry 仍可达，失败发生在 2 m 内的控制/信用。
- 最便宜证伪：只用同一 trace 的 `entry2_to_closest_s`、vertical/pitch error、
  base actuator 包络做静态可达性计算，不新增 rollout。

### H117b-B：2 m 内可达，但 dense reward 把策略拉向“路径好看”

- 成立：多数 failure 在 2 m entry 可达；且 failure 的 terminal share < 0.2、
  dense/terminal > 3，同时 residual 与 dense path 项的相关性高于与 closure。
- 否定：terminal share 不低，或 residual 与 dense 项无相关性。
- 最便宜证伪：在现有 trace 上计算 residual 与 `reward_path_*` /
  `reward_terminal*` 的逐 step 相关，不训练。

### H117b-C：2 m 内可达、方向正确，但 PPO 样本/更新不足

- 成立：2 m entry 可达；residual alignment >= 0.5；closure 仍 < 0.3 m/s；
  且同协议 PPO 增加 episode 覆盖后仍学不出增量。
- 否定：方向不对、geometry 不可达、或 reward credit 主导。
- 最便宜证伪：先做 H117b-A/B 的只读分析；只有两者都否定，才预注册 PPO
  样本预算对照。仍不直接开 SAC。

## 5. 当前决策

```text
decision = continue
next = Goal118 approach-phase reachability + terminal-credit decomposition
```

不换算法、不改 reward/terminal/success/goal radius、不打开 held-out。Goal118
必须先做只读 reachability，再决定是否值得改 reward 或动作参数化。
