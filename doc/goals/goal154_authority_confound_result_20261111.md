# Goal154 Phase 1b：残差权限配置混杂审计结果

**日期**：2026-11-11<br>
**状态**：`COMPLETED / EVAL_ONLY / DEV40 / HELDOUT_CLOSED`<br>
**上游**：Goal151 near10_2/shell、Goal152 train-time gate、Goal86 T2-SAC<br>
**数据根**：`artifacts/lowlevel/goal154_residual_authority_20261111/eval`<br>
**分析器**：`scripts/analyze_pc_goal153_eval_grid.py`

## 1. 审计问题

Goal151 曾观察到 `near10_2` 比 `full` 的 path 代价更低，并把收益口头归因给
“近段 residual”。但这个比较混淆了至少四个因素：

1. 位置：`near10_2` 关闭了 >10 m 的 residual；
2. 幅度：`near10_2` 把远段 effective residual 压到 0；
3. 区段定义：`near10_2` 实际包含 2–10 m 的 approach shell，不是只含 <=2 m；
4. checkpoint/环境：不同 run 可能使用不同 checkpoint、goal radius 或 success frame。

本审计固定同一批 3 个 Goal86 T2-SAC checkpoint、同一 Goal85 default/strict125
环境、同一 dev seeds `20–59`、同一 `eta_tight_at_done` success 契约，只改 residual
authority。

## 2. 变体

| variant | >10 m | 2–10 m | <=2 m | scale | 说明 |
|---|---:|---:|---:|---:|---|
| `full` | 1 | 1 | 1 | 1.0 | 全开 |
| `near10_2` | 0 | smooth 0→1 | 1 | 1.0 | 2–10 m ramp |
| `far_off_only` | 0 | 1 | 1 | 1.0 | 只关 >10 m |
| `near_only` | 0 | 0 | 1 | 1.0 | 只保留 terminal 2 m |
| `mid_only` | 0 | 1 | 0 | 1.0 | 只保留 2–10 m shell |
| `full_scale_0.5` | 0.5 | 0.5 | 0.5 | 0.5 | 只降幅度，不改位置 |
| `full_scale_0.75` | 0.75 | 0.75 | 0.75 | 0.75 | 只降幅度，不改位置 |

trace 已核对 `gate` 实际生效：`full` mean `1.0000`，`near10_2` `0.1092`，
`far_off_only` `0.2106`，`near_only` `0.0018`，`mid_only` `0.1066`。

## 3. 主结果（3 checkpoint × dev40 = 120 episodes）

| variant | default success | Δ vs zero | default path3d | Δ path | strict success | Δ strict | strict path3d | Δ path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero | 78/120 | 0 | 2.3375 | 0 | 42/120 | 0 | 2.3323 | 0 |
| full | 85/120 | +7 | 2.6152 | **+0.2777** | 46/120 | +4 | 2.6087 | **+0.2764** |
| near10_2 | 88/120 | +10 | 2.3367 | -0.0007 | 48/120 | +6 | 2.3309 | -0.0013 |
| far_off_only | 87/120 | +9 | 2.3347 | -0.0027 | 50/120 | +8 | 2.3283 | -0.0040 |
| near_only | 78/120 | **0** | 2.3374 | **0.0000** | 42/120 | **0** | 2.3322 | **0.0000** |
| mid_only | 88/120 | +10 | 2.3361 | -0.0014 | 48/120 | +6 | 2.3299 | -0.0023 |
| full_scale_0.5 | 87/120 | +9 | 2.4495 | **+0.1121** | 46/120 | +4 | 2.4443 | **+0.1121** |
| full_scale_0.75 | 85/120 | +7 | 2.5309 | **+0.1935** | 45/120 | +3 | 2.5249 | **+0.1926** |

配对 bootstrap 95% CI（default）：

| variant | Δsuccess CI | Δpath CI |
|---|---|---|
| full | [-0.025, +0.142] | [+0.193, +0.370] |
| near10_2 | [+0.025, +0.142] | [-0.003, +0.001] |
| far_off_only | [+0.008, +0.142] | [-0.009, +0.003] |
| near_only | [0, 0] | [0, 0] |
| mid_only | [+0.025, +0.150] | [-0.005, +0.002] |
| full_scale_0.5 | [+0.008, +0.142] | [+0.070, +0.156] |
| full_scale_0.75 | [-0.017, +0.142] | [+0.131, +0.261] |

注：3 个 checkpoint 共用同一组 dev seeds，CI 只作描述性配对区间，不当作独立 120 个新样本。

## 4. 逐 checkpoint 结果

default success / path3d：

| variant | seed 86201 | seed 86202 | seed 86203 |
|---|---:|---:|---:|
| full | 28 / 2.354 | 28 / 2.760 | 29 / 2.732 |
| near10_2 | 28 / 2.333 | 28 / 2.342 | 32 / 2.335 |
| far_off_only | 28 / 2.325 | 27 / 2.346 | 32 / 2.334 |
| near_only | 26 / 2.338 | 26 / 2.338 | 26 / 2.338 |
| mid_only | 28 / 2.330 | 28 / 2.344 | 32 / 2.335 |
| full_scale_0.5 | 28 / 2.323 | 30 / 2.530 | 29 / 2.496 |
| full_scale_0.75 | 27 / 2.334 | 29 / 2.642 | 29 / 2.617 |

success 差异主要由 seed 86203 贡献；path 惩罚在 full/scale 上跨 seed 更稳定。
因此不能把 `+10/120` 写成稳健的 success 提升，能稳健写的是：
**near/mid 把 full 的 path 惩罚收回，且不牺牲 success。**

## 5. 结论

### 5.1 `TERMINAL_ONLY_NULL`

`near_only` 与 zero 在 120 episodes 上逐项几乎完全相同：success `78/120` 对
`78/120`，path `2.3374` 对 `2.3375`，Δpath `0.0000`。说明：

> 只允许 <=2 m 的 actuator residual，对 terminal capture 没有可测收益。

不是 residual 幅度不够（<=2 m 内 learned residual RMS 仍约 `1.0` 归一化），而是
权限出现得太晚：进入 2 m 后只剩很少步数，residual 已无法改变 capture 结果。
这直接支持“先查动作参数化和 terminal capture，而不是继续调 terminal residual”。

### 5.2 `SHELL_AUTHORITY_CONFIRMED`

`mid_only`（只保留 2–10 m）与 `near10_2`、`far_off_only` 的 success/path 基本重合：
`88/120`、path `2.336`，path 与 zero 无差异。说明此前所谓“近段 residual 有用”
实际是 **2–10 m approach shell 有用**；`near10_2` 的标签具有误导性。

### 5.3 `MAGNITUDE_ONLY_INSUFFICIENT`

`full_scale_0.5` 把 success 拉回 `87/120`，但 path 仍为 `+0.112`；`full_scale_0.75`
为 `85/120`、path `+0.194`。所以：

- 降幅度可以救一部分 success；
- 但不能收回 path 惩罚；
- path 惩罚是位置性的，主要来自 >10 m 和/或 terminal 区的 residual。

### 5.4 vertical 不是本轮主收益

default vertical RMSE（越低越好）：zero 约 `1.0110`，`full` `1.0074`，
`near10_2` `1.0054`，`far_off_only` `0.9972`，`mid_only` `1.0010`，
`near_only` `1.0110`。full 的纵向收益只有约 `0.004 m`，却付出 `+0.278 m` path；
不值得。`far_off_only` 的纵向最好（约 `-0.014 m`），说明 shell + terminal 的组合
可能有一点纵向收益，但幅度很小。

### 5.5 不是 noisenet/高频噪声主导

对 `learned_residual_action` 按距离分箱做通道 RMS 和 lag-1 自相关：

| variant | >10 m RMS (3ch) | >10 m lag-1 | 2–10 m RMS | 2–10 m lag-1 |
|---|---|---|---|---|
| full | 0.73 / 0.82 / 0.75 | 0.39 / 0.25 / 0.39 | 0.76 / 0.86 / 0.76 | 0.14 / 0.03 / 0.29 |
| near10_2 | 0 / 0 / 0 | 0 | 0.50 / 0.55 / 0.50 | 0.17 / 0.05 / 0.25 |
| mid_only | 0 / 0 / 0 | 0 | 0.48 / 0.55 / 0.48 | 0.12 / 0.02 / 0.10 |
| full_scale_0.5 | 0.37 / 0.41 / 0.37 | 0.37 / 0.21 / 0.39 | 0.39 / 0.43 / 0.38 | 0.10 / -0.03 / 0.19 |

远段 residual 的 lag-1 自相关约 `0.25–0.39`，不是白噪声式的逐帧乱跳；三个归一化
通道的 RMS 也接近，没有单一通道异常主导。物理量上 RPM 通道数值最大，是因为
actuator scale 的 RPM 量纲远大于舵角，不代表策略只改 RPM。

因此“noisenet 主导”目前没有证据。更符合数据的解释是：

> 远段 residual 幅度大且持续，足以持续改变航迹；它不是高频抖动，而是有方向的
> 偏置性动作。问题在 authority 位置和幅度，不在噪声频率。

### 5.5.1 lag-1 口径复核（2026-11-11）

上述 `0.25–0.39` 是跨 episode 拼接后的 lag-1，会被 episode 边界和 reset 稀释。
按 `episode_id` 分组、在每条 episode 内去均值后重算，`full` 3 seeds 的结果是：

| 区段 | 通道 RMS（3 seeds 池化） | `|mean|/RMS` | 逐 episode lag-1 |
|---|---|---:|---:|
| 2–10 m | `0.755 / 0.859 / 0.756` | `0.51 / 0.40 / 0.30` | `0.846–0.936` |
| >10 m | `0.734 / 0.819 / 0.746` | `0.57 / 0.40 / 0.33` | `0.983–0.990` |

因此“不是白噪声”的结论不变，而且更强：远段 residual 在单条 episode 内高度持续，
同时有 `30–57%` 的幅度来自常数偏置。它不是 NoisyNet 式逐帧随机抖动，而是会持续
把航迹推偏的平滑动作。原表中的 `0.25–0.39` 不再作为正式数值使用。

另外，Goal86 T2-SAC 使用 SB3 标准 SAC MLP，不包含 `NoisyLinear`；评估时
`model.predict(..., deterministic=True)` 取的是策略均值。因此本轮讨论的 residual
本来就没有 NoisyNet 分量，不能把远段退化归因于 NoisyNet 探索噪声。

### 5.5.2 远段偏置与 path 退化的对应

把 `full` 的 >10 m residual 按 seed 拆开，偏置占比定义为
`||mean(residual)|| / RMS(||residual||)`：

| seed | >10 m residual RMS | 偏置占比 | default path3d |
|---|---:|---:|---:|
| 86201 | `1.126` | `0.225` | `2.354` |
| 86202 | `1.345` | `0.889` | `2.760` |
| 86203 | `1.451` | `0.694` | `2.732` |

86202/86203 同时有更高的远段 RMS 和更高的偏置占比，因此这组 seed 对比不能把
“幅度”和“偏置”完全拆开。但 scale sweep 显示：`full_scale_0.5` 的 >10 m RMS
`0.661`、偏置占比 `0.421`、path `2.4495`；`full_scale_0.75` 为 `0.994 / 0.428 /
2.5309`；`full` 为 `1.329 / 0.440 / 2.6152`。在偏置占比基本不变时，RMS 与 path
单调同增，说明幅度本身也有代价。

因此更准确的表述是：**远段 path 代价由 residual 的幅度和持续偏置共同决定，而不是
高频噪声**。Goal155 若仍出现 `PATH_TRADEOFF_REMAINS`，下一步应同时查远段幅度与
偏置的 credit/正则（例如零均值或偏置 barrier），而不是只做全局降 scale。

## 6. 裁决

`VERDICT = SHELL_AUTHORITY_CONFIRMED / TERMINAL_ONLY_NULL / MAGNITUDE_ONLY_INSUFFICIENT`

后续不应再笼统比较“near vs full”，而应固定：

- 训练和评估都记录 `shell_authority`（2–10 m）与 `terminal_authority`（<=2 m）
  两个独立开关；
- 主线优先 `mid_only` 或 `far_off_only`，不要再把 `near_only` 当成候选；
- 若纵向 reward 要扩展到其他路段，先看它能否在 `mid_only` 上改善 vertical，
  再看 `full`/`far_off` 是否还能保持 path；
- 不再用“terminal residual 更大”解释 capture 失败；那属于动作参数化/terminal
  guidance 问题。

## 7. 复现命令

```bash
# 42 个 eval：3 seeds × 2 scenarios × 7 variants
GPUS="3 4 6" MAX_PARALLEL=3 SEEDS="86201 86202 86203" EVAL_SEED_COUNT=40 \
VARIANTS="full near10_2 far_off_only near_only mid_only full_scale_0.5 full_scale_0.75" \
ROOT=artifacts/lowlevel/goal154_residual_authority_20261111/eval \
LOG_ROOT=logs/goal154_authority_full \
bash scripts/run_goal154_authority_eval.sh

python scripts/analyze_pc_goal153_eval_grid.py \
  --root artifacts/lowlevel/goal154_residual_authority_20261111/eval \
  --zero-tag zero \
  --tags full near10_2 far_off_only near_only mid_only full_scale_0.5 full_scale_0.75
```

zero 基线复用 200k scratch 的 `--action-mode zero-residual` dev40；它不依赖
checkpoint，环境、seeds、success 契约与 authority 矩阵一致。

## 8. 多指标口径修正（不改变主结论）

2026-11-11 统一分析器升级后重新生成了 `goal154_authority_summary_v2.md`：

- `terminal_kind` 已从数字编码改为 `GOAL / GOAL_PLANE_MISS / TIME_LIMIT / safety` 名称；
  default 的 `full` 为 85 GOAL + 35 plane miss，不再是全 0；
- route-table 的 pooled active-step RMSE 与主表 episode-mean path3d 已明确分开；
- 旧 `action sat` 显示 `full_scale_0.5/0.75 = 0%`，是因为当时统计的是乘以 global
  residual multiplier 后的有效动作；新报告区分 `policy-head sat pre-scale` 与
  `residual sat post-scale`。旧 trace 没有 `policy_action_raw`，因此 pre-scale 显示 `NA`，
  不能把缺失当 0%；
- actuator/material saturation 对旧 trace 也显示 `NA`，因为当时没有写入饱和 mask。

这些修正只影响指标解释，不改变 `SHELL_AUTHORITY_CONFIRMED / TERMINAL_ONLY_NULL /
MAGNITUDE_ONLY_INSUFFICIENT` 的裁决。
