# Goal126：SAC P-A 能否突破 fixed-bias 的 24/40 上限

**日期**：2026-11-11<br>
**状态**：`COMPLETED_SCREEN / SAC_PA_HARM / HELDOUT_CLOSED`<br>
**上游**：Goal125 `PPO_BIAS_MIXED / PPO_BIAS_CEILING`；Goal124 `BIAS_GUARDRAIL_PASS`<br>
**性质**：预注册算法对照 screen；只把 PPO 换成 SAC，不改任务契约。

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal126 --goal-doc doc/goals/goal126_sac_pa_bias_prior_20261111.md --strict --require-result
```

测量开始后不得修改训练 seed、预算、eval dev40、success frame、goal radius、
checkpoint 选择规则或 Gate 阈值。

## 1. 背景与证据

Goal125 的固定偏置控制为 strict `24/40`、default `33/40`。PPO 结果：

| arm | strict mean | default mean | 结论 |
|---|---:|---:|---|
| Z15 `(0,0,0)` | 20.0 | 33.0 | 无偏置时 PPO 丢 4 个 success |
| B15 `(0,-0.25,0)` | 24.0 | 33.0 | 精确停在控制组 |
| B20 `(0,-0.25,0)` | 24.0 | 33.0 | 精确停在控制组 |
| BP3 `(0,-0.25,0)` | 23.5 | 33.0 | reward/advantage 改动无增益 |

PPO 的 deterministic residual 只有 base L2 的 `0.7%–4.0%`，`approx_kl` 约 `1e-4`。
Goal91 已排除低噪声，Goal92 已排除更新调度，Goal120 已判
`PPO_OPTIMIZATION_LIMITED`；Goal121/122 已排除“动作通道完全无权限”。

因此本 Goal 回答一个明确问题：

```text
固定 stern-0.25 先验已经给出 24/40；
换用 off-policy SAC，能否在相同 1M transitions 内超过它？
```

这不是“SAC 一定更好”的声明，也不把 PPO 的失败归因为 NoisyNet。当前 PPO
没有 NoisyNet，初始化与 Goal119/Goal108 的 63/63 tensor parity 已通过。

## 2. 假设

### H126-A：SAC 能在同协议下超过 fixed-bias control

- 成立：8 个 1M screen seed 的 strict mean `>=26/40`，至少 4 个 seed `>=26`，
  且每个 seed default `>=31/40`。
- 失败：strict mean `<=24.5/40`，或没有任何 seed 超过 control。
- 最便宜证伪：1M × 2 seeds 的 init-parity + screen；若明显退化，不扩 5M。

### H126-B：Goal108 SAC 的退化来自未对齐的 path 初始化

- 成立：使用 `bc_path_dim=8`、独立 endpoint4、加载 BC encoder 的 P-A SAC 初始
  zero-residual 精确复现 `24/33`，且 trained deterministic action 不出现
  `abs_max≈1` 的全通道饱和。
- 失败：P-A SAC 仍出现同样的动作饱和或 init parity 不成立。
- 最便宜证伪：init-only checkpoint + dev40 zero-residual；训练前必须完成。

### H126-C：SAC 也停在 fixed-bias 上限，瓶颈不在算法族

- 成立：SAC strict mean `<=24.5/40`，default 不掉，且 residual 未提供新的
  success seed。
- 失败：至少一个 screen 或 formal arm 达到 H126-A。
- 最便宜证伪：完整 8-seed 1M screen。

### H126-D：SAC 会把固定偏置破坏成大体量饱和动作

- 成立：任一 seed timeout/safety 非零，或 default 低于 control 2 个以上，
  或已执行 residual 的 `abs_max_mean > 0.8` 且 success 大幅下降。
- 失败：所有 seed 的 timeout/safety 为 0，default 不低 control−2。
- 最便宜证伪：先 2-seed screen，再决定是否用满 8 卡。

## 3. 架构与动作契约（冻结）

必须与 Goal86 P-A / 当前 PPO 的 z_t 契约一致，禁止 Goal108 的 path12 随机初始化：

| 组件 | 冻结值 |
|---|---|
| observation | state12 + path12 + command3 + current3，history 10 |
| path 拆分 | `bc_path_dim=8` + endpoint4 |
| state encoder | `20→128→128→64`（60 active + 4 reserved） |
| path encoder | `8→64→64`，加载 BC encoder |
| endpoint encoder | `4→32→64`，零末层 |
| current encoder | `3→64→60` + 4 reserved |
| actor | `131→256→128→3`，`mu` 零初始化 |
| critic | `134→256→128→1`，twin Q |
| 随机性 | SB3 SAC `log_std_init=-3.0`，评价 deterministic mean |
| NoisyNet | 禁止；不存在 `rho/sigma/noise` 参数 |
| 环境 base | Goal86 frozen residual BC，SHA `80024e...f9dd` |
| base bias | `(0,-0.25,0)` |
| residual terminal gate | 与 Goal125 fixed-bias control 一致：`10 m -> 0`、`3 m -> 1` |
| eval scale | actuator residual scale `2.00` |

`base_residual_bias` 与 SAC 学习残差一起经过 terminal gate 后，再加到 frozen BC
基座上，语义为：

```text
executed = frozen_BC + 2.0 * gate * (SAC_action + base_bias)
```

`base_residual_bias` 不得进入 SB3 policy 构造器、不得作为可训练参数，也不得
重复叠加到 BC checkpoint。`gate` 只作用于学习残差和 bias，不作用于 frozen BC；
这与 Goal125 fixed-bias control 的 `BatchTorchPCFrozenBCResidualVecEnv` 完全一致。

## 4. 数据与隔离

| 项目 | 值 |
|---|---|
| screen 训练 seed | 86201–86208，8 卡各 1 个 |
| screen 预算 | 每 seed `1,000,192` transitions；256 envs，batch 256，UTD 0.125 |
| formal 预算（仅通过后） | 每 seed `5,000,000` transitions；4 seeds |
| eval dev seeds | 20–59，40 episodes/scenario |
| eval scenarios | strict `goal_radius=1.25`、default `2.0` |
| success frame | `both` |
| control | Goal125 fixed-bias control，strict 24/default 33；必须再次复现 |
| held-out | `100–149` 关闭 |

训练超参数固定为 Goal86 P-A SAC 基线：

```text
actor_lr=3e-4, critic_lr=3e-4, gamma=0.995, tau=0.005
learning_starts=20_000, batch_size=256, buffer_size=500_000
ent_coef=auto, target_entropy=-3
```

## 5. 执行计划

- [x] Step 0：实现 SAC `base_residual_bias` 与 terminal gate，加入 config、training manifest 和
      evaluation 的严格一致性校验；运行单元测试和 1-iteration smoke；
- [x] Step 1：生成 init-only checkpoint，确认 `bc_path_dim=8`、endpoint4、
      BC encoder 已加载、`mu=0`、无 NoisyNet；
- [x] Step 2：用 init-only policy + bias 在 dev40 复现 strict 24/default 33；
      若不精确复现则 `INVALID`，不改 Gate；
- [x] Step 3：8 卡并行 8×1M screen；
- [x] Step 4：16 个 deterministic dev40 评估；旧重复 scale 评估已归档，修正后重跑；
- [x] Step 5：按 Gate 判读为 `SAC_PA_HARM`，不扩 4×5M formal；
- [x] Step 6：写 result/blocker/decision、LEDGER、STATE；
- [x] Step 7：运行 checker 通过；不自动 commit/push。

## 6. Gate（1M screen）

### Init parity 审计（训练前）

第一次 parity 为 strict `22/40`、default `29/40`。逐输入对照后确认：

- PPO `policy.base_action()` 与 SAC `PCBCPolicy.act_inference()` 的基座动作
  最大差 `2.4e-7`，state/path/command/current 归一化均一致；
- 差异来自 PPO fixed-bias control 的 `BatchTorchPCFrozenBCResidualVecEnv`：
  `bias` 和学习残差经过 `10 m -> 0`、`3 m -> 1` 的 terminal gate，而 SAC
  adapter 最初把 bias 全程无衰减施加；
- 补齐 gate 后，zero-residual dev40 精确复现 strict `24/40`、default
  `33/40`，timeout/safety 为 0。该修复不改变算法、success、goal radius、
  seed、预算或 Gate 阈值，只是恢复控制组已有的动作契约。

证据：

```text
artifacts/lowlevel/goal126_sac_pa_bias_20261111/init_parity_eval/strict/summary.json
artifacts/lowlevel/goal126_sac_pa_bias_20261111/init_parity_eval/default/summary.json
artifacts/lowlevel/goal126_sac_pa_bias_20261111/network_audit.json
```

网络自检同时确认：BC state/path encoder 参数与 Goal26 源 checkpoint 完全一致
（max diff `0.0`），endpoint4 与 current encoder 末层全零，actor `mu` 全零，
`log_std=-3.0`，无 `rho/sigma/noise` 参数，features dim 为 `131`。

### 有效性

| 检查 | 要求 |
|---|---|
| init parity | zero-residual strict `24/40`、default `33/40` |
| 架构 | `bc_path_dim=8`；endpoint4；BC encoder loaded |
| 参数 | `mu=0`；`log_std=-3.0`；无 NoisyNet |
| 数值 | 全部 finite；无 NaN/Inf |
| 安全 | timeout=0，safety=0 |
| 隔离 | 未打开 held-out |

### 判读

| 观察 | 判读 | 下一步 |
|---|---|---|
| strict mean `>=26`，至少 4/8 seed `>=26`，每 seed default `>=31` | `SAC_PA_GAIN` | 扩 4 seeds × 5M formal |
| strict mean `<=24.5`，default 不低 control−2，安全为 0 | `SAC_PA_CEILING` | 不再换算法；回到动作参数化/reward/terminal capture |
| strict mean `<22`，或任一 default `<=30`，或 safety>0，或 timeout>0 | `SAC_PA_HARM` | 写 blocker；不扩 formal |
| 介于以上之间 | `SAC_PA_MIXED` | 只做配对方向/残差审计，不扩 formal |
| init/架构/数值/control 不成立 | `INVALID` | 修实现后重跑，不改阈值 |

## 7. 停止与自治边界

- 不因一次 screen 失败扫 seed、lr、UTD 或 entropy；
- 不把 `SAC_PA_CEILING` 解释成“需要 REDQ/DroQ”；只有新的样本效率证据才考虑；
- 不打开 held-out；不用 validation 结果直接报 formal；
- 不自动 commit/push；禁止 `git add -A`。

## 8. Git

只 add 本 Goal 相关文件：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal126 \
  --goal-doc doc/goals/goal126_sac_pa_bias_prior_20261111.md \
  --strict --require-result
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存。

## 9. Goal 提示词

```text
你现在执行 Goal126：SAC P-A 同协议对照。

先读：
- doc/goals/goal126_sac_pa_bias_prior_20261111.md
- doc/goals/goal125_result_20261111.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md
- doc/goals/goal108_result_20261111.md

硬约束：
1. 不改 success、goal radius、terminal capture、eval split；
2. 不打开 held-out 100–149；
3. 必须使用 Goal86 P-A：bc_path_dim=8、endpoint4、加载 BC encoder、mu=0；
4. 禁止 Goal108 path12 随机初始化 SAC；
5. base_residual_bias=(0,-0.25,0) 只在环境动作组合处加入；
6. init parity 不通过就停止；不扩 5M；
7. 8 个 1M screen 任务并行用 8 张 4090；
8. Gate 记录前不改 seed、预算、阈值或 checkpoint 选择规则。

执行顺序：
- 实现 bias 契约与测试；
- init-only checkpoint 和 zero-residual dev40 parity；
- 生成 8 个训练配置；
- 8 卡并行训练；
- 16 个 strict/default dev40 评估；
- Gate、result、blocker/decision、LEDGER、STATE、checker。
```
