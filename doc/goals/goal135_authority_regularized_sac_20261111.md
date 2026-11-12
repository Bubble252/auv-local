# Goal135：authority-regularized SAC 的跨 seed screen

**日期**：2026-11-11<br>
**状态**：`AUTHORITY_SAC_HARM / NO_FORMAL / RESULT_RECORDED / HELDOUT_CLOSED`<br>
**上游**：Goal134 `SINGLE_SEED_ONLY / SAC_STRESS_HARM_AT_HIGH_STRESS`；Goal126 `SAC_PA_HARM`；Goal129/130 `ACTION_PARAM_NO_EFFECT`<br>
**性质**：保持 SAC 与现有 z_t 架构，只加入训练侧 learner residual authority 正则；先 1M × 8 seeds screen，不直接上 formal。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal135 --goal-doc doc/goals/goal135_authority_regularized_sac_20261111.md --strict --require-result
```

测量开始后不得修改训练 seed、预算、正则形式、rho/beta、eval dev40、success frame、
goal radius、terminal 契约或 Gate 阈值。实现和 smoke 只用于验证接线与数值有限性，
不得用正式 eval 结果反过来挑选 rho/beta。

### 1.1 预注册前 engineering calibration

在正式 screen 开始前，使用 seed `86201` 和正式 1M 预算做了四组工程校准，只验证
hinge 是否激活、authority 是否随 rho 形成剂量顺序，以及训练是否数值有限。该批结果
**不是正式 eval，不参与 Gate**，也没有用于修改 rho/beta：

| arm | rho | RMS last | RMS mean | penalty mean | train success | transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| `disabled` | — | — | — | — | `307/713` | `435.0` |
| `R020` | `0.20` | `0.1830` | `0.1291` | `2.89e-5` | `347/684` | `388.7` |
| `R025` | `0.25` | `0.2426` | `0.1615` | `3.01e-5` | `328/672` | `390.9` |
| `R030` | `0.30` | `0.2945` | `0.1838` | `2.78e-5` | `328/660` | `383.6` |

结论：hinge 在 1M 训练中确实激活；三个开启正则的 arm 最终 RMS 随 rho 单调升高；
`R020` 已把 deterministic residual RMS 压到 rho 以下。训练 success 只作接线诊断，
不能替代 dev40 闭环评估。基于此冻结原定 `R030/R025/R020`、`beta=2.0`、
`86201–86208` 和 1M screen 预算。

## 1. 背景与要回答的问题

Goal134 的 90-run 压力测评给出了两个同时成立的事实：

```text
压力有效：zero strict 在 current15/current20/combined20 为 14/16/13（nominal 24）
但 SAC 没有跨 checkpoint 优势：只有 86201 在 current15/plant20 获益
current20/combined20：0/8 checkpoint 获益，8/8 harm
```

Authority dose 进一步显示：

```text
86202/86204 的 harm 大部分随 residual scale 降低而消失
86201 在 scale 0.25 仍有 current15 strict/default +2/+2
86201 的 learned residual RMS 是 8 个 checkpoint 中最小
```

所以本 Goal 只回答：

> 训练时直接约束 learner residual 的 RMS，能否把“小 authority + 正确方向”从
> 86201 一个 seed 扩展到多个 seed，并在 nominal/current15 同时保留收益？

这不是重开 eval-only cap。Goal129/130 已证明对旧 checkpoint 做事后缩放只能
减少伤害，不能恢复跨 seed 优势；本 Goal 改的是训练目标，让策略在训练中学到
更小的 residual authority。

## 2. 假设与可证伪判据

### H135-A：训练侧 authority 正则带来跨 seed 增益

- 机制：SAC 在 1M 内可以找到方向，但 Q 目标没有惩罚“残差替代基座”；给
  deterministic mean action 加 RMS hinge 后，多个 seed 会保留 86201 式的小修正。
- 成立：至少一个 arm 满足
  `>=4/8` seed nominal strict `>=26` 且 default `>=33`，
  同时 `>=4/8` seed current15 strict `>=16` 且 default `>=24`。
- 失败：没有任何 arm 同时满足 nominal 和 current15 两个条件。
- 最便宜证伪：3 arms × 8 seeds × 1M；先看 nominal，再跑 current15。

### H135-B：authority 降低但方向不足会退化成 zero

- 机制：若策略没有可泛化的修正方向，正则会把动作压小，success 回到 zero。
- 成立：某 arm 的训练后 action RMS 明显降低，但 nominal strict 均值 `<=24.5`，
  current15 strict 均值 `<=14.5`，且没有 seed 超过 zero+2。
- 失败：至少一个 arm 在 authority 降低的同时出现 `>=2` 个 seed 的 strict `>=26`。
- 最便宜证伪：同一次 screen 的 nominal/current15 paired 结果和 authority 统计。

### H135-C：存在 authority 剂量窗口，而不是越强越好

- 机制：rho 太小会把有用的接近段修正一起压掉；rho 太大则等于没有正则。
- 成立：三个 arm 的 residual RMS 随 rho 单调变化，但 success 在中间档最好，
  或至少出现“最强档低于较弱档”的可解释顺序。
- 失败：三个 arm 的 success 与 RMS 都没有单调/剂量关系，差异落在 seed 噪声内。
- 最便宜证伪：同一批 8 个训练 seed 的三个 rho 档。

### H135-D：86201 的收益只是 seed 特例

- 机制：Goal134 的唯一增益来自单个初始化，authority 正则不能改变 seed 方差。
- 成立：只有 86201 的对照 seed 达标，其余 7 个 seed 仍在 zero 附近或更低。
- 失败：`>=4/8` seed 达标。
- 最便宜证伪：逐 seed 报告，不看 aggregate mean。

## 3. 方法：只改 actor loss 的 authority 正则

训练目标保持 SAC：

```text
L = L_SAC + beta * mean(ReLU(RMS(tanh(mu(s))) - rho)^2)
RMS(a) = sqrt(mean_i a_i^2)
```

其中 `mu(s)` 是 SAC actor 的 deterministic mean 输出，`tanh(mu)` 是实际动作空间；
`L_SAC = ent_coef * log_prob - min(Q1,Q2)` 不变。正则只进入 actor loss，不进入
reward、environment step、critic target 或 evaluation。

冻结：

| 组件 | 冻结值 |
|---|---|
| 算法 | SAC；不换 PPO，不开 REDQ/DroQ |
| observation | state12 + path12 + command3 + current3，history 10 |
| path 拆分 | `bc_path_dim=8` + endpoint4 |
| state encoder | `20 -> 128 -> 128 -> 64`（60 active + 4 reserved） |
| path encoder | `8 -> 64 -> 64`，加载 BC encoder |
| endpoint encoder | `4 -> 32 -> 64`，零末层 |
| current encoder | `3 -> 64 -> 60` + 4 reserved |
| actor | `131 -> 256 -> 128 -> 3`，ELU，`mu` 零初始化 |
| critic | twin Q，`134 -> 256 -> 128 -> 1` |
| 随机性 | `log_std_init=-3.0`，评价 deterministic mean |
| NoisyNet | 禁止 |
| base | Goal86 frozen residual BC，`base_residual_bias=(0,-0.25,0)` |
| terminal gate | 10 m -> 0，3 m -> 1，与 Goal126/134 一致 |
| eval scale | `actuator_residual_scale=2.0`，`--residual-scale 1.0` |
| reward / terminal / success / radius | 完全沿用 Goal126/134 |

正则实现要求：

- 默认关闭时与 Goal126/134 完全等价；
- 训练 manifest 必须记录 `form/rho/beta/target`；
- 记录训练期 deterministic action RMS 和 penalty 均值；
- 不把 penalty 写进环境 reward，不改变环境返回；
- 评价仍用 `SAC.load` 的 deterministic mean，不依赖训练时的自定义类。

## 4. 数据与隔离

| 项目 | 值 |
|---|---|
| 训练环境 | Goal125 `B15.json`，与 Goal126 完全相同 |
| 训练 seeds | `86201–86208`，与 Goal126 配对 |
| screen transitions | `1,000,192` per seed（与 Goal126 一致） |
| eval seeds | dev `20–59`，40 个 |
| eval scenarios | nominal + current15，strict/default |
| success frame | `both` |
| held-out | `100–149` 保持关闭 |
| zero control | Goal134 nominal `24/33`；current15 `14/24` |
| 86201 reference | Goal134 nominal `30/33`；current15 `16/29` |

训练与评估环境配置：

- 训练：`artifacts/lowlevel/goal125_bias_ppo_20261111/environment_configs/B15.json`；
- nominal eval：`configs/lowlevel/goal134/pc_goal134_nominal_{strict,default}_20261111.json`；
- current15 eval：`configs/lowlevel/goal134/pc_goal134_current15_{strict,default}_20261111.json`。

## 5. 预注册 screen 矩阵

| arm | rho | beta | 说明 |
|---|---:|---:|---|
| `R030` | 0.30 | 2.0 | 温和 authority 约束 |
| `R025` | 0.25 | 2.0 | 中间档 |
| `R020` | 0.20 | 2.0 | 较强约束，接近 86201 的低 authority |

每个 arm × 8 个训练 seed 都是独立的 1M run：

```text
3 arms × 8 seeds × 1,000,192 transitions = 24,004,608 env transitions
```

评估：

```text
3 arms × 8 checkpoints × 2 levels × 2 scenarios × 40 seeds = 96 runs
```

8 张 4090 按 run 分配；训练使用与 Goal126 相同的 `num_envs=256`、`batch_size=256`、
`update_to_data_ratio=0.125`。不得给某个 arm 额外样本、额外 batch 或更长训练。

## 6. Gate

### 6.1 Validity

| 检查 | 要求 |
|---|---|
| 架构 | 与 Goal126/134 的 tensor layout 一致 |
| 正则关闭 | 与 Goal126 动作差 `<=2.2e-16` |
| manifest | 记录 rho/beta/form 和 authority 统计 |
| 数值 | 所有训练/评估 finite，无 NaN/Inf |
| 安全 | 评估 timeout=0、safety=0 |
| 隔离 | held-out 未打开；未改 success/radius/terminal/reward |

### 6.2 Per-arm Gate

`zero_nominal=(24,33)`，`zero_current15=(14,24)`。

| 观察 | 判读 | 下一步 |
|---|---|---|
| `>=4/8` seed nominal strict `>=26` 且 default `>=33`；且 `>=4/8` seed current15 strict `>=16` 且 default `>=24` | `AUTHORITY_SAC_GAIN` | 单 arm 扩 8 seeds × 5M formal |
| `>=4/8` nominal 达标，但 current15 未达标 | `NOMINAL_ONLY_GAIN` | 不扩 formal；审计压力迁移和 authority 分布 |
| 只有 `1/8` seed nominal 达标 | `SINGLE_SEED_ONLY` | 不扩 formal；查初始化/数据覆盖/seed 方差 |
| action RMS 下降但 nominal mean `<=24.5`，且 current15 mean `<=14.5` | `DIRECTION_SIGNAL_INSUFFICIENT` | 停止 cap/regularization 家族，回 reward/state/terminal |
| nominal mean `<24` 或 default mean `<32`，或 safety/timeout >0 | `AUTHORITY_SAC_HARM` | 写 blocker；先查正则实现和 Q/entropy，不换算法 |
| 三个 arm 差异全部落在 seed 噪声内 | `AUTHORITY_DOSE_AMBIGUOUS` | 不继续扫 beta；按 blocker 进入 seed/信号审计 |
| manifest/架构/有限性不成立 | `INVALID` | 修实现后重跑，不改阈值 |

### 6.3 formal 条件

只有 `AUTHORITY_SAC_GAIN` 才允许对**同一个 arm**扩到 5M formal。formal 仍用
8 个训练 seed、同一 dev40、同一 success/terminal/radius；不得在 screen 和 formal
之间换 seed、改 rho/beta 或改 Gate。

## 7. 执行计划

- [x] Step 0：写本 Goal，冻结 rho/beta、seed、预算和 Gate；
- [x] Step 1：实现 `AuthorityRegularizedSAC`、配置解析、manifest 和单元测试；
- [x] Step 2：本地测试 + 远端 smoke + 1M engineering calibration，验证正则关闭等价、正则开启有限且 hinge 激活；
- [x] Step 3：生成 3 arms × 8 seeds 的 training configs；
- [x] Step 4：8 卡并行训练 24 个 1M screen；
- [x] Step 5：跑 96 个 nominal/current15 × strict/default dev40 评估；
- [x] Step 6：按 per-arm Gate 判读，决定 formal/pivot/stop；
- [x] Step 7：写 result/blocker/decision，追加 LEDGER，更新 STATE，运行 checker；
- [x] Step 8：不自动 commit/push。

## 7.1 结果

裁决：`AUTHORITY_SAC_HARM`。三个 arm 都没有达到跨 seed 目标；R020 nominal strict
mean 为 `22.125`，低于 zero 的 `24`。RMS 随 rho 下降但 success 没有恢复，说明
“压低幅度”没有产生可迁移方向。

详见 `doc/goals/goal135_result_20261111.md`、`research/blockers/2026-11-11-goal135-authority-sac-harm.md`
和 `research/decisions/2026-11-11-goal135-pivot-goal136-diversity.md`。

## 8. 停止规则

- 不因一次 screen 失败扫 seed、lr、entropy、UTD 或 batch；
- 不把 `DIRECTION_SIGNAL_INSUFFICIENT` 解释成需要换 PPO/REDQ/DroQ；
- 不继续微调 eval-only cap，Goal129/130 已覆盖；
- 不改 reward/terminal/success/goal radius；
- 不打开 held-out，不用 validation 结果直接报 formal；
- 不自动 commit/push，禁止 `git add -A`。

## 9. Git

只 add 本 Goal 相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal135 \
  --goal-doc doc/goals/goal135_authority_regularized_sac_20261111.md \
  --strict --require-result
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志、缓存或数据集。

## 10. Goal 提示词

```text
你现在执行 Goal135：authority-regularized SAC 的跨 seed screen。

先读：
- doc/goals/goal135_authority_regularized_sac_20261111.md
- doc/goals/goal134_result_20261111.md
- research/blockers/2026-11-11-goal134-stress-single-seed-harm.md
- research/decisions/2026-11-11-goal134-pivot-authority-regularized-sac.md

硬约束：
1. 保持 SAC 和现有 z_t64 + zero-DADE64 + command3、path8 + endpoint4、BC encoder；
2. 不换 PPO，不开 REDQ/DroQ，不加 NoisyNet；
3. 只改 actor loss 的 authority RMS hinge，不改 reward/terminal/success/radius；
4. 不打开 held-out 100–149；
5. 3 arms 固定为 R030/R025/R020，rho=0.30/0.25/0.20，beta=2.0；
6. 8 个训练 seed 必须与 Goal126 的 86201–86208 配对；
7. screen 未达 AUTHORITY_SAC_GAIN 不扩 formal、不扫 seed、不改阈值；
8. 训练型任务放远端 8×4090；本地只做小测试和 smoke。

执行顺序：
- 实现并测试 authority loss；
- init-only / penalty-off parity；
- 生成 24 个 1M training configs；
- 8 卡并行训练；
- nominal + current15 的 96 个 dev40 评估；
- per-arm Gate、result、blocker/decision、LEDGER、STATE、checker。
```
