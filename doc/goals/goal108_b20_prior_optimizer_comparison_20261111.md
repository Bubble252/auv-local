# Goal108：B20 先验上的 PPO/SAC 小增量对照

**日期**：2026-11-11<br>
**状态**：`COMPLETED / NO_PRIOR_OPTIMIZER_GAIN / HELDOUT_CLOSED`<br>
**上游**：Goal85/85b/86 的 unbiased 契约与 PPO/SAC 适配；Goal100/101 的 BC-only 结论；Goal107 的 credit/优化归因<br>
**性质**：训练型预注册 Goal；不改 reward、terminal、success、goal radius、dev seeds 或 held-out。新增 arm 不复用旧 checkpoint 作为结果。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal108 --goal-doc doc/goals/goal108_b20_prior_optimizer_comparison_20261111.md --strict --require-result
```

记录前不得修改 base checkpoint、B20 scale、训练预算、seed、scenario、success frame、path 阈值或 Gate 判读规则。

## 1. 背景：为什么从 B20 开始，而不是再从零训练

Goal100/101 已经给出两个关键事实：

```text
B20 = frozen BC residual + actuator_residual_scale 2.00
B20 default 33/40, strict125 20/40
ALOS   default 27/40, strict125 14/40
```

也就是说，当前 BC 先验本身已经明显超过传统 ALOS；真正的问题不是“网络容量不够”
或“没有 BC”，而是**在 BC 先验上继续加 PPO 增量会破坏 capture**：

- Goal95：监督残差先验 + 1M PPO 从 31/20 退到 30/19；
- Goal100：同一 checkpoint 关闭 PPO 得到 33/20，开 PPO 的 `P0_20` 只有 31/32 与 18/18；
- Goal107：C0 失败种子最后 2 s 的 applied residual 反而更大（strict 359 vs 282），
  U050 把 residual 压到约 160–170 后 strict 反而 +3。

因此 Goal108 不再从零训练，也不继续扫 residual 幅度，而是把 **B20 作为共同起点**：

```text
action = ALOS + 2.00 * (r_BC + r_learned)
r_BC      = frozen BC residual, 不更新
r_learned = zero-initialized PPO 或 SAC 增量
```

这样 step 0 就是 B20（33/20），任何闭环变化都能归因到“增量学习”，而不是
“初始化把先验弄坏”。

## 2. 假设

### H108-A：B20 先验 + 小增量能提升 strict capture

- 机制：从 B20 出发，失败种子不需要再学“整段控制”，只需要修正最后阶段的
  闭合方向；Goal107 已证明失败种子的 residual 更大且方向不利，因此小增量比
  大幅度 residual 更合理。
- 成立：至少一个 arm 在 3 seeds 上达到
  `mean default >= 33`、`mean strict125 >= 23`，至少 2/3 seeds strict >=23；
  path 不恶化超过 5%，timeout/safety 为 0。
- 失败：两个 arm 的 strict 都停在 20–22，或 default 掉到 32 以下。
- 最便宜证伪：1M、3 seeds 的 B20-PPO / B20-SAC 配对；不先跑 5M。

### H108-B：同一 B20 先验下，SAC 比 PPO 更能利用 replay 修正 credit

- 机制：PPO 是 on-policy，当前失败更像 credit/优化方向；SAC 的 replay 可以在
  B20 附近反复利用少数近终端 transition，可能更早学到闭合修正。
- 成立：SAC 的 strict 比 PPO 高 `>=3/40`，或达到同 success 使用更少 transitions。
- 失败：两者差异小于 seed 噪声，或 SAC 出现更大的 path/safety 退化。
- 最便宜证伪：同 base、同 scale、同 1M、同 seeds 的 B20-PPO vs B20-SAC。

### H108-C：如果 B20+小增量仍不动，瓶颈不在优化器/先验

- 机制：B20 已经是当前最强基线；如果两种算法都无法从它出发改善 strict，
  说明限制在 terminal reward/geometry，而不是 PPO/SAC 或样本效率。
- 成立：两个 arm 的 strict 都 <=20，或改善只出现在 default 且 strict 回退。
- 失败：至少一个 arm 达到 H108-A 的 Gate。
- 最便宜证伪：本 Goal 的 6 个 1M run；失败后不扩 5M，转 Goal109 reward/terminal。

## 3. 固定契约

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt` |
| base SHA256 | `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd` |
| environment | Goal85 `unbiased_noise_v1` default；strict125 只用于评价 |
| position source | `eta`（真值） |
| sensor delay | 0 |
| eval scale | `2.00` |
| training scale | `2.00` |
| dev seeds | `20–59`，每 run 40 episodes |
| success frame | `both` |
| held-out | `100–149` 关闭 |
| PPO transitions | `1000 envs × 25 steps × 40 iterations = 1,000,000` |
| SAC transitions | `256 envs × 1,000,000`；batch 256；UTD 0.125；learning starts 20k |
| training seeds | `86201/86202/86203` |

### 3.1 动作组合

```text
PPO: ALOS + 2.00 * (r_BC + r_PPO)
SAC: ALOS + 2.00 * (r_BC + r_SAC)

r_BC      = frozen BC residual（不更新）
r_PPO     = PPO residual，output_std=0.0 初始化
r_SAC     = SAC deterministic mu，zero_residual_init=true 初始化
```

两个 arm 的 step 0 都必须精确或近似复现 B20：

```text
default 33/40, strict125 20/40
```

SAC 的 BC base 在 `PCCommandResidualOffPolicyEnv` 中按 normalized observation
计算；已用 40-env 首步对齐 Goal104 C0 trace，`max_abs_diff ≈ 5e-4`。

## 4. 实验矩阵

| arm | base | 增量 | scale | 算法 | seeds | 预算 |
|---|---|---|---:|---|---|---:|
| `B20` | frozen BC | 无 | 2.00 | eval-only | — | 复用 Goal104/100 |
| `B20-PPO` | frozen BC | PPO residual | 2.00 | PPO | 86201/02/03 | 1M |
| `B20-SAC` | frozen BC | SAC residual | 2.00 | SAC | 86201/02/03 | 1M |
| `ALOS-zero` | 无 | 无 | — | eval-only | — | 复用 Goal101 |

新跑训练：

```text
2 arms × 3 training seeds = 6 runs
```

新跑评价：

```text
2 arms × 3 seeds × 2 scenarios = 12 dev40 eval runs
```

B20 与 ALOS 只做同协议参照，不重新训练。

## 5. Gate

### 5.1 `INVALID`

- base checkpoint SHA 不匹配；
- step-0 / iteration-0 不能复现 B20（default 33、strict 20）且偏差不可解释；
- 缺 summary/trace、出现 NaN/Inf、seed 不完整；
- 读取 held-out，或改了 success/goal radius/reward/terminal。

### 5.2 `PASS_PRIOR_INCREMENT`

至少一个 arm 同时满足：

```text
mean default >= 33/40
mean strict125 >= 23/40
至少 2/3 seeds strict >= 23/40
path RMSE <= 1.05 * B20
timeout = 0, safety = 0
increment L2 ratio > 0.01
action_abs_max < 0.999（未全面饱和）
```

通过后只允许扩到 5M formal，仍固定 3 seeds；不得直接报正式结论。

### 5.3 `PARTIAL_PRIOR_INCREMENT`

- default 保住 33，但 strict 只有 21–22；或 strict 到 23 但 default 掉到 32；
- path 或 saturation 护栏失败。

此时不扩 5M，先做只读 trace 审计；只有明确机制（例如只用 rudder/只改近终端）
才另开新 Goal。

### 5.4 `NO_PRIOR_OPTIMIZER_GAIN`

两个 arm 都：

```text
mean strict <= 20/40
或 strict 改善 < 2/40
或 default < 33
```

此时停止 PPO/SAC 优化器轴，写 blocker，转 Goal109：
terminal reward / capture 几何 / residual direction，不再扩大 residual 或换 REDQ/DroQ。

## 6. 统计与报告

- 每个 arm 报 3 个训练 seed 的 success、timeout、safety、path RMSE、residual L2、
  action abs max、increment 激活率；
- 逐 seed 配对 B20 vs arm，报告 gained/lost，不只报均值；
- 3 seeds 只作 screen；formal 5M 通过后仍需扩到 5 seeds 才能写最终结论；
- 不把单 seed 最好结果写成正式 PASS。

## 7. 执行计划

- [x] Step 0：写预注册、Research Contract、configs；
- [x] Step 1：实现 SAC 的 frozen-BC base + zero-init increment；
- [x] Step 2：单测 + 40-env 首步 parity + SAC initial / PPO iteration-0 B20 parity；
- [x] Step 3：6 个 1M training run（PPO/SAC × 3 seeds）；
- [x] Step 4：12 个 dev40 eval run（default + strict125）；
- [x] Step 5：按 Gate 写 result/decision/blocker/LEDGER/STATE，运行 checker；
- [x] Step 6：Gate 为 `NO_PRIOR_OPTIMIZER_GAIN`，按预注册不扩 5M formal；
- [x] Step 7：未自动 commit/push。

## 8. 运行命令

```bash
# SAC init parity
PYTHONPATH=src:/data/yhx/underwater-robotics/src/stable-baselines3 \
  python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal108/pc_goal108_b20_sac_default_20261111.json \
  --output artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_SAC_init \
  --seed 86201 --device cuda --init-only --overwrite

# PPO 1M
PYTHONPATH=src python scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/goal108/pc_goal108_b20_ppo_default_20261111.json \
  --output artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_PPO_s86201 \
  --device cuda

# SAC 1M
PYTHONPATH=src:/data/yhx/underwater-robotics/src/stable-baselines3 \
  python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal108/pc_goal108_b20_sac_default_20261111.json \
  --output artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_SAC_s86201 \
  --seed 86201 --total-env-transitions 1000000 --device cuda --engineering-smoke --overwrite
```

## 9. 不做什么

- 不改 reward、terminal、success、goal radius；
- 不打开 held-out；
- 不从零训练 scratch residual；
- 不继续扫 residual scale / channel gain / rpm 常数；
- 不在本 Goal 引入 REDQ/DroQ/DADE；
- 不用单 seed 或最好 seed 写正式结论；
- 不覆盖 Goal86/95/100/104 的旧 checkpoint 和结果。

## 10. 为什么这个新尝试有机会出效果

B20 已经把“网络 + BC 先验”做到 33/20，明显高于 ALOS 的 27/14。Goal107 又证明：

```text
失败不是 reward 没信号，也不是几何不可分；
失败种子的 residual 更大，缩小 residual 反而更好。
```

因此最合理的下一步不是换掉网络，也不是继续放大 residual，而是：

```text
保留 B20 的强先验，只学习一个零初始化的、小步的、有 replay/credit 修正的增量。
```

如果这样仍不能把 strict 从 20 推到 23，才说明瓶颈确实在 terminal reward/geometry，
届时再进入 reward/terminal Goal，而不是继续在优化器上试错。
