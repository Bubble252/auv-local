# Goal155：残差权限训练 screen（near / mid / far-off × 3 seeds）

**日期**：2026-11-11<br>
**状态**：`COMPLETED / CAPTURE_TRADEOFF / 9x1M / CPU_FLOAT64_FORMAL / HELDOUT_CLOSED`<br>
**结果**：[goal155_result_20261112.md](goal155_result_20261112.md)<br>
**上游**：Goal86 T2-SAC、Goal151 near10_2、Goal152 train-time gate、Goal154 authority confound<br>
**性质**：探索型训练 screen。固定 SAC、z_t、Goal85 位置契约、reward、terminal、success、goal radius、训练预算；只改 residual authority 的位置与幅度。目标不是直接写 formal，而是判断“训练时把 residual 限制在 approach shell 或关掉远段”能否把 Goal86 的 path 代价收回来。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal155 --goal-doc doc/goals/goal155_residual_authority_training_screen_20261111.md --strict --require-result
```

## 1. 证据与问题

Goal86 T2-SAC 的 3 个 full-authority checkpoint 有弱 success 信号，但 path RMSE 稳定退化：

| 指标 | Goal86 T2 full |
|---|---:|
| default success | `28.67/40` vs zero `27/40` |
| strict125 success | `15.33/40` vs zero `14/40` |
| default path 退化 | `+0.277 m`，CI 不含 0 |
| residual L2 | `0.20–0.28`，动作接近饱和 |

Goal154 在**同一批 checkpoint**上做了 eval-only 权限拆解：

| variant | default success | Δ vs zero | default path3d | Δ path |
|---|---:|---:|---:|---:|
| zero | `78/120` | 0 | `2.3375` | 0 |
| full | `85/120` | +7 | `2.6152` | `+0.2777` |
| near10_2 | `88/120` | +10 | `2.3367` | `-0.0007` |
| far_off_only | `87/120` | +9 | `2.3347` | `-0.0027` |
| mid_only | `88/120` | +10 | `2.3361` | `-0.0014` |
| near_only | `78/120` | 0 | `2.3374` | `0.0000` |

结论是：

1. `near_only`（仅 <=2 m）没有可测收益，权限出现太晚；
2. 真正有效的是 2–10 m approach shell；
3. full 的 path 惩罚是位置性的，不能只靠全局 scale 收回；
4. 现在缺的不是“再扫 reward 权重”，而是**训练时** residual 权限是否应固定在 shell 或关闭远段。

Goal152 曾预注册 `near10_2` 的 train-time gate，但没有实际启动。本 Goal 把它作为
control 一并执行，并新增 `mid_only` 与 `far_off_only` 两条训练臂。

## 2. 假设

### H155-A：训练时 `mid_only` 能保留 success 并收回 path

成立应看到：pooled default `Δsuccess >= +6/120`、strict125 `>= +4/120`；两个 scenario
`Δpath <= +0.05 m`；至少 2/3 个训练 seed 的 default `Δsuccess >= 0`；2–10 m residual
RMS `>=0.5`，10 m 外 `<=0.05`。

失败应看到：success 跌回 zero，或 path 仍 `>=+0.10 m`。此时不能扩 seed，先归因。

### H155-B：`far_off_only` 至少不差于 `mid_only`

如果 far 的 success/path 与 mid 持平，说明收益主要来自关闭 >10 m 的偏置性 residual，
而不是 shell 内的精细控制；如果 mid 更好，说明 2 m 内 residual 也有价值，但 Goal154
已经证明 `near_only` 单独无效。

成立应看到：far 与 mid 的 pooled success 差在 `±2/120` 内，path 差在 `±0.02 m` 内。

### H155-C：门控降低远段 residual 与动作饱和，而不是把策略学成零残差

成立应看到：gated arms 的 10 m 外 residual RMS 明显低于 full；2 m 内或 2–10 m 内仍有
`>=0.5` 的 residual；policy-head 饱和比例低于 full 的约 `85%`。

失败应看到：所有距离段 residual 都接近 0；判 `INVALID_NO_RESIDUAL`，不能把 ALOS
zero 的结果算作 gate 收益。

### H155-D：如果 near/mid/far 都不动，瓶颈不是 authority

成立应看到：三条 gated arm 的 success 都在 zero 附近，path 也没有改善；residual
方向和幅度没有随 gate 改变。此时停止权限轴，转动作参数化、terminal capture 和
reward/critic credit，不换算法、不扫 seed。

## 3. 实验臂

统一使用 Goal86 T2-SAC 的 scratch actuator residual、`z_t` 架构、Goal85
`unbiased_noise_v1`、`position_source=eta`、零传感器延迟、C3 固定环境。

| arm | gate | 学习到的有效 residual | 训练预算 | seeds |
|---|---|---|---:|---|
| `full` | 无 | 全距离 1.0 | 复用 Goal86 3×1M | `86201–86203` |
| `near10_2` | smooth `10→2` | >10 m 0，2–10 m ramp，<=2 m 1.0 | 新跑 3×1M | `86201–86203` |
| `mid_only` | smooth `10→2` + shell 形状 `4g(1-g)` | 仅 2–10 m，<=2 m 0，>10 m 0 | 新跑 3×1M | `86201–86203` |
| `far_off_only` | smooth `10.0001→10` | >10 m 0，<=10 m 1.0 | 新跑 3×1M | `86201–86203` |

固定值：

| 项目 | 值 |
|---|---|
| 算法 | SAC，不换 PPO/REDQ/DroQ |
| 动作接口 | actuator residual `[rudder, stern, rpm]` |
| `actuator_residual_scale` | `0.25` |
| `actuator_max_action` | `1.0` |
| `num_envs` / `batch_size` | `256` / `256` |
| buffer / learning starts | `500000` / `20000` |
| total transitions | `1,000,000` per seed |
| reward / terminal / success / radius | 不改 |
| actor 真值观测 | `false` |
| held-out | `100–149` 关闭 |

`full` 不重训，只读取 Goal86 已完成的 3 个 checkpoint；所有新臂在独立目录训练，
不覆盖 Goal86/Goal152/Goal153 的任何产物。

## 4. 评价

| 项目 | 值 |
|---|---|
| dev seeds | `20–59`，40 个 |
| scenarios | default `2.0 m`、strict125 `1.25 m` |
| 对照 | zero-residual；full 3-seed 历史对照 |
| 精度 | CPU / float64 |
| 主指标 | `eta_tight_at_done` |
| 风险/健康指标 | path3d RMSE、vertical RMSE、plane miss、timeout、safety、residual L2、policy-head saturation、material/actuator saturation |
| 终端分布 | `eta_min_distance_m`、near-miss shell、annulus entry、capture funnel |
| 多环境 screen | nominal、current10、noise_n2、plant20（default radius） |

多环境 screen 只作稳健性筛选，不替代主 Gate。所有输出必须写清 success、timeout、
plane miss、主指标、风险指标、seed 数和评估协议。

## 5. Gate

主 screen 以 pooled `3 seeds × 40 episodes = 120` 配对结果判断。

| check | 阈值 |
|---|---:|
| default success | `policy - zero >= +6/120` |
| strict125 success | `policy - zero >= +4/120` |
| path cost | 两个 scenario 都 `<= +0.05 m` |
| seed consistency | 至少 `2/3` seed 的 default `Δsuccess >= 0` |
| residual activity | near/far：<=2 m RMS `>=0.5` 且 >10 m `<=0.05`；mid：2–10 m RMS `>=0.5` 且 <=2 m、>10 m 都 `<=0.05` |
| vertical guard | policy vertical RMSE `<= zero + 0.03 m` |
| safety | timeout/safety 不高于 zero |
| multi-env | 至少 3/4 压力档 success 不比 zero 低超过 `2/40` |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| mid 或 far 过 success/path/vertical，且优于 near | `AUTHORITY_TRAIN_PASS` | 另开 5M formal 预注册 |
| success 过、path 不过 | `PATH_TRADEOFF_REMAINS` | 查 gate 内 residual 幅度、critic credit，不扩 seed |
| path 过、success 不过 | `CAPTURE_TRADEOFF` | 查 terminal capture 与动作参数化 |
| 只有 86203 有增益 | `SEED_CONCENTRATED` | 查训练分布/探索，不 promotion |
| residual 全零 | `INVALID_NO_RESIDUAL` | 不把 zero 算作 gate 收益 |
| 三条 gated arm 全部不动 | `AUTHORITY_NOT_BOTTLENECK` | 停止权限轴，转动作参数化/terminal capture |
| 纵向 RMSE 恶化超过阈值 | `VERTICAL_DOMINANT` | 另开 Goal140 reward pilot，不在本 Goal 改 reward |

## 6. 执行计划

- [x] Step 0：核对 Goal86 checkpoint、Goal154 结果、远端 GPU/内存；
- [x] Step 1：生成 3 arms × 3 seeds 的冻结配置和 SHA manifest；
- [x] Step 2：本地静态测试 + 每臂 2k smoke，确认 `learner_terminal_gate_mode`、gate trace 和 manifest；
- [x] Step 3：8 卡并行训练 9 个 1M runs，剩余 1 个排队；不给任何 arm 偷加预算；
- [x] Step 4：26 个 eval run（4 arms × 3 seeds × 2 scenario + zero × 2），dev40；
- [x] Step 5：汇总 success/path/vertical/residual bins/saturation/terminal distribution；
- [x] Step 6：跑 multi-env A_screen，按 §5 裁决；
- [x] Step 7：写 result、blocker/decision、LEDGER、STATE；失败不扩 seed、不换算法；
- [x] Step 8：通过 checker 后才考虑 5M formal。

### 6.1 正式评估的设备口径

训练 runner 在训练结束后会自动跑一轮 `--device cuda --dtype float64` 的评估。该轮结果
只作为**诊断**，不得直接写正式 Gate 结论，因为本 Goal 预注册的评估精度是 CPU /
float64。正式复评使用独立输出根：

```bash
cd settled_局部规划器_反应式
MAX_PARALLEL=8 bash scripts/run_goal155_eval_cpu.sh
```

输出写到 `artifacts/lowlevel/goal155_authority_training_screen_20261111/eval_cpu/`，
分析器通过 `--eval-root` 指定该目录，审计汇总也写到该目录，不会与 CUDA 诊断结果混算：

```bash
.venv/bin/python scripts/analyze_pc_goal155_authority_training.py \
  --root artifacts/lowlevel/goal155_authority_training_screen_20261111 \
  --eval-root artifacts/lowlevel/goal155_authority_training_screen_20261111/eval_cpu
```

CPU/float64 单 run 已验证可跑通；2-episode smoke 约 11 s，40-episode 的 8 路并行
复评不会成为瓶颈。

### 6.2 中途诊断（非 Gate，8/9 checkpoint）

2026-11-11 23:42 的 CPU/float64 复评已完成 8 个新 checkpoint（`far_off_only_seed86203`
仍在训练），zero 为 `27/40 default`、`14/40 strict125`，与 Goal86 历史 zero 一致。
下列数字只用于判断方向，**不替代正式 9-checkpoint Gate**：

| arm | default success (pooled) | Δ vs zero | strict success (pooled) | Δ vs zero | default Δpath |
|---|---:|---:|---:|---:|---:|
| `full` | `86/120` | `+5` | `46/120` | `+4` | `+0.277 m` |
| `near10_2` | `80/120` | `-1` | `44/120` | `+2` | `~0 m` |
| `mid_only` | `75/120` | `-6` | `42/120` | `0` | `~0 m` |
| `far_off_only` | `49/80` | `-5` | `28/80` | `0` | `~0.006 m` |

关键观察：

1. CPU/float64 的 `full` 对照精确复现了 Goal86 的 success/path 方向，说明新评估管线可信；
2. 训练时 gate 确实收回了 path，但也把 success 压回 zero 附近甚至以下；
3. gated arms 的 learned residual RMS 只有 `0.20–0.44`，而 `full` 为 `0.46–0.90`；
   gate 不是“没学到”，而是把可用于 terminal capture 的 residual 幅度压小了；
4. 这与 Goal154 的 eval-only gate 不同：Goal154 是在 **full-trained checkpoint** 上
   后置 gate，保留了 shell 内的大 residual；本 Goal 是 **训练时 gate**，actor 从一开始
   就没有学到远段探索产生的 shell 动作。

因此若正式 Gate 失败，优先解释不是“shell 位置错了”，而是 **训练时 gate 抑制了
探索/信用分配**。下一步应保留 full-residual 训练，再在评估或短程 fine-tune 阶段加
gate，而不是继续用训练时硬 gate 扫 seed。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、配置生成器、runner、分析器和测试；
不提交 artifacts、checkpoint、日志或缓存。提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal155 \
  --goal-doc doc/goals/goal155_residual_authority_training_screen_20261111.md \
  --strict --require-result
git diff --cached --check
```

用户明确要求前不 commit / push。
