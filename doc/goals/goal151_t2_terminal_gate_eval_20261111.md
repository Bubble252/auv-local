# Goal151：T2-SAC 残差近终端门控 eval-only 证伪

**日期**：2026-11-11<br>
**状态**：`PREREGISTERED / FROZEN_BEFORE_MEASUREMENT / EVAL_ONLY / HELDOUT_CLOSED`<br>
**上游**：Goal85 位置契约、Goal85b plant mismatch、Goal86 T2 3-seed、Goal145 T2 路径分解<br>
**优先级**：`P1`。Goal86 T2 已经有弱正向 success 信号，但全路径 RMSE 稳定退化；本 Goal
先验证“减少远处残差权限”能否保留终端收益，而不是继续扩 seed 或直接进 5M formal。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal151 --goal-doc doc/goals/goal151_t2_terminal_gate_eval_20261111.md --strict --require-result
```

`required: true`：本 Goal 复用已有 checkpoint 做新的机制评估，不训练新策略；如果
Gate 失败，必须留下 blocker 和至少两个可证伪 hypothesis，不能通过扩 seed 或换算法救。

## 1. 背景与证据

### 1.1 Goal85/85b 已固定的契约

- 训练观测：`p_nav = eta + zero-mean noise`；
- reward / termination：真值 `eta`；
- 官方 success：`eta_tight_at_done`；
- sensor delay 为 0，`position_source=eta`；
- plant mismatch ±10% 已实现，但主要改变 path RMSE，未形成稳定 terminal success 变化。

因此本轮不再重复位置 bias/delay 或 ±10% plant mismatch；这些变量保持冻结。

### 1.2 Goal86 T2 的弱正向与 path 代价

T2-SAC 在 3 个训练 seed 上的 dev40 结果：

| 场景 | T2 success | zero | Δsuccess | pooled Δpath RMSE |
|---|---:|---:|---:|---:|
| default | `28.67/40` | `27/40` | `+1.67/40` | `+0.277 m`，CI `[+0.190,+0.366]` |
| strict125 | `15.33/40` | `14/40` | `+1.33/40` | `+0.275 m` |

PPO 同协议全部 `0/0/40` 平局。因此 T2 的弱正向来自 SAC 的实际残差，不是动作接口
本身；失败机制更像 terminal capture 与 path tracking 冲突，而不是残差没动。

### 1.3 现有代码已具备的权限工具

`PCCommandResidualOffPolicyEnv` 已支持：

- `residual_terminal_gate_enabled`：按 `goal_distance` 给出 0（远处）到 1（近处）
  的光滑 gate；
- `learner_terminal_gate_mode=shell`：只在 3–10 m 的过渡壳层放权，进入近距离捕获区
  后再次抑制；
- `residual_scale`：只改推理倍率，不改 checkpoint。

这些工具此前没有在 T2 3-seed 上做过 eval-only 对照。本 Goal 只测试权限位置，不改
reward、terminal、success、checkpoint、训练数据或算法。

## 2. 问题与可证伪假设

### H151-A：近终端 gate 能保留 T2 success，并显著收回 path 代价

做法：gate 在 10 m 外关闭，10 m 到 2 m 平滑打开，2 m 内保持 1。

成立应看到：

- 3 seeds pooled success 相对 zero 至少 `+3/120`，且相对 full T2 不低于 `-4/120`；
- pooled path RMSE 相对 zero 不超过 `+0.12 m`，且相对 full T2 改善至少 `0.12 m`；
- 2 m 外 residual 明显低于 full，2 m 内仍非零。

失败应看到：success 保留但 path 仍 `+0.2 m` 以上，或 path 收回但 success 跌回 zero。

### H151-B：3–10 m shell gate 比 10→2 m gate 更稳

目的：检验 T2 的收益是否来自进入 capture radius 前的 approach shell，而不是 2 m 内的
末端修正。

成立应看到：shell 的 success 不低于 near gate `-2/120`，且 path 不高于 near gate
`+0.03 m`；3–10 m 内 residual 非零，10 m 外和 3 m 内接近 zero。

失败应看到：shell 与 near gate 的 success/path 都接近 full，说明 gate 没有作用；或
shell 明显 loss success，说明 T2 的翻转依赖 capture radius 内的 residual。

### H151-C：若两种 gate 都不能降低 path 代价，则冲突不只来自远处残差

成立应看到：两个 gate 的 pooled path Δ 与 full 接近，CI 重叠；此时下一步转向
reward/terminal credit 或 residual channel，而不是继续扫 gate 位置。

失败应看到：至少一个 gate 的 path Δ 明显低于 full，说明权限位置是有效杠杆。

### H151-D：gate 不是通过“把残差全关掉”获得低 path 代价

成立应看到：gate 变体的 near-goal residual L2 仍显著非零，并且 success 高于 zero；
如果 residual 全零，应按 zero 基线处理，不能算 gate 收益。

## 3. 数据与冻结矩阵

复用 Goal86 screen 的 3 个 T2-SAC final checkpoint：

| train seed | checkpoint SHA256 |
|---:|---|
| 86201 | `10044cb037eec9232e1d361479bdfdf526dfa4b1efde76d255b3b077971d3ed4` |
| 86202 | `c2c90ee252994274206a35957aa240bd6b66b2a2b912773e5ab3eafaecd1a08a` |
| 86203 | `3b68081d20679edc9b0e322e47fda6952ceed26ce458fc54b8fbb49bbe1f2adb` |

| 项目 | 值 |
|---|---|
| dev seeds | `20–59`，每个 run 40 episodes |
| scenarios | default、strict125（Goal85 candidate 契约） |
| algorithms | 只评估 SAC；不训练、不新增 PPO |
| variants | `full`（复用 Goal145）、`near10_2`、`shell10_3` |
| full / zero 数据 | 复用 `artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111/t2/` |
| 新 run 数 | 2 variants × 3 checkpoints × 2 scenarios = 12 |
| timeout/safety | 必须记录；无 NaN/Inf |
| held-out | `100–149` 关闭 |
| reward / terminal / success / radius | 不改 |

`near10_2`：

```text
residual_terminal_gate_enabled = true
enter_m = 10.0
full_m = 2.0
learner_terminal_gate_mode = none
```

`shell10_3`：

```text
residual_terminal_gate_enabled = true
enter_m = 10.0
full_m = 3.0
learner_terminal_gate_mode = shell
```

`full` 与 `zero` 必须来自同一 Goal86 T2 3-seed、同一 dev seeds、同一 success frame；
不重新选 checkpoint，不根据 transfer 结果改 Gate。

## 4. 指标

主指标：

- `success_eta_tight_at_done`；
- `path_three_d_rmse_m`；
- 配对 Δsuccess、Δpath RMSE；
- `residual_l2_ratio` 和 trace 的 distance-bin residual RMS；
- `eta_done_distance_m`、plane miss、timeout、safety。

距离 bin：

```text
<=2 m, 2–10 m, >10 m
```

用于证明 gate 是否真的改变了远处/近处权限，而不是只改了一个配置字段。

## 5. Gate

按 scenario 分别计算 3 个 checkpoint 的 pooled 120 对；两个 scenario 都参与裁决。

| check | 阈值 |
|---|---|
| success effect | variant - zero `>= +3/120` |
| success preservation | variant - full `>= -4/120` |
| path cost | variant - zero `<= +0.12 m` |
| path improvement | full - variant `>= +0.12 m` |
| residual activity | 2–10 m 或 <=2 m 的 residual RMS 非零；不能是全零 |
| safety | timeout/safety 不高于 zero |
| 数值 | 无 NaN/Inf；否则 INVALID |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 任一 variant 两个 scenario 都过 success/path | `GATE_PASS_READY_FOR_TRAINING_PREREG` | 另开训练型 Goal，固定同一 gate |
| success 保留、path 代价仍高 | `PATH_TRADEOFF_REMAINS` | 转 reward/channel audit，不扩 seed |
| path 收回、success 跌回 zero | `CAPTURE_TRADEOFF` | 先用 trace 查 gate 内/外 credit，不扫 gate |
| 两个 variant 都接近 full | `GATE_NOT_MECHANISM` | 停止 gate 方向，转 reward/terminal |
| residual 全零 | `INVALID_NO_RESIDUAL` | 不把 zero 结果算作 gate 收益 |

本 Goal 只产出 eval-only 机制结论，不能写成 formal success 收益。

## 6. 执行计划

- [ ] Step 0：核对 Goal86 checkpoint SHA、Goal145 full/zero 数据、dev seeds 和 GPU；
- [ ] Step 1：生成本 Goal 的两个派生配置和 SHA manifest；
- [ ] Step 2：本地跑配置/路径/runner 静态测试；
- [ ] Step 3：远端同步脚本与配置；
- [ ] Step 4：12 个 eval run（2 variants × 3 seeds × 2 scenarios）；
- [ ] Step 5：配对汇总 full / near / shell，检查 distance-bin residual；
- [ ] Step 6：按 §5 Gate 裁决并写 result；
- [ ] Step 7：失败则写 blocker、hypotheses、decision、LEDGER、STATE，运行 checker。

## 7. 复现入口

```bash
# 生成派生配置
python scripts/build_goal151_t2_gate_configs.py

# 远端评估
GPUS="0 3 4 6" MAX_PARALLEL=6 \
  bash scripts/run_goal151_t2_gate_eval.sh

# 汇总
python scripts/analyze_pc_goal151_t2_gate.py \
  --root artifacts/lowlevel/goal151_t2_terminal_gate_eval_20261111
```

## 8. Git

只 add 本 Goal 的文档、配置生成器、runner、analyzer、测试、LEDGER、STATE；不提交
`artifacts/`、checkpoint、日志或缓存。禁止 `git add -A`，不自动 commit/push。

