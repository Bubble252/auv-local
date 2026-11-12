# Goal156：full-residual SAC 训练 + 部署时 shell gate 的 5M formal

**日期**：2026-11-11<br>
**状态**：`COMPLETED / GATE_FAIL / CAPTURE_TRADEOFF / HELDOUT_CLOSED`<br>
**上游**：Goal154 eval-only authority、Goal155 train-time gate screen、Goal86 T2-SAC<br>
**性质**：预注册 formal。固定 SAC、`z_t`、Goal85 位置契约、reward、terminal、success、goal radius；只改变训练时 authority 与部署时 gate 的组合。

**结果**：[goal156_result_20261112.md](goal156_result_20261112.md)

**运行标识**：runner 使用 `--engineering-smoke` 只是关闭 tiny-smoke caps；实际配置仍为 `3 seeds × 5M`、`num_envs=256`、`batch_size=256`、`UTD=0.125`。因此 manifest 中的 `run_kind=engineering_smoke` 是诚实标签，不是训练降级。结果必须同时核对 `total_env_transitions_actual=5,000,000`、config SHA 和上述超参数；任一不符判 `INVALID`，不能靠改标签补救。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal156 --goal-doc doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md --strict --require-result
```

## 1. 证据与问题

Goal154 在同一批 full-trained Goal86 T2-SAC checkpoint 上做 eval-only authority 拆解：

| variant | default success | Δ vs zero | default path3d | Δ path |
|---|---:|---:|---:|---:|
| zero | `78/120` | 0 | `2.3375` | 0 |
| full | `85/120` | `+7` | `2.6152` | `+0.2777` |
| near10_2 | `88/120` | `+10` | `2.3367` | `-0.0007` |
| mid_only | `88/120` | `+10` | `2.3361` | `-0.0014` |
| far_off_only | `87/120` | `+9` | `2.3347` | `-0.0027` |

Goal155 的中途 CPU/float64 结果显示：训练时加 gate 会把 success 压回 zero 附近，
而 full-residual 对照仍复现弱正 success 和 path 代价。更符合数据的解释是：
full-authority 训练先探索到 shell 动作，再在部署时关闭远段 residual；不是让 actor
从零开始只在 shell 里探索。

## 2. 假设

### H156-A：full 训练 + eval-time mid gate 保留 success 并收回 path

成立应看到：default `Δsuccess >= +6/120`、strict125 `>= +4/120`；两个 scenario
`Δpath <= +0.05 m`；至少 2/3 seed default `Δsuccess >= 0`；2–10 m residual 活跃、
>10 m 约为 0。

失败应看到：success 回到 zero，或 path 仍 `>=+0.10 m`。此时转 terminal capture /
动作参数化或 reward/critic credit，不扩 seed。

### H156-B：收益来自 full-authority 训练，而不是部署 gate 本身

对照 Goal155 的 `mid_only` 训练臂。若 full-train + mid-gate 明显优于 mid-train +
mid-gate，而两者部署 gate 相同，则差异归因于训练时探索/信用分配。最便宜的
falsification 是复用两边已有 1M checkpoint 做同协议 dev40 对比。

### H156-C：5M 不会把 shell residual 学成饱和的远段偏置

成立应看到：部署 mid gate 后 >10 m residual `<=0.05`；2–10 m residual `>=0.5`，
bias fraction 不继续上升；policy-head saturation 不高于 Goal155 full 的约 `0.74`。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 训练 seeds | `86201–86203`，3 个 |
| 每 seed transitions | `5,000,000` |
| dev seeds | `20–59`，40 个 |
| eval scenarios | default `2.0 m`、strict125 `1.25 m` |
| held-out | `100–149` 关闭 |
| 评估精度 | CPU / float64 |
| 位置契约 | Goal85 `position_source=eta`、`unbiased_noise_v1`、无额外位置偏移/延迟 |
| 速度范围 | 本 Goal 固定 C1 的 `1.2 m/s`；REMUS100 平台硬上限为 `2.5 m/s` |

Goal156 不启用高速课程 v2，因此不能把它的结果写成 `2.5 m/s` 或高速泛化证据。
未来若启用 v2，C6 必须覆盖 `0.90–2.50 m/s`，并与同预算的 v1 课程对照；
强洋流下 `2.5 m/s` 不可达时报告饱和/timeout，不删档、不改 success 口径。

## 4. 方法

### 4.1 训练

- 算法：SAC，复用 `configs/lowlevel/goal86/pc_goal86_t2_sac_default_20261110.json`；
- 动作：actuator residual `[rudder, stern, rpm]`，`actuator_residual_scale=0.25`；
- `z_t` 架构、`num_envs=256`、`batch_size=256`、`update_to_data_ratio=0.125`；
- 训练时 residual authority = full，不加 `terminal_gate`；
- 每个 training seed 独立进程，`CUDA_VISIBLE_DEVICES` 固定。

### 4.2 评估 gate

只改变部署 gate，不重训。至少评估：

| eval variant | >10 m | 2–10 m | <=2 m |
|---|---:|---:|---:|
| `zero` | 0 | 0 | 0 |
| `full` | 1 | 1 | 1 |
| `near10_2` | 0 | smooth 0→1 | 1 |
| `mid_only` | 0 | 1 | 0 |
| `far_off_only` | 0 | 1 | 1 |

`mid_only` 使用 `learner_terminal_gate_mode=shell`（`4g(1-g)`），配置复用
`configs/lowlevel/goal154/authority/mid_only.json`。所有 eval 共用同一 checkpoint；
不按 dev success 选 checkpoint。

### 4.3 诊断指标

- success、timeout、plane-miss、safety；
- path3d/vertical RMSE、terminal distance、near-goal distribution；
- residual RMS 按 `<=2 m / 2–10 m / >10 m` 分箱；
- 每通道 RMS、bias fraction、逐 episode lag-1；
- policy-head/effective/actuator/material saturation。

## 5. 执行计划

- [x] Step 0：Goal155 正式 Gate 记录完成，确认没有改冻结参数；
- [x] Step 1：生成 3 个 5M 训练配置和 SHA manifest；
- [x] Step 2：本地静态测试 + 单 seed smoke；`smoke/full_25k` 已存在；
- [x] Step 3：3 seeds × 5M 完成；manifest 均为 `5,000,192` transitions、`num_envs=256`、`batch_size=256`、`UTD=0.125`；
- [x] Step 4：CPU/float64 完成 zero/full/near/mid/far × 2 scenarios，共 26 个正式单元；
- [x] Step 5：完成 success/path/terminal/residual 汇总；verdict `CAPTURE_TRADEOFF`；
- [x] Step 6：完成 standing `A_screen` 104 个策略单元和 8 个 zero 基线；4/4 压力档未见 near 相对 zero 超过 `2/40` 的下降；
- [x] Step 7：按 §6 Gate 裁决为 `GATE_FAIL / CAPTURE_TRADEOFF`；失败进入 blocker，不扩 seed；
- [x] Step 8：已写 result、blocker、decision、reading note、LEDGER、STATE，并跑 checker。

## 6. Gate

主 screen 以 pooled `3 seeds × 40 episodes = 120` 配对结果判断：

| check | 阈值 |
|---|---:|
| default success | `policy - zero >= +6/120` |
| strict125 success | `policy - zero >= +4/120` |
| path cost | 两个 scenario 都 `<= +0.05 m` |
| seed consistency | 至少 `2/3` seed default `Δsuccess >= 0` |
| residual activity | 2–10 m RMS `>=0.5`，>10 m `<=0.05` |
| vertical guard | policy vertical RMSE `<= zero + 0.03 m` |
| safety | timeout/safety 不高于 zero |
| multi-env | 至少 3/4 压力档 success 不比 zero 低超过 `2/40` |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| mid/near 过 success/path/vertical | `FULL_TRAIN_EVAL_GATE_PASS` | 写方法级 result，进入 B/C/D tier |
| success 过、path 不过 | `PATH_TRADEOFF_REMAINS` | 查 2–10 m 幅度/偏置和 critic credit |
| path 过、success 不过 | `CAPTURE_TRADEOFF` | 转 terminal capture/动作参数化 |
| 只有 full 过、gated 不过 | `GATE_NOT_TRANSFERABLE` | 查 gate 位置和部署状态分布 |
| residual 全零 | `INVALID_NO_RESIDUAL` | 不把 zero 算作增益 |
| 纵向 RMSE 恶化 | `VERTICAL_DOMINANT` | 另开 Goal140 reward pilot |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、配置生成器、runner、分析器和测试；
不提交 artifacts、checkpoint、日志或缓存。提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal156 \
  --goal-doc doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md \
  --strict --require-result
git diff --cached --check
```

用户明确要求前不 commit / push。

## 8. Goal 提示词

```text
你现在执行 Goal156：full-residual SAC 训练 + 部署时 shell gate 的 5M formal。

先读：
- doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md
- doc/goals/goal154_authority_confound_result_20261111.md
- doc/goals/goal155_residual_authority_training_screen_20261111.md
- research/EVALUATION_PROTOCOL.md

硬约束：
1. 固定 SAC、z_t、Goal85 位置契约、reward、terminal、success、radius；
2. 训练时 full residual，不在训练时加 gate；gate 只在部署/评估时打开；
3. 3 seeds × 5M，同预算；不扩 seed、不换算法、不用 held-out 选 checkpoint；
4. 正式 Gate 用 CPU/float64；CUDA 只作诊断；
5. 失败先归因，按 autoresearch 写 blocker/hypothesis/reading note/LEDGER。

执行顺序：
- 审计 Goal155 和 Goal154 的 checkpoint/配置；
- 生成 5M 配置与 manifest；
- smoke；
- 3 seeds × 5M；
- zero/full/near/mid/far × default/strict125 的 CPU/float64 eval；
- Gate、result、ledger、checker。
```
