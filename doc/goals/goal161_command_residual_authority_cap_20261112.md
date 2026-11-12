# Goal161：command residual authority cap，固定 SAC/z_t/H3 的同协议对照

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED_AFTER_GOAL160_FAIL / SAC_ONLY / HELDOUT_CLOSED`<br>
**上游**：Goal160 `GATE_FAIL / COMMAND_RESIDUAL_HARM`；Goal154 authority schedule；Goal156/159c residual credit

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal161 --goal-doc doc/goals/goal161_command_residual_authority_cap_20261112.md --strict
```

## 1. 背景与证据

Goal160 固定 SAC、`z_t`、H3、C3/ALOS、Goal85 位置契约、reward/terminal/success/
radius，只改变动作参数化，仍然 `GATE_FAIL`：

- current10 default：A actuator `91/120`（+1 vs zero），B command `83/120`
  （-7），C terminal-only command `84/120`（-6）；
- current10 strict125：A `53/120`（-4），B `49/120`（-8），C `43/120`（-14）；
- 三臂 residual 活跃、timeout/safety=0，说明不是“残差没打开”；
- C 比 B 更差，说明 terminal-only 训练没有解决 command residual 的闭环问题。

Goal160 后做了**只读 scale 诊断**（不训练、不改 Gate）：对 B/C 的
current10 default 复用同一 checkpoint，只把部署 `residual_scale` 从
`1.0` 降到 `0.25`：

| arm | scale 1.0 | scale 0.5 | scale 0.25 | zero |
|---|---:|---:|---:|---:|
| B command | 83/120 | 95/120 | 95/120 | 90/120 |
| C terminal command | 84/120 | 93/120 | 96/120 | 90/120 |

> **2026-11-12 Phase A 口径修正**：上表来源是
> `diag_scale_20261112`，其 summary 明确记录
> `curriculum_current_speed_range_override_mps=null`、
> `curriculum_shear_scale_override=null`，环境默认洋流是 `0.25–0.45 m/s`。
> 因此这组 `95/96` 是 **默认洋流诊断**，不是 `current10` 的
> `0.30–0.55 m/s + shear 1.0`，不能作为 H161-A 已经成立的证据。
> 真正的 current10 结论只以 Phase A 的显式 override 结果为准；Gate
> 阈值、seed、level 和 radius 均不因此修改。

同时 path 3D RMSE 从 zero `2.062` 降到约 `1.963`，final distance 从
`2.573` 降到约 `2.36`。这说明失败主因不是优化/探索或样本效率，而是
command residual 的**权限过大**：策略输出接近满幅，经 `0.35 rad /
0.17 rad / 0.60 m/s` 的 command scale 放大后破坏 H3/ALOS 的闭环结构。

Goal161 因此只做 authority cap / channel restriction，不换 SAC，不改
reward/terminal/success/radius，不扩 seed 救旧 Gate。

## 2. 假设

### H161-A：训练时把 command residual cap 到 0.25，能保留近区纠错而不破坏 ALOS

- 机制：`0.25 × command scale` 给出 heading `±0.0875 rad`、pitch
  `±0.0425 rad`、speed `±0.15 m/s`，接近当前 actuator near10_2 的有效权限；
  足以修正终端误差，但不会覆盖 ALOS 的相位/阻尼结构。
- 成立应看到：`B_command_cap025` 或 `C_command_cap025` 在 current10
  default 相对 zero/A 至少 `+4/120`，strict 至少 `+2/120`；其他层不低
  于 `-2/120`；path 3D/vertical 不恶化超过 `+0.05 m`。
- 失败应看到：cap 训练后仍无增益，或 path/entry 明显恶化；此时不是
  authority 幅度，而是 command-space 方向/credit 不适用于 H3。
- 最便宜证伪：先完成 eval-only cap 0.25 的全 level/radius 确认，再跑
  3 seeds × 1M cap 训练；不先上 5M。

### H161-B：channel-restricted cap 比全通道 cap 更稳

- 机制：速度环由 base controller 保持更稳，RL 只改 pitch/stern 或只改
  heading；这符合 AUV 文献中“速度环 PI + RL 舵/升降舵”的分解。
- 成立应看到：pitch/stern-only 或 heading-only 的 success/path 优于
  三通道 cap，且 speed 通道 residual 接近零。
- 失败应看到：通道限制后 success 不升，说明三通道 cap 0.25 已经是足够
  保守的表示；不要再加结构复杂度。
- 最便宜证伪：复用 cap 0.25 checkpoint 做 eval-only channel mask
  （`--learner-channel-cap-ratios` 或 command channel gate），不重训。

### H161-C：失败是 terminal reward/credit，而不是 authority

- 机制：residual 方向与 signed terminal error 不一致，降低 authority 只是
  把扰动变小，不能产生正增益。
- 成立应看到：cap 0.25 只回到 zero，不能超过；residual-error 相关跨 seed
  符号不一致；entry p90 改善但 final capture 不变。
- 失败应看到：cap 0.25 稳定超过 zero；此时 terminal reward 不是首要矛盾。
- 当前证据：eval-only cap 0.25 已到 `+5/+6` default，H161-C 暂不支持，
  但仍需 strict/其他层确认。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 算法 | SAC，`pc_z_t`，零初始化 residual |
| 训练 seeds | `86201–86203`（与 Goal160 相同，不扩 seed） |
| 预算 | 每 seed `1,000,192` transitions；Phase B 6 runs |
| 训练环境 | H3/C3/ALOS，current `[0.275,0.55]`，shear `1.0`，goal radius `2.0 m` |
| 训练速度 | 固定 `1.2 m/s`；REMUS100 地速上限 `2.5 m/s` |
| 训练 cap | `command residual scale=0.25`（B/C 两臂） |
| eval | dev seeds `20–59`；current10 为主，其他三层为 guard |
| held-out | `100–149` 关闭 |
| 禁止 | 用 eval success 选 checkpoint；改 reward/terminal/success/radius；扩 seed；把 eval-only 诊断写成 formal |

## 4. 方法

### Phase A：eval-only authority confirmation（不训练）

复用 Goal160 的 B/C full-authority checkpoints，在 `scale={0.25,0.5}` 下
补齐 4 levels × default/strict125 × 3 seeds。只用于确认 authority 方向，
不直接作为正式训练结论。

### Phase B：pre-registered cap training

| arm | 接口 | 训练 cap | 训练 terminal gate | 部署 cap |
|---|---|---:|---|---:|
| `B_command_cap025` | command | `0.25` | 无（全程） | `0.25` |
| `C_command_terminal_cap025` | command | `0.25` | `10 m -> 2 m` | `0.25` |

其余全部复用 Goal160：H3、C3、ALOS、`z_t`、reward、terminal、success、
radius、256 envs、batch 256、UTD 0.125、3 seeds。

> **Phase B eval 契约修正（2026-11-12，Gate 不变）**：B 训练时
> `command_residual.terminal_gate_enabled=false`，C 训练时为 `true`
> （`10 m -> 2 m`）。为避免部署 gate 与训练契约错配，Phase B 必须分臂评估：
> B 用 `env_h3_b_command_cap025_eval.json`（gate=false），C 用
> `env_h3_c_command_terminal_cap025_eval.json`（gate=true）。两臂仍使用相同的
> dev seeds、success 定义、radius、level 和 cap；该修正只消除训练/评估
> 分布错配，不改变任何 Gate 阈值。

### Phase C：channel restriction（仅 Phase B 失败时）

在 cap 0.25 checkpoint 上做 eval-only channel mask：pitch/stern-only、
heading-only、speed-only；不训练，不新增 seed。

## 5. 执行计划

- [x] Step 0：写 Goal161 文档、Research Contract、config/manifest；
- [x] Step 1：补齐 Phase A eval-only cap 全 level/radius（96/96，0 FAILED；结果见
  `goal161_phaseA_result_20261112.md`）；
- [x] Step 2：本地 pytest + 远端 smoke（两臂 4 envs × 2048 transitions，cap/gate/
  speed contract 全部通过）；
- [ ] Step 3：Phase B 3 seeds × 1M × 2 arms（2026-11-12 已启动，6 runs）；
- [ ] Step 4：dev40 × 4 levels × default/strict125，H3 zero 配对；
- [ ] Step 5：Gate 裁决；失败写 blocker、两个 hypothesis、reading note、
  最便宜 falsification，再决定 continue/pivot/stop；
- [ ] Step 6：写 result、LEDGER、STATE，跑 checker；不自动 commit/push。

## 6. Gate

| check | 阈值 |
|---|---:|
| 主指标 | current10 default：best cap arm vs zero `>= +4/120`，vs A `>= +4/120` |
| strict | current10 strict125：best cap arm vs zero `>= +2/120`，vs A `>= +2/120` |
| 其他层 guard | nominal/current07/current15 不得低于 `-2/120` |
| seed | 至少 `2/3` seeds 相对 zero `Δsuccess >= 0` |
| path | `Δpath3d <= +0.05 m`，`Δvertical <= +0.05 m` |
| 终端 | 2 m entry 纵向或横向 p90 至少一项改善；final-distance bins 不恶化 |
| residual | `<=2 m` 可见，`>10 m` 接近 0 |
| 风险 | timeout/safety 不增加；0 NaN/飞点；0 超包线速度 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| cap 0.25 训练后仍无增益 | authority 不是唯一问题 | 转 terminal reward/credit 或 channel restriction，不换算法 |
| cap 0.25 有增益但 path 恶化 | 近区权限仍过大 | 降 cap 或 channel cap，单独预注册 |
| cap 0.25 回到 zero 但不能超过 | 基控制器接近局部最优 | 转 terminal capture / base controller，先不训练 |
| strict 显著恶化 | 终端相位问题 | 查 delay/rate/terminal gate，不扩 seed |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal161 \
  --goal-doc doc/goals/goal161_command_residual_authority_cap_20261112.md \
  --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal161：固定 SAC、z_t、H3、Goal85 位置契约、reward/terminal/success/radius，
只研究 command residual authority cap，不换算法、不扩 seed。

先读：
- doc/goals/goal161_command_residual_authority_cap_20261112.md
- doc/goals/goal160_action_parameterization_terminal_capture_20261112.md
- research/blockers/2026-11-12-goal160-command-residual-harm.md
- research/paper_reading/2026-11-12-goal160-command-residual-authority.md

硬约束：
1. 只训练 B_command_cap025 和 C_command_terminal_cap025，3 seeds × 1M；
2. 训练和部署 cap 都是 0.25；速度固定 1.2 m/s，地速上限 2.5 m/s；
3. 不改 reward/terminal/success/radius，不打开 held-out 100–149；
4. 先补齐 Phase A eval-only cap，再做 Phase B 训练；
5. Gate 失败先归因，不换 REDQ/DroQ，不扩 seed。

执行顺序：
- 生成 cap 配置和 manifest；
- 本地 pytest 与远端 smoke；
- Phase A 全 level/radius eval-only；
- Phase B 6 runs × 1M；
- dev40 × 4 levels × default/strict125；
- 按 Gate 裁决并写 result/blocker/ledger/state。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research`
查论文并写 reading note；先跑最便宜 falsification，再决定
`continue` / `pivot` / `stop`；`pivot` 时写新 Goal 文档和 prompt；持续迭代，
直到满足成功标准或触发必须向用户报告的停止条件。不自动 commit/push，
不打开 held-out，不改变任务成功定义或主算法族。
