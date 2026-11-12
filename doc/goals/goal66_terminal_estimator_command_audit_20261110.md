# Goal66：terminal estimator 与 command 的只读因果筛查

**日期**：2026-11-10<br>
**状态**：`TERMINAL_GEOMETRY_UNRESOLVED`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal64 `GEOMETRY_SUPPORTED`、Goal65 `DWELL_NOT_SUFFICIENT`<br>
**服务器**：`yhx@152.136.183.51 -p 43127`，8×RTX 4090，384 CPU

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal66 --goal-doc doc/goals/goal66_terminal_estimator_command_audit_20261110.md --strict
```

## 1. 背景与问题

Goal65 已经排除“dwell 太短”：50 步窗口内，strict125 的 6 个 baseline true-success
丢失 episode 没有一个进入 `1.00 m`，default 的 5 个丢失 episode 也没有进入
`1.75 m`。继续等待只会增加终点距离。

Goal66 只问一个诊断问题：

**丢失 episode 的剩余障碍到底是 `p_nav` 估计误差，还是 terminal command 没有
朝向真实 goal，还是速度没有收敛？**

本 Goal 不训练，不新增 rollout；直接复用 Goal60 baseline trace 与 Goal65
`_dwell50_m025` candidate trace。重点检查 candidate terminal 前最后 51 步，因为
这是 Goal65 已经记录、且覆盖 post-crossing 行为的固定窗口。

## 2. 可证伪假设

### H66-A：`p_nav` 估计器单独截断了 true capture

机制：如果 true `eta` 在窗口内已经进入收紧球，而 termination 使用的 `p_nav`
始终在球外，那么失败是估计器/校准问题；只改 command 不能修复。

支持条件（pooled lost true）：

```text
p_nav-only capture fraction >= 0.50
p_nav-only = (min eta distance <= r_tight) and (min p_nav distance > r_tight)
```

若这个比例低于 `0.50`，则 H66-A 不支持，不能把问题写成“只是 p_nav 估计偏差”。

### H66-B：terminal command 没有朝向真实 goal

机制：如果 controller 收到的 corrected command 在最近接近点仍不朝向 goal，则即使
估计和终止规则正确，载具也会继续离开捕获球。方向错误应表现为 command-to-goal
cosine 低，且在 lost/retained 之间存在可分辨差异。

支持条件（pooled lost vs retained，最近 `eta` 点）：

```text
lost corrected-command LOS cosine median < 0.95
retained median - lost median >= 0.05
```

若 lost command 已经高度对准 goal，H66-B 不支持，下一步应优先查速度/动力学。

### H66-C：terminal speed command 没有收敛

机制：turn radius 与速度成正比。如果最近接近点的 speed command 仍接近 episode
初始命令速度，载具可能已经没有足够距离/转弯能力进入球。

支持条件（pooled lost vs retained）：

```text
lost speed-command ratio median >= 0.75
lost median - retained median >= 0.10
ratio = corrected speed command / episode initial base speed command
```

若速度已经明显下降，H66-C 不支持。

### Reward 复核（描述性，不单独设 Gate）

记录最后 51 步的 `reward_terminal_progress`、`reward_goal_plane_miss`、
`reward_terminal`、path/command/total reward 的累计值和终点处分量占比，检查
terminal penalty 是否仍然只在最后一帧才出现。该部分用于约束下一步，不单独宣称
reward 是因果主因。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal60 的 10 个 trace，dev `20–59` |
| candidate | Goal65 的 4 checkpoint × 2 scenario policy + 2 zero-residual |
| candidate suffix | `_dwell50_m025` |
| trace window | candidate 每个 seed 最后 51 行 |
| checkpoint | Goal59 `240/752/880/1008` |
| scenario | default `r_tight=1.75 m`；strict125 `r_tight=1.00 m` |
| 精度 / 终止 | CPU / float64；`endpoint_stop` |
| 操作 | 只读分析；不训练、不 rollout、不改 checkpoint/config/reward |
| held-out | `100–149` 关闭 |

## 4. 计算方法

对每个 seed：

1. 从 baseline episodes 选出 `success & terminal_eta_capture` 的 true success；
2. 按 seed 对齐 candidate episode；
3. 在 candidate 最后 51 行中找 true `eta` 到 goal 的最小距离点 `i*`；
4. 在 `i*` 读取 `p_nav` 距离、corrected command、speed command 和 reward；
5. corrected command 由 `base_command + residual` 恢复，heading 使用 wrap，
   pitch 使用代码中的 `[-pi/2, pi/2]` 限幅；
6. command LOS 是 `eta(i*)` 到 goal 的 3D 单位向量；
7. 对 lost 与 retained 分别汇总中位数和逐 seed 值。

不改变任何运行时状态；分析器只读取 `episodes.json`、`trace.npz` 和
`summary.json`。

## 5. Gate（测量前冻结）

本 Goal 是分类 Gate，不是性能 Gate：

| 观察 | 判定 | 下一步 |
|---|---|---|
| H66-A 支持 | `ESTIMATOR_ENTRY_LIMITED` | 预注册 `p_nav` calibration / eta-aware 捕获诊断 |
| H66-A 不支持且 H66-B 支持 | `TERMINAL_COMMAND_MISALIGNED` | 预注册 oracle direct-LOS command 反事实 |
| A/B 不支持且 H66-C 支持 | `TERMINAL_SPEED_NOT_DECAYING` | 预注册 terminal speed/braking profile 反事实 |
| A/B/C 均不支持 | `TERMINAL_GEOMETRY_UNRESOLVED` | 转执行器/动力学 reachability，不扫训练参数 |
| 数据/实现不完整 | `INVALID` | 只修测量实现，不改阈值 |

如果多个 hypothesis 同时支持，按 A → B → C 的因果顺序报告，不把相关指标写成
独立因果证据。

若 default 与 strict125 指向不同机制，总状态记为 `MIXED_TERMINAL_MECHANISM`；
下一 Goal 优先采用 strict125（部署更严格场景）的机制，并在文档中保留 default
反例，不把两个场景强行合并。

## 6. 算力与验证

- 分析在服务器 CPU 上进行，直接读取已有 trace；
- 不新增 `num_envs`、seed、训练 batch 或 GPU 任务；
- 分析器必须对 seed、SHA、finite、dwell metadata、scenario radius 做校验；
- 定向测试覆盖 trace 对齐、最近点选择、corrected command 和 Gate 边界；
- 结果写 JSON/Markdown，并拉回本地 `artifacts/lowlevel/goal66_terminal_estimator_command_20261110/`。

## 7. 执行计划

- [x] Step 0：复核 Goal60/64/65、trace schema 和服务器状态；
- [x] Step 1：实现只读 analyzer，校验 baseline/candidate 对齐与 dwell metadata；
- [x] Step 2：定向测试、Ruff 和 1 个 run 的结果结构 smoke；
- [x] Step 3：服务器运行 analyzer，生成 JSON/MD 与 H66-A/B/C 判读；
- [x] Step 4：写 result、blocker/reading note、LEDGER、两份 STATE；
- [x] Step 5：strict checker、全量 pytest、Ruff、`git diff --check`；
- [ ] Step 6：只 add 相关文件；用户明确要求后才 commit/push。

**结果**：`doc/goals/goal66_result_20261110.md`。H66-A/B/C 均不支持，总状态
`TERMINAL_GEOMETRY_UNRESOLVED`；下一步是 Goal67 执行器/动力学 reachability 审计。

**验证**：strict ledger checker PASS；full pytest `1257 passed`；touched-file
Ruff PASS；`git diff --check` PASS。JSON SHA256
`f1a72828028150ad036a90799c999de5550b807b61747a6a300a66af98811926`。

运行命令：

```bash
python3 scripts/analyze_pc_goal66_terminal_estimator_command.py \
  --baseline-root artifacts/traces \
  --candidate-root artifacts/goal65_plane_miss_dwell \
  --candidate-suffix _dwell50_m025 \
  --output artifacts/goal66_terminal_estimator_command/analysis/goal66.json \
  --markdown artifacts/goal66_terminal_estimator_command/analysis/goal66.md
```

## 8. 明确禁止

- 不重开 Goal27 的 residual/reward/gamma/seed 搜索；
- 不训练，不改 controller/reward/config/checkpoint；
- 不扫描 termination、margin、speed profile 或 command 阈值；
- 不打开 held-out `100–149`；
- 不启动 SAC/REDQ/DroQ；
- 不 `git add -A`，不自动 commit/push。
