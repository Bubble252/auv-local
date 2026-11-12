# Goal168：只读 terminal horizontal capture / homing authority 审计

**日期**：2026-11-12<br>
**状态**：`AUDIT_COMPLETE / TERMINAL_HEADING_LAG_DOMINANT / NO_TRAINING / NO_CONTRACT_CHANGE / HELDOUT_CLOSED`<br>
**上游**：Goal167 `GATE167_B_FAIL_TERMINAL_HORIZONTAL_CAPTURE`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal168 --goal-doc doc/goals/goal168_terminal_horizontal_capture_authority_20261112.md --strict
```

`required: true`：本 goal 是 Goal167 失败后的只读归因审计。不得在审计完成前
训练新策略、改 reward/terminal/success/radius 或切换算法族。

## 1. 背景与证据

Goal167 的 A（actuator residual）和 B（ALOS 3D command residual）在 primary
`current10` 上都是逐 seed 同分：

- default `102/120 -> 102/120`；
- strict125 `99/120 -> 99/120`；
- success 集合 only-policy/only-zero 均为 `0/0`。

但动作 authority 并不为零：

- A near-goal requested action 改变 `[0.0593, 0.0805, 287.8]`，
  policy saturation `0.853`；
- B 在 outer gate 关闭、full command authority 下，requested action 改变
  `[0.0398, 0.0562, 31.52]`，policy saturation `0.980`；
- 两臂 final-distance 绝对变化平均只有 A `0.0136 m`、B `0.0111 m`。

18 个 primary 失败全部是 `GOAL_PLANE_MISS`，横向 `2.2–11.15 m`，纵向
`<0.31 m`。A 中 11/18 的失败在 gate 完全打开前越平面；另外 7/18 在 gate
接近 1 时仍以 `2.2–2.7 m` 横向误差失败。B 所有失败都有完整 command 权限。

因此当前问题不是“残差幅度为 0”，而是：**终端横向修正为什么没有转化为
净位置收敛**。在继续训练前，必须先判断是 guidance 没生效、projection 跳段，
还是物理 turn-rate/lateral acceleration 本身不足。

## 2. 可证伪假设

### H168-A：guidance 没有真正持续收敛

- 成立：失败 episode 在 `goal_distance<=20 m` 时，robust-terminal
  direct-homing 未持续生效；或 projection 在终点附近跳段/回退，导致命令
  仍指向路径切线而不是 goal。
- 失败：terminal phase 一直生效、projection 单调稳定，但横向误差仍不收敛。
- 最便宜 falsification：只读 trace 的 `guidance_terminal_phase`、
  `path_progress`、`path_horizontal_error`、`command_physical`，不需要训练。

### H168-B：物理横向 authority 不足

- 成立：在最后 20 m 内，把横向误差压到 2 m 所需的 turn-rate/lateral
  acceleration 超过实测可用值（rudder、turn rate、crab、current 包线）。
- 失败：至少 2/3 seeds 的失败 episode 中，required 的 p90 小于 available
  的 0.8 倍。
- 最便宜 falsification：用现有 trace 计算 required vs available，不训练。

### H168-C：authority 足够，但 residual 的 credit/direction 错

- 成立：terminal phase 生效、projection 稳定、required authority 在包线内，
  但 residual action 与 signed cross-track error 的相关性接近 0 或符号不稳。
- 失败：B 在 full authority 下仍只改变 final distance 约 1 cm，说明不是
  单纯 credit 不足。

## 3. 数据与隔离

- 数据：Goal167 `eval/{A_actuator_terminal,B_command_terminal}/default/current10/`
  的 `zero_seed86201` 与 3 个 policy seeds；
- 失败集合：primary 18 个 `GOAL_PLANE_MISS`（每 arm 3 seeds × 6）；
- eval：dev seeds `20–59`；
- held-out `100–149`：关闭；
- 训练：本 goal 不启动任何训练；
- oracle fallback：禁止把真值 current/plant 注入 actor；只允许在审计中用于
  计算 required authority。

## 4. 方法

对每个失败 episode，在 `goal_distance<=20 m` 的每个 step 计算：

1. **required turn-rate**：由横向误差 `e`、剩余沿轨距离 `s` 和时间
   `t=s/max(ground_speed,eps)` 估计
   `omega_req ≈ atan2(e, s) / t`；
2. **required lateral acceleration**：`a_req ≈ 2e/t^2`；
3. **available turn-rate**：使用同一 episode 的实测 yaw rate/`nu` 角速度，
   取 p90/p95，并记录 rudder/stern 饱和、crab 角度、current 速度；
4. **available lateral acceleration**：`a_avail ≈ ground_speed * omega_avail`；
5. **authority margin**：`m = a_avail/a_req` 与 `omega_avail/omega_req`；
6. **guidance 状态**：`guidance_terminal_phase` 是否在 `goal_distance<=20 m`
   持续为真；`command_physical` 的 heading 是否指向 goal bearing；
7. **projection 稳定性**：`path_progress` 是否单调，是否在终点附近发生
   超过 `5 m` 的单步跳变或回退；
8. **residual 方向**：residual/requested action 与 signed cross-track error
   的 per-seed 相关性。

输出：

- `artifacts/lowlevel/goal168_terminal_horizontal_authority_20261112/audit.json`
- `artifacts/lowlevel/goal168_terminal_horizontal_authority_20261112/audit.md`
- 每个失败 episode 一行的 required/available/margin 表；
- 机制分类计数。

## 5. 执行计划

- [x] Step 0：写 Goal168、Research Contract、冻结约束和 Gate；
- [x] Step 1：实现只读审计脚本，跑 1 个失败 episode 的 smoke；
- [x] Step 2：审计 18 个 primary 失败 episode；
- [x] Step 3：按 seed 汇总 required/available/margin 与机制分类；
- [x] Step 4：写 result、blocker/decision、ledger、STATE、checker；
- [x] Step 5：根据 H168-A/B/C 裁决下一步；不自动 commit/push。

## 6. Gate

### Gate168-A：审计完整性

- 18/18 失败 episode 都有 trace；
- 每 episode 至少 5 个 `goal_distance<=20 m` 的 step；
- 无 NaN/Inf；held-out 未打开；无训练启动。

### Gate168-B：机制分类

按跨 seed 多数判读，且每类必须给出至少 2 个 episode 证据：

| 分类 | 判据 | 含义 |
|---|---|---|
| `GUIDANCE_NOT_ACTIVE` | terminal phase 在 `<=20 m` 中 <90% 为真 | guidance 切换/生效问题 |
| `PROJECTION_JUMP` | `path_progress` 单步跳变/回退 >5 m | 终点投影不稳定 |
| `AUTHORITY_INSUFFICIENT` | `a_req/a_avail` 或 `omega_req/omega_avail` 的 p90 >1 | 物理机动不足 |
| `AUTHORITY_SUFFICIENT` | required p90 <0.8×available | 物理可达，下一步查 credit/direction |

裁决：

- `AUTHORITY_SUFFICIENT`：预注册 deterministic terminal homing 对照，
  不改 success/radius/reward；
- `GUIDANCE_NOT_ACTIVE` / `PROJECTION_JUMP`：预注册最小 guidance 修复，
  不做 RL 训练；
- `AUTHORITY_INSUFFICIENT`：记录当前 success 契约在 P2 终点不可达；
  不放大 radius/success 救，需用户决策是否改课程/路径族。

## 7. 运行与验证

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_sampler.py

# 远程只读审计（示意，脚本在 Step 1 实现）
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_goal168_terminal_horizontal_authority.py \
  --eval-root artifacts/lowlevel/goal167_action_parameterization_20261112/eval \
  --output artifacts/lowlevel/goal168_terminal_horizontal_authority_20261112

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal168 \
  --goal-doc doc/goals/goal168_terminal_horizontal_capture_authority_20261112.md \
  --strict --require-result
```

## 8. 算力与 Git

- 本 goal 只读 trace，CPU 即可；不使用 8×4090 训练；
- 不扩 seed、不改预算、不改主算法族；
- 禁止 `git add -A`；只 add 本 goal 相关文件；
- 提交前运行 `git diff --cached --check`；不自动 commit/push。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 用 `paper-search` / `deep-research` 查论文，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal 和 prompt，不修改失败 Gate 的阈值；
6. 不自动 commit/push，不打开 held-out，不改变成功定义或主算法族。

## 10. Goal 提示词

```text
你现在执行 Goal168：只读 terminal horizontal capture / homing authority 审计。

先读：
- doc/goals/goal168_terminal_horizontal_capture_authority_20261112.md
- doc/goals/goal167_result_20261112.md
- research/blockers/2026-11-12-goal167-terminal-horizontal-capture-null.md
- research/paper_reading/2026-11-12-goal167-terminal-capture-authority.md

硬约束：
1. 不训练、不换 PPO/SAC/REDQ/DroQ、不扩 seed；
2. 不改 reward/terminal/success/radius，不放宽 REMUS100 2.5 m/s；
3. 不打开 held-out 100–149；
4. 只使用 Goal167 dev traces，先跑最便宜的只读 falsification；
5. 失败后写 blocker/decision/ledger，再决定 continue/pivot/stop。

执行顺序：
1. 实现 required vs available turn-rate/lateral acceleration 审计；
2. 审计 18 个 GOAL_PLANE_MISS episode；
3. 检查 terminal phase、path_progress 跳变、residual 方向；
4. 按 Gate168-B 分类；
5. 写 result 和下一步预注册，不自动 commit/push。
```

## 11. 结果

审计完成：A/B 两臂各 18 个失败，15/18 为 `TERMINAL_HEADING_LAG`，
3/18 为 `PROJECTION_JUMP`。terminal phase 在 20–15 m 的 duty 为
`0.96–1.00`，20 m 入口 authority margin 为 `2.17–33.76`；但命令需要转
`0.74–1.47 rad`，实测 yaw 仅 `0.061–0.087 rad/s`，15/18 的转向时间
达到剩余时间的 `0.87–1.88×`。

裁决：H168-A PARTIAL/SUPPORTED（terminal 切换时序太晚），H168-B 在 20 m
处 NOT SUPPORTED，H168-C 不需要。详见
`doc/goals/goal168_result_20261112.md`。

下一步：Goal169 deterministic terminal homing timing / blending 对照；
不训练 residual、不改 success/radius/reward/terminal、不换算法。

