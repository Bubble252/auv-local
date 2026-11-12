# Goal96：terminal capture / reward-credit-assignment 只读审计（预注册）

**日期**：2026-11-11<br>
**状态**：`COMPLETED / DIRECTION_NOT_SUPPORTED / AUTHORITY_SUFFICIENT / PPO_DEGRADES_PRIOR / NO_SAC`<br>
**上游**：Goal95 `SCREEN_FAILED / SUPERVISED_PRIOR_INSTALLED / PATH_INDEPENDENCE_NEGATIVE / NO_SAC`<br>
**性质**：只读回放与离线分析；不训练、不换算法、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal96 --goal-doc doc/goals/goal96_terminal_credit_audit_20261111.md --strict
```

本 Goal 在记录结果前不得修改审计窗口、指标定义、成功/失败分组或 bootstrap 规则。
实现修复可以迭代，但不能根据 dev 结果改变判据。

## 1. 要回答的问题

Goal95 已经证明：

- 监督残差先验安装成功：residual/base 约 `0.548–0.561`；
- path/endpoint/current encoder 全非零，frozen BC drift `0.0`；
- 但闭环 success 没有提高：`P_D3`/`P_BC15` 都是 default `30/40`、
  strict125 `19/40`；
- 同 seed S0 是 default `31/40`、strict `20/40`；
- 终端漏斗损失集中在 `3 m -> 2 m` 和 `2 m -> 1.25 m`。

本 Goal 只回答一个问题：

```text
为什么 residual 非零、方向看起来也不像随机，
但最后几米的 terminal capture 仍然不改善？
```

四个候选机制：

1. terminal window 的 residual 方向不对；
2. terminal window 的 residual 被执行器包线/速率限制截断；
3. reward / credit assignment 只在 done 行给终端信号，pre-terminal closure 没有被
   有效优化；
4. PPO 更新在破坏 frozen BC / 监督先验，而不是改进它。

## 2. 数据与范围

使用 Goal95 已有产物，不新增训练：

- `P_D3`、`P_BC15` 的 initial、final、frozen-base；
- dev40 default + strict125；
- 每个 trace 包含 `eta`、`p_nav`、`goal_ned_m`、`goal_distance_eta_m`、
  `residual_action`、`frozen_residual_action`、`actuator_base_physical`、
  `actuator_residual_applied`、`actuator_residual_effective`、
  `actual_actuator`、`requested_action`、`terminal_kind` 等字段；
- 不打开 held-out `100–149`；
- 不改变 success frame、goal radius、terminal capture radius。

## 3. 审计窗口

对每个 episode，按 `goal_distance_eta_m` 截取四个窗口：

| 窗口 | 用途 |
|---|---|
| `10–5 m` | 远终端接近段，作为对照 |
| `5–3 m` | 进入 gate 后的早期 |
| `3–2 m` | default capture 前最后一米 |
| `2–1.25 m` | strict125 capture 前最后一段 |

如果某 episode 没有进入某个窗口，记为 `not_entered`，不能当成“方向错误”。

## 4. 指标

### 4.1 方向（动作空间，不做 NED 直接 cos）

`r_PPO` 是舵/翼/转速三通道，不是 NED 空间向量；不能直接把
`cos(r_PPO, goal_direction)` 当成方向指标。本 Goal 用两类动作空间代理：

1. **先验对齐**：`r_BC_phys = frozen_residual_base_action - actuator_base_physical`；
   `cos_prior = cos(r_PPO_phys, r_BC_phys)`。它只说明 PPO 是否偏离 BC 先验，
   不单独证明“朝目标”。
2. **经验闭合效应**：对每个窗口计算 next-step
   `radial_closure = -(||e[t+1]|| - ||e[t]||) / dt`；把窗口内 residual 通道与
   `radial_closure` 做相关/线性回归，得到逐通道经验系数。定义
   `action_effect_score = sum_c sign_c * corr(r_c, radial_closure)`，其中
   `sign_c` 由同一窗口的 frozen BC 通道-闭合关系校准，避免把舵/翼符号写死。

计算：

- `radial_closure` 的窗口均值；
- `action_effect_score`；
- `cos_prior`；
- success/failure 两组的均值差与 bootstrap 95% CI（episode 为重采样单位）。

注意：`r_PPO` 的 normalized 与 physical 尺度不能混用；`action_effect_score` 用
normalized residual 做相关即可，`cos_prior` 必须统一到 physical 或统一除以
full-scale。

### 4.2 动作权限

对每个窗口计算：

- `requested_l2`：策略请求的 PPO residual；
- `applied_l2`：进入包线后的 residual；
- `effective_l2`：`actuator_residual_effective`；
- `survival = effective_l2 / max(applied_l2, eps)`；
- 逐通道 survival：rudder / stern / rpm；
- magnitude clip、rate clip、material saturation；
- success/failure 的 clip 差异。

### 4.3 Reward / credit

如果 trace 没有 reward 分项，允许做一次只读 evaluation（仍不训练）来记录
`last_reward_breakdown`。分项至少包括：

- progress；
- terminal_progress；
- track_3d / track_yaw / track_u；
- action / saturation；
- terminal（goal bonus / timeout penalty / out-of-bounds penalty）；
- total / ppo_scaled。

计算：

- 每个分项的 episode return 和方差占比；
- terminal 项在总正 return 中的占比；
- pre-terminal radial closure 与 success 的 episode-level Spearman 相关；
- success/failure 的 reward-to-go 在最后 `5 m/3 m/2 m` 是否提前分离。

### 4.4 先验保持

比较 initial、final、frozen：

- success；
- terminal funnel；
- terminal-window `action_effect_score` 与 `cos_prior`；
- terminal-window residual 与 initial 的 L2 分叉；
- final 是否在失败 episode 上更远离 initial 的成功方向。

## 5. 预注册判据

### 5.1 方向

`DIRECTION_POSITIVE` 当且仅当：

- success 组窗口平均 `action_effect_score` 减去 failure 组均值 `> 0`；
- episode bootstrap 95% CI 下界 `> 0`；
- 至少在 `3–2 m` 或 `2–1.25 m` 窗口成立。

如果 `action_effect_score` 因样本不足不可估，才允许用 `cos_prior` 的
success/failure 分离作为次要证据；不能把 `cos_prior` 单独写成“朝目标方向正确”。

否则记 `DIRECTION_NOT_SUPPORTED`。

### 5.2 权限

`AUTHORITY_SUFFICIENT` 当且仅当：

- terminal window 的 `median(effective_l2 / applied_l2) >= 0.90`；
- success 与 failure 的 clip rate 差 `<= 0.10`。

否则记 `AUTHORITY_CLIPPED`。

### 5.3 Reward

`REWARD_ALIGNED` 当且仅当：

- success 的 pre-terminal radial closure 高于 failure，bootstrap 95% CI 下界 `> 0`；
- terminal 项占总正 return 的比例 `< 0.80`。

否则记 `REWARD_CREDIT_MISALIGNED`。

### 5.4 先验保持

`PRIOR_PRESERVED` 当且仅当：

- final success `>=` initial success（同 arm、同 scenario）；
- final terminal funnel 不差于 initial。

否则记 `PPO_DEGRADES_PRIOR`。

## 6. 决策矩阵

| 方向 | 权限 | Reward | 先验 | 决策 |
|---|---|---|---|---|
| positive | sufficient | aligned | preserved | 才允许预注册 SAC 同协议对照，检查探索/样本效率 |
| positive | clipped | any | any | 先修动作参数化/末端权限，不换算法 |
| positive | sufficient | misaligned | any | 先修 reward/credit assignment，不换算法 |
| not_supported | any | any | any | 先查 terminal residual 方向/观测，不换算法 |
| any | any | any | degraded | 先查 PPO 更新/先验保持，不换算法 |

只有第一行同时成立，才允许把“样本效率不足”作为 SAC 的理由。其余情况都先修
当前 PPO/frozen-BC 契约。

## 7. 执行计划

### Step 0：预注册与审计脚本

- [x] 写本 Goal96 文档；
- [x] 确认 trace 字段和窗口截取逻辑；
- [x] 固定 bootstrap seed、重采样次数和 success/failure 分组规则；
- [x] 写只读审计脚本和单测；
- [x] 在本地 smoke 上验证不读取 held-out。

### Step 1：Phase A trace-only audit

- [x] 对 8 组 Goal95 trace 计算方向、权限、漏斗；
- [x] 输出 per-window、per-arm、per-scenario 表；
- [x] 检查 success/failure 分离和先验保持；
- [x] 如果 Phase A 已能明确归因，直接写 result，不进入 Phase B。

### Step 2：Phase B reward audit（仅在需要时）

- [x] 用同 checkpoint、同 dev40、同场景做只读 evaluation；
- [x] 记录 `last_reward_breakdown`；
- [x] 计算 reward 分项、terminal share、pre-terminal closure 相关；
- [x] 不训练、不改 reward、不打开 held-out。

### Step 3：裁决

- [x] Phase C 逐 seed 配对与 terminal-window initial→final 分叉；
- [x] 按第 6 节决策矩阵写 `doc/goals/goal96_result_20261111.md`；
- [x] 写 blocker / decision；
- [x] 追加 `research/LEDGER.jsonl`；
- [x] 更新 `research/STATE.md` 和根目录 `research/STATE.md`；
- [x] 运行 checker。

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不扩 Goal95 seed；
- 不扫 lr/epoch/gain/seed；
- 不改 reward、terminal、success、goal radius；
- 不打开 held-out；
- 不把 Phase A 的相关性写成因果，除非有窗口内方向/权限的直接证据。

## 9. 预期证据路径

- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/trace.npz`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/terminal_proximity.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40_initial/*/summary.json`
- 新增：`artifacts/lowlevel/goal96_terminal_credit_audit_20261111/`

## 10. 检查命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
python ../scripts/check_research_ledger.py \
  --goal Goal96 \
  --goal-doc doc/goals/goal96_terminal_credit_audit_20261111.md --strict
```
