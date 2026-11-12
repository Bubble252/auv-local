# Goal101：同 checkpoint 的 ALOS / BC-only 配对基线

**日期**：2026-11-11<br>
**状态**：`COMPLETE / BC_PRIOR_BEATS_ALOS / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal100 `BC_SCALE_ONLY_SUPPORTED / PPO_INCREMENT_HARMFUL / NO_SAC`<br>
**性质**：eval-only、冻结 Gate；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal101 --goal-doc doc/goals/goal101_alos_same_checkpoint_pairing_20261111.md --strict
```

## 1. 背景

Goal100 发现 `B20`：

```text
frozen BC only + actuator_residual_scale 2.00
default 33/40, strict125 20/40, timeout/safety 0
path 1.013x B15
```

但 `B20` 相对“传统控制器”的正式增益还没有同 checkpoint、同配置、同 seeds
的配对结果。Goal85 的历史 zero-residual candidate 是 `0.655/0.265`，`B20`
是 `0.825/0.500`；两者谱系不同，不能直接当正式结论。

本 Goal 只做一件事：在同一个 Goal95 checkpoint、同一 Goal85 unbiased
candidate 环境、同一 dev seeds 上，跑真正的 ALOS / zero-residual 基线，
再与 Goal100 的 `B20` 逐 seed 比较。

## 2. 方法

新增 eval-only `alos` action mode：

```text
跳过 frozen-BC wrapper
BatchedPCActuatorResidualEnv 接收全零 residual
=> ALOS + PD/PI 传统控制
```

该模式不读取 PPO actor 的动作，不改变 checkpoint、reward、terminal、success
或环境配置；只用于评价传统基线。

## 3. 数据与隔离

- 训练：不训练；
- eval：dev seeds `20–59`，每 run 40 episodes；
- scenario：Goal85 unbiased default、strict125；
- checkpoint：Goal95 `P_D3/final.pt`、`P_BC15/final.pt`（只用于通过加载契约；
  `alos` 不执行 PPO 动作）；
- success frame：`both`；
- held-out `100–149` 禁止打开；
- `B20` 复用 Goal100 结果，不重跑。

新跑矩阵：

```text
2 arms × 2 scenarios = 4 eval runs
```

Step 2 smoke 已验证：`action_mode=alos`、attenuation disabled、trace 无 attenuation/
frozen-residual 字段、`actuator_residual_applied` 全零；正式 4-run 前冻结 Gate。

## 4. Gate

### 4.1 `INVALID`

- `B20` 不能复现 33/20；
- ALOS run 缺 summary/trace；
- checkpoint SHA、seed、success frame 不匹配；
- NaN/Inf 或读取 held-out。

### 4.2 `BC_PRIOR_BEATS_ALOS`

`B20` 同时满足：

- 两臂 default `> ALOS default`；
- 两臂 strict `> ALOS strict`；
- path 3D RMSE 相对 ALOS 平均恶化 `<=5%`；
- timeout/safety 不增加。

### 4.3 `BC_PRIOR_PARITY`

`B20` 与 ALOS 的 success 差不超过 `1/40`，且 path 不恶化超过 `5%`。

### 4.4 `BC_PRIOR_BELOW_ALOS`

任一主场景 `B20` 低于 ALOS 超过 `1/40`。

### 4.5 `INCONCLUSIVE`

不满足 `BEATS`、`PARITY` 或 `BELOW`，且没有 `INVALID` 证据时，记为
`INCONCLUSIVE`，不把它改写成任何有利结论。

## 5. 执行计划

- [x] Step 0：确认 Goal100 结果与网络 parity；
- [x] Step 1：实现 eval-only `alos` 模式；
- [x] Step 2：单测 + 1 seed smoke；
- [x] Step 3：远端跑 4 个 dev40 ALOS eval；
- [x] Step 4：按冻结 Gate 判断；
- [x] Step 5：补 result/decision、LEDGER、STATE，并运行 checker。

## 6. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不改 reward、terminal、success、goal radius；
- 不扩 seed，不打开 held-out；
- 不用 Goal85 历史数字冒充本 Goal 的同 checkpoint 配对；
- 不把 `BC_PRIOR_PARITY` 写成 `BC_PRIOR_BEATS_ALOS`。

## 7. Goal 提示词

```text
你现在执行 Goal101：同 checkpoint 的 ALOS / BC-only 配对基线。

先读：
- doc/goals/goal101_alos_same_checkpoint_pairing_20261111.md
- doc/goals/goal100_result_20261111.md
- doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md

硬约束：
1. 固定 Goal95 checkpoint、Goal85 unbiased candidate 环境、dev seeds 20-59、success frame both；
2. 不训练、不扩 seed、不打开 held-out、不换算法；
3. 不改 reward/terminal/success/goal radius；
4. ALOS 必须是同配置下的全零 actuator residual，不允许用历史不同谱系数字代替；
5. 失败后先归因，不直接改算法。

执行顺序：
- 实现/验证 eval-only alos；
- 单测 + 1 seed smoke；
- 远端跑 4 个 dev40 ALOS eval；
- 与 Goal100 B20 逐 seed 比较；
- 按 Gate 判 BC_PRIOR_BEATS_ALOS / BC_PRIOR_PARITY / BC_PRIOR_BELOW_ALOS / INVALID；
- 写 result/decision、LEDGER、STATE，并运行 checker。
```

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪
hypothesis、查论文/代码、跑最便宜 falsification、写 LEDGER decision；不自动
commit/push，不打开 held-out，不改变成功定义或主算法族。
