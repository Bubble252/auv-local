# Goal167：直接动作残差对照——actuator vs ALOS 3D command

**日期**：2026-11-12<br>
**状态**：`GATE167_B_FAIL_TERMINAL_HORIZONTAL_CAPTURE / ACTION_AUTHORITY_PRESENT_BUT_NET_TERMINAL_EFFECT_NULL / NO_ALGORITHM_CHANGE / HELDOUT_CLOSED`<br>
**上游**：Goal166 `GATE166_B_FAIL_ACTION_AUTHORITY_NULL`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal167 --goal-doc doc/goals/goal167_action_parameterization_screen_20261112.md --strict
```

`required: true`：Goal167 是比较动作参数化的 screen；必须保持算法、horizon、
reward/terminal/success/radius 和评测协议不变，只改变 residual 进入闭环的位置。

## 1. 背景

Goal165 修复了 `nu_ground` 控制器反馈，速度包线 PASS。Goal166 修复了
1M/1024 envs 的 terminal 覆盖（3 seeds completed `591/582/595`，
success `525/516/523`），但 controller-gain residual 仍然是 no-op：

- 8 个 radius×level 格子逐 seed `Δ=0/120`；
- near-goal residual norm `1.28`，policy action saturation `0.85–0.98`；
- requested action 平均差只有 `[0.012, 0.029, 0.173]`；
- eta 平均差 `4.5e-3 m`。

因此本 goal 不再调 gain scale，而是比较两种**直接动作残差**：

| arm | 动作空间 | 物理含义 | 关键参数 |
|---|---|---|---|
| `A_actuator_terminal` | actuator | 直接改 rudder/stern/rpm | `residual_scale=0.25`，terminal gate `10→2 m` |
| `B_command_terminal` | command | 直接改 ALOS 的 heading/pitch/speed 指令 | `0.0875 rad / 0.0425 rad / 0.15 m/s`，terminal gate `10→2 m` |

## 2. 冻结约束

- 算法：SAC，`z_t`，H3/C3/ALOS；
- 训练：2 arms × 3 seeds × 1M，seeds `86201–86203`，`num_envs=256`；
- 唯一改动：`action_interface` 及对应 residual 参数；
- reward、terminal、success、goal radius、observation、训练 horizon 不变；
- REMUS100 地速硬上限 `2.5 m/s`；本 goal 不激活速度课程 v2；
- dev seeds `20–59`；held-out `100–149` 关闭；
- zero 复用 Goal165 corrected baseline；不重复跑 zero；
- 不换 PPO/SAC/REDQ/DroQ，不扩 seed，不改 Gate 阈值。

## 3. 可证伪假设

### H167-A：actuator residual 有足够 authority

- 成立：near-goal requested action 改变达到可解释量级（rudder/stern 至少
  `0.05 rad` 或 rpm 至少 `50`），success 在当前压力层改变，且不增加
  timeout/safety。
- 失败：动作改变仍接近 Goal166，或直接导致安全/稳定性退化。

### H167-B：corrected contract 下 command residual 能修正终端

- 成立：command residual 出现 signed-error 相关，current10 达到
  `>=+4/120` default、`>=+2/120` strict125，guards 不显著退化。
- 失败：与 Goal161 一样出现 command-residual harm，或仍无可测收益。

### H167-C：如果两种直接动作残差都无 authority

- 成立：两者 requested action/eta 改变仍接近 0，success Δ 全为 0。
- 结论：residual 被 base controller/actuator 饱和吞掉，下一步查
  reward/terminal credit 或控制频率/authority，不换算法。

## 4. 方法与隔离

- A arm：`action_interface=actuator`，`actuator_residual_scale=0.25`，
  `residual_terminal_gate_enabled=true`，enter/full `10/2 m`；
- B arm：`action_interface=command`，`command_residual.enabled=true`，
  scales `0.0875/0.0425/0.15`，wrapper terminal gate `10/2 m`，
  outer residual gate 关闭，避免 gate 平方；
- 两 arm 的 base controller、plant、current、sensor、reward、terminal 完全一致；
- 训练输出到 `artifacts/lowlevel/goal167_action_parameterization_20261112/train/`；
- 评估输出到同根 `eval/`，zero 用 symlink 指向 Goal165 corrected zero。

## 5. 执行计划

- [x] Step 0：写 Goal167、Research Contract、冻结约束和 Gate；
- [x] Step 1：生成两 arm 配置，校验只有动作接口/residual 参数不同；双臂 4-env smoke 通过；
- [x] Step 2：2 arms × 3 seeds × 1M 训练，检查 terminal 覆盖；
- [x] Step 3：policy-only paired dev40 评估，复用 Goal165 zero；
- [x] Step 4：success/速度/action-authority/signed-error 分析；
- [x] Step 5：result、ledger、STATE、checker；不自动 commit/push。

## 6. Gate

### Gate167-A：训练覆盖

- 两 arm 的 3 个 seed 都 `completed_episodes>=256`、`success_episodes>=128`；
- 无 NaN/Inf；`speed_contract.ok=true`。

### Gate167-B：闭环收益

每个 arm 单独判定：

- current10 default `>=+4/120`；
- current10 strict125 `>=+2/120`；
- nominal/current07/current15 不显著退化；
- 所有 policy/zero `eta<=2.5 m/s`；
- timeout/safety 不高于 zero；
- path/vertical/entry/near/final bins 不被系统性破坏。

本 goal 的 screen 结论：

- `PASS`：至少一个 arm 通过 Gate167-B，且没有安全/速度退化；
- `FAIL`：两 arm 都未通过；
- 若两 arm 都通过，优先 strict125 更稳、requested action 更小、safety 更好的 arm。

### Gate167-C：机制

- signed lateral/vertical 至少一个通道跨 seed 稳定相关；
- near-goal requested action/eta 改变明显大于 Goal166；
- 如果 action authority 仍接近 0，判 `ACTION_AUTHORITY_NULL`，进入 reward/terminal credit 审计。

## 7. 运行与验证

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_pc_goal164_velocity_contract.py \
  tests/test_pc_goal165_controller_feedback.py \
  tests/test_lowlevel_curriculum_sampler.py

ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
GPUS="0 1 2 3 4 5" PARALLEL=6 bash scripts/run_goal167_train.sh
SEEDS="86201 86202 86203" GPUS="0 1 2 3 4 5 6 7" PARALLEL=8 bash scripts/run_goal167_eval.sh
bash scripts/run_goal167_analysis.sh

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal167 \
  --goal-doc doc/goals/goal167_action_parameterization_screen_20261112.md \
  --strict --require-result
```

## 8. 算力与 Git

- 6 个训练 job 可同时在 6 张 4090 上跑；48 个评估 job 用 8 张卡并行；
- 不因为“用满 GPU”而扩 seed 或改预算；
- 禁止 `git add -A`；只 add 本 goal 相关文件；
- 提交前运行 `git diff --cached --check`；不自动 commit/push。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：先看 action authority，再看 success；
如果两 arm 都无 authority，查 reward/terminal credit；不换主算法族，
不打开 held-out，不放宽 2.5 m/s。

## 10. 结果

两臂在 primary `current10` 上逐 seed 同分：
default `102/120 -> 102/120`、strict125 `99/120 -> 99/120`，均未通过
Gate167-B。A 的 actuator authority 已达到 `[0.059, 0.080, 288]` 量级，
B 的 command residual 在 full authority 下仍只改变 final distance 约
`1.1 cm`。

失败机制不是纵向、NoiseNet 或样本效率，而是横向 `GOAL_PLANE_MISS`：
18/120 失败全部 kind 9，横向 `2.2–11.15 m`、纵向 `<0.31 m`。详见
`doc/goals/goal167_result_20261112.md`。

裁决：`pivot` 到 Goal168 只读 terminal horizontal capture /
homing authority 审计；不换算法、不扩 seed、不改 reward/terminal/
success/radius、不打开 held-out。
