# Goal153 C1 gate：direct command 评估契约审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_READ_ONLY / INVALID_GATE_CONFIRMED / RUNNER_REPAIRED / HELDOUT_CLOSED`<br>
**上游**：`doc/goals/goal153_reward_curriculum_multienv_20261111.md`<br>
**优先级**：`P0`。200k pilot 的 C1 gate 连续两次为 `0/40`，且 40/40 都是
约 17 m 外的 `GOAL_PLANE_MISS`；在把这次结果解释成 reward/课程失败前，先排除
评估契约错误。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal153 --goal-doc doc/goals/goal153_reward_curriculum_multienv_20261111.md --strict --require-result
```

## 1. 现象

200k scratch pilot 中，R0/R1/R2 的 C1 gate 在 100k 与 200k 都是：

```text
default_success = 0.0
strict_success = 0.0
plane_miss_rate = 1.0
eta_done_distance_mean ~= 17 m
steps_mean ~= 1415
```

同一批 checkpoint 在 nominal default dev40 上仍有 `R0 27/40`、`R1/R2 25/40`。
因此 `0/40` 不是简单的“C1 稍难”，而是评估条件与 nominal 发生了结构性变化。

## 2. 可证伪假设

### H-A：curriculum direct command 与 path-success 契约不一致

`BatchedPCLowLevelEnv` 在 C0/C1 且 `pc_control_mode=alos` 时默认启用 direct command；
训练配置也保持 `curriculum_direct_commands_enabled=true`。direct command 替换参考
heading/pitch/speed，但 success 仍按路径终点计算，于是评估 episode 可以在跟随随机
指令时正常走到离终点很远的 plane-miss。

预期证据：同一个 200k checkpoint、同一 C1 env、同一 seeds，只切换
`--disable-curriculum-direct-command`：

- 关闭 direct command 后 success 显著高于 `0/40`；
- final goal distance 从约 17 m 回到 path-end 附近；
- residual 幅度不是主因。

若关闭后仍 `0/40`，则 H-A 不成立，转查 goal plane/终止/路径终点。

### H-B：C1 `pitch_gain=2` 是主因

当前 `configs/lowlevel/goal153/env/R0_C1.json` 与
`configs/lowlevel/goal136/curriculum/C1.json` 都是 `pitch_gain=2`；Goal147 的 K4
则是 `4`。但 Goal147 的 Z（gain 2）在 nominal C1 仍有 `26/40`，单靠 gain 2
无法解释 `0/40`。本审计只记录该差异，不先改 controller。

## 3. 实验

- checkpoint：`pilot/train/R0_C1/curriculum_checkpoints/C1_000200192.zip`；
- env：`configs/lowlevel/goal153/env/R0_C1.json`；
- stage：`C1`；
- seeds：`20–59`，40 episodes；
- action mode：`policy`，`residual-scale 1.0`；
- 输出：`artifacts/lowlevel/goal153_reward_curriculum_20261111/c1_direct_command_audit/`；
- 不改 reward、success、termination、goal radius、seed 范围或 held-out。

## 4. 执行计划

- [x] Step 0：读取 200k initialization、curriculum decisions、C1 episodes 与配置；
- [x] Step 1：跑 `direct_on` control（当前 gate 口径）；
- [x] Step 2：跑 `direct_off` treatment（Goal147/148 口径）；
- [x] Step 3：比较 success、plane miss、final distance、path error、residual；
- [x] Step 4：写 result，判 200k C1 gate `INVALID_GATE_DIRECT_COMMAND`；
- [x] Step 5：修 runner 并加回归测试；更新 `research/STATE.md` 与 `research/LEDGER.jsonl`。

## 5. 运行命令

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
export PYTHONPATH="src:/data/yhx/underwater-robotics/src/stable-baselines3"
python scripts/evaluate_pc_sac.py \
  --config configs/lowlevel/goal153/train/R0_C1_sac.json \
  --checkpoint artifacts/lowlevel/goal153_reward_curriculum_20261111/pilot/train/R0_C1/curriculum_checkpoints/C1_000200192.zip \
  --environment-config configs/lowlevel/goal153/env/R0_C1.json \
  --curriculum-stage C1 \
  --output artifacts/lowlevel/goal153_reward_curriculum_20261111/c1_direct_command_audit/direct_on \
  --seed-start 20 --seed-count 40 --success-frame both \
  --device cuda --dtype float64 --action-mode policy --residual-scale 1.0 \
  --overwrite
```

`direct_off` 复用同一命令，只增加 `--disable-curriculum-direct-command`。

## 6. 判读规则

| 结果 | 判读 |
|---|---|
| `direct_off` success 明显高于 `direct_on`，final distance 回到 path-end | `INVALID_GATE_DIRECT_COMMAND`；200k reward/课程 Gate 不成立 |
| 两者都 `0/40` | direct command 不是充分原因；转查 C1 路径终点与 plane-miss 契约 |
| 两者都低且 residual 很小 | 才进入 reward/credit/action-authority blocker |

## 7. 结果

固定 200k `R0_C1/curriculum_checkpoints/C1_000200192.zip`，C1 env，dev seeds 20–59：

| mode | success | final distance | path3d | residual L2 ratio | terminal |
|---|---:|---:|---:|---:|---|
| `direct_on`（原 gate） | **0/40** | 15.22 m | 8.38 m | 0.0307 | 40/40 plane miss |
| `direct_off`（Goal147/148 口径） | **26/40** | 1.37 m | 0.86 m | 0.0310 | 26 GOAL / 14 plane miss |

同一个 checkpoint 只切换 direct-command 语义，success 从 `0/40` 变成 `26/40`，
final distance 从 `15.22 m` 回到 `1.37 m`。因此 200k C1 gate 的 `0/40` 不是策略
能力，也不是 reward，而是评估契约错误。

R1/R2 在同一 `direct_off` 口径下也都是 `26/40`，path3d 分别 `0.854/0.854 m`，
与 R0 的 `26/40 / 0.862 m` 无可测差异。R1 的 vertical potential 每步均值
`+0.0026`、标准差 `0.0042`；R2 的 heave potential 每步均值约 `1e-6`。信号确实
存在，但相对 `path_vertical`（std 约 `0.85`）和 terminal `±100/300` 很弱。

## 8. 训练预算的独立限制

200k pilot 为 `256` envs：

```text
200192 / 256 = 782 steps/env
```

而 C1 episode 约 `1160–1350` steps；`training_manifest.json` 记录
`completed_episodes=0`。因此整轮 200k 中没有任何环境真正经历一次回合终止，
`goal_bonus=300`、plane-miss penalty 和 terminal capture 都没有进入训练样本。
即使 reward 设计正确，200k 也不可能学会 success。

另外，200k `initialization.json` 的 `bc_checkpoint=null`，这轮是 scratch；后来的
1M 才真正加载 D3 BC（`encoder_loaded=true`、`head_loaded=true`）。

## 9. 裁决与修复

- 200k C1 gate：`INVALID_GATE_DIRECT_COMMAND`，不能用于否定 reward/课程或残差头。
- 训练 runner 已改为无条件传 `--curriculum-disable-direct-command`，与
  Goal144/147/148 保持一致；新增回归测试防止再次漏掉。
- 无效的 1M direct-on 运行已停止；修正后的 `pilot_bc_1m_alos` 正在运行，100k
  C1 为 `0.65/0.65`，与 Goal144 的有效口径一致。
- 下一步不是冻结/不冻结编码器，而是等修正后的 1M 走完；若 R1/R2 仍与 R0 持平，
  再查 residual 头是否利用纵向表征或 terminal capture，而不是继续扫 reward 权重。

## 10. Commit / Push

本审计只新增文档与只读 artifact。确认结果后再分别提交：

```bash
git add doc/goals/goal153_c1_gate_direct_command_audit_20261111.md
git diff --cached --check
git commit -m "docs(goal153): audit c1 direct-command gate contract"
git push
```

不提交 `artifacts/`、checkpoint、trace 或日志。
