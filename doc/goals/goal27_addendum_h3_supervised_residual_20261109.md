# Goal27 Addendum：H3-SR 监督残差预注册

日期：2026-11-09<br>
状态：`COMPLETED_H3_SR_FAIL`<br>
父 Goal：`doc/goals/goal27_command_residual_credit_assignment_20261109.md`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal27 --goal-doc doc/goals/goal27_addendum_h3_supervised_residual_20261109.md --strict
```

## 1. 范围与命名消歧

父 Goal 的 §2 把“无侧向推进器/舵角饱和”编号为 H3；阻塞记录
`research/blockers/2026-11-09-goal27-command-residual-collapse.md` 和
`research/LEDGER.jsonl` 则把“on-policy PPO 样本效率/探索不足”编号为 H3。本追加
使用 **H3-SR** 专指后者，避免把两个不同机制混在一起：

- **H3-SR**：PPO 的 on-policy 探索和样本效率不足，导致残差均值塌缩；用
  teacher-minus-D3 的监督标签直接预训练残差，应能学出可用修正。
- **父 Goal H3（执行器饱和）**：本追加不检验。它仍是独立问题，不能把本追加结果
  追认成该假设的结论。
- **C1/R1/H1/G1/A1**：原有 FAIL/PIVOT 结论保持不变；本追加不是命令残差 PPO 的
  seed 救结果，也不是继续扫 reward 权重。

本追加只做一个最便宜的闭环 falsification：在 frozen D3 上，用 D3 自己访问到的
状态、由 ALOS/PC teacher 标注动作，监督训练一个动作残差，然后在固定 dev 矩阵上
与同 checkpoint 的 frozen-base 逐 seed 配对比较。

## 2. 可证伪假设

**H3-SR 成立的必要证据**：监督残差在闭环中有非零幅度，并且至少一个主场景的
pre-goal 3D RMSE 相对 paired frozen-base 改善 `>=0.10 m`，另一个场景不恶化超过
`3%`；同时 success、safety、plane-miss、timeout 不恶化。

如果残差幅度非零但闭环 RMSE 不变或恶化，则 H3-SR 不成立：on-policy 样本效率不是
唯一或主要限制，监督标签本身也不足以产生可靠的闭环修正。

如果残差幅度仍接近零，则说明本轮监督数据/标签/训练预算不足以检验 H3-SR，记录为
`INVALID`，不能写成 H3-SR 被证伪。

## 3. 固定数据与隔离

| 项目 | 固定值 |
|---|---|
| base | `checkpoints/goal26/d3/pc_bc_actor.pt` |
| base SHA256 | `7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73` |
| 训练配置 | `configs/lowlevel/pc_goal27_a1_ppo_default_27006.json` |
| 训练环境 | `configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json` |
| 示范来源 | `scripts/collect_pc_bc_dagger_demos.py`，D3 执行动作、teacher 标注 |
| 训练 seeds | `60–99`（40 个 env，均不在 dev `20–59` 或 held-out `100–149`） |
| 采集规模 | `num_envs=40`，`steps_per_env=8192`，共 327,680 个 transition |
| 动作噪声 | `0.0`（本轮只检验监督标签，不把探索噪声混入归因） |
| 评估 dev seeds | `20–59`，40 seeds，固定 |
| held-out | `100–149` 继续冻结，禁止打开 |

示范只在 default 环境采集；strict125 只用于闭环评估，不参与训练。所有输出写入
远程隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/artifacts/h3_sr/
```

不覆盖 Goal24/25/26 的 checkpoint、配置、数据或 artifacts。

## 4. 训练与评估合同

### 4.1 DAgger 示范

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/repo
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/collect_pc_bc_dagger_demos.py \
  --checkpoint checkpoints/goal26/d3/pc_bc_actor.pt \
  --config configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json \
  --output artifacts/h3_sr/demos_d3_dagger_default_s60_99 \
  --num-envs 40 --steps-per-env 8192 \
  --seed-start 60 --episode-id-offset 700000 \
  --action-noise-std 0.0 --seed-role h3_sr \
  --device cuda --dtype float64
```

采集后检查 `demos.manifest.json`：`student.checkpoint_sha256` 必须等于 D3 SHA256，
`path_dim=8`、`actor_command_dim=0`、无 NaN/Inf；样本数必须为 `40 × 8192`。

### 4.2 监督残差训练

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/pretrain_pc_residual.py \
  --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
  --demos artifacts/h3_sr/demos_d3_dagger_default_s60_99/demos.npz \
  --output artifacts/h3_sr/pretrain_residual_sr01 \
  --batch-size 4096 --max-epochs 200 --patience 30 \
  --learning-rate 1e-3 --weight-decay 1e-5 \
  --output-l2 1e-4 --smoothness-l2 0.0 \
  --validation-fraction 0.2 --seed 27007 --device cuda
```

固定使用 `final.pt`，不使用 best-epoch 事后挑选。训练 manifest 必须满足
`status=completed`、所有 loss finite、frozen BC 参数漂移为 `0`、残差通道非零。

### 4.3 闭环评估

每个 scenario 都评估同一个 `final.pt` 的三种模式：

- `--action-mode policy`：D3 + 监督残差；
- `--action-mode frozen-base`：同 checkpoint 的 D3 零残差配对基线；
- `--path-ablation path-zero`（default 至少一组）：检查残差是否真的使用路径输入。

```bash
for scenario in default strict125; do
  env=configs/lowlevel/pc_goal27_a1_reward_v2_${scenario}_20261109.json
  out=artifacts/h3_sr/eval_${scenario}_40
  PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
    scripts/evaluate_pc_ppo.py \
    --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
    --checkpoint artifacts/h3_sr/pretrain_residual_sr01/final.pt \
    --environment-config "$env" --output "$out/policy" \
    --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
    --action-mode policy
  PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
    scripts/evaluate_pc_ppo.py \
    --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
    --checkpoint artifacts/h3_sr/pretrain_residual_sr01/final.pt \
    --environment-config "$env" --output "$out/frozen_base" \
    --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
    --action-mode frozen-base
done
```

## 5. Gate（测量前冻结）

所有比较均为同 checkpoint、同 seeds、同传感器 profile、`endpoint_stop`、
CPU/float64、第 200 轮 `final.pt`。

| check | 通过条件 |
|---|---|
| 主效果 | default 或 strict125 的 pre-goal 3D RMSE 相对 paired frozen-base 改善 `>=0.10 m` |
| 另一场景 | 另一个 scenario 的 pre-goal RMSE 恶化不超过 `3%` |
| success | 两个 scenario 都不低于 paired frozen-base 超过 `0.025`；且 default `>=0.695`、strict125 `>=0.470` |
| safety | 相对 paired frozen-base 恶化不超过 `0.02` |
| plane miss / timeout | 相对 paired frozen-base 恶化不超过 `0.05` |
| 残差非零 | `residual_to_base_l2_ratio_mean >= 0.01`，且 residual abs mean `>0` |
| 路径归因 | default 的 `path-zero` pre-goal RMSE 相对正常 path 恶化至少 `5%`；否则残差不能声称使用了路径信息 |
| 稳定性 | 0 NaN/Inf、0 safety、0 timeout 新增超门禁 |

判定规则：

- 主效果通过且护栏全通过：`H3_SR_PASS`；
- 主效果失败：`H3_SR_FAIL`，不继续扫 lr/epoch/seed；
- 残差近零或数据/manifest 不合法：`INVALID`，修复数据链路后只重跑一次，不把
  `INVALID` 写成 `FAIL`；
- 两个场景的配对差小于 seed noise、CI 跨零且不足 `0.10 m`：`AMBIGUOUS`，按
  Research Escalation 写 blocker，不直接换主线。

## 6. 执行计划

- [x] Step 0：确认 D3 SHA、A1 config hash、远程目录和 GPU；未触碰 held-out；
- [x] Step 1：运行 smoke 采集，检查 manifest；2-seed 版本因单 episode 无法切分而改为
      `2 env × 128 steps`，仅用于链路验证；
- [x] Step 2：运行完整 40×8192 DAgger 采集并验证样本数/哈希；
- [x] Step 3：运行监督残差预训练，检查 manifest、loss、冻结参数漂移；
- [x] Step 4：在 dev `20–59` 运行 default/strict125 的 policy 与 frozen-base 配对评估；
- [x] Step 5：运行 default `path-zero` 归因评估；
- [x] Step 6：按 §5 写 result、更新 ledger/STATE，运行 checker，只 add 相关文件；
- [ ] Step 7：用户明确要求后才 commit/push。

## 6.1 结果摘要

完整结果见 `doc/goals/goal27_h3_sr_result_20261109.md`。监督残差确实非零
（default/strict125 的 residual/base L2 为 `16.14%/16.33%`），但相对同 checkpoint
的 frozen-base，pre-goal RMSE 只改善 `0.0231/0.0231 m`，success 从
`0.725/0.500` 降到 `0.650/0.350`，plane-miss 增加 `0.075/0.150`。因此
**H3-SR FAIL**；不继续扫 lr/epoch/seed，也不 warm-start PPO。

## 7. Git 步骤

禁止 `git add -A`。只添加本追加文档、结果文档、必要的脚本/测试和 research 记录：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git add doc/goals/goal27_command_residual_credit_assignment_20261109.md \
  doc/goals/goal27_addendum_h3_supervised_residual_20261109.md \
  doc/goals/goal27_h3_sr_result_20261109.md \
  doc/goals/goal27_result_20261109.md \
  research/blockers/2026-11-09-goal27-command-residual-collapse.md \
  research/decisions/2026-11-09-goal27-h3sr-stop.md \
  research/LEDGER.jsonl research/STATE.md ../research/STATE.md
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal27 \
  --goal-doc doc/goals/goal27_addendum_h3_supervised_residual_20261109.md --strict
git commit -m "research: preregister and evaluate Goal27 H3 supervised residual"
```

推送前先确认远端和分支：

```bash
git remote -v
git branch --show-current
git push origin "$(git branch --show-current)"
```

本回合不自动 commit/push；只有用户明确要求才执行上述提交或推送。

## 8. 失败后的唯一允许分支

- `H3_SR_PASS`：才允许讨论 supervised residual warm-start 或 off-policy residual；
- `H3_SR_FAIL`：记录监督标签不足，停止本 Goal 的 residual 训练；下一 Goal 转向
  terminal capture/plane-miss 或执行器饱和（父 Goal H3）；本次实测为 `FAIL`；
- `INVALID`：只修数据/加载链路并重跑同一预注册实验，不改阈值、seed 或 eval split。
