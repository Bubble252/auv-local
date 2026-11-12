# Goal26：Success-First 基线与残差 PPO screen

日期：2026-11-08<br>
状态：`COMPLETED_FAIL_STOP`；协议在测量前已冻结

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal26 --goal-doc doc/goals/goal26_success_first_residual_ppo_20261108.md --strict
```

- **Research question**：在 success-first 约束下，给 frozen D3 BC 加小权重 3D 路径
  奖励并做一次冻结残差 PPO screen，能否在不降低 success、safety 或终点捕获的前提下，
  改善 pre-goal 3D RMSE？
- **H1**：固定 D3 为 frozen base，残差从零开始，只加小权重 3D 路径奖励，可以保留
  D3 的 success 和终点捕获，并至少让一个主场景的 pre-goal RMSE 改善 `>=0.10 m`。
- **H1 falsification**：若 zero-residual 与 frozen-base 动作不一致，或任一场景
  success 低于 D3 减 `0.03` / safety 恶化 / plane miss 或 timeout 恶化，则停止，
  不扩 seed，不通过改 reward 权重追结果。
- **H2**：若零残差基线通过，`w_path_3d=0.15` 的小权重增量足以产生方向正确的路径信号；
  若没有，最多只再测预注册的 `w_path_3d=0.30`，不存在第三档。
- **Primary metric**：dev seeds `20–59` 的 default/strict125 success、safety、
  plane miss、timeout。
- **Secondary metric**：pre-goal 3D RMSE、success-only 3D RMSE、full RMSE、
  post-goal overshoot、terminal longitudinal RMSE、material saturation。
- **Evaluation contract**：同一评估矩阵、同一 seed、`endpoint_stop`、path_dim 8、
  `realistic_calibrated`、CPU/float64；default/strict125 normal 为主，path-zero 只做
  诊断。
- **Data contract**：D3 只读，不重采 BC，不进入 held-out `100–149`，不覆盖 Goal24/25。
- **Budget contract**：zero-residual 一次；PPO 最多两个预注册奖励档位，每档 1 个训练
  seed；固定 `4096 env × 128 steps × 200 iterations`。不扫 lr、residual scale、
  reward 权重、DAgger 比例或 seed。
- **Negative-result contract**：任何门禁失败都写负结果并停止；不把 PASS 改成“趋势
  向好”，不事后改 success 口径或终点协议。

## 1. 背景与裁决

Goal25 的 D3 在旧协议下是 **FAIL**，因为 default/strict125 pre-goal RMSE 均未达到
旧阈值。该历史结论不追溯修改，也不把 D3 追认成 PASS。

用户最新决定采用 **success-first**：

- success、safety、plane miss/timeout 是主门禁；
- pre-goal 3D RMSE 是次级指标和候选选择依据；
- 不得通过放宽终点捕获、隐藏 timeout 或更换 seed 提升 success。

项目 `AGENTS.md` 仍保留“主指标是 3D path RMSE”的历史文字；本 Goal 明确以本文件的
success-first 契约为准，**不修改 AGENTS.md 历史内容**。

D3 当前只读基线（来自 Goal25 的 40-seed dev 评估，checkpoint SHA256
`7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73`）：

| scenario | success | pre-goal RMSE | full RMSE | plane miss | timeout | safety | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 0.7250 | 2.849170 | 2.763411 | 0.2750 | 0.0000 | 0.0000 | 0.191812 |
| strict125 | 0.5000 | 2.829135 | 2.752011 | 0.5000 | 0.0000 | 0.0000 | 0.191880 |

D3 是当前 success 最高的 BC 底座；不能用 Goal25 的旧 pre-goal 失败阻止本 Goal 的
success-first 探索。

## 2. 技术栈与隔离

- 动力学：`batch_torch_remus100`，不依赖 IsaacLab/IsaacSim；
- 控制：D3 frozen BC actor + PPO residual，残差加在归一化低层动作
  `[rudder, stern, rpm]` 上；
- actor 输入：12D state + 8D path；command=0，current encoder=0，DADE=off；
- BC checkpoint：D3，只读；
- 评估：CPU/float64，`endpoint_stop`，dev `20–59`；
- 不使用 held-out `100–149`；
- 大型训练在服务器执行，本地只做审计、smoke 和 CPU 评估；
- 服务器隔离目录：
  `/data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/`；
- 新产物统一使用 `goal26_*`，不覆盖 Goal24/25 的配置、数据、checkpoint、
  manifest、evaluation。

## 3. 预注册实验臂

### Phase 0：只读基线

复用 Goal25 D3 的 40-seed 评估结果，必须确认：

- checkpoint SHA256 与 Goal25 report 一致；
- eval matrix、seed 范围、`p_nav` success 口径和 sensor profile 一致；
- 原始 D3 evaluation 不被改写；
- 记录 `ps`、`nvidia-smi`、磁盘和 git status。

不重跑 D3，除非上述一致性检查失败。

### Phase 1：zero-residual 基线

以 D3 初始化 frozen base；`frozen_pc_bc_residual` 分支自身强制零初始化残差，因此
配置中 `training.initial_residual_zero=false`（该字段表示额外的通用输出清零，
不能与 frozen BC 的专用零初始化同时启用）。检查：

- 初始 residual 的绝对值最大值为 0；
- policy 动作与 frozen-base 动作在同一观测上逐元素一致；
- frozen BC 参数不参与梯度、不发生漂移；
- path encoder alpha、归一化和 checkpoint 结构不漂移；
- 无 NaN/Inf。

不通过即停止，不启动 PPO。

### Phase 2：PPO screen

固定：

- `num_envs=4096`；
- `num_steps_per_env=128`；
- `iterations=200`；
- `num_mini_batches=128`；
- 训练 seed `26001`；
- `residual_scale=0.25`；
- DADE off，current encoder off，actor command dim 0；
- actor hidden `[256,128]`，critic hidden `[256,128]`，`d_z=64`；
- lr actor/critic `3e-4`，gamma `0.995`，GAE `0.95`，clip `0.02`；
- 不做 lr、residual scale、seed 或 budget 扫描。

只允许两个预注册路径奖励档位：

| arm | `w_path_3d` | `w_3d` | `w_prog` | `w_yaw` |
|---|---:|---:|---:|---:|
| P1 | 0.15 | 0 | 0 | 0 |
| P2 | 0.30 | 0 | 0 | 0 |

说明：`w_path_3d` 是本 Goal 新增的、默认关闭的 ALOS 路径误差奖励项；代码实际使用
`-w_path_3d * sqrt(cross^2 + vertical^2) / 5`。P1/P2 只改这一个档位，其他奖励项固定。
不切换到 `path_learned_speed`，以保持 D3 的 ALOS 环境完全可比。
如果 P1 已经失败，是否测 P2 只看本文件预算，不允许加第三档。

## 4. 执行计划

### Step 0：执行前审计

- [x] 记录 `ps`、`nvidia-smi`、`df -h /`、`df -h /dev/shm`、`git status --short`；
- [x] 把 D3 checkpoint 复制到服务器隔离目录并校验 SHA256；
- [x] 记录 D3 training manifest、evaluation comparison 和 config hash；
- [x] 创建隔离服务器代码副本，不覆盖旧服务器目录；
- [x] 写只读 `goal26_phase0_audit.md`；
- [x] 不启动训练。

### Step 1：配置与 smoke

- [x] 新增 Goal26 default/strict125 environment config；
- [x] 新增 P1/P2 PPO training config；
- [x] 在 `/dev/shm` 本地跑 2 seed smoke，确认 D3 checkpoint 可加载；
- [x] 验证 `pc_control_mode=alos` 时 actor 输入仍为 12D + 8D path；
- [x] 检查 reward breakdown：`path_cross/path_vertical/path_progress` 非空且有限；
- [x] 运行相关单测和 config validation。

### Step 2：zero-residual

- [x] 用 D3 初始化 frozen base；
- [x] 验证 residual 初始为 0；
- [x] 同一 dev seed 上比较 `--action-mode frozen-base` 与 `--action-mode policy`；
- [x] 检查 NaN/Inf、BC 权重漂移和 path alpha；
- [x] 写 `goal26_zero_residual.json`；
- [x] FAIL 则停止并写负结果。

### Step 3：PPO screen

- [x] 在服务器空闲 GPU 上跑 P1：1 seed、200 iterations；
- [x] 输出 checkpoint、manifest、TensorBoard、日志；
- [x] 评估 dev `20–59` 的 default/strict125 normal；
- [x] path-zero 只做诊断；
- [x] 写 `goal26_p1_screen.md/json`；
- [x] 若 P1 失败则按预注册预算决定是否只测 P2，不扩 3 seeds；
- [x] 只有 screen 通过才允许扩到 3 个训练 seed。

### Step 4：Gate 与收口

- [x] 对所有候选使用同一 40-seed 评估矩阵；
- [x] 写 `goal26_result_20261108.md`；
- [x] 写 `research/LEDGER.jsonl`；
- [x] 更新 `research/STATE.md`；
- [x] 运行 ledger 检查；
- [x] 只暂存本次相关文件，不自动 commit/push。

## 5. Gate

### success-first 主门禁

| check | threshold |
|---|---:|
| default success | `>= 0.695` |
| strict125 success | `>= 0.470` |
| safety delta | `<= 0.02` |
| plane miss / timeout delta | `<= 0.05` |
| pre-goal RMSE delta | `>3%` 恶化只记风险，不一票否决 |

### PPO screen 效果门禁

- 至少一个主场景 pre-goal RMSE 改善 `>=0.10 m`，或 success 提升 `>=0.03`；
- 另一个主场景 pre-goal RMSE 恶化不超过 `3%`；
- path-zero 诊断不得显示成功完全依赖 path 输入；
- 若出现减速、悬停、绕路、终点捕获下降，立即回退该权重；
- 失败即写负结果，不扩 seed，不追扫参数。

## 6. 运行命令

服务器工作副本：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/repo
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
```

zero-residual：

```bash
$PY scripts/train_batch_lowlevel_ppo.py \
  --training-config /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/configs/ppo_p1_default.json \
  --iterations 1 \
  --num-envs 64 \
  --num-steps 8 \
  --device cuda \
  --output /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/artifacts/zero_residual \
  --record-only
```

PPO screen：

```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_batch_lowlevel_ppo.py \
  --training-config /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/configs/ppo_p1_default.json \
  --iterations 200 \
  --num-envs 4096 \
  --num-steps 128 \
  --device cuda \
  --output /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/artifacts/ppo_p1 \
  --seed 26001 \
  --record-only
```

评估：

```bash
$PY scripts/evaluate_pc_ppo.py \
  --training-config /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/configs/ppo_p1_default.json \
  --checkpoint /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/artifacts/ppo_p1/model_200.pt \
  --environment-config /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/configs/env_default.json \
  --output /data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/artifacts/eval_p1_default \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
  --action-mode policy
```

## 7. 产物与验证

必须保存：

- `configs/`：Goal26 环境、P1/P2 PPO 配置；
- `checkpoints/`：D3、P1/P2 checkpoint 及 SHA256；
- `artifacts/`：zero-residual、PPO、评估；
- `logs/`：训练和评估日志；
- `goal26_phase0_audit.md`；
- `goal26_zero_residual.json`；
- `goal26_result_20261108.md`；
- `research/LEDGER.jsonl` 和 `research/STATE.md` 更新。

验证命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python -m pytest -q \
  tests/test_lowlevel_pc_residual_pretrain.py \
  tests/test_lowlevel_residual.py \
  tests/test_lowlevel_residual_eval.py \
  tests/test_lowlevel_pc_policy.py
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal26 \
  --goal-doc doc/goals/goal26_success_first_residual_ppo_20261108.md \
  --strict
```

若服务器 isolated repo 的代码尚未包含 Goal26 所需的可选配置字段，只允许做默认关闭、
向后兼容的扩展；必须先补单测，不得破坏 Goal24/25 的旧路径。

## 8. Git

禁止 `git add -A`。只暂存本次相关文件：

```bash
git add \
  doc/goals/goal26_success_first_residual_ppo_20261108.md \
  doc/goals/goal26_success_first_protocol_20261108.md \
  configs/lowlevel/pc_goal26_*.json \
  configs/lowlevel/eval/pc_v1_nominal_eval_matrix_goal26_*.json \
  scripts/... \
  tests/...
git diff --cached --check
```

不自动 commit/push。服务器的隔离副本不是 Git 工作区时，只记录文件清单和 SHA256。

## 9. 停止条件

- D3 checkpoint 无法加载或 SHA256 不一致；
- zero-residual 动作不一致；
- 训练/评估出现 NaN/Inf；
- success 低于 D3 减 `0.03`；
- safety 恶化超过 `0.02`；
- plane miss 或 timeout 恶化超过 `0.05`；
- 路径奖励导致减速、悬停、绕路或终点捕获下降；
- 需要 held-out 才能决策；
- 试图扫 lr、residual scale、奖励权重、seed 或追加 budget 追结果。
