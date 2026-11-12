# Goal58：Off-policy 命令残差的信用分配检验

**日期**：2026-11-09<br>
**状态**：`FORMAL_COMPLETE_FAIL_PIVOT`；结果见 `doc/goals/goal58_result_20261109.md`<br>
**上位记录**：Goal27 已完成并停在 `H3_SR_FAIL_STOP`，见
`doc/goals/goal27_command_residual_credit_assignment_20261109.md` 和
`doc/goals/goal27_h3_sr_result_20261109.md`。<br>
**目的**：只检验 Goal27 明确推迟的 off-policy residual 分支；不重开 PPO、
reward、gamma/lambda、seed rescue 或 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal58 --goal-doc doc/goals/goal58_offpolicy_residual_credit_assignment_20261109.md --strict
```

本 Goal 是探索型实验，`required: true`。一旦预注册 Gate 失败，或出现
seed-noise ambiguous，必须写 blocker、至少两个可证伪 hypothesis、reading note
和最便宜的 falsification experiment，再在 `research/LEDGER.jsonl` 记录
`continue` / `pivot` / `stop`。Gate 记录前不修改确认参数、eval split 或 held-out。

## 1. 背景与证据

### 1.1 Goal27 已经给出的结论

Goal27 在 frozen base 上检验了 ALOS 命令残差和 action residual：

- C1/R1/H1/G1/A1 均未达到 pre-goal RMSE 改善 `>=0.10 m` 的效果门禁；
- C1/R1 的命令残差只有约 `0.0026–0.0032`，scale/rate clip 为 `0.0`；
- H3-SR 监督动作残差确实非零（`16.14%/16.33%`），但相对配对 frozen-base
  只改善 `0.0231 m`，success 从 `0.725/0.500` 降到 `0.650/0.350`；
- Goal27 因此裁决 `H3_SR_FAIL_STOP`，并明确：若继续 residual，只能在新 Goal
  预注册 off-policy baseline、eval split 和 pass/fail threshold。

本 Goal 不推翻 Goal27，也不把历史 C1/R1 结果重新解释成 PASS。

### 1.2 为什么还值得做一次 off-policy

Goal27 的 blocker 把 residual collapse 分成四条路径：命令惩罚冲突、GAE horizon、
on-policy 样本效率/探索、residual scale/初始化。前两条已由 H1/G1 证伪，
监督残差（H3-SR）也没有改善闭环。剩下的唯一未被正式检验的分支是：

```text
off-policy replay + NoisyNet + TD3-style critic
  -> 是否有足够样本效率让 3-D 命令残差离开零附近
  -> 若离开零附近，是否转化为 pre-goal 路径误差改善
```

这不是“再加训练预算”，而是改变 credit assignment 机制。若 off-policy 仍然失败，
就不能再把 residual 主线归因于 on-policy 样本效率。

### 1.3 路径 provenance 前置条件

Goal38 已确认并修复 `BatchedPCCommandResidualEnv` 旧实现缓存 stale `path` 的根因。
当前 wrapper 不保存 `path` 属性，`__getattr__` 动态委托到内部环境；回归测试
`tests/test_lowlevel_pc_command_residual.py::test_wrapper_path_tracks_seed_selection_after_reset_and_reset_indices`
要求 reset 后 `wrapper.path is env.path`。本 Goal 的 Step 2 必须重新跑该测试和
一个 freshness smoke；不通过则本 Goal 为 `INVALID`，不得训练。

### 1.4 修正后的 B0 基线与历史数值

Goal27 的 `B0 = 4.9713 m` 是用 command-residual wrapper 的
`zero-command-residual` 评估得到的；Goal38 已确认该 wrapper 在旧实现中缓存
stale path 0。Goal58 的 Step 3 在修复后的 wrapper 上重跑 40-seed dev
`20–59`，得到：

| scenario | success | plane miss | safety | material sat. | pre-goal 3D RMSE |
|---|---:|---:|---:|---:|---:|
| default | `0.725` | `0.275` | `0.0` | `0.2695` | `2.6431 m` |
| strict125 | `0.500` | `0.500` | `0.0` | `0.2699` | `2.6287 m` |

success、plane-miss、safety 和 material saturation 与 Goal27 B0 逐项一致，
只有被 stale path 污染的 path-derived pre-goal RMSE 改变。因此本 Goal 的所有
配对差值都以修复后的零残差 wrapper 为 baseline，不使用历史 `4.97 m` 作为
效果大小的分母或对照。

## 2. 研究问题与可证伪假设

### H1：off-policy replay 能让命令残差离开零并改善 pre-goal RMSE

机制：DDPG/TD3 从 replay 中反复利用同一批闭环转移，NoisyNet 提供持续探索，
比 on-policy PPO 更可能在长时路径误差上分配信用。

可证伪预测：在固定 B0 base、固定 reward-v2、固定 dev `20–59` 下，至少
2/3 个独立训练 seed 的 pre-goal 3D RMSE 相对零残差配对基线改善 `>=0.10 m`，
且 success/plane-miss/safety 护栏不恶化。若残差仍近零，或改善低于门禁，
H1 不成立。

最便宜 falsification：1 个训练 seed、1M transitions、40 个 dev seeds 的 screen；
screen 只检查接口、残差幅度和 finite，不作为正式 Gate。

### H2：残差能长大但闭环指标不动，说明样本效率不是瓶颈

机制：如果 off-policy 学到非零修正（applied residual L2 ratio `>=1%`），但
pre-goal RMSE 仍无改善，失败点就更可能在 residual 的动作方向、奖励目标或
可控性，而不是 replay 样本效率。

可证伪预测：H1 失败时，若 residual L2 ratio `<1%`，则 H2 不能解释失败，
应检查 scale/初始化或优化；若 ratio `>=1%` 且 pre-goal 改善 `<0.10 m`，
则 H2 成立，停止“换一个更 sample-efficient 的 off-policy 算法”作为主线。

### H3：路径输入确实被 residual 使用

机制：Goal27 H3-SR 的 path-zero 诊断已显示路径输入会影响行为，但那是
frozen D3 action residual，不是本 Goal 的 off-policy 命令残差。

可证伪预测：对通过 H1 主效果的 checkpoint，在 default 场景把 actor 的 path
特征置零（环境真实路径不变），pre-goal RMSE 应恶化至少 `5%`。若 path-zero
不恶化，不能声称 residual 使用了路径信息；该 checkpoint 只能判为无效归因。

### H4：scale/零初始化是残差坍缩的唯一解释

机制：若 off-policy 的 applied residual 仍 `<1%`，可能是 actor 零初始化后
NoisyNet 探索不足，或命令残差 scale 太小。该假设只在 H1 的残差近零分支中做
一次 1-seed/短程 falsification，不进入 reward/gamma sweep。

可证伪预测：保持 reward、数据、算法和训练预算不变，只把 residual scale 和
输出噪声放大 2 倍；若 residual L2 ratio 仍 `<1%`，H4 不成立；若 residual
显著变大但闭环指标仍不改善，H4 不是主因，按 H2 处理。该分支即使让残差变大，
也不能自动判 H1 PASS，仍需重新满足主 Gate。

## 3. 技术栈与实现边界

### 3.1 环境与控制链

```text
参考路径 + 海流/传感器随机化
  -> ALOS 3-D guidance
  -> [psi_d, theta_d, u_d] base command
  -> 3-D command residual (actor)
  -> PD/PI base controller
  -> [rudder, stern, rpm]
  -> REMUS100 6-DOF batch dynamics
```

- base：ALOS + deployable PD/PI，零残差等价 B0；不使用 D3 checkpoint 作为
  off-policy base。
- command residual 的物理 scale 固定为 Goal27 C1/R1：
  `heading 0.35 rad`、`pitch 0.17 rad`、`speed 0.60 m/s`；
  rate limit 分别为 `0.25 rad/s`、`0.15 rad/s`、`0.80 m/s^2`。
- reward：`pc_command_track_v2`，`pc_ppo_scale=0.01`，固定 Goal27 R1 的
  `goal_bonus=300`、`timeout_penalty=-100`、`terminal_progress_weight=2.0`、
  `goal_plane_miss_penalty=-100`、`goal_plane_miss_distance_weight=10.0`。
- 训练固定 default 环境；strict125 只用于配对评估，不调参。

### 3.2 Off-policy 网络与观测

- actor 输入：归一化 `actor_state(12) + path(8) + command(3) + current_estimate(3)`
  = `26` 维；DDPG 内部保留 10 步 GRU history。
- critic 输入：actor 观测加 privileged
  `critic_state(15) + true_current_ned(3)` = `44` 维。
- actor 输出：3 维 normalized command residual，范围 `[-1, 1]`。
- actor 使用 NoisyNet；最终层 mean 初始化为 0，使 deterministic 初始策略严格
  等于零残差。NoisyNet 只负责探索，不改变评估时的零残差基线。
- 算法：DDPG actor + twin critics + target policy smoothing + delayed policy
  update（TD3-style），replay 使用现有 `PrioritizedSequenceReplayBuffer`。
- 固定训练参数：`gamma=0.995`、`tau=0.005`、`batch_size=256`、
  `buffer_size=500000`、`learning_starts=20000`、`update_to_data_ratio=0.125`、
  `actor_lr=2e-4`、`critic_lr=2e-4`、`policy_delay=2`、
  `target_policy_noise_sigma=0.1`、`target_policy_noise_clip=0.25`、
  `history_length=10`、`use_pper=true`。
- checkpoint 只保存 final；不按 dev 结果选 best。

### 3.3 服务器

服务器为 `zhulab3-AS-4125GS-TNRT`，8 × RTX 4090，每卡 24564 MiB。
独立训练 seed 分布到不同 GPU，不把一个 seed 当成自动的 8-GPU 分布式训练。
正式隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal58_offpolicy_residual_20261109/
```

不得覆盖 Goal27 的 `goal27_command_residual_20261109/`。本地只做实现、单元测试、
零残差 parity 和小 smoke；正式训练与 dev 评估在服务器执行。

## 4. 数据、隔离与评估协议

| 项目 | 固定值 |
|---|---|
| 训练环境 | default command-residual reward-v2 |
| 训练 seeds | S1 `240–279`；S2 `280–319`；S3 `320–359` |
| 训练 agent seeds | `240 / 280 / 320` |
| screen | 只跑 S1，1M transitions，40 envs |
| formal | 3 个独立训练 seed，各 5M transitions，40 envs |
| dev eval | seeds `20–59`，严格配对 |
| held-out | `100–149` 关闭，不读取 |
| eval 场景 | default 与 strict125（仅 eval） |
| eval 精度 | CPU/float64 |
| eval 终止 | `endpoint_stop` |
| eval checkpoint | 每个训练 seed 的 final，不选 best |
| 路径归因 | default `path-zero`，只对通过主效果的 checkpoint |

pre-goal 指标按 evaluator 既有定义：真实位置到参考路径的 3-D 误差，在到达
goal radius 或 terminal tangent plane 之前累计；success 只认 `GOAL` 终止，
plane-miss/timeout/safety 分开记录。所有结果必须写 success、timeout、主指标、
风险指标、seed 数和评估协议。

## 5. Gate（测量前冻结）

### 5.1 每个训练 seed 的配对比较

对每个训练 seed，用同一 checkpoint 评估 `policy` 与 `zero-residual` 两种模式：

- `policy`：DDPG actor 输出命令残差；
- `zero-residual`：同一个环境、同一 seeds，强制 residual 为 0，等价 B0。

| check | 通过条件 |
|---|---|
| 主效果 | default 或 strict125 的 pre-goal 3D RMSE 相对 zero-residual 改善 `>=0.10 m` |
| 另一场景 | 另一场景 pre-goal RMSE 恶化不超过 `3%` |
| success | 两场景都不低于 zero-residual 超过 `0.025`；default `>=0.695`，strict125 `>=0.470` |
| safety | 相对 zero-residual 恶化不超过 `0.02` |
| plane miss / timeout | 相对 zero-residual 恶化不超过 `0.05` |
| 残差非零 | applied residual L2 / base command L2 `>=1%`，且 applied abs mean `>0` |
| 路径归因 | default 的 `path-zero` pre-goal RMSE 相对正常 path 恶化 `>=5%` |
| 稳定性 | 0 NaN/Inf、0 崩溃、0 非有限 reward/action |

### 5.2 三个 formal seed 的总体裁决

- **PASS**：至少 `2/3` 个训练 seed 在同一场景满足主效果，三个 seed 的配对
  均值也改善 `>=0.10 m`；每个 seed 都满足 success/safety/plane-miss/timeout
  护栏；残差非零和路径归因通过。
- **FAIL**：主效果或 success-first 护栏失败，且不满足 PASS；`INVALID` 只用于
  path provenance、数据/配置、checkpoint 或链路错误，不写成 FAIL。
- **AMBIGUOUS**：配对差 CI 跨零、seed 间方向不一致，或差异小于 seed noise；
  进入 Research Escalation，不能直接换主线或扫 seed。

### 5.3 失败分支

| 观察 | 判读 | 下一步 |
|---|---|---|
| residual `<1%`，指标不动 | H1 失败，H2 不能解释 | 先做 H4 的 1-seed scale/init falsification；仍失败则 stop residual 主线 |
| residual `>=1%`，指标不动 | H2 成立 | 不再换 off-policy 算法；写 blocker，考虑动作语义/目标函数，不再扫 reward/gamma/seed |
| residual 长大但 success 变差 | 残差方向错误 | 停止该 checkpoint 主线，检查 terminal/plane-miss 归因，不调 success 口径 |
| path-zero 不恶化 | H3 证伪 | checkpoint 不能声称使用路径，按 INVALID attribution 处理 |
| 2/3 seed 方向不一致 | seed-noise ambiguous | 写 ledger，最多做一次独立 replication；不通过扩 seed 或放宽门禁救结果 |

### 5.4 Screen 后的非 Gate 诊断：NoisyNet 是否主导残差

在 Step 7 正式结果出来前，只用 screen checkpoint 和训练 seeds `240–279` 做了
200 control steps 的诊断，不读取 dev `20–59`，不改变任何训练参数或 Gate：

| 诊断量 | 值 |
|---|---:|
| deterministic action abs mean（heading/pitch/speed） | `1.0000 / 0.9997 / 0.5252` |
| deterministic 饱和比例（任一维 / heading / pitch） | `100% / 100% / 99.5%` |
| deterministic applied residual / base command L2 | `29.84%` |
| deterministic rate-clip 比例 | `34.5%` |
| NoisyNet noise std / deterministic action norm | `0.535` |
| NoisyNet sample 与 deterministic mean 的 cosine | `0.865` |

结论：训练日志中的大残差不能仅归因于 NoisyNet。1M screen checkpoint 的
deterministic policy 本身已经把 heading/pitch 推到接近 `±1` 的饱和区；NoisyNet
只在其上增加约半倍动作范数的探索。正式 final checkpoint 是否同样饱和，以
Step 7 的确定性 `policy` vs `zero-residual` 为准；该诊断不参与 Gate 判定。
如果正式结果同样“残差很大但闭环不动”，优先检查动作参数化（tanh 饱和、rate
limit）、reward 与 terminal capture，而不是先换算法。

### 5.5 后续算法分支（只记录，当前不执行）

以下分支不改变本 Goal 的预注册 Gate，也不在 formal 运行中替换算法：

- 若 Step 7 明确把失败归因于优化/探索：新开一个预注册 goal，用 SAC 在同样
  的 base、数据、dev/held-out 和配对评估协议下做对照；不把 SAC 结果混入 Goal58。
- 若失败归因于样本效率：新 goal 中再考虑 REDQ/DroQ；同样必须重新预注册 Gate。
- 若残差很大但闭环不动：不要先换算法，先查动作参数化、reward 和 terminal
  capture，确认残差是否被 tanh/rate limit/终止条件吞掉。
- 任何新算法都需要独立 goal 文档、Research Contract 和 Gate；Goal58 本身不重开。

## 6. 执行计划

- [x] Step 0：审计 Goal27 终态、Goal27 blocker、reading note、Goal38 provenance；
- [x] Step 1：写本 Goal58 预注册文档并冻结 Gate；
- [x] Step 2：实现 PC off-policy adapter、训练器、评估器和单元测试；
- [x] Step 3：本地运行 path freshness、零残差 parity、adapter smoke、checkpoint roundtrip；
- [x] Step 4：同步隔离目录到服务器，核验 8 × 4090、venv、config/hash 和 GPU 占用；
- [x] Step 5：服务器跑 S1 1M-transition screen，检查 residual/finite/吞吐；
- [x] Step 6：通过 screen 后并行跑 S1/S2/S3 各 5M transitions；
- [x] Step 7：对每个 final checkpoint 在 dev `20–59` 跑 default/strict125 的
      `policy` 与 `zero-residual` 40-seed 评估；对候选追加 `path-zero`；
- [x] Step 8：按 §5 写 result、blocker/decision、ledger 和 STATE，运行完整
      pytest、Ruff、strict ledger checker、`git diff --cached --check`；
- [ ] Step 9：只有用户明确要求时才 commit/push。

### 6.1 Step 2 运行与验证

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python -m pytest -q \
  tests/test_pc_goal58_offpolicy_env.py \
  tests/test_pc_goal58_offpolicy_trainer.py \
  tests/test_lowlevel_pc_command_residual.py
.venv/bin/ruff check \
  src/auv_rl/ddpg/pc_command_residual_env.py \
  src/auv_rl/ddpg/pc_vector_trainer.py \
  scripts/train_pc_command_residual_offpolicy.py \
  scripts/evaluate_pc_command_residual_offpolicy.py
```

### 6.2 Step 3 本地 smoke

```bash
.venv/bin/python scripts/train_pc_command_residual_offpolicy.py \
  --config configs/lowlevel/pc_goal58_offpolicy_default_20261109.json \
  --output /dev/shm/goal58_local_smoke \
  --num-envs 2 --total-env-transitions 64 --device cpu \
  --train-seed-start 240 --smoke --seed 240
```

验证：zero-residual deterministic actor 与 B0 stepwise 一致；observation shape
为 `(2, 26)`；privileged shape 为 `(2, 18)`；replay/training-state roundtrip
通过；checkpoint 重新加载后 deterministic action 不变。

### 6.3 Step 4 服务器同步

```bash
sshpass -p 'Bq9068w1' rsync -az --delete \
  -e 'ssh -p 43127 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=no' \
  --include='src/***' --include='scripts/***' --include='configs/***' \
  --include='checkpoints/***' --include='tests/***' --exclude='*' \
  /home/bubble/水下机器人/settled_局部规划器_反应式/ \
  yhx@152.136.183.51:/data/yhx/underwater-robotics/src/plan_p1/goal58_offpolicy_residual_20261109/
```

同步后用服务器 venv 运行 import/smoke，不覆盖 Goal27 目录。

### 6.4 Step 5 screen

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/train_pc_command_residual_offpolicy.py \
  --config configs/lowlevel/pc_goal58_offpolicy_default_20261109.json \
  --output artifacts/screen_s240 \
  --num-envs 40 --total-env-transitions 1000000 \
  --train-seed-start 240 --seed 240 --device cuda
```

screen 通过条件：finite、residual 有非零读数、吞吐和显存允许 3 个并行 run。
screen 不用于调 reward/gamma/scale，也不用于选 checkpoint。

Screen 记录（2026-11-09）：`artifacts/screen_s240/` 完成 1,000,000 transitions、
122,500 gradient updates，wall time `2360.10 s`，`423.71 transitions/s`，GPU 0
显存约 `2.2 GiB`；`residual_abs_mean=0.31244`，`residual_l2_ratio=0.37996`，
`residual_scale_clip_rate=0.0`，`residual_rate_clip_rate=0.75496`；training
curve 与 836 个 episode 全部 finite，0 safety/timeout。通过 screen，不据此调参。

### 6.5 Step 6 formal

```bash
for spec in '0 240 240' '1 280 280' '2 320 320'; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=src \
  /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
    scripts/train_pc_command_residual_offpolicy.py \
    --config configs/lowlevel/pc_goal58_offpolicy_default_20261109.json \
    --output artifacts/formal_s$3 \
    --num-envs 40 --total-env-transitions 5000000 \
    --train-seed-start $2 --seed $2 --device cuda \
    > artifacts/formal_s$3.log 2>&1 &
done
wait
```

每个 run 的 `final.pt` 和 manifest 必须记录 config hash、code hash、train seed、
seed block、transitions、gradient steps、wall time 和 device。

### 6.6 Step 7 评估

```bash
for seed_dir in artifacts/formal_s240 artifacts/formal_s280 artifacts/formal_s320; do
  for scenario in default strict125; do
    for mode in policy zero-residual; do
      PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
        scripts/evaluate_pc_command_residual_offpolicy.py \
        --config configs/lowlevel/pc_goal58_offpolicy_default_20261109.json \
        --environment-config configs/lowlevel/pc_goal27_r1_reward_v2_${scenario}_20261109.json \
        --checkpoint "$seed_dir/final.pt" --output "$seed_dir/eval_${scenario}_${mode}" \
        --seed-start 20 --seed-count 40 --device cpu --dtype float64 --action-mode "$mode"
    done
  done
done
```

`path-zero` 只在候选达到主效果后运行，并写进同一 result，不用于挑 best。

### 6.7 Step 8 收口命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal58 \
  --goal-doc doc/goals/goal58_offpolicy_residual_credit_assignment_20261109.md --strict
git diff --cached --check
```

完整测试或 Ruff 失败时不能宣告完成；若失败属于本 Goal 改动，先修复再收口。

## 7. Git 步骤

禁止 `git add -A`。只添加本 Goal 的文档、脚本、配置、测试和 research 记录；
不添加 `artifacts/`、checkpoint、日志、缓存或大 PDF。提交前运行：

```bash
git add doc/goals/goal58_offpolicy_residual_credit_assignment_20261109.md \
  doc/goals/goal58_result_20261109.md \
  configs/lowlevel/pc_goal58_offpolicy_default_20261109.json \
  src/auv_rl/ddpg/pc_command_residual_env.py \
  src/auv_rl/ddpg/pc_vector_trainer.py \
  scripts/train_pc_command_residual_offpolicy.py \
  scripts/evaluate_pc_command_residual_offpolicy.py \
  tests/test_pc_goal58_offpolicy_env.py \
  tests/test_pc_goal58_offpolicy_trainer.py \
  research/blockers/<date>-<slug>.md \
  research/decisions/<date>-<slug>.md \
  research/paper_reading/<date>-<slug>.md \
  research/LEDGER.jsonl research/STATE.md ../research/STATE.md
git diff --cached --check
```

本回合不自动 commit/push；只有用户明确要求才执行：

```bash
git commit -m "research: evaluate Goal58 off-policy command residual"
git push origin "$(git branch --show-current)"
```

## 8. 明确禁止

- 不重开 Goal27 的 C1/R1/H1/G1/A1 或 H3-SR；
- 不扫 `w_path_3d`、command reward weight、gamma/lambda、success 口径或 seed；
- 不把 validation/dev 结果用于 best-epoch 选择；
- 不打开 held-out `100–149`；
- 不把 screen 结果写成正式 Gate；
- 不把一次 `INVALID` 写成 `FAIL`，也不把 `FAIL` 写成 `PASS`；
- 不覆盖 Goal27 的服务器 artifacts。
