# Goal59：终端门控的有界命令残差与 12D endpoint 观测

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal58 已完成 `OFFPOLICY_RESIDUAL_FAIL_PIVOT`；Goal27 停在
`H3_SR_FAIL_STOP`；Goal30/31 已停止 terminal window 与 path-lookahead 参数路线。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal59 --goal-doc doc/goals/goal59_bounded_terminal_residual_20261109.md --strict
```

本 Goal 是探索型实验，`required: true`。预注册 Gate 记录前不得改训练 seeds、
dev split、阈值、checkpoint 规则或 held-out；失败后必须写 blocker、至少两个
可证伪 hypothesis、reading note 和最便宜 falsification，再在
`research/LEDGER.jsonl` 记录 `continue` / `pivot` / `stop`。

## 1. 背景与证据

Goal58 的 3 个 5M off-policy seed 没有达到 pre-goal RMSE 改善 `>=0.10 m`：
default 配对均值 `-0.0051 m`、strict125 `-0.0150 m`；success 分别下降
`0.5167/0.4667`，plane-miss 同幅上升。关键机制不是 residual 太小，而是：

- 三个 final deterministic policy 在 heading/pitch/speed 上 **100% 饱和**；
- applied residual 元素均值约 `0.327`，L2 ratio `42–47%`；
- NoisyNet 不是评估时主因：deterministic mean 已饱和，final sigma 仍在初始化附近；
- eval-time scale `0.50` 能改善 path RMSE，却同时恶化 success/plane-miss；
- Goal30/31 已证明 path RMSE 与 terminal capture 可以解耦，不能再用 path 改善
  冒充终端捕获改善。

本 Goal 不再重开 Goal27 的 PPO、reward、gamma/lambda、seed rescue；不再扫
Goal30 的 terminal window；不再扫 Goal31 的 path-lookahead。只检验两个剩余机制：

1. **残差参数化**：把 `tanh` 的 ±1 边界改成有界 softsign，并只在终端阶段施加；
2. **终端观测**：让 actor 看到 12D path endpoint context，而不再只有 8D path。

算法保持 Goal58 的 DDPG/TD3-style off-policy 不变。SAC/REDQ/DroQ 只在后续
诊断明确证明优化/探索或样本效率是主因时，另开预注册 Goal。

## 2. 可证伪假设

### H1：终端门控的有界残差能避免“全程最大修正”

机制：Goal58 的 `tanh` actor 在三个通道长期顶到 ±1，物理残差变成贯穿全程的
最大修正。Goal59 使用固定 scale `0.5` 的 softsign 输出，并把残差乘以
`goal_distance 20 m -> 5 m` 的 smoothstep 门控；远离终点时残差为 0，接近终点
才允许局部修正。

若成立应看到：

- final deterministic raw saturation fraction `<=0.50`；
- gate-active applied residual L2 ratio `>=1%`；
- 至少一个 scenario 的 true-state `terminal_eta_capture_rate` 相对 zero-residual
  改善 `>=0.05`，另一个 scenario 不恶化；
- success/safety/plane-miss/timeout 不越过护栏。

若失败应看到：deterministic 仍长期顶到门控后的边界，或 gate-active residual
非零但 terminal capture 不动/变差。此时 H1 不成立，不能把失败归因于样本效率。

最便宜 falsification：1-seed/1M screen 的 deterministic 诊断；不进入 formal 就
能排除“softsign 只是换了一个饱和边界”的情况。

### H2：12D endpoint context 是把局部残差转化为 terminal capture 的必要条件

机制：8D path 只有路径投影误差，没有 goal 平面的切向偏移、goal heading/pitch
误差和 goal distance proximity；残差可能只学会沿路径飞行，不会在终点平面收敛。
12D endpoint context 在最后 20 m 提供这些量。

若成立应看到：在同样的 H1 参数化下，G12 的 screen terminal capture 比 G8 高
`>=0.05`，或 plane-miss 低 `>=0.05`；formal G12 至少达到 H1 的主效果门禁。

若失败应看到：G12 与 G8 的 screen 方向不一致或差异小于 seed 噪声，或者 formal
G12 在 gate-active residual 非零时仍没有 terminal capture 改善。此时 endpoint
context 不是主因，不能用“再加特征”继续救。

最便宜 falsification：G8/G12 同 1M screen、同 seed、同 reward；screen 只作机制
诊断，不把单 seed 差异写成正式 PASS。

### H3：优化/探索是后续失败原因

机制：如果 H1 的参数化和 H2 的观测都已成立，残差非零且不饱和，但闭环 terminal
capture 仍不改善，则剩余问题可能是 actor 优化/探索，而不是动作语义。

若成立应看到：gate-active residual 有非零读数、raw saturation 低、训练曲线已
离开零附近，但 paired terminal capture 无改善；此时才允许新开预注册 Goal 用 SAC
做同协议对照。

若失败应看到：residual 仍坍缩，或 terminal capture 随 residual 方向系统性变差；
这说明不是优化/探索，SAC 不是下一步。

### H4：reward-v2 已经覆盖终端目标，缺的是动作/观测

机制：Goal27 R1 reward-v2 已包含 `goal_bonus`、`terminal_progress_weight`、
`goal_plane_miss_penalty` 和 `goal_plane_miss_distance_weight`。Goal58 的失败不是
缺少一个终端奖励项，而是残差把 dense path 项放大成全程最大修正。

若成立应看到：不修改 reward，只做 H1/H2 后 terminal capture 改善，且 gate-active
residual 与 terminal capture 同向。

若失败应看到：gate-active residual 非零、不饱和，但 terminal capture 不动；此时
才允许另开 Goal 检验 dense terminal-offset reward，而不是先换算法。

### H5：NoisyNet 主导当前残差

机制：训练时日志里的 residual 很大，可能是 NoisyNet 的随机权重把动作推到门控
边界；评估时关闭噪声后，确定性残差应该显著变小。

Goal58 已经用 1M screen checkpoint 和 3 个 final checkpoint 做过同协议审计：

- deterministic action abs mean 为 `1.0000 / 0.9997 / 0.5252`，heading/pitch
  饱和比例 `100% / 99.5%`；
- NoisyNet noise std / deterministic action L2 为 `0.535`，且
  `noise_std > deterministic` 的状态比例为 `0.0`；
- final checkpoint 的三个维度饱和比例均为 `1.0`，hidden/output sigma 仍在初始化
  附近 `0.300 / 0.100`。

因此 H5 在 Goal58 上 **REJECTED**：NoisyNet 可以影响探索，但不是确定性饱和和
闭环失败的主因。Goal59 不扫描、不关闭、不缩小 NoisyNet 作为正式变量；只把它作为
非 Gate 审计，防止新参数化下再次出现“噪声把 deterministic mean 淹掉”的假象。

若成立应看到：deterministic 动作接近零、不饱和，而 noisy 动作显著更大，且
`noise_std_larger_than_det_fraction` 明显大于零；若失败应看到：deterministic
已经饱和，或噪声只增加探索而不改变闭环方向。此时不能把 NoisyNet 当作换算法的
理由，也不能用关闭噪声来救 Gate。

## 3. 实验臂与隔离

### 3.1 基线

- `B0`：ALOS + 可部署 PD/PI，`zero-residual`；同一 checkpoint 配对评估。
- 历史 Goal58 checkpoint 只作对照，不覆盖、不重训、不重新解释为 PASS。

### 3.2 Screen 臂

| arm | residual 参数化 | terminal gate | path_dim | 作用 |
|---|---|---|---:|---|
| `G8` | softsign, scale `0.5` | 20→5 m | 8 | 隔离门控+有界参数化 |
| `G12` | softsign, scale `0.5` | 20→5 m | 12 | **主候选** |
| `U12` | softsign, scale `0.5` | 关闭 | 12 | 隔离 terminal gate |
| `N12` | 同 G12，`use_noisy=false` | 20→5 m | 12 | 只做 NoisyNet 诊断，**不进入 formal** |
| `T12` | tanh, scale `1.0` | 20→5 m | 12 | 只做参数化对照，**不进入 formal** |

Screen 只跑 S1、1M transitions、128 envs。Screen 通过条件只检查工程有效性：
finite、无 NaN/Inf、无 OOM、gate-active residual 有非零读数、吞吐和显存允许
8 个 formal run。Screen 不按效果选 best；G8/U12 只作机制诊断，不能被提升为
formal 候选。

`N12` 是为了回答“残差是否被 NoisyNet 主导”的单 seed 诊断臂：它不改变 G12 的
动作参数化、门控和观测，只关闭 NoisyNet。它不能作为 formal 候选，也不能用它的
好坏来选算法；只用于检查“关闭噪声后 deterministic residual 是否仍然饱和/闭环
不动”。若 N12 仍然饱和或仍然不改善 terminal capture，则 NoisyNet 不是主因。

`T12` 是 softsign/scale 的参数化对照：门控、endpoint、噪声和其余训练配置都与
G12 相同，只把 actor 换回 Goal58 的 `tanh` + scale `1.0`。它只用于判断“有界
softsign 是否真的缓解饱和”，不作 formal 候选，也不据此选 best。

每个 screen checkpoint 额外运行一次 NoisyNet 主导性审计。该审计不改变训练参数，
不参与 Gate；只记录 deterministic 动作/饱和、noise std、noise-to-det ratio、
cosine 和 hidden/output sigma。若 deterministic 已经饱和，则不能把大 residual
归因于 NoisyNet。

### 3.3 Formal

只跑 `G12`。8 个独立训练 seed，各 5M transitions、128 envs：

| run | train seeds | agent seed | GPU |
|---|---|---:|---:|
| S1 | `240–367` | 240 | 0 |
| S2 | `368–495` | 368 | 1 |
| S3 | `496–623` | 496 | 2 |
| S4 | `624–751` | 624 | 3 |
| S5 | `752–879` | 752 | 4 |
| S6 | `880–1007` | 880 | 5 |
| S7 | `1008–1135` | 1008 | 6 |
| S8 | `1136–1263` | 1136 | 7 |

每个 run 只保存 final checkpoint，不按 dev 结果选 best。8 个 run 同时跑，预计
wall time 与 Goal58 单 seed 同量级；若 128 envs 导致显存或吞吐不达标，先停，
记录 `INVALID_ENGINEERING`，不能私改协议后继续。

## 4. 数据与评估协议

| 项目 | 固定值 |
|---|---|
| 训练环境 | default，reward-v2，command residual |
| 训练 seeds | 上表 8 个 block |
| screen | S1，1M transitions，128 envs |
| formal | 8 seeds，5M transitions，128 envs |
| dev eval | seeds `20–59`，40 seeds，严格配对 |
| held-out | `100–149` 关闭，不读取 |
| eval 场景 | default 与 strict125（仅 eval） |
| eval 精度 | CPU/float64 |
| eval 终止 | `endpoint_stop` |
| eval checkpoint | 每个训练 seed 的 `final.pt` |
| 配对模式 | `policy` vs `zero-residual` |
| 路径归因 | 只对通过主效果的 seed 跑 default `path-zero` |

主指标是 true-state `terminal_eta_capture_rate`：episode 终止时
`||eta_true - goal|| <= goal_radius`。协议 success、plane-miss、timeout、safety
分开记录；pre-goal 3D RMSE 只作风险护栏，不再作为唯一主指标。

所有结果必须写 success、timeout、主指标、风险指标、seed 数和评估协议；不得把
protocol success 写成 true-state capture，也不得把 path RMSE 改善写成 terminal
capture 改善。

## 5. Gate（测量前冻结）

### 5.1 每个 formal seed 的配对比较

对每个训练 seed，用同一 checkpoint 评估 `policy` 与 `zero-residual`：

| check | 通过条件 |
|---|---|
| 主效果 | default 或 strict125 的 `terminal_eta_capture_rate` 改善 `>=0.05`；另一场景 `>=0.0` |
| success 护栏 | 两场景相对 zero-residual 下降不超过 `0.025` |
| plane-miss 护栏 | 两场景相对 zero-residual 上升不超过 `0.05` |
| safety 护栏 | 相对 zero-residual 上升不超过 `0.02` |
| timeout 护栏 | 相对 zero-residual 上升不超过 `0.05` |
| path 风险 | pre-goal 3D RMSE 恶化不超过 `0.10 m` |
| 残差非零 | gate-active applied residual L2 / base command L2 `>=1%` |
| 残差有界 | deterministic raw saturation fraction `<=0.50`，bounded saturation fraction `<=0.50` |
| 门控有效 | gate-active steps `>0`，且 gate 在终端阶段有非零读数 |
| 稳定性 | 0 NaN/Inf、0 崩溃、0 非有限 reward/action |

### 5.2 8 个 formal seed 的总体裁决

- **PASS**：至少 `6/8` 个 seed 在同一场景满足主效果和全部护栏；8-seed 配对均值
  terminal capture 改善 `>=0.05`，95% CI 下界 `>0`；mean success delta
  `>=-0.025`，mean plane-miss delta `<=+0.05`；路径归因通过。
- **FAIL**：主效果或 success-first 护栏失败，且不满足 PASS。
- **AMBIGUOUS**：方向不一致、CI 跨零，或差异小于 seed 噪声；进入 Research
  Escalation，不能直接换主线。
- **INVALID**：配置/checkpoint/链路/控制链错误，不写成 FAIL。

### 5.3 失败分支

| 观察 | 判读 | 下一步 |
|---|---|---|
| raw saturation 仍 `>0.5` | softsign 只是换边界，H1 失败 | 停 residual 参数化主线；不要换 SAC |
| gate-active residual `<1%` | 残差坍缩/门控过强 | 检查门控和初始化；不扩 seed、不加预算 |
| residual 非零、不饱和，capture 不动 | H3 或 H4 | 另开预注册 Goal：先 SAC 或 dense terminal reward，二选一，不能同时换 |
| capture 改善但 success/plane-miss 恶化 | 方向错误 | 停该 checkpoint 主线，检查 reward/terminal capture 口径，不放宽护栏 |
| G12 不如 G8 | endpoint context 不是主因 | 记录 H2 失败；不继续加特征 |
| 6/8 未达到但均值方向为正 | seed-noise ambiguous | 进入 autoresearch；最多一次独立 replication，不扩 seed 救 PASS |

### 5.4 路径归因

只对通过主效果的 seed 跑 default `path-zero`：把 12D path block 全部置零，环境真实
路径不变。通过条件：terminal capture 相对正常 path 下降 `>=0.05`，或 plane-miss
上升 `>=0.05`，或 pre-goal RMSE 恶化 `>=5%`。若 path-zero 不恶化，该 checkpoint
不能声称使用了 path/endpoint 信息，按 `INVALID_ATTRIBUTION` 处理，不写成 PASS。

### 5.5 后续算法分支提醒（不在本 Goal 执行）

以下分支只在 Goal59 结果和归因诊断明确后启用，不能在 formal 中途切换：

- **优化/探索是主因**：另开预注册 Goal，使用 SAC 做同协议对照。至少保持同一
  训练 seed 集合、数据预算、dev `20–59`、default/strict125、`endpoint_stop` 和
  final checkpoint 规则；算法差异要单独标注，不能和 Goal59 结果混在一起选 best。
- **样本效率是主因**：才考虑 REDQ/DroQ。它们提高 update-to-data 或 Q 集成效率，
  会改变计算预算和优化协议，因此同样必须另开 Goal 并冻结新的 Gate。
- **残差很大但闭环不动**：不要先换算法。先按“动作参数化 → reward → terminal
  capture 口径”排查，确认 residual 不是换了一个边界继续饱和，也不是优化了错误
  的终端目标。
- 任何一个分支都不得打开 held-out、不得扩 seed 救 PASS、不得把 `INVALID` 或
  `FAIL` 改写成 `PASS`。

## 6. 实现合同

### 6.1 Actor 有界输出

- 在 `DDPGConfig` 和 `PolicyActor` 增加 `actor_output_scale`（默认 `1.0`）和
  `actor_output_squash`（默认 `"tanh"`，Goal59 用 `"softsign"`）。
- 归一化动作：`a = scale * softsign(raw)`；物理残差仍由环境 scale
  `[0.35 rad, 0.17 rad, 0.60 m/s]` 和 rate limit 决定。
- 默认 `scale=1.0, squash="tanh"` 必须与旧 checkpoint 行为一致。
- diagnostic 必须报告 raw 分布、bounded saturation、squash derivative；不能继续用
  `1 - tanh(raw)^2` 冒充 softsign 的梯度。

### 6.2 终端门控

- 在 `BatchedPCCommandResidualEnv` 增加默认关闭的
  `terminal_gate_enabled/enter_m/full_m`；Goal59 固定 `20 m -> 5 m` smoothstep。
- 门控使用 actor-visible `p_nav` 和 goal，不使用 true `eta`；零残差时仍与 B0
  逐元素一致；关闭门控时旧行为逐位不变。
- info 必须记录 pre-gate residual、gate 值、applied residual；trainer/evaluator
  记录 gate-active residual L2 ratio。

### 6.3 12D endpoint 观测

- 环境配置打开 `observation.path_endpoint_context_enabled=true`，固定
  `path_endpoint_enter_m=20`、`path_endpoint_full_m=5`。
- off-policy `path_dim=12`；actor 观测维度从 26 变 30，critic privileged 不变。
- 不修改 8D path 的语义；G8/G12 只差 endpoint block。

### 6.4 不改变的东西

- 算法仍是 Goal58 的 DDPG + twin critics + target smoothing + delayed actor；
- reward-v2、gamma、tau、batch size、UTD、PER、NoisyNet 均不在本 Goal 扫描；
- 不打开 held-out；不做 eval-time scale rescue；不按 dev 选 best checkpoint。

## 7. 执行计划

- [x] Step 0：写 Run Contract，复核 Goal58 checkpoint、Goal30/31 终止结论和服务器；
- [x] Step 1：实现 actor scale/squash、terminal gate、12D endpoint 配置和 terminal 指标；
- [x] Step 2：本地单测、zero-residual parity、Ruff；
- [x] Step 3：同步服务器，dry-run + 128-env microbenchmark；
- [x] Step 4：screen `G8/G12/U12`（并跑非 Gate 的 `N12/T12` 诊断），只做工程有效性检查；
- [x] Step 5：formal `G12` 8 seeds × 5M（先占 6 张空闲卡，余下 2 seed 等卡）；
- [x] Step 6：dev `20–59` policy/zero 配对评估，terminal capture 汇总；
- [x] Step 7：对通过主效果的 seed 跑 `path-zero`；
- [x] Step 8：写 result、blocker/reading note、ledger、两份 STATE，运行 strict checker；
- [ ] Step 9：只 add 本 Goal 文件，`git diff --cached --check`，不自动 commit/push（本轮用户未要求 commit，保持未 stage）。

### 7.1 本地验证

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_pc_goal59_terminal_bounded.py \
  tests/test_pc_goal58_offpolicy_env.py tests/test_m4_ddpg.py
.venv/bin/ruff check src tests scripts
```

### 7.2 服务器同步

```bash
sshpass -p 'Bq9068w1' rsync -az --delete \
  -e 'ssh -p 43127 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=no' \
  --include='src/***' --include='scripts/***' --include='configs/***' \
  --include='tests/***' --exclude='*' \
  /home/bubble/水下机器人/settled_局部规划器_反应式/ \
  yhx@152.136.183.51:/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/
```

服务器路径：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/
```

### 7.3 Screen

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/train_pc_command_residual_offpolicy.py \
  --config configs/lowlevel/pc_goal59_offpolicy_g8_20261109.json \
  --output artifacts/screen_g8 --num-envs 128 \
  --total-env-transitions 1000000 --train-seed-start 240 --seed 240 --device cuda
```

G12/U12 同命令，分别换 config 和 GPU 1/2。Screen 记录 wall time、transitions/s、
GPU 显存、residual/gate 读数；不据此调 reward/gamma/seed。

NoisyNet 审计（不参与 Gate，使用 screen checkpoint）：

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/probe_pc_command_residual_noise.py \
  --config configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json \
  --environment-config configs/lowlevel/pc_goal59_terminal_bounded_default_20261109.json \
  --checkpoint artifacts/screen_g12/final.pt \
  --output artifacts/screen_g12/noisynet_probe.json \
  --seed-start 240 --seed-count 128 --control-steps 200 \
  --noise-samples 32 --device cpu --dtype float64
```

判读只看“deterministic 是否已经饱和、noise 是否大于 deterministic”，不据审计结果
调 reward、seed、scale 或算法。

### 7.4 Formal

```bash
for spec in '0 240 240' '1 368 368' '2 496 496' '3 624 624' \
            '4 752 752' '5 880 880' '6 1008 1008' '7 1136 1136'; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=src \
  /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
    scripts/train_pc_command_residual_offpolicy.py \
    --config configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json \
    --output artifacts/formal_s$3 --num-envs 128 \
    --total-env-transitions 5000000 --train-seed-start $2 --seed $2 --device cuda \
    > artifacts/formal_s$3.log 2>&1 &
done
wait
```

### 7.5 评估

```bash
for seed_dir in artifacts/formal_s{240,368,496,624,752,880,1008,1136}; do
  for scenario in default strict125; do
    for mode in policy zero-residual; do
      PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
        scripts/evaluate_pc_command_residual_offpolicy.py \
        --config configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json \
        --environment-config configs/lowlevel/pc_goal59_terminal_bounded_${scenario}_20261109.json \
        --checkpoint "$seed_dir/final.pt" --output "$seed_dir/eval_${scenario}_${mode}" \
        --seed-start 20 --seed-count 40 --device cpu --dtype float64 --action-mode "$mode"
    done
  done
done
```

## 8. Git 步骤

禁止 `git add -A`。只添加本 Goal 的文档、脚本、配置、测试和 research 记录；
不添加 `artifacts/`、checkpoint、日志、缓存或大 PDF。提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal59 \
  --goal-doc doc/goals/goal59_bounded_terminal_residual_20261109.md --strict
git diff --cached --check
```

本回合不自动 commit/push；只有用户明确要求才执行 commit/push。

## 9. 明确禁止

- 不重开 Goal27 的 PPO、reward、gamma/lambda、seed rescue；
- 不扫 Goal30 的 terminal alignment window；
- 不扫 Goal31 的 path-lookahead/terminal reference 参数；
- 不启动 SAC/REDQ/DroQ；只有 H3/H4 明确成立后才另开 Goal；
- 不打开 held-out `100–149`；
- 不把 eval-time scale 诊断写成正式 Gate；
- 不因为 screen 效果不好就换 seed、加预算或放宽门限；
- 不把 `INVALID` 写成 `FAIL`，不把 `FAIL` 写成 `PASS`。

## 10. Goal 提示词

```text
你现在执行 Goal59：终端门控的有界命令残差与 12D endpoint 观测。

先读：
- doc/goals/goal59_bounded_terminal_residual_20261109.md
- doc/goals/goal58_result_20261109.md
- doc/goals/goal30_result_20261109.md
- doc/goals/goal31_result_20261109.md
- research/blockers/2026-11-09-goal58-offpolicy-residual-saturation.md
- research/STATE.md
- AGENTS.md

硬约束：
1. 不换算法；保持 Goal58 的 DDPG/TD3-style off-policy；
2. 只改 actor 有界输出、terminal gate、12D endpoint 观测和诊断指标；
3. 不扫 reward/gamma/seed/window/lookahead；不打开 held-out；
4. 先 screen 后 formal；screen 只做工程有效性检查；
5. 失败写 blocker、至少两个可证伪 hypothesis、reading note、ledger；
6. 每个 Gate 结果写清 success、terminal capture、plane-miss、timeout、safety、
   residual/gate 读数和评估协议；
7. 不自动 commit/push。

执行顺序：
- Step 0：Run Contract 与只读审计；
- Step 1：实现并测试 actor scale/squash、terminal gate、12D endpoint；
- Step 2：本地 parity/Ruff；
- Step 3：服务器 dry-run + microbenchmark；
- Step 4：G8/G12/U12 screen；
- Step 5：G12 8-seed formal；
- Step 6：dev 20–59 配对评估；
- Step 7：path-zero 归因；
- Step 8：result、ledger、STATE、checker。
```
