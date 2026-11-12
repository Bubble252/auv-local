# Goal95：监督残差预训练 + 同协议 PPO 有界微调

**日期**：2026-11-11<br>
**状态**：`SCREEN_FAILED / SUPERVISED_PRIOR_INSTALLED / PATH_INDEPENDENCE_NEGATIVE / NO_SEED_EXPANSION / NO_SAC`<br>
**结果**：`doc/goals/goal95_result_20261111.md`<br>
**上游**：Goal94 `PHASE1_PARTIAL / PHASE2_RATE_AWARE_FALSIFIED / NO_SAC`<br>
**性质**：只改 PPO residual actor 的初始化；不换 PPO 主算法，不改 reward、terminal、success、goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal95 --goal-doc doc/goals/goal95_supervised_residual_pretrain_20261111.md --strict
```

Gate 记录前不得改 seed、阈值、arm 定义、场景、动作尺度或 held-out。实现修复可以
迭代；一旦开始记录 screen，不得根据 dev 结果改确认参数。

## 1. 背景

Goal94 已经把“动作参数化/速率截断”这条线排除得比较干净：

- `S15` 三 seed strict125 为 `20/21/21`，均值 `20.67/40`，default `31/31/31`；
- rate-aware 让 post-clamp PPO survival 到 `1.0`，但 raw fin rate clip 仍为
  `0.266–0.886`，PPO/BC fin L2 ratio 只有 `0.037–0.052`；
- `F15` 用 fin gain 4 把 PPO/BC fin ratio 提到 `0.19–0.21`，success 仍是
  `31/21`，total fin clip `0.53–0.54`；
- 网络结构与 Goal86/87/89 parity 一致；冻结 BC 的 22 个 state entry 与
  Goal86 checkpoint `max abs diff = 0.0`。不是接线、维度、checkpoint 错配或
  NoisyNet 主导。

因此当前主瓶颈是 **residual 的幅度/方向/信用分配**，而不是 rate limiter。
Goal95 的做法是先用已有 D3/ALOS 标签给 PPO residual actor 一个强先验，再做
完全相同的 1M PPO 微调。它不是“再调一个 gain”，也不是换算法。

## 2. 网络结构对照与本次修正

Goal95 开始前重新对照了 Goal86 P-A BC、Goal94 S15 和当前 policy：

| 项目 | Goal86 BC | Goal94 S15 | Goal95 |
|---|---|---|---|
| `d_z_total` | 64 | 64 | 64 |
| path | 12（前 8 列有效，endpoint4 零） | 12 | 12 |
| command | 3（列权重为零） | 3 | 3 |
| current encoder | 存在但权重全零 | 存在但权重全零 | 存在，预训练可学 |
| actor | `[256,128]` ELU | `[256,128]` ELU | 同 |
| state encoder | `[128,128]` | `[128,128]` | 同 |
| DADE | off | off | off |
| frozen BC 22 entries | 基准 | `max abs diff=0.0` | 同 |

对比中发现并修复了两个真实链路缺口：

1. `train_pc_supervised_residual` 只优化 `actor_encoder` 和 `actor`，没有把
   `path_encoder`、`path_endpoint_encoder`、`actor_current_encoder` 放进
   optimizer；
2. `load_pc_supervised_residual` 只加载 actor encoder/actor，不加载上述
   path/endpoint/current 权重；且 `scripts/pretrain_pc_residual.py` 的 layout
   漏了 `current_dim`。

修复后，监督预训练学到的路径/终端特征会真正进入 PPO residual actor；这也是
Goal95 与历史 H3-SR/Goal86 “只热启动 actor 主干”的关键区别。修复已补单测，
不是只改文档。

## 3. 动作组合代数

环境实际执行：

```text
executed = ALOS + s_env * (r_BC + s_ppo * r_PPO)
```

- `ALOS`：actor 可观测的 PD/PI 传统控制器；
- `r_BC`：Goal86 P-A 冻结 BC 输出的 residual，SHA
  `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`；
- `s_env = 1.5`：`pc.actuator_residual_scale`，与 Goal94 S15 完全相同；
- `s_ppo`：`base_policy.residual_scale`，PPO residual 的有效尺度。

为了让监督残差在 gate 全开时复现 D3，PPO target 必须是：

```text
r_PPO = ((D3 - ALOS) / full_scale - s_env * r_BC) / (s_env * s_ppo)
```

不是简单 `D3 - ALOS`。如果直接拟合 `D3 - ALOS`，冻结 BC 的 `r_BC` 会再被加
一次，闭环动作会过冲；这正是 Goal86/Goal94 组合语义里最容易犯的错。

## 4. 假设

### H95-A：监督先验能打破 PPO residual 的零解锁定

- 机制：先用标签直接拟合一个有用的 `r_PPO`，再让 PPO 从该点微调，而不是从
  近零输出开始探索。
- 成立应看到：PPO/BC fin L2 ratio `>=0.15`，且监督初始化时 composed action
  与 D3 的逐通道误差 `<1e-6`（离线检查）。
- 失败应看到：初始化 residual 仍近零，或 fin ratio `<0.10`。

### H95-B：路径/终端编码器必须参与预训练

- 机制：terminal capture 依赖 path12 的后 4 个 endpoint 特征；如果只热启动
  actor 主干，prior 无法按目标方位修正。
- 成立应看到：path/endpoint encoder 在预训练后非零，且 path-zero 诊断使
  pre-goal RMSE 明显恶化；P-* 的 strict125 `>=21/40`。
- 失败应看到：path encoder 仍为零，或 path-zero 与正常 path 无差异。

### H95-C：如果监督先验仍不改善闭环，问题不是初始化

- 机制：如果 composed action 已经能离线复现 D3，但 1M PPO 微调后 strict125
  仍 `<=20/40`，则剩余瓶颈是 reward/terminal/credit assignment 或闭环分布
  偏移，而不是 residual 初始化。
- 成立应看到：离线 target 误差小，但 dev40 success 不移动，且 trace 显示
  PPO 微调把监督 residual 拉回近零或方向反转。
- 失败应看到：P-* 在 1-seed screen 就达到 strict `>=21`、default `>=31`。
- 该假设成立时，不直接开 SAC；先按 Research Escalation 写 blocker，再决定
  是否需要新的预注册 off-policy 对照。

## 5. 固定契约

| 项目 | 固定值 |
|---|---|
| 环境契约 | Goal85 `unbiased_noise_v1`，`position_source=eta`，零传感器延迟 |
| path | 12 维；BC 使用前 8 列，endpoint4 由 PPO endpoint encoder 处理 |
| frozen BC | Goal86 P-A，SHA `80024ef...ff9dd` |
| PPO | 1000 envs × 25 steps × 40 iterations = 1M transitions |
| PPO 超参 | 与 Goal94 S15 完全相同：clip 0.02、KL 0.005、lr 3e-4、1 epoch、100 mini-batches |
| 动作尺度 | `s_env=1.5`；`s_ppo=1.0`（S0/P-D3）或 `2/3`（P-BC15） |
| gate | `enter=10 m`、`full=3 m`；只 gate PPO residual |
| 训练 seeds | screen `86201`；扩 seed 时 `86202/86203` |
| 评价 | CPU/float64，dev seeds `20–59`，default + strict125 |
| held-out | `100–149` 关闭 |
| reward / terminal / success / goal radius | 不改 |
| 算法 | PPO；SAC/REDQ/DroQ 不在本 Goal 内 |

## 6. 实验矩阵

### 6.1 Screen（先跑）

| arm | residual 初始化 | `s_ppo` | PPO | seeds | 预算 |
|---|---|---:|---|---:|---:|
| `S0` | 当前零初始化（Goal94 S15，复用） | 1.0 | 1M | 86201/2/3 | 已存在 |
| `P-D3` | 监督 target 使 composed = D3 | 1.0 | 1M | 86201 | 1M |
| `P-BC15` | 监督 target 使 BC 1.5× + PPO 1.0× = D3 | 2/3 | 1M | 86201 | 1M |

1-seed screen 只用于判断是否值得扩 3 seeds；不把 screen 写成 formal 结论。

### 6.2 扩 seed 条件

`P-D3` 或 `P-BC15` 满足以下全部条件才扩 `86202/86203`：

- strict125 `>=21/40`；
- default `>=31/40`；
- path 3D RMSE 相对同 seed S0 不差超过 `0.02 m`；
- timeout 不增加；
- PPO/BC fin L2 ratio `>=0.15`；
- 无 NaN/Inf、无 safety 恶化。

### 6.3 评价指标

每个 run 同时报告：

- success、timeout、safety、terminal kind；
- `eta_min`、`eta_done`、path 3D RMSE；
- PPO/BC fin L2 ratio、raw fin rate clip、post-clamp survival；
- direction：成功/失败样本的 residual cos 均值与 bootstrap CI；
- near-miss shell：`+0.25/+0.50/+1.00 m`；
- annulus entry、conditional capture；
- radius sensitivity（只作诊断，不改训练/终止契约）。

## 7. 预注册判据

### 7.1 1-seed screen

`P-D3` / `P-BC15` 任一通过：

- strict125 `>=21/40`；
- default `>=31/40`；
- path 相对 S0 同 seed 差值 `<=0.02 m`；
- timeout `0`；
- fin ratio `>=0.15`；
- 无 NaN/Inf。

通过则扩 3 seeds；否则记录 `FAIL`，不扫 lr/epoch/gain/seed。

### 7.2 3-seed formal

至少一个 P-* arm 满足：

- 3-seed strict125 均值 `>=21/40`；
- 3-seed default 均值 `>=31/40`；
- path 相对 S0 三 seed 均值不差超过 `0.02 m`；
- timeout 不增加；
- 方向分离的 bootstrap CI 下界 `>0`；
- fin ratio `>=0.15`。

若通过，才允许进入 5M formal；若失败，按 Research Escalation 写 blocker 和
`research/LEDGER.jsonl`，不直接换 SAC/REDQ/DroQ。

## 8. 执行计划

### Step 0：预注册与实现

- [x] 对照 Goal86/87/89 网络结构；
- [x] 修复监督残差 path/endpoint/current 训练与加载缺口；
- [x] 添加 path/endpoint/current 迁移单测；
- [x] 生成 `configs/lowlevel/goal95_20261111/`；
- [x] 写本 Goal95 文档并运行 checker；
- [x] 同步代码与标签到远端。

### Step 1：监督标签与预训练

- [x] 用 `scripts/build_pc_goal95_residual_supervision.py` 生成 `P-D3` 与
      `P-BC15` 的 `goal95_supervision.npz`；
- [x] 校验 composed action 与 D3 的逐通道误差；
- [x] 用 `scripts/pretrain_pc_residual.py` 在远端 GPU 预训练两个 residual actor；
- [x] 检查 path/endpoint/current encoder 非零、frozen BC drift `0`。

### Step 2：1-seed PPO screen

- [x] `P-D3` seed86201、1M；
- [x] `P-BC15` seed86201、1M；
- [x] dev40 default+strict125，policy + frozen-base 配对；
- [x] trace 计算 fin ratio、rate clip、direction、shell 指标；
- [x] 与 S0 seed86201（strict 20、default 31）比较；原始 summary 已校正。

### Step 3：扩 seed 或裁决

- [x] screen 未通过，不扩 86202/86203；
- [x] 已写 result + blocker + ledger，停止该 arm；
- [x] 已更新 `research/STATE.md` 和根目录 `research/STATE.md`；
- [x] 运行 checker（见 result 文档）。

## 9. 证据路径（实际）

- `configs/lowlevel/goal95_20261111/s0_ppo.json`
- `configs/lowlevel/goal95_20261111/p_d3_pretrain.json`
- `configs/lowlevel/goal95_20261111/p_d3_ppo.json`
- `configs/lowlevel/goal95_20261111/p_bc15_pretrain.json`
- `configs/lowlevel/goal95_20261111/p_bc15_ppo.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/pretrain_P_D3/final.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/pretrain_P_BC15/final.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/P_BC15/final.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/trace.npz
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40_initial/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40_pathzero/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/labels/P_D3/goal95_supervision.npz`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/labels/P_BC15/goal95_supervision.npz``

## 10. Git 步骤

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git add src/auv_lowlevel/batch/pc_residual_pretrain.py \
  scripts/pretrain_pc_residual.py \
  scripts/build_pc_goal95_residual_supervision.py \
  scripts/build_pc_goal95_configs.py \
  tests/test_lowlevel_pc_residual_pretrain.py \
  configs/lowlevel/goal95_20261111 \
  doc/goals/goal95_supervised_residual_pretrain_20261111.md
git diff --cached --check
python ../scripts/check_research_ledger.py \
  --goal Goal95 \
  --goal-doc doc/goals/goal95_supervised_residual_pretrain_20261111.md --strict
```

只有用户明确要求才 commit / push；禁止 `git add -A`。
