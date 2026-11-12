# Goal27：ALOS 命令残差、残差接口对照与信用分配诊断

日期：2026-11-09<br>
状态：`H3_SR_FAIL_STOP`（C1/R1/H1/G1/A1 仍为 FAIL；H3-SR 追加实验 FAIL）

> 2026-11-09 追加：原 `FINAL_FAIL_PIVOT` 对 C1/R1/H1/G1/A1 仍然有效。阻塞记录中的
> H3（on-policy PPO 样本效率/探索不足）已通过 H3-SR 直接检验并判为 FAIL；预注册、
> 数据、命令和门禁见
> `doc/goals/goal27_addendum_h3_supervised_residual_20261109.md`。追加结果见
> `doc/goals/goal27_h3_sr_result_20261109.md`。本追加不修改
> C1/R1/H1/G1/A1 的确认参数、seed、eval split 或既有结论。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal27 --goal-doc doc/goals/goal27_command_residual_credit_assignment_20261109.md --strict
```

## 1. 背景与裁决

Goal26 已完成 `FAIL / STOP`：在 D3 frozen BC 上直接学习
`[rudder, stern, rpm]` action residual，并把 `w_path_3d` 从 0.15 提到 0.30，
pre-goal 3D RMSE 只改善约 `0.014–0.017 m`，success 没有提升，residual/base
L2 比例只有 `1.8%–2.7%`。因此不能继续用“再加一点路径奖励权重”作为主线。

本 Goal 的核心判断改为：

1. **残差空间可能比残差大小更重要。** 直接在执行器动作上加残差，策略容易学到
   速度补偿或小幅抖动，而不是路径修正；在 ALOS 的
   `[psi_d, theta_d, u_d]` 命令空间加残差，动作语义更接近“怎么修路径”，而且
   仍由已知的 PD/PI 控制器生成执行器命令。
2. **信用分配不能只靠 gamma。** 当前 `gamma=0.995, lambda=0.95` 时，GAE 有效
   权重视野约为 `1/(1-gamma*lambda)≈18.3` 步，即 50 Hz 下约 `0.37 s`；
   `num_steps_per_env=128` 的 2.56 s rollout 并没有转化为有效的长时信用分配。
3. **无侧向推进器是硬约束。** REMUS100 只有舵、艉水平舵和螺旋桨；横向修正必须
   通过艏向/航迹角和速度间接完成。若所有残差臂都受舵角、舵速率或速度限制，
   继续加 PPO 预算不会产生路径跟踪收益。
4. **BC 修复必须有上限。** 本 Goal 不无上限重采 BC；只做一次教师/数据一致性
   审计。之后把预算放在残差接口、奖励和信用分配上。

## 2. 研究问题与可证伪假设

### H1：ALOS 命令残差优于执行器 action residual

机制：命令残差在 3D 引导空间内改变 heading/pitch/speed，经固定控制器落成执行器
动作；探索更结构化，且不会直接产生高频执行器抖动。

可证伪预测：在同一 reward、同一 eval matrix、同一 1-seed screen 下，
`command_residual` 的 pre-goal RMSE 改善至少比 Goal26 P2 的 `0.017 m` 大一个
可分辨量级（目标 `>=0.10 m`），且 success/safety 不恶化；若没有，H1 不成立，
残差接口不是主因。

### H2：信用分配窗口是主因之一

机制：`gamma=0.995, lambda=0.95` 的 GAE 有效视野只有约 0.37 s，无法解释
20–30 s 的路径误差。

可证伪预测：在最佳残差接口上，把 `gamma=0.999, lambda=0.99`、rollout 扩到
256 steps（约 5.12 s）后，path reward 带来的 pre-goal 改善显著超过当前设置；
若只改 gamma 不改 lambda/rollout，改善应很小。

### H3：无侧向推进器的执行器约束是主因之一

机制：`src/auv_lowlevel/batch/dynamics.py:531-543` 中只有
`rudder -> sway/yaw`、`stern -> heave/pitch`、`propeller -> surge`，没有独立的
sway/heave 推力器。横向误差需要靠艏向和速度间接修正，低速时舵效随速度平方下降，
高速时转弯半径又变大。

可证伪预测：若命令残差在不增加舵角/舵速率饱和的情况下仍能显著降低 cross-track
error，H3 不是主因；若误差与曲率、舵角饱和、heading rate 强相关，且所有残差臂都
在饱和附近，H3 成立，下一步应测试弯道限速/曲率调度，而不是继续加 PPO。

### H4：奖励目标仍不完整

Goal26 只改变了 `w_path_3d`，没有真正加入 success、terminal progress、plane miss、
终点距离/方位误差等终止项。H4 的可证伪预测：在最佳接口上加入 reward-v2 后，
success 或 pre-goal RMSE 至少一个达到效果门禁；若没有，则不是奖励项缺失。

### H5：增大 PPO 探索噪声不能修复系统性基座偏差

机制：高斯 action noise 是零均值、逐步独立的；基座误差是系统性偏差。提高噪声
主要增加抖动和饱和风险，而不是把偏差消掉。

可证伪预测：`init_noise_std` 从 0.1 提到 0.2 的对照臂若没有超过 0.10 m 改善，
且 saturation/success 变差，则 H5 成立；该臂只作对照，不作为主方案。

## 3. 实验臂与隔离

### 3.1 基线

- `B0`：ALOS + 可部署 PD/PI 控制器，无残差；
- `D3`：Goal25/26 的 frozen BC，不重训、不覆盖；
- `G26-P2`：Goal26 的 action residual + `w_path_3d=0.30`，作为历史对照，不重复跑。

### 3.2 新臂

| arm | base | residual space | 说明 |
|---|---|---|---|
| `C1` | ALOS + 可部署 PD/PI | `delta[psi_d, theta_d, u_d]` | 主候选；零初始化、限幅、限速 |
| `C1p` | ALOS + privileged teacher | `delta[psi_d, theta_d, u_d]` | 仅作 upper bound；不得当作部署结果 |
| `Z1` | frozen D3 | 64D `z_t` | 对照；零初始化 `delta_z`，由冻结 decoder 出动作 |
| `A1` | frozen D3 | `[rudder, stern, rpm]` | 只在 reward-v2 或信用分配变化时重跑；否则用 G26-P2 |
| `S1` | ALOS + 可部署 PD/PI | 无残差 | 曲率/饱和感知限速，用于检验 H3 |
| `R1` | 最佳接口 | 同接口 | reward-v2：success/terminal/progress/plane-miss/near-goal 项 |
| `G1` | 最佳接口 | 同接口 | `gamma=0.999, lambda=0.99, num_steps=256` |
| `N1` | 最佳接口 | 同接口 | `init_noise_std=0.2`，只作探索噪声对照 |

### 3.3 数据与隔离

- 训练数据：D3 只读；C1 不需要新 BC 数据；
- eval：dev seeds `20–59`，default/strict125，`endpoint_stop`，
  `realistic_calibrated`，CPU/float64；
- held-out `100–149` 禁止进入训练、调参和模型选择；
- 不覆盖 Goal24/25/26 的配置、checkpoint、数据和评估；
- 大型训练只放服务器隔离目录，本地只做 smoke、审计和评估。

## 4. 方法合同

### 4.1 ALOS 命令残差

基础命令来自现有 ALOS：

```text
c0 = [psi_d, theta_d, u_d]
```

策略只输出零初始化残差：

```text
delta_c = [delta_psi, delta_theta, delta_u]
c = c0 + scale_c * delta_c
```

这里的命令残差是**策略输出**，不是把 `[psi_d, theta_d, u_d]` 作为 actor 输入；
现有 `pc.actor_command_dim` 是输入合同，二者不能混用。`C1` 的基线必须是可部署
控制器；如果先用现有 privileged teacher 做快速 upper bound，必须单独记为 `C1p`，
不能把 `C1p` 的结果写成部署策略收益。

然后依次执行：

1. heading/pitch/speed 限幅；
2. 现有 guidance rate limit（`pc_env.py:943`）；
3. 固定 PD/PI 控制器生成 `[rudder, stern, rpm]`；
4. 现有执行器幅值/速率检查。

要求：

- 零残差时，`C1` 与 `B0` 的 118 个闭环指标逐元素一致；
- 基础控制器必须使用 actor-visible 的 12D state，不得使用真实
  `env.eta/env.nu/current` 作为部署输入；真实状态只允许进 critic 或诊断日志；
- 命令残差必须有 `scale_heading/scale_pitch/scale_speed`、绝对限幅和速率限制；
- `delta_u` 不允许把速度降到低于任务最小值，除非显式进入 terminal/curve 减速模式；
- 不改变 D3、Goal26 checkpoint 和现有 `action residual` 默认路径。

### 4.2 `z_t` 残差

冻结 D3 的 state/path encoder 和 actor decoder，只训练：

```text
z_t' = z_t + beta * delta_z(o_t)
```

`delta_z` 零初始化；先验证 `beta=0` 与 D3 完全一致，再允许 PPO 更新。该臂用于判断
“潜空间残差”是否比命令空间残差更有效，不作为默认部署路径。

### 4.3 奖励 v2

只在接口 screen 选出一个候选后测试，避免同时改接口和奖励。候选奖励项：

- success / terminal progress；
- goal-plane miss 惩罚；
- 终点距离、终点切向偏移、终点 heading/pitch 误差；
- pre-goal 3D RMSE 的稠密项。

终点相关项只在接近终点时增强，不能在整条路径上压过路线跟踪项。所有奖励改动必须
保持 success-first 护栏，不能通过放宽终点捕获或隐藏 timeout 提升 success。

### 4.4 Research Escalation

本 Goal 必须遵守 `AGENTS.md` 的 Research Escalation。遇到以下任一 hard blocker，
不能只换参数、加预算或反复试错：

- 同一 Gate 连续失败 2 次；
- 连续 3 次实现修复后指标没有移动；
- held-out / OOD 出现 100% 系统性错误；
- 两个方法差异小于 seed 噪声，无法判断；
- 现有 hypothesis 无法解释失败，且本地 ablation 无法区分；
- 需要引入外部论文方法，但本地没有 reading note。

触发后必须按顺序执行：

1. 读 `AGENTS.md`、`research/README.md`、`research/STATE.md`；
2. 写 `research/blockers/<date>-<slug>.md`，记录现象、证据、已排除项；
3. 写至少 2 个可证伪 hypothesis，以及每个 hypothesis 的最便宜 falsification；
4. 按“数据 → 标签 → 评测 → baseline → 架构 → 文献”的顺序排查；
5. 用 `paper-search` / `deep-research` 调研，写 `research/paper_reading/` note；
6. 跑最便宜的 falsification experiment，先做只读或 1-seed；
7. 在 `research/LEDGER.jsonl` 写 `decision`（continue / pivot / stop）和 `next`；
8. 只有 `pivot` 或 `stop` 才允许改主线；预注册 Gate 记录前不改确认参数。

不允许把“再扫一个权重”“再换一个 seed”或“再加一点预算”当作 autoresearch。

### 4.5 主动调研（未构成 blocker 时）

有些“不足”在触发 4.4 之前就能看见：本地 ablation 无法区分两个假设、计划借用外部
方法但没有 reading note、或 Step 1 的分解结果解释不了残差为什么学不动。此时允许
先做一次小范围主动调研，不必等到 Gate 失败：

1. 写 `research/paper_reading/<date>-<slug>.md`，记录问题、检索词、候选方法与
   它们对本场景的适用条件；
2. 用 `paper-search` / `deep-research` 至少交叉验证两篇来源；
3. 只跑最便宜的 falsification（只读统计或 1-seed），不直接开 4096 环境长训练；
4. 在 `research/STATE.md` 记一行“主动调研 + 结论 + 是否影响 Step 顺序”。

约束：主动调研不得修改预注册 Gate 的阈值、seed、eval split 或 checkpoint 选择
规则；若结论要改主线，仍必须补 4.4 的 blocker、至少 2 个可证伪 hypothesis 和
`research/LEDGER.jsonl` 的 `decision`。主动调研不能替代 Step 1/Step 3 的对照实验。

## 5. 执行计划

### Step 0：只读审计与 Run Contract

- [x] 记录 `ps`、`nvidia-smi`、`df -h /`、`df -h /dev/shm`、`git status --short`；
- [x] 复核 Goal26 P1/P2、D3 baseline、eval matrix 和 checkpoint SHA256；
- [x] 复核 `dynamics.py` 的执行器映射，确认无 sway/heave 独立推力器；
- [x] 写 `goal27_phase0_audit.md`，不启动训练。

### Step 1：离线奖励/优势分解

- [x] 在 D3 的 dev 20–39 固定轨迹上分解 path / terminal / progress / command /
      saturation 各 reward 项的 return 与信用代理份额；G26-P2 本地 checkpoint
      不可用，改用 Goal26 的 40-seed 汇总作历史对照，不重跑；
- [x] 统计 residual 幅度与 speed error、cross-track error、commanded turn rate、
      rudder saturation 的相关性；
- [x] 输出 `artifacts/lowlevel/goal27_offline_decomposition{,_strict125}/`；
- [x] 根据结果确认 H1/H2/H3/H4 的优先级，不训练；报告见
      `doc/goals/goal27_step1_offline_decomposition_20261109.md`；
- [x] §4.5 主动调研未触发：证据足以区分主因，无需外部文献裁决。

### Step 2：实现命令残差与零残差测试

- [x] 新增非破坏性的 `command_residual` 模式，默认关闭；
      `configs/lowlevel/*.json` 未设置该块时保持 `enabled=false`；
- [x] 实现可部署 PD/PI 控制器上下文，输入只来自 actor-visible state；
      `src/auv_lowlevel/batch/pc_command_residual.py` 只使用 `actor_state`，
      并复用 `BatchedRemus100Teacher`；
- [x] 加单元测试：shape、NaN、限幅、rate limit、zero-residual 等价；
      `tests/test_lowlevel_pc_command_residual.py`，全套 43 tests passed；
- [x] 2-seed smoke：确认 `C1` 与 `B0` 在零残差时一致；
      `artifacts/lowlevel/goal27_c1_zero_smoke_20261109/`，
      `max_abs_diff = {eta: 0, nu: 0, actuator: 0, reward: 0}`；
- [x] 不通过即停止，不启动 PPO；本项通过，允许进入 Step 3。

### Step 3：1-seed 接口 screen

- [x] `C1`：ALOS 命令残差，当前 reward，1 seed；40-seed dev 结果见
      `doc/goals/goal27_step3_interface_screen_20261109.md`；
      success `0.700/0.500`，pre-goal RMSE `4.9805/4.9667`，
      残差 `0.00289`，接口效果门禁 FAIL；
- [x] `C1p`：privileged teacher upper bound，仅用于判断接口潜力，不进入部署结论；
      success `0.850/0.500`，RMSE `4.7935/4.9575`；同 checkpoint 零残差对照
      `0.875/4.7920`，证明收益来自 privileged controller，不是 residual；
- [x] `Z1` 决策：不运行。H1 命令权重清零臂没有把残差从近零拉出，未满足
      “H1 证明 reward 是主因”这一前置条件；明确记为未测试，不伪装成已完成实验；
- [x] `A1`：frozen D3 action residual + reward-v2，40-seed dev 完成；
      相对 Goal26-P2 的 pre-goal RMSE 仅改善 `0.0213/0.0200 m`，success
      `-0.025/+0.050`，仍为 FAIL；见 `goal27_result_20261109.md`；
- [x] 最便宜 falsification：H1（命令权重清零）、G1（长 horizon）和 A1
      均完成；H1/H2 被证伪，A1 只有弱信号；
- [x] 同一 dev `20–59`、default/strict125；
- [x] 记录 success、pre-goal RMSE、plane miss、timeout、safety、saturation、
      rudder rate、crab angle、命令残差幅度；
- [x] 写 `goal27_interface_screen.md/json`；见
      `doc/goals/goal27_step3_interface_screen_20261109.md` 和
      `artifacts/lowlevel/goal27_interface_screen.json`。

### Step 4：条件分支

- [x] `C1`/`Z1` 效果门禁未通过；不扩 3 个训练 seed；
- [x] 接口失败后先跑 `R1` reward-v2，再跑 `G1`。Step 1 已证明 path 项
      只占 3.7% 奖励方差且 terminal 项全为零；G1 不能创造缺失的 success 信号；
- [x] `R1` 未通过，因此不启动“R1 + G1”叠加；改为 G1 单独对照；
- [x] G1 单独对照完成，残差 `0.00307`、success 与 B0 持平，H2 被证伪；
      裁决为 FAIL / PIVOT，停止命令残差 reward/gamma/seed 搜索；
- [x] `S1` 未触发：Step 1 显示饱和存在（约 19% material / 21% rate clip），
      但 G1/A1 没有显示饱和是 residual 收益的唯一瓶颈；留下新 Goal 处理；
- [x] `N1` 未运行：H1/H2 已失败，没有理由把探索噪声当作主方案；
- [x] 负结果已写入 `goal27_result_20261109.md`；没有换 seed、加预算或扫第三档权重。

### Step 5：Gate、收口与 Git

- [x] 所有候选使用同一 40-seed dev 矩阵（C1/C1p/R1/H1/G1/A1，seeds `20–59`）；
- [x] 写 `goal27_result_20261109.md`；
- [x] 更新 `research/STATE.md`、`research/LEDGER.jsonl`，并写 decision 记录；
- [x] 运行 ledger 检查：`research ledger check: PASS (goal=Goal27, entries=1)`；
- [x] 只 add 本次相关文件，运行 `git diff --cached --check`，不自动 commit/push。

## 6. Gate

### success-first 主门禁

| check | threshold |
|---|---:|
| default success | `>= 0.695` |
| strict125 success | `>= 0.470` |
| safety delta | `<= 0.02` |
| plane miss / timeout delta | `<= 0.05` |
| pre-goal RMSE delta | `>3%` 恶化只记风险，不一票否决 |

### 接口效果门禁

- 至少一个主场景 pre-goal RMSE 改善 `>=0.10 m`，或 success 提升 `>=0.03`；
- 另一个主场景 pre-goal RMSE 恶化不超过 `3%`；
- 不能靠减速、悬停、绕路或降低终点捕获换 RMSE；
- `C1` 的命令残差必须保持有界，不能长期顶住 heading/speed rate limit。

### 执行器约束门禁

- material saturation 不得比基线恶化超过 `0.02`；
- rudder/stern rate clip 不得显著增加；
- 若误差与曲率、舵角饱和强相关，记录为 H3 证据；下一阶段优先测试弯道限速/
  曲率调度，而不是继续增加 PPO 预算。

### 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| C1 零残差不等价 | 实现错误 | 修复后重跑，不训练 |
| C1/Z1 都无改善 | 接口不是主因 | 进入 G1 信用分配 |
| G1 仍无改善 | horizon 不是主因 | 进入 R1 reward-v2 |
| 所有臂都受饱和限制 | H3 成立 | 转曲率限速/执行器评估 |
| N1 只增加抖动 | H5 成立 | 不把噪声当主方案 |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal27 \
  --goal-doc doc/goals/goal27_command_residual_credit_assignment_20261109.md \
  --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal27：ALOS 命令残差、残差接口对照与信用分配诊断。

先读：
- doc/goals/goal27_command_residual_credit_assignment_20261109.md
- doc/goals/goal26_result_20261108.md
- research/blockers/2026-11-08-goal26-residual-path-reward.md
- research/STATE.md
- AGENTS.md

硬约束：
1. 不破坏现有 action residual、D3 checkpoint、Goal24/25/26 配置和评估；
2. 新 `command_residual` 模式默认关闭，零残差必须与 ALOS 基线逐元素一致；
3. 基础控制器只能使用 actor-visible 12D state，真实 eta/nu/current 只进 critic 或诊断；
4. 不碰 held-out 100–149，不覆盖旧数据/checkpoint，不使用 git add -A；
5. 先做 Step 0/1 只读审计和离线分解，再决定是否训练；
6. 每次只改变一个因素；接口 screen 通过前不跑 reward-v2 或 gamma 扫描；
7. 失败即写负结果并停止，不换 seed、不补预算、不放宽门禁；
8. 能在服务器跑的任务尽量在服务器跑；本地只做只读审计、CPU 评估和 2-seed
   smoke，服务器使用隔离目录，不覆盖旧结果。
9. 触发 hard blocker 时必须进入 autoresearch loop：写 blocker、至少 2 个可证伪
   hypothesis、检索文献并写 reading note、跑最便宜 falsification，最后在
   `research/LEDGER.jsonl` 写 continue/pivot/stop；不得用反复调参代替调研。
   未构成 blocker 但证据不足或计划引入外部方法时，按 §4.5 主动调研，先写
   reading note 再做最便宜验证，但不得修改预注册 Gate。

执行顺序：
- Step 0：记录环境、复核 Goal26/D3、确认无侧向推进器；
- Step 1：离线 reward/advantage 分解与残差-误差相关性；
- Step 2：实现 command_residual + 可部署 PD/PI，跑零残差 smoke；
- Step 3：C1 与 Z1 的 1-seed screen；A1 只在需要时重跑；
- Step 4：按 Gate 分支进入 3-seed/40-seed；接口失败时先 R1，再 G1；
- Step 5：写 result、ledger、state，运行检查器，只提交本次相关文件。
```
