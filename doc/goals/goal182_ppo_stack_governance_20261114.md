# Goal182：PPO 训练栈与 residual authority 治理实验

**日期**：2026-11-14<br>
**状态**：`ACTIVE_PLANNING / C3_FIXED_600IT_DONE / FULL_CURRICULUM_FORMAL_PENDING / HELDOUT_CLOSED`<br>
**仓库**：`settled_局部规划器_反应式`

```yaml
goal_id: Goal182
repo: settled_局部规划器_反应式
status: ACTIVE_PLANNING / C3_FIXED_600IT_DONE / FULL_CURRICULUM_FORMAL_PENDING / HELDOUT_CLOSED
phase: P7_PPO_STACK_GOVERNANCE
created: 2026-11-14
updated: 2026-11-16
parents:
  - Goal92
  - Goal176
  - Goal177
  - Goal178
  - Goal179
  - Wheel-Legged-Lab reference
matrix: configs/governance/goal182_core_composition_index.json
core_matrix_pe2e: configs/governance/goal182_core_pe2e_matrix.json
core_matrix_alosb: configs/governance/goal182_core_alosb_matrix.json
matrix_status: NOT_GENERATED
eval_matrix: configs/governance/goal182_eval_matrix.json
eval_matrix_status: NOT_GENERATED
axis_registry: configs/governance/goal182_axis_registry.json
legacy: false
```

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal182 --goal-doc doc/goals/goal182_ppo_stack_governance_20261114.md --strict
```

- 本 goal 是训练型因子实验；所有因子必须在测量前预注册，不能边跑边改。
- 任一 Gate 失败两次、或连续三次修复后指标不移动，必须进入 autoresearch loop。
- held-out `100–149` 全程关闭；不读取、不评价、不写入 cache。

## 1. 背景与证据

### 1.1 为什么需要这个 goal

Goal177/178 已经把问题定位到 residual 权限，而不是 latent 信息或单纯 MLP 容量：

- Goal176：`z_t` 对纵向/pitch/q/w 的 held-out probe R² 为 `0.9967–0.99996`；
  late KL 只有 `4.31e-5–5.23e-5`，约为 `desired_kl=0.005` 的 `1%`；
- Goal177：最大 `head residual/action_std` 只有 `0.0737 < 0.20`，Gate A 判
  `RESIDUAL_TOO_SMALL_FOR_PPO`；
- Goal178：`A_S150` 的闭环改善来自冻结 BC authority `0.75→1.50`；PPO rudder 增量
  只占 composed rudder delta 的 `0.31%`。

Goal179 是诊断型 goal，只回答“head 是否有可见幅度、BC/PPO 是否抵消、reward/update
方向是否正确”。它不负责把 PPO 训练栈修好，也不负责比较多组训练配置。

与此同时，`/home/bubble/港中深暑研/Wheel-Legged-Lab` 提供了一个成熟的 PPO 训练栈
参考：独立 authority、显式探索边界、observation normalization、非对称 critic、
adaptive KL 和 metric-gated staged training。这些不能只停留在“参考/建议”，必须变成
有基线、有处理组、有 Gate 的正式实验。

### 1.2 本 goal 要回答的问题

| 问题 | 对应因子 | 主要判据 |
|---|---|---|
| PPO 的有效动作权限是否足以产生闭环影响 | authority 组成、通道范围 | residual/action_std、applied share、route dose response |
| 输入尺度是否让 head 难以学习 | observation / advantage normalization | 同预算下的 residual share、KL、route |
| 更新是否太保守或噪声太大 | clip / KL / adaptive LR | KL、clip fraction、grad norm、route per transition |
| 探索幅度是否与 residual 空间不匹配 | init std、min/max std、anneal | 有效 std、entropy、residual share、safety |
| critic 是否缺少足够信息 | privileged critic | value loss、explained variance、advantage 相关性 |
| 固定 iteration 是否浪费预算 | metric gate、staged promotion | 达到同等指标所需 transitions、晋级稳定性 |

## 2. 实验边界与上下游

### 2.1 与 Goal179 / Goal181 的关系

| Goal | 职责 | 与 Goal182 的边界 |
|---|---|---|
| Goal179 | head-only 幅度、BC/PPO 抵消、reward/update credit 诊断 | Goal179 Gate A 通过后，Goal182 才能进入 Phase 1 训练 |
| Goal182 | PPO authority、归一化、探索、信任域、critic、训练组织；D13 route-centric reward/StageGate 重构 | 固定使用 M 档 MLP（actor/critic `[512,256,128]`），不做架构搜索；D13=C 已选定，两种控制模式的 reward 目标表已确认（§4.9），scale audit 与 StageGate 细节待定，未批准前不生成 matrix、不训练 |
| Goal181 | MLP / Transformer actor 架构与容量配对 | Goal182 在 M 档 MLP 上选出稳定 PPO profile 后，Goal181 再比较 M 档 Transformer 与 M 档 MLP |

这条顺序是硬约束：如果 PPO 训练栈本身没有可用权限或 credit，先比较架构会把
“训练栈失败”误判成“架构失败”。

### 2.2 不改变的东西

- **D13 是显式例外**：已选 C，Goal182 后续会同时重构 route-centric reward 和
  StageGate。纯端到端与 ALOS+本体控制的两套 reward 修改目标表已确认（§4.9）；标注
  “审计起点”的权重、scale audit、晋级指标和阈值尚未定稿，termination 逻辑保持不变。
  权重和 StageGate 细节批准前，现有 reward/StageGate 继续冻结，不得据此生成或启动训练；
- D13 定稿前不改变 termination、success、goal radius、held-out；
- 不改变 BC checkpoint、BC 训练数据、held-out 划分；
- Goal182 内固定使用 M 档 MLP actor/critic `[512,256,128]`；旧 `[256,128]` 只作
  `historical_only`，不进入新 arm；不改变 encoder 层数或宽度，也不做 MLP/Transformer
  架构搜索，架构比较留给 Goal181；
- 不改变动作语义；authority、channel gain、归一化、探索和 critic 输入都作为显式因子；
- 不把多个因子塞进同一个 arm；每个 Phase 只允许一个因子变化；
- 不用 success 单项代替 route 3D/cross/vertical、preterminal、completion/progress。

D8 的 training-only 有界续路是一个显式例外：它只改变训练 episode 内的路线长度和
终止位置，不改变评价路线、评价 termination、success、goal radius 或当时的奖励公式。
D13=C 是第二个显式例外，但必须等 reward/StageGate 方案单独批准后才生效；不能用
D13 的未冻结内容提前改动训练。任何续路逻辑泄漏到评价环境，判
`INVALID_ROUTE_PROTOCOL`。

### 2.3 单因子原则

每个 Phase 的结构固定为：

```text
baseline = 上一 Phase 的 winner
treatment arms = baseline + 只改变一个因子
screen = 1 seed × 5 iterations
formal = 3 seeds × 整次 run 600–2600 iterations；首波 C6 不早停，第二波再加参考式早停
```

只有 screen 方向正确、且 instrumentation 完整的 arm 才进入 formal。formal 的 winner
才允许成为下一 Phase 的 baseline。禁止“authority + obs norm + clip + entropy 一起改”。

### 2.4 参考代码快照与复用规则

参考工程按 `/home/bubble/水下机器人/总协调/doc/governance/REFERENCE_CODE_POLICY.md`
治理：保留只读快照，不把参考仓库直接复制进 `src/`。本 Goal 使用的参考是
`Wheel-Legged-Lab@e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`，计划快照位置为：

```text
/home/bubble/水下机器人/third_party/reference/wheel_legged_lab_e61bfe1/
```

参考仓库根 `LICENSE` 是 Apache-2.0，但 `rsl_rl_ppo_cfg.py`、`train.py` 等关键文件
单独标记 BSD-3-Clause；必须逐文件保留 copyright/SPDX，禁止删掉文件头。快照只复制
需要的源码/配置/测试，不复制 `.git/`、checkpoint、视频、日志或训练产物。

生产代码不得直接 import `third_party`。优先把参考逻辑改写成本项目接口并补测试；只有
许可证兼容且必须保留原实现时才 vendoring，并增加 `SOURCE.json`、
`THIRD_PARTY_NOTICES.md`、来源路径、commit、SHA-256 和修改说明。参考快照本身不进入
Goal182 的 matrix、包入口或部署依赖。

## 3. 假设与证伪

### H182-A：独立 per-channel PPO authority 能恢复可用 residual

- 机制：当前 `0.75×(r_BC+0.25×r_PPO)` 把 PPO 权限压到 `0.1875`，再乘
  `[1,1e-6,1e-6]`，vertical/cross 实际关闭。
- 成立预期：至少一个 authority arm 的 `residual/action_std >= 0.20`，applied share
  `>=1e-2`，且 route 至少一个方向出现跨 authority 档位的单调剂量响应。
- 失败预期：所有 authority 档位下 applied share 和 route 仍不移动；此时不是
  authority 单独能解释，转查 plant、动作方向或 credit。
- 最便宜证伪：Phase 0 eval-only，复用 Goal179 checkpoint，不训练。

### H182-B：PC 固定尺度 + empirical 只在 authority 可见后才有收益

- 机制：固定物理 scale 先把量纲压到 O(1)，empirical running stats 再消除剩余均值/
  方差差异；两者都改善输入条件数，但不能替代动作权限。
- 成立预期：在 Phase 1 winner 上，`N1_FIXED_EMPIRICAL_BOTH` 的同预算 residual share、
  KL 或 route 优于 `N0_CURRENT_FIXED`，且 improvement 大于 seed 噪声。
- 失败预期：差异落在 seed 噪声内，或只改变 loss 不改变 route；此时保留
  `N0_CURRENT_FIXED`，删除 empirical 因子。
- 最便宜证伪：1 seed × 5 iterations screen；同时记录 running mean/std 漂移和
  `N0`/`N1` 的 post-normalization std。

### H182-C：参考 trust-region profile 只在 authority 修复后有收益

- 机制：参考工程把 `clip=0.2`、`desired_kl=0.01`、`schedule=adaptive` 和初始
  `lr=3e-4` 作为同一个 PPO profile；Goal92 已证明单独放宽 clip 不能改变
  success/path，因此首轮不拆成“先 adaptive、再宽 clip”的组合。
- 成立预期：`T1_REFERENCE_TRUST_REGION` 的 route per transition 优于
  `T0_CURRENT_PROFILE`，且 KL/clip telemetry 显示更新没有被长期压死。
  `T3_CLIP_ONLY` 只作后续 Goal92 复核，不应优于 `T1`；否则与 Goal92 结论冲突。
- 失败预期：`T1` 落在 seed 噪声内或只增大 clip fraction；此时保留
  `T0_CURRENT_PROFILE`，不继续放宽 trust region。
- 最便宜证伪：同一 authority winner 上跑 1 seed × 5 iterations，比较 KL/clip/route。

### H182-D：有效探索必须按执行器空间校准

- 机制：当前 mean 与 noise 的缩放不对称；`policy.residual_action()` 缩放 mean，
  但采样 noise 仍来自未缩放的 `policy.std`，因此不能把 `0.1×0.75×0.25` 当成有效
  探索 std。必须先实测 per-channel applied normalized delta 的 `sigma_eff`；它可能
  低到无法探索，也可能被 authority 放大到不安全，不能仅凭 raw `init_std` 判断。
- 参考 `0.3` 作用在直接 normalized action 上；本项目是带 per-channel authority、
  channel gain 和 rpm 2x 归一化 Jacobian 的 residual。D5 首轮仍按参考值采用
  `init_std=0.3`，但必须记录实际 applied std，不能把参考值当成等价幅度。
- 成立预期：`E1_REFERENCE_STD` 在不增加 safety/timeout 的前提下提高 residual share
  或 route；`E0_CURRENT` 仍接近零。
- 失败预期：提高探索只增加 clip/safety，不改善 route；此时保留低探索并转查 credit。
- 最便宜证伪：Phase 4 screen，记录有效 std、entropy、clip rate 和 safety。

### H182-E：privileged critic 改善 credit，但不改变 actor 部署

- 机制：critic 看到 full state/current/disturbance/route error，value 估计更准，
  advantage 更干净；actor 输入不变。
- 成立预期：`C1_DUAL_PATH_PRIVILEGED` 的 value loss、explained variance、advantage
  相关性或 route 优于 `C0_CURRENT_PARTITION`；actor 部署接口只使用非特权 path
  block，不读取真实状态、真实 current 或 privileged route 真值。
- 失败预期：value 指标和 route 都无差异；此时 critic 信息不是瓶颈。
- 最便宜证伪：Phase 5 screen，固定 actor checkpoint，只训练 critic 或同预算对照；
  同时检查 actor path block 中不存在 progress/remaining/curvature/endpoint 泄漏。

### H182-F：参考式早停比固定短预算更可靠

- 机制：固定 16–32 iterations 远短于参考工程，不能证明收敛；整次 run 使用
  `min_iterations=600`、`max_iterations=2600`，并在 20-iteration 滑窗连续 3 次满足
  route-health/safety 规则后早停。
- **首波不验证该假设**：CU7 已选 B，首波正式 run 不启用 C6 early stop，直接跑到
  2600 上限；该假设留到第二波启用早停时再检验。
- 成立预期：同一 arm 在 600–2600 iterations 内可通过早停达到稳定 route，且不依赖
  逐参数扫 iteration。
- 失败预期：20/3 规则永远不满足而只能跑满 2600；此时修规则或训练栈，不扫 iteration。
- 最便宜证伪：用已有训练曲线离线回放早停规则，不启动新训练。

### H182-G：3D 需要非对称 per-channel authority

- 机制：rudder、stern、rpm 的 plant 敏感度和安全边界不同；把同一个 authority
  乘到三个通道上，会让某个通道先饱和，不能把 3D 的失败归因于“3D 没用”。
- 成立预期：`A3_SEP_3D` 在每通道 applied share 可见的同时，safety/timeout 不恶化；
  若等权 3D 先饱和，则 `A4_SEP_3D_HALF` 或非对称 authority 能恢复稳定性。
- 失败预期：无论等权还是非对称，3D 都只增加 clip/safety，不改善 route；此时判
  `THREE_D_AUTHORITY_UNUSABLE`，保留 rudder/stern 作为中间结论，但 3D 目标不关闭。
- 最便宜证伪：Phase 0 的 `A3_SEP_3D` vs `A4_SEP_3D_HALF`，只做 eval-only。

## 4. 因子与 arm 设计

### 4.0 参考快照前置检查（生成 matrix 前）

- [ ] 固定 `Wheel-Legged-Lab` 仓库 URL、commit `e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`
  和拉取日期；
- [ ] 在 `third_party/reference/wheel_legged_lab_e61bfe1/` 建立只读快照，只复制本 Goal
  实际引用的文件；
- [ ] 为每个复制文件记录原路径、许可证、SHA-256 和用途，写成 `SOURCE.json`；
- [ ] 逐文件核对 Apache-2.0 / BSD-3-Clause 文件头，保留版权和 SPDX，不把根许可证当成
  所有文件的许可证；
- [ ] 在 `总协调/doc/governance/REFERENCE_SOURCES.md` 登记该快照和适用范围；
- [ ] 确认没有复制 `.git/`、checkpoint、视频、日志、缓存或数据集；
- [ ] 若需要复用函数或文件，先写适配/改写方案和测试；生产代码不得直接 import 快照。

### 4.1 Phase 0：authority + channel calibration（eval-only）

动作通道顺序固定为 `[rudder, stern, rpm]`。`1e-6` 表示该通道关闭；不使用 `0.0`，
因为 wrapper 要求 channel gain 为正。三组通道范围是本 Goal 的主因子：

| arm | authority 公式 | channel gain | 用途 |
|---|---|---|---|
| `A0_HISTORICAL` | `0.75*(r_BC + 0.25*r_PPO)` | `[1,1e-6,1e-6]` | 历史基线，复现 Goal177/178 |
| `A1_SEP_RUDDER` | `0.75*r_BC + a_ppo*[1,1e-6,1e-6]*r_PPO` | `[1,1e-6,1e-6]` | rudder-only 安全对照 |
| `A2_SEP_STERN` | 同上，只开 stern | `[1e-6,1,1e-6]` | stern-only 对照 |
| `A3_SEP_3D` | 同上，三通道独立 authority | `[1,1,1]` | **3D 最终目标** |
| `A4_SEP_3D_HALF` | `A3` 的 `a_ppo/2` | `[1,1,1]` | 3D 剂量下界/稳定性对照 |

`a_ppo` 不拍脑袋给定值；按“目标 `residual/action_std` 落在 `[0.20,0.35]`”校准，
并记录实际 `a_ppo`、每通道 applied share、clip rate。若无法在该区间内找到稳定点，
Phase 0 判 `AUTHORITY_CALIBRATION_FAILED`，不进入训练。

3D 是最终目标，但 3D 不能跳过 rudder/stern 对照：三组必须在同一 checkpoint、
同一 dev seeds、同一评价矩阵下比较，才能知道 3D 的收益是否来自 stern/rpm，
以及 3D 是否引入 safety/timeout 回归。

`bc_authority=0.75` 只作为 Phase 0/1 的历史可比锚点，不是 BC 的最优值。它是
Goal173 的 `actuator_residual_scale` screen 产物；当时 `P_RUD_025` 与 `P_RUD_075`
同时改变了 BC 和 PPO authority，属于混合干预。BC authority 必须另立因子，单独测试
`0.75 / 1.0 / 1.25 / 1.5`，不能和 PPO authority 一起锁死。

### 4.2 Phase 1：channel scope（一个因子，三组）

| arm | channel scope | channel gain | 角色 |
|---|---|---|---|
| `P0_HISTORICAL` | 历史 rudder-only composition | `[1,1e-6,1e-6]` | 历史基线 |
| `P1_RUDDER` | rudder-only | `[1,1e-6,1e-6]` | 安全/可比对照 |
| `P2_STERN` | stern-only | `[1e-6,1,1e-6]` | 纵向通道对照 |
| `P3_3D` | 3D all-channel | `[1,1,1]` | **最终目标** |

`P0_HISTORICAL` 只作跨协议参考，不参与 channel-scope 的因果对比。真正的
channel-scope contrast 是 `P1_RUDDER` vs `P2_STERN` vs `P3_3D`：三者使用同一套
独立 authority composition、同一校准规则、同一预算，只改变通道范围。

Phase 1 的 PPO profile、归一化、探索、clip/KL、critic、iteration 预算全部固定，
只允许 channel scope 变化。三组使用同一套 per-channel `ppo_authority` 校准规则；
不得给 3D 更大预算或更大探索。

### 4.2b Phase 1b：3D authority schedule（一个因子）

只有在 Phase 1 的 `P3_3D` 通过安全门后，才比较：

| arm | 做法 | 与 baseline 的差异 |
|---|---|---|
| `P3_3D_FIXED` | Phase 1 的固定 3D authority | 基线 |
| `P3_3D_SCHED` | 3D authority warmup/anneal | 只加 schedule |

如果 `P3_3D` 安全门失败，Phase 1b 不启动，写 `THREE_D_SAFETY_BLOCKED`；不要把
rudder/stern 的短期收益当成 3D 已解决。

### 4.3 Phase 2：normalization（一个因子）

| arm | 做法 | 说明 |
|---|---|---|
| `N0_CURRENT_FIXED` | 现有 `PCNormalization` 固定 scale | Phase 1 winner；基线 |
| `N1_FIXED_EMPIRICAL_BOTH` | `PCNormalization` + rsl_rl empirical，actor/critic 都开；`until=1e8`、`eps=1e-2` | 主处理组；最接近参考实现 |
| `N2_FIXED_EMPIRICAL_CRITIC` | `PCNormalization` + empirical 只开 critic | 需要先拆 actor/critic 开关 |
| `N3_EMPIRICAL_FREEZE` | `N1` + 在 curriculum 最终 stage 稳定后，再累计固定 transitions 冻结 running stats | 单独因子；只在 `N1` 后再测；不要在早期 path block 尚未激活时冻结 |
| `N4_ADV_SCOPE` | advantage 归一化范围：full rollout vs minibatch | 参考工程默认 full-rollout；不进入首批 matrix，只作未来 cheap ablation |

`N1/N2/N3/N4` 不得组合；return/value normalization 不在本 Phase，另列待定。
D3 的方向改为 **A+E**：保留 `PCNormalization` 作为物理前端，再叠加 rsl_rl 的
empirical running mean/std。顺序必须是：

```text
raw observation
  -> PCNormalization（固定物理 scale，保持原作用位置）
  -> EmpiricalNormalization（actor/critic 各自的 running mean/std）
  -> PCPolicy split / encoder
```

这样既保留 PC 尺度语义，又完整采用参考实现的自适应标准化；不是用 empirical
替换 PC。物理域 clip 默认不做；只保留 outlier/clip-rate telemetry，若 empirical
统计被极端值拖坏，再另立安全因子。

**推荐默认**：按当前方向，首轮主对照改成 `N0_CURRENT_FIXED` vs
`N1_FIXED_EMPIRICAL_BOTH`。`N2` 只在需要判断 actor/critic 谁受益时跑；`N3` 在
`N1` 显示 running stats 漂移明显时再跑；`N4` 独立于归一化，不混跑。

#### 4.3.1 参考实现核对（Wheel-Legged-Lab / IsaacLab 2.3.2 / rsl_rl）

以下结论来自只读代码核对，不改变本项目当前配置：

本项目实际使用的是 `rsl_rl 2.3.3` 接口：`scripts/train_batch_lowlevel_ppo.py` 从
`rsl_rl.runners` 导入 `OnPolicyRunner`，`BatchTorchPCVecEnv` 实现其 `VecEnv`
契约。2.3.3 的 `empirical_normalization` 是 **runner 级**开关：同时创建
`obs_normalizer`（actor policy observation）和 `privileged_obs_normalizer`
（critic privileged observation）；它作用在 VecEnv 返回的、已经过
`PCNormalization` 固定缩放的 observation 上，**不是 encoder 后的 `z_t`**。
参考工程使用的是较新的 per-model `obs_normalization` / `actor_obs_normalization`
接口。因此“服务器已经有 rsl_rl”不等于“参考配置可以直接照搬”。

| 参考实现 | 具体做法 | 可借鉴 | 不照搬 |
|---|---|---|---|
| 版本/接口差异 | 参考工程用较新的 per-model normalizer；本项目适配 `rsl_rl 2.3.3`，只有一个 `empirical_normalization` 布尔值，但同时控制 actor/critic 两个 normalizer | 先确认服务器实际 `rsl_rl_version`；若要 critic-only，改 runner/config 的字段拆分，不改 PPO 算法 | 不把参考工程的 YAML 字段名直接写进 2.3.3 配置 |
| actor/critic 独立归一化开关 | `RslRlMLPModelCfg.obs_normalization` 分别配置 actor 和 critic；旧接口也有 `actor_obs_normalization` / `critic_obs_normalization` | 我们的 `runner.empirical_normalization` 是单一全局开关；Phase 2 若要测试 critic-only，必须先拆成 `actor_*` / `critic_*` 两个字段 | 不把“全局开/关”当成 actor/critic 对照 |
| 物理域逐项 clip + scale | IsaacLab 先对原始观测 `clip`，再乘 `scale`（`observation_manager.py:356-407`） | 只把 clip 当可选安全措施；保留 outlier/clip-rate telemetry，默认不做 | 不把物理 clip 当成 normalization 的必要组成部分；也不对归一化后的 `z` 做全局裁剪 |
| normalizer 随模型保存 | `_mean/_var/_std/count` 是 buffer，进入 actor/critic `state_dict`，随 checkpoint 保存/恢复，并导出到 JIT/ONNX | 固定 scale 也要写入 checkpoint metadata，并带 version/hash；部署与训练必须使用同一份 scale | 不在推理时重新估计统计量 |
| 观测维度迁移 | `expand_rsl_checkpoint_for_jump.py` 对新增维度显式扩展 `_mean/_var/_std`，并拒绝静默改 shape | 任何 observation schema 变化都要显式迁移或拒绝加载；不能靠广播/截断“碰巧能跑” | 不把新增特征默认 mean=0、std=1 当成无需验证 |
| train/eval 冻结 | normalizer 只在 `training` 模式更新；runner 在推理前调用 `eval_mode()`；统计量随 checkpoint 保存 | `N1` 完整采用该机制；eval 必须冻结 actor/critic 两个 normalizer | Wheel 配置 `until=None` 会无限更新；本项目 2.3.3 runner 是 `until=1e8`，在 3M 预算内等价于全程更新，不能无脑照搬 |
| 分母稳定项 | `(x - mean) / (std + eps)`，参考实现 `eps=1e-2` | 固定 scale 增加下限校验（例如 `scale >= 1e-3`）或显式 denominator floor，防止小尺度特征放大噪声 | 不直接套用 `eps=1e-2`；我们的 scale 是物理量程，应单独审计 |
| advantage 归一化 | 参考工程未设置 `normalize_advantage_per_mini_batch`，IsaacLab/rsl_rl 默认 `False`，即在整段 rollout 上归一化；设为 `True` 才切到 minibatch | 参考工程和本项目当前配置都是 full-rollout；首批 matrix 保持 full-rollout | 不把 `N4` 写成“启用 advantage normalization”，也不把 minibatch 当成参考默认 |
| reward/return 归一化 | rsl_rl 只有 RND intrinsic reward 的 discounted-variation normalization，不是标准 PPO extrinsic return 归一化 | 暂无直接可借的 return normalization | 本轮不做 return/value normalization |
| 非对称观测组 | actor 用 policy group，critic 用 critic/privileged group；每个模型独立 normalizer | 与 C1 privileged critic 兼容；归一化可以按 actor/critic 分开治理 | 不让 actor 看到 critic 的 privileged 字段 |

本 Phase 的前置实现（在生成 matrix 前完成）：

- [ ] 把 `runner.empirical_normalization` 拆成 actor/critic 两个独立字段，或明确禁止
  critic-only 归一化 arm；2.3.3 内部已有两个 normalizer，拆的是 runner/config 的
  开关，不是重写 PPO；
- [ ] 保留 `PCNormalization` 原作用位置和固定 scale；只增加尺度下限校验，不默认加
  raw-domain clip；outlier/clip-rate 只作为 telemetry；
- [ ] 在 checkpoint metadata 中记录 normalization `version`、`source`、`hash` 和逐特征
  scale/clip；加载时严格比对 schema；
- [ ] 确认 `obs_norm_state_dict` / `privileged_obs_norm_state_dict` 随 checkpoint 保存和
  恢复，并记录 running mean/std/count 的漂移；
- [ ] 增加单元测试：train/eval 前后 scale 不变、checkpoint round-trip 一致、部署
  controller 与训练使用同一份 normalization；
- [ ] 增加 telemetry：每个 block 的 raw min/max/percentile、clip rate、归一化后
  min/max/std，用于决定 `N1`/`N2` 是否真的需要跑。

### 4.4 Phase 3：trust region（一个参考 profile）

本 Phase 把参考工程的 clip、KL 目标和 adaptive LR 作为一个相互耦合的
trust-region profile 整体替换。首轮不再拆 `T1_KL_ADAPT` / `T2_WIDE`：参考实现中
adaptive LR 的触发阈值就是 `desired_kl`，单独改其中一项会产生参考工程并不存在的
组合。除该 profile 外，authority、entropy、normalization 和 exploration 均保持不变。
其中 D14 已定所有新 arm 共用 `gamma=0.995`、`lam=0.95`，D15 已定共用
`entropy_coef=0.005`；这两项都是公共基线，不是本 Phase 的 treatment 因子。

| arm | clip | desired_kl | schedule | 初始 LR | 首轮 |
|---|---:|---:|---|---:|---|
| `T0_CURRENT_PROFILE` | `0.02` | `0.005` | fixed | 当前 actor/critic LR | 是，基线 |
| `T1_REFERENCE_TRUST_REGION` | `0.20` | `0.01` | adaptive | `3e-4`，actor/critic 共享 | 是，主处理组 |
| `T3_CLIP_ONLY` | `0.20` | `0.005` | fixed | 当前 actor/critic LR | 否，后续 Goal92 诊断 |

`T1` 完全沿用 rsl_rl 2.3.3 的内置 adaptive 规则；参考 cfg 只设置
`schedule="adaptive"` 与 `desired_kl=0.01`，没有另行覆盖该规则：

```text
每个 minibatch 更新后：
  KL > 2 * desired_kl        -> lr = max(1e-5, lr / 1.5)
  0 < KL < desired_kl / 2    -> lr = min(1e-2, lr * 1.5)
  其他                        -> lr 不变
```

来源：`Wheel-Legged-Lab/.../agents/rsl_rl_ppo_cfg.py:37-49` 给出参考 profile；
本项目 `rsl_rl` 2.3.3 的 `algorithms/ppo.py:271-305` 给出上述 adaptive 规则。

参考实现使用单一 optimizer 和单一 LR，因此 `T1` 的 actor/critic 必须使用相同的初始
LR `3e-4`。首轮不允许 separate actor/critic LR、DADE-head 独立 LR 或 critic warmup；
这些能力当前配置也要求 `schedule="fixed"`。若将来要拆 actor/critic 自适应速率，必须
另立实现因子，不能混进 `T1`。

所有 arm 固定 canonical profile `PPO-FLAT-4096x48-E5-MB4`，即 `5 epochs / 4
minibatches / 49,152 minibatch`。`epochs` 和 `num_mini_batches` 是 profile 的一部分，
不是本 Phase 的 trust-region 因子。若将来要测 `E1` 或 `MB64`，必须另立
profile/batch-organization 因子，不能和 clip/KL 混跑；Goal92 的 1-epoch 历史配置
只作参考，不作为新 arm。

`T3` 是 Goal92 的对照，不是推荐配置，也不进入首轮 matrix；它只用来验证“只放宽
clip 是否仍然无效”。若首轮 `T1` 失败，不自动回退到 `T3`，而是保留 `T0` 并记录
`TRUST_REGION_NO_GAIN`。

### 4.5 Phase 4：exploration（一个因子）

**D5 已定：仿照参考工程**。首轮直接采用参考配置的探索值：

```text
init_std = 0.3
noise_std_type = scalar
clip_actions = 1.0
```

首轮不加 min/max std，不做 anneal；`E2_BOUNDED` 和 `E3_ANNEAL` 只作为后续独立
因子。参考的 `0.3` 作用在直接 normalized action 上，本项目作用在 residual 上，
所以必须同时记录每通道实际作用到执行器上的有效 std，确认 `0.3` 在 residual 空间
放大到多少；但首轮不据此静默调参，除非触发 safety/clip 停止条件。

有效探索量的记录口径为 per-channel applied normalized delta：

```text
applied_ppo_delta_norm_mean_abs_c
applied_ppo_delta_norm_std_c
applied_ppo_delta_norm_p95_c
applied_ppo_snr_c = mean_abs / std
```

其中 `applied_ppo_delta_norm_c = ppo_delta_c × action_normalization_jacobian_c`，
`action_normalization_jacobian=[1/rudder_max, 1/stern_max, 2/rpm_max]`。rpm 的
wrapper 映射是 `2*rpm/rpm_max-1`，不能和舵角共用 `1/full_scale`。这是 telemetry
修正，不改变 D5 的参考值。

| arm | 做法 | 说明 |
|---|---|---|
| `E0_CURRENT` | 当前 `init_std=0.1` | 基线 |
| `E1_REFERENCE_STD` | 参考工程 `init_std=0.3`、scalar std、`clip_actions=1.0` | 主处理组；与 E0 只差探索幅度 |
| `E2_BOUNDED` | `E1` + min/max std | 后续因子，首轮不跑 |
| `E3_ANNEAL` | `E2` + 达标后退火 | 后续因子，首轮不跑 |

`E1_REFERENCE_STD` 是只改变探索幅度的一个因子；不得同时改 authority、entropy、
clip/KL 或 observation normalization。若 `0.3` 造成明显 safety 或 clip 回归，结果按
`EXPLORATION_UNSTABLE` 记录，不回头改 D5 的参考值。
D15 已定 `E0_CURRENT` 与 `E1_REFERENCE_STD` 都使用 `entropy_coef=0.005`；本 Phase
只改变 `init_std`。

### 4.6 Phase 5：critic information（一个因子）

本 Phase 按用户已确认的“一步到位”实现，不拆 C1/C2/C3 三批训练臂。因子定义是
“actor 非特权 path block + critic 特权 path block”的联合 observation partition；
字段组的 mask 只用于实现审计和失败诊断，不进入首批 matrix。

| arm | actor path block | critic 输入 |
|---|---|---|
| `C0_CURRENT_PARTITION` | 当前 8-D path block；仅作历史/回退对照 | 当前 15-D ideal state + true current history |
| `C1_DUAL_PATH_PRIVILEGED` | 5-D deployable block：cross、vertical、tangent heading/pitch、lookahead heading | 现有 ideal state/true current + 独立 privileged path encoder + actuator/disturbance/sensor-safety 特权 |

actor path block 固定为：

```text
signed_cross_track_m
signed_vertical_error_m
tangent_heading_error_rad
tangent_pitch_error_rad
lookahead_heading_delta_rad
```

actor 不得读取 `progress_ratio`、`remaining_ratio`、`curvature_1_per_m` 或
`endpoint_context`。`remaining_ratio` 与 progress 等价，不能因只删 progress 而保留。

critic privileged path block 使用真实 `eta` 和真实路径投影，至少包含上述 5 维真值、
`curvature_1_per_m` 和 path lookahead；不包含 progress、remaining、goal distance 或
endpoint context。其余特权按参考工程映射为四组：

- route truth：真实 path projection、切向、曲率和 lookahead；
- actuator：上一时刻实际 stern/rudder/rpm、上一 residual、变化率、饱和/限幅；
- disturbance：真实 current、current innovation、current 估计误差；
- sensor/safety：USBL/DVL 有效性/age、导航创新、深度/横滚/边界裕度。

privileged path 必须进入独立 encoder，不能直接拼进现有 15-D critic state。所有字段
必须在 matrix/metadata 中逐项列出并带 schema hash；禁止未来帧、held-out 或评价专用
标签。首轮不比较字段组收益，只在 Gate 失败时用 mask 做诊断。

### 4.7 Phase 6：training organization（一个因子）

D9 已确认“模仿参考代码”的 min/max/window/consecutive；CU7 进一步确认首波不实现
C6 early stop，正式 run 先跑满 2600，第二波再启用参考式早停。

| arm | 做法 | 说明 |
|---|---|---|
| `G0_FIXED_LEGACY` | 固定短 iteration | 只作历史诊断，不进入首批 matrix |
| `G1_REFERENCE_EARLY_STOP` | `min_iterations=600`、`max_iterations=2600`、`window=20`、`consecutive=3` | 第二波协议；首波 C6 明确关闭 early stop，跑满上限 |

首波中，20/3 只用于 routine curriculum promotion；C6 没有 early-stop counter，按
hard cap 跑满。第二波启用 C6 early stop 后，20/3 才同时服务 routine promotion 和
C6 early stop，且两套 counter 必须独立。C0–C6 每个 stage 用训练 rollout 的
20-iteration 滑动平均，连续 3 次满足该 stage 门槛才晋级；每个 stage 保留自己的
min/max iteration。frozen-dev 只在 stage 边界和最终 formal 使用，不参与每 20
iterations 的 routine 晋级。Phase 6 只确认早停状态、hard cap、课程 provenance 和
边界评价可审计，不新增 `G2_STAGED_GATE` 训练臂。

### 4.8 Phase 7：integration confirmation

| arm | 做法 | 用途 |
|---|---|---|
| `I0_OLD_STACK` | Goal173 历史 PPO 栈 | 历史对照 |
| `I1_BEST_STACK` | Phase 1–5 的 winner 组合 + D9 早停协议 | 主候选 |
| `I2_BEST_SAFE` | `I1` + 更保守的 channel/authority | 安全候选 |

Phase 7 不做新因子探索，只做组合确认。每个 arm 至少 3 个训练 seed，跑完整评价矩阵。

### 4.9 Phase 8：D13 reward 修改目标（已确认）

用户已于 2026-11-15 确认下表作为 Goal182 后续 reward 重构的主要修改目标。它冻结的是
**奖励目标结构**和**首轮权重起点**，不等于 StageGate 已定稿，也不授权跳过只读 scale
audit 直接改配置或启动训练。纯端到端和 ALOS+本体控制必须分别使用各自一列，不能把两套
策略的 heading/pitch/speed 目标混在一起。

| 奖励项 | 纯端到端 `path_*`（learned speed 仅次要子实验） | ALOS + 本体 `alos` | 说明 |
|---|---:|---:|---|
| `w_prog` | 0 | 0 | progress 奖励完全删除 |
| `w_3d` | 0 | 0 | 不再使用 3D 合成误差奖励 |
| `w_path_3d` | 0 | 0 | ALOS 旧路径 3D 项也关闭；仅保留 3D 评价指标 |
| `w_path_cross` | 3.5 | 3.5 | 横向路线误差；本轮加强 |
| `w_path_vertical` | 1.5 | 1.5 | 纵向路线误差 |
| `w_path_progress` | 0 | 0 | 删除路线 progress 奖励 |
| `route_middle_gate` | 0 | 0 | 中间段不特殊加权 |
| `w_yaw` | 1.0 | 0 | 纯端到端使用路线切向 heading；指数核 `std=0.5 rad` |
| `w_cmd_yaw` | 0 | 1.0 | ALOS 使用指令 heading；指数核 `std=0.5 rad` |
| `w_path_pitch` | 1.0（需新增字段） | 0 | 纯端到端使用路线切向 pitch；指数核 `std=0.5 rad` |
| `w_cmd_pitch` | 0 | 1.0 | ALOS 使用指令 pitch；线性惩罚，R6-A 固定 |
| `w_cmd_speed` | 2.5 | 2.5 | 速度跟踪；指数核 `std=0.25 m/s`；目标使用上下限 clamp 后的 `reference_ground_velocity_ned` |
| `w_route_speed_pc` | 0 | 0 | legacy 字段；首轮移除 reward 侧速度地板，Goal182 中非零即配置错误 |
| `w_route_ground_speed_stall` | 0 | 0 | 低速停滞接口必须实现，但首轮权重严格为 0；未来启用建议从 5.0 开始 |
| `route_ground_speed_stall_threshold_mps` | 0.25 | 0.25 | `v_route >= 0.25 m/s` 时不罚 |
| `route_ground_speed_stall_gate_low_mps` | 0.2 | 0.2 | `v_ref <= 0.2 m/s` 时 gate=0 |
| `route_ground_speed_stall_gate_high_mps` | 0.4 | 0.4 | `v_ref >= 0.4 m/s` 时 gate=1 |
| `w_u` | 0 | 0 | legacy 字段；Goal182 必须为 0，速度奖励统一使用 `w_cmd_speed` |
| `w_action_pc` | 0.1 | 0.1 | 一阶动作平滑 |
| `w_action_smooth_pc` | 0.005 | 0.005 | 二阶动作平滑；只罚舵和艉翼，不罚 RPM |
| `w_action_clip_attempt_pc`（legacy `w_sat_pc`） | 0 | 0 | 裁剪前动作尝试项；Q3-A 首轮关闭 |
| `w_actuator_material_sat_pc`（legacy `w_material_sat_pc`） | 0.02 | 0.02 | 实际执行器材料饱和事件；首轮唯一保留的饱和奖励 |
| `w_path_speed_smooth` | 0.025 | 0.025 | 速度突变惩罚；Q8-B 使用 `-w*((Δv)/0.25)²`，与旧 `0.1*(Δv²/0.25)` 等强度 |
| `w_path_energy` | 0.01 | 0.01 | 能耗项 |
| `w_phi` | 3.0 | 3.0 | 横滚角安全项；本轮加强 |
| `w_p` | 0.02 | 0.02 | 横滚角速率安全项；本轮保持 |
| `goal_bonus` | 0 | 0 | 到达奖励失效 |
| `goal_plane_miss_penalty` | 0 | 0 | 终点平面 miss 仅终止/诊断，不再给 reward |
| `goal_plane_miss_distance_weight` | 0 | 0 | 超出 goal radius 的距离不再进入 reward |
| `terminal_progress_weight` | 0 | 0 | 终局 progress 奖励失效 |
| `timeout_penalty` | 0 | 0 | 不作为到达激励 |
| `w_approach_vertical` / `w_approach_pitch` | 0 | 0 | 终点接近项失效 |
| `w_vertical_potential` / `w_heave_potential` | 0 | 0 | 首轮不做 |
| `pc_ppo_scale` | 暂保留 0.01 | 暂保留 0.01 | 单独作为后续尺度因子，不与 reward 形状混改 |

纯端到端的首轮主臂 `path_fixed_speed` 同样使用 `w_cmd_speed=2.5`；旧 `w_u` 在
Goal182 中固定为 0，不能再用它替代速度奖励。`path_learned_speed` 只是很次要的
速度参考变体，区别不在 reward：它允许速度参考在上下限内变化，而 `path_fixed_speed`
保持固定 episode speed。两者都必须在同一 `[v_min, vehicle.max_speed_mps]` clamp
之后才生成 command 和 reward reference。

跟踪核函数已固定为：

```text
纯端到端 heading:
  -w_yaw * (1 - exp(-(wrap(tangent_yaw - yaw) / path_heading_std_rad)^2))
  w_yaw=1.0, path_heading_std_rad=0.5

ALOS heading:
  -w_cmd_yaw * (1 - exp(-(heading_error / cmd_heading_std_rad)^2))
  w_cmd_yaw=1.0, cmd_heading_std_rad=0.5

速度跟踪:
  -w_cmd_speed * (1 - exp(-((reference_speed - ground_surge) / cmd_speed_std_mps)^2))
  w_cmd_speed=2.5, cmd_speed_std_mps=0.25
  reference_speed = world_to_body(reference_ground_velocity_ned, attitude)[:, 0]
  ground_surge = ground_velocity_body_mps[:, 0]
  reference_ground_velocity_ned 必须由物理上下限 clamp 后的 v_ref 构造，不用原始命令

路径模式速度上下限（已确认）:
  v_cmd_episode = _episode_command_speed_mps
  v_min = vehicle.min_speed_mps
  v_ref = clamp(v_cmd_episode, v_min, vehicle.max_speed_mps)

  path_fixed_speed:
    command.speed_mps = v_ref
    reference_ground_velocity_ned = v_ref * path_tangent_ned
    速度奖励使用 w_cmd_speed

  path_learned_speed:
    v_cmd_episode 可在上下限内变化；仍先经过同一 vehicle.min/max_speed_mps clamp
    command.speed_mps = v_ref
    reference_ground_velocity_ned = v_ref * path_tangent_ned
    速度奖励使用 w_cmd_speed

ALOS 速度上下限（已确认）:
  v_min = vehicle.min_speed_mps
  v_max = vehicle.max_speed_mps
  command.speed_mps = clamp(solution.command_speed_mps, v_min, v_max)
  terminal_ground_speed = max(terminal_ground_speed, v_min)
  guidance.minimum_speed_ratio 在 Goal182 中不再参与任何速度限制

低速停滞占位（首轮 w=0）:
  v_ref = ||reference_ground_velocity_ned||
  v_route = max(0, dot(ground_velocity_ned, path_tangent_ned))

  t = clamp(
      (v_ref - route_ground_speed_stall_gate_low_mps)
      / (route_ground_speed_stall_gate_high_mps
         - route_ground_speed_stall_gate_low_mps),
      0, 1)
  gate = t^2 * (3 - 2*t)

  stall_ratio = clamp(
      (route_ground_speed_stall_threshold_mps - v_route)
      / route_ground_speed_stall_threshold_mps,
      0, 1)

  -w_route_ground_speed_stall * gate * stall_ratio^2
  w_route_ground_speed_stall=0
```

以上均为零误差为 0 的惩罚形式，与参考工程正向 `exp(-e²/std²)` 对误差的梯度同形，
只差常数偏移。`speed` 只指速度跟踪 `w_cmd_speed`。Q1 已确认首轮移除奖励侧速度地板：
`w_route_speed_pc=0`，guidance 的保底加速仍作为控制器先验保留。stall 只保留接口、
breakdown 和 telemetry；其权重为 0，首轮不产生任何 reward 贡献，也不进入扫参。

2026-11-16 进一步冻结：`alos`、`path_fixed_speed`、`path_learned_speed` 三个模式
统一只使用 REMUS100 的物理速度包络 `[vehicle.min_speed_mps, vehicle.max_speed_mps]`。
ALOS 不再用 `guidance.minimum_speed_ratio * cruise_speed` 抬高下限，终点制动后的地速
也不能低于 `vehicle.min_speed_mps`。`guidance.minimum_speed_ratio` 只保留 legacy
解析，Goal182 config 中即使存在也不得改变速度；实现时必须移除
`terminal_speed_ratio >= minimum_speed_ratio` 的旧约束，否则 legacy 字段仍会间接
影响可配置性。ALOS 的 `_route_speed_floor()` 是独立的保底加速逻辑，不属于最低速度
包络，Q1-E 的决定不变。

本轮同时冻结以下 Goal182 canonical 终止/评价边界：

- goal radius 只使用 default `2.0 m`；`strict125 1.25 m` 完全退出 Goal182 active
  protocol、matrix 和评价，不再作为待选半径。历史 Goal 文档中的 `strict125` 仍是历史
  记录，不回写；
- `termination.max_depth_m=510.0`，触发条件是真实 `z > 510 m`；`min_depth_m=0.0`
  保持不变；
- `plant.bounds_m` 的 z 上界设为 `520 m`，确保 `z>510 m` 时 `DEPTH_LIMIT` 优先于
  `OUT_OF_BOUNDS` 被记录；x/y bounds 保持 `[-200, 200]`。若代码或 config 仍使用 z
  上界 `300 m`，该终止协议视为未实现。

实现和治理规则：

- **“删除”一律解释为权重置 0、接口保留**：不删除 reward 字段、breakdown、telemetry 或
  配置解析；这样既能关闭旧目标，也能保留审计和回滚能力。
- Q3-A 的 canonical 命名是 `w_action_clip_attempt_pc` 和
  `w_actuator_material_sat_pc`。前者表示“策略在裁剪前试图越界的连续量”，首轮固定为
  0；后者表示“实际发生了执行器材料裁剪”的二元事件，首轮为 0.02。旧名 `w_sat_pc` /
  `w_material_sat_pc` 只作兼容别名，parser 应接受但映射到 canonical 字段并给出迁移
  warning；新 matrix、breakdown 和 telemetry 只能写新名。
- 由于首轮 `w_action_clip_attempt_pc=0`，不需要为了恢复“尝试项”而保留当前会把
  `applied.normalized` 当输入的死代码路径。若未来启用该权重，必须从动作管线取得
  **任何 clip 之前的 raw action**，并同时记录 `action_clip_attempt_excess`；否则该权重
  视为配置错误。实际材料饱和仍由 `w_actuator_material_sat_pc` 惩罚。
- Q4-C 新增 `w_action_smooth_pc=0.005`，公式为
  `-w * Σ_{c∈rudder,stern}(a_norm_t - 2a_norm_{t-1} + a_norm_{t-2})²`。它只作用于
  舵和艉翼，RPM 不参与；`a_norm` 与 `w_action_pc` 使用同一执行器来源。实现必须保存
  至少两步历史执行器并在 episode reset 时清零，同时记录分通道二阶差分和加权贡献。
- **控制模式名词**：`alos` 表示 heading/pitch/speed 由 ALOS 求解器生成；
  `path_fixed_speed` 和 `path_learned_speed` 都跳过 ALOS，heading/pitch 直接取路线
  切向。三者都使用 `w_cmd_speed=2.5` 的显式速度奖励；`w_u` 只是 Goal182 中必须
  置 0 的 legacy 字段。纯端到端首轮主臂使用 `path_fixed_speed`；
  `path_learned_speed` 只允许速度参考在物理上下限内变化，是次要速度参考实验。
- **速度上下限（已确认）**：`alos`、`path_fixed_speed`、`path_learned_speed` 都只
  使用机器人物理包络：`v_min=vehicle.min_speed_mps`、
  `v_max=vehicle.max_speed_mps`。path 模式令
  `v_ref=clamp(v_cmd_episode, v_min, v_max)`；ALOS 令
  `command.speed_mps=clamp(solution.command_speed_mps, v_min, v_max)`，终点地速也
  不得低于 `v_min`。`guidance.minimum_speed_ratio` 在 Goal182 中不再生效。`_path_command()`
  返回的 `speed_mps`、`reference_ground_velocity_ned` 和 reward 的参考速度都只能用
  `v_ref`，不能用未 clamp 的 `_episode_command_speed_mps`。不新增独立扫参速度地板，
  也不重新启用 `w_route_speed_pc` 或 stall。telemetry 至少记录
  `episode_command_speed_mps`、`path_min_speed_mps`、`path_reference_speed_mps` 和
  `path_speed_clamp_delta_mps`。
- **速度数值必须来自变量**：运行时不得写死 `1.0` / `2.5`；所有速度包络必须读取
  `self.vehicle.min_speed_mps` 和 `self.vehicle.max_speed_mps`。测试也应优先引用同一
  变量，只有验证外部契约时才允许带理由的字面量。该规则纳入
  `总协调/doc/governance/MAGIC_NUMBER_PROVENANCE.md` 的 magic-number provenance
  checker；Goal182 的速度包络不得使用 inline exemption 绕过。错误码：
  `MAGIC_NUMBER_WITHOUT_PROVENANCE`。
- **`path_learned_speed` 优先级**：它是纯端到端的**次要速度参考实验**，不占用首轮
  主矩阵预算；只有主 E2E/ALOS 臂完成并通过协议检查后，才允许追加。它必须保留
  `[v_min, vehicle.max_speed_mps]` 上下限 clamp，不能因“learned”而让速度命令落到
  机器人不可执行区间。它与 `path_fixed_speed` 的区别只是速度参考是否可变，reward
  都使用 `w_cmd_speed=2.5`。
- **速度 reward 唯一入口**：Goal182 的 `alos`、`path_fixed_speed`、
  `path_learned_speed` 都使用 `w_cmd_speed`；`w_u` 是 legacy 字段，任何新 config
  非零都应报配置错误。若 parser 同时发现 `w_u` 与 `w_cmd_speed` 非零，使用
  `DUPLICATE_SPEED_TRACKING` 拒绝，避免同一速度误差被两套项重复计权。
- **Q5-A 模式不变量**：`path_*` 模式必须 `w_cmd_yaw=w_cmd_pitch=0`；`alos` 模式必须
  `w_yaw=w_path_pitch=0`。parser 和 matrix checker 以
  `DUPLICATE_PATH_COMMAND_TRACKING` 拒绝混用；speed 权重不受这条规则限制。历史
  config 扫描目前没有发现已存在的冲突组合，但必须用可执行检查防止新矩阵引入。
- **ALOS terminal alignment（2026-11-16 已确认）**：robust terminal 的
  `terminal_alignment_min_m` 和 `terminal_alignment_max_m` 都设为 `5.0 m`。现有
  35–50 m 路线因此只在最后 5 m 进入 terminal phase，不再像旧配置 `50 m` 那样从
  episode 起点就覆盖路径切向制导。`terminal_capture_distance_m` 是非 robust 分支的
  legacy 字段，本条不隐式改动；若后续统一 terminal 阈值，必须单独确认。
- **ALOS 在线路线契约（2026-11-16 已确认）**：训练阶段由 D8 维护一条持续增长的
  path，按同 family 在剩余 `<=5 m` 时追加；terminal 只依据真实 mission goal，也即
  最终 path endpoint。部署阶段可以使用 rolling path buffer，但不得把 buffer 末端
  自动当作 mission goal；terminal 必须由独立标志或真实任务终点决定。rolling buffer
  的前方长度至少覆盖 `max_lookahead_m` 加 terminal margin，避免 `point_at_s()` 在
  buffer 末端被截断。追加同一条 path 时保留 `_projection_hint`；若未来路径被替换、
  重规划或切向不连续，必须重新全路径投影并重置 hint。评价仍使用固定路线，不续路。
- **路线来源（2026-11-16 已确认，同日晚些时候修订垂向与速度课程）**：C0–C6 的水平
  path family、长度和起始深度沿用现有；C2 改为 50% 纯垂向 `1.75 m / 1 cycle`
  加 50% 纯水平 `1.75 m / 1 cycle`，C3/C4 加 `2.0 m / 2 cycles` 垂向，C5 加
  `3.0 m / 1.5 cycles`，C6 加 `5.0 m / 1 cycle`；详见 §4.12.7.1。训练速度改用
  `SPEED_V3_CENTER_OUT_5S`：从 `1.5 m/s` 附近交替向两侧扩档，并在 episode 内每
  `5 s` 重采样；详见 §4.12.7.2。评价使用固定路线和固定 route speed，不续路，也不把
  评价路线末端扩展成新的 mission goal。训练续路按 CU3 B 第一波关闭，D8 的 append
  只作为第二波独立因子。
- `goal_plane_miss_penalty` 和 `goal_plane_miss_distance_weight` 首轮都为 0；
  `pc_goal_plane_miss_termination_enabled` 保持现状，终止事件继续计数，但不产生 reward。
- **纯端到端主目标**是路线空间的 cross、vertical、切向 heading、切向 pitch，外加
  speed tracking、动作平滑和饱和保护；速度地板关闭，stall 仅留权重为 0 的接口；
  不读取 progress、终点距离或到达奖励。
- **ALOS+本体主目标**是指令空间的 heading、pitch、speed；route cross/vertical 只作
  路线 guard，不与指令跟踪争抢同一权重。
- **path health 只保留 cross + vertical**；progress 和 middle gate 都关闭。评价中的
  completion/progress 仍可作防停滞 guard，但不得回流成训练奖励。
- **3D 只保留为评价/StageGate 候选指标**，不再作为奖励合成项；`w_3d` 和 ALOS 的
  `w_path_3d` 必须同时为 0，避免只关掉一个入口。
- `w_path_pitch` 当前尚未实现，必须新增 `RewardConfig` 字段、配置解析、breakdown 和
  telemetry；在字段落地前，纯端到端的 pitch reward 不能算完成。
- heading/speed 的指数核必须新增 `path_heading_std_rad`、`cmd_heading_std_rad` 和
  `cmd_speed_std_mps` 的配置解析、breakdown 与 telemetry；不能把 std 写死在 reward
  函数里。ALOS heading 使用指令 heading 误差，纯端到端 heading 使用路线切向误差，两者
  不得混用。
- stall 接口必须新增 `w_route_ground_speed_stall`、阈值和两个 gate 边界字段，并在
  breakdown/telemetry 中记录 `reference_ground_speed_mps`、`route_ground_speed_mps`、
  `stall_gate`、`stall_ratio` 和 raw/加权贡献。首轮 `w_route_ground_speed_stall=0`；
  `w_route_speed_pc` 仅作 legacy 解析字段，canonical Goal182 config 非零时启动器报错。
- `w_cmd_speed` 的实际速度输入固定为 `ground_velocity_body_mps[:,0]`，不得使用
  `water_velocity_body_mps[:,0]`。`privileged_surge` 仍可能服务 hold/critic，实现时
  应给 reward 传入独立的 `reward_ground_surge`，避免把 reward 的地速语义扩散到其他
  消费者；若要让 hold/critic 也统一为地速，必须另行记录并确认。
- 表中标“审计起点”的权重先按该值进入实现；只读 scale audit 只用于核对现行 reward
  聚合、`pc_ppo_scale` 和各项量纲。若审计发现必须改值，必须写回本表并再次得到用户确认，
  不得在训练脚本里静默改权重。
- `pc_ppo_scale=0.01` 暂不随本表改动，作为独立尺度因子另立对照；在 StageGate 和
  scale audit 完成前，不比较 `0.01 / 0.1 / 1.0`。
- 本表是 reward 冻结表；termination 和 goal radius 的例外以上一段的 canonical
  `2.0 m / 510 m / 520 m` 边界为准。safety terminal penalty 和 success 派生规则不变。

#### 4.9.1 StageGate v1：timeout 只诊断，不 gate

用户已确认：D8 的 training-only 续路会让训练 episode 的路线长度变化，timeout 与路线
长度耦合，不能作为课程晋级或 early stop 的通过条件。StageGate v1 固定为：

| 层级 | 指标 | 处理 |
|---|---|---|
| 硬否决 | `safety_rate` | 超过当前 stage `safety_max` 即不晋级 |
| 防停滞 | `path_completion_ratio` | 低于 `completion_min[stage]` 即不晋级 |
| 主 route | `path_horizontal_rmse_m`、`path_vertical_rmse_m` | 分别低于 `cross_rmse_max[stage]`、`vertical_rmse_max[stage]` |
| 仅诊断 | `timeout_rate`、`plane_miss_rate`、`success_rate`、`path_three_d_rmse_m`、`pre_goal_3d_rmse_m`、overshoot、terminal longitudinal | 写入 summary/decision log，但不参与 pass/fail |

补充规则：

- `timeout_rate` 仍保留在 `summary.json` 和逐 episode 记录中，用于区分“卡住”“慢但跟踪好”
  等情形；但 StageGate 和 C6 early-stop 都不得因 timeout 直接失败。
- 本规则只约束 curriculum StageGate 和 C6 early-stop；Phase 实验的 treatment Gate
  仍可把 timeout 作为回归诊断，但不得用 timeout 单项代替 route 改善判定。
- `plane_miss` 和 `success` 默认降级为诊断；首轮不设低 success floor。若 P0 baseline
  显示 completion guard 不足以排除病态策略，再单独提出低 success floor 并重新确认。
- StageGate v1 使用 RMSE；cross/vertical p90 先作为报告指标，不进入首轮 gate。若要
  p90 进 gate，必须先扩展 evaluator 的逐 episode p90 字段并重新冻结协议。
- `path_three_d_rmse_m` 只作为综合报告，不单独决定晋级，避免 cross/vertical 一项很差
  但被另一项平均掉。
- 课程 routine 晋级使用训练 rollout 的 20/3 规则；frozen-dev 只在 stage 边界和最终
  formal 评价运行，不承担每 20 iterations 的晋级判断。
- routine 晋级使用的 `completion_min`、cross/vertical 上限，由 P0
  baseline 的训练 rollout 20-iteration 分布标定；边界/final 的 frozen-dev 阈值另表
  标定。两张表都写入并冻结后使用，不得混用，也不得用 treatment 结果回调阈值。
- 首轮先运行 route-gate **shadow mode**：计算 `route_gate_would_pass` 并写日志，但不
  阻止晋级；阈值冻结后再切换为 enforce。

#### 4.9.2 阈值标定与 shadow（已确认）

用户已确认三项：

1. **纯端到端与 ALOS 分开标定绝对阈值**，但使用同一套相对规则；不能拿 ALOS 的绝对
   误差去卡纯端到端。
2. **首轮只做总体 route gate**；per-stratum 指标先作为 worst-stratum 诊断报告，不
   直接做硬门控。若 P0 显示某个 stratum 被总体平均掩盖，再单独升级。
3. **不设到达/success floor**；`path_completion_ratio` + `safety` 是防病态的唯一硬
   条件，到达终点不参与晋级。

标定流程：

- 用冻结的 BC/ALOS reference，在与训练同控制模式下逐 C0–C6 stage 评估固定评价路线，
  不做 training-only 续路；收集 `path_horizontal_rmse_m`、`path_vertical_rmse_m`、
  `pre_goal_3d_rmse_m`、`path_completion_ratio` 的 mean/std。该表只服务 stage 边界和
  最终 formal；routine 晋级另用 P0 训练 rollout 的 20-iteration 统计标定。
- 初始阈值公式固定为：

```text
route_threshold = max(reference_mean * (1 + margin), reference_mean + 2 * reference_std)
```

  cross/vertical 的 `margin` 初始建议 `0.10`；pre-goal 3D 不设 margin，只作
  telemetry；
  `completion_min` 初始建议 `min(0.90, reference_completion - 0.05)`，且不低于
  `0.80`。这些系数在 P0 baseline 后冻结，不得根据 treatment 结果改。
- 纯端到端和 ALOS 各自生成 routine 训练 rollout 表与边界 frozen-dev 表，分别写入
  matrix/metadata；四张表的公式相同、绝对值和统计来源不同，不能交叉套用。
- 首轮进入 route-gate **shadow mode**：计算并记录 `route_gate_would_pass`、各分量值、
  threshold 和失败原因，但继续使用当前 gate 决定晋级。
- shadow 至少覆盖一个完整 stage 边界和一次最终 formal 评价；若出现“全部通过”或
  “全部不通过”，只允许回到 reference 标定和 margin 检查，不能按 treatment 结果调阈值。
- 阈值表冻结并写入本文件后，才切换为 route-gate enforce；切换前不生成
  `goal182_core_pe2e_matrix.json` / `goal182_core_alosb_matrix.json` /
  `goal182_eval_matrix.json`，不启动 formal。

仍待定：

1. 标定后的具体阈值和最终 margin；
2. shadow 切换的确切 iteration/transition 点；
3. 是否在后续阶段增加 per-stratum 硬门控或低 success floor；首轮均不启用。

### 4.10 ALOS guidance pre-screen（先于 PPO 修改执行，eval-only）

2026-11-16 用户决定：在继续 PPO/reward 实现和训练前，先做一轮 ALOS-only 快速预实验。
本轮只评价冻结的 guidance profile，不训练、不改运行时代码、不生成 formal matrix，
目的是先确认“缩短 lookahead + 提高纵向 gain”是否能让 ALOS 本身的路线跟踪变好。

**共同冻结条件**

- 使用 `scripts/evaluate_pc_ppo.py --action-mode alos`；该模式绕过 actor，向 ALOS +
  PD/PI 发送全零 residual，只评价 ALOS 与本体控制。
- 必须带 `--disable-curriculum-direct-command`，防止 C0/C1 回退到随机 direct command。
- `--guidance-config` 使用完整 guidance profile；profile 整块替换环境 config 的
  `guidance`，不直接改默认 config。除下表的四个字段外，其余 guidance 字段保持基线。
- `terminal_alignment_min_m = terminal_alignment_max_m = 5.0`；不改
  `terminal_capture_distance_m` 的 legacy 语义。
- 同一 arm 使用相同评价 seeds、相同固定路线、相同 current 和相同 curriculum stage；
  只改变 guidance profile。训练 seeds 不参与本轮。
- checkpoint 只用于 evaluator 的 metadata 校验和模型壳，不产生动作；仍必须选择与
  `training-config`、`environment-config`、`pc_stage` 和 observation schema 匹配的
  checkpoint。优先使用 C3/C4 匹配 checkpoint；若只有 C0 smoke checkpoint，只能做
  机制验证，不能外推到复杂路线。
- `--trace-output` 只对单个 seed 开启；其余 seed 不写大 trace。

**预实验 arm（六组，快速筛选，不是 formal）**

| arm | horizontal lookahead | vertical lookahead | horizontal gain | vertical gain | 作用 |
|---|---:|---:|---:|---:|---|
| `A0_BASE` | `3.0 m` | `2.0 m` | `2.5` | `1.0` | 旧 ALOS 基线 |
| `A1_REQUESTED` | `1.5 m` | `1.5 m` | `2.5` | `2.5` | 用户指定组合 |
| `A2_LOOKAHEAD_ONLY` | `1.5 m` | `1.5 m` | `2.5` | `1.0` | 隔离 lookahead 变化 |
| `A3_VERTICAL_GAIN_ONLY` | `3.0 m` | `2.0 m` | `2.5` | `2.5` | 隔离纵向 gain 变化 |
| `A4_LOW_LOOKAHEAD` | `1.0 m` | `1.0 m` | `2.5` | `2.5` | 更短 lookahead 的方向边界 |
| `A5_HIGH_LOOKAHEAD` | `2.0 m` | `2.0 m` | `2.5` | `2.5` | 中间 lookahead 的方向边界 |

六个 profile 已生成到 `configs/lowlevel/goal182/guidance_alos_c3_*.json` 和
`configs/lowlevel/goal182/guidance_alos_c4_*.json`，并已用 `load_config()` 校验 guidance
字段；尚未启动任何评估。`A4/A5` 只用于判断 lookahead 的方向边界，不替代
`A1 - A0`、`A2 - A0`、`A3 - A0` 的因果对照。

**今晚扩展执行口径（2026-11-16，seed 上限 4）**

- **不扩 seed**：所有 arm 固定 `seed_start=20`、`seed_count=4`（seeds `20–23`）。需要
  “多做一点”时增加 arm/stage 覆盖，不增加单臂 seed 数。
- C3：`A0–A5` 六组全跑，每组 4 seeds，共 24 个 seed-episode。
- C4：`A0–A5` 六组全跑，每组 4 seeds，共 24 个 seed-episode。
- 今晚优先完成 C3+C4 共 12 个 arm、48 个 seed-episode；如果时间不够，先删 C4 的
  `A4/A5`，不删 C3，也不把剩余时间用来加 seed。
- 不为了追数量拆掉 arm/seed 配对；C3 与 C4 必须分开汇总，不能混成 pooled mean。
- 每个 arm 一次 evaluation 调用、一个独立 output 目录；一个进程负责该 arm 的全部
  seeds，不用每个 seed 单独起进程。最多四个并行 worker，分别绑定空闲 GPU。
- 服务器只用独立 worktree 和本次专用 run 目录；不改旧部署目录、不覆盖已有正式结果。
- 先跑 `C3/A1_REQUESTED` 的单 seed smoke，确认 `summary.json`、episode 产物和关键
  telemetry 可读后再扩到四 worker。smoke 失败只修入口，不把一次崩溃写成方法失败。
- 本轮仍然是 screen：route/completion/success/timeout/饱和率和单个 episode 的
  safety termination 都是软指标/正常结局，擦线只记 warning，不停止整个 arm；只有
  运行级 NaN/Inf、checkpoint/config/stage/hash 不匹配、非课程入口或输出覆盖才停止。
- ALOS 预实验本身仍然只做 eval、不训练；今晚的 BC/PPO 训练在 §4.11 单独发起，
  不能把 ALOS 结果写成 PPO 结论。

**评价与判读**

- 主指标：`path_horizontal_rmse_m`、`path_vertical_rmse_m`、`path_three_d_rmse_m`；
  报告逐 seed paired delta、均值和 median。4 seeds 下 bootstrap 95% CI 只作诊断，
  不作显著性声明。
- 辅助指标：`preterminal_three_d_rmse_m`、completion、terminal phase 占比、speed
  tracking error、heading/pitch rate clip 比例、material saturation 和 timeout。
- 以同一 seed 的 paired delta 比较，不用单次最好结果；`A1 - A0` 看总效果，
  `A2 - A0` 看 lookahead，`A1 - A2` 看 vertical gain。
- 若主要 RMSE 差异落在 seed 噪声 CI 内，判为“无证据”，不扩成正式矩阵；若只有纵向
  改善而横向退化，保留分轴判断，不用 3D 单指标掩盖。
- success 只作诊断，不作为本轮唯一通过条件；本轮不调 reward、不调 PPO、不启动训练。

**计划产物（执行后）**

- profile：`configs/lowlevel/goal182/guidance_alos_<arm>.json`；
- 评估输出：C3/C4 分目录，例如
  `artifacts/lowlevel/goal182_alos_lookahead_prescreen_20261116/<stage>/<arm>/`；
- 汇总表：每个 arm 一行，附 matched checkpoint、stage、seeds、profile sha256 和
  paired delta；同时给出每个 stage 的逐 seed delta、均值和 median。CI 只作诊断；
  结论只写“方向性提示/无证据”，不写 formal PASS 或显著性结论。

### 4.11 今晚联合训练批次（2026-11-16，用户授权 BC/PPO）

用户允许今晚同时推进 ALOS、BC 和 PPO。执行顺序固定为：先起 ALOS 评估，再补 5D BC
schema 并重训，最后用新 BC 启动 canonical-profile PPO。三条流可以使用不同 GPU，但
总并发不超过服务器空闲 GPU 数；不杀别人任务、不覆盖已有结果。

**A. ALOS guidance screen**

- 按 §4.10 执行：C3/C4 六臂、每组 4 seeds（`20–23`），共 12 arm / 48 seed-episode。
- 这是 eval-only，不训练；C3/C4 分开汇总。

**B. BC 5D 重训**

- canonical path schema 为 5D：`signed_cross_track_m`、`signed_vertical_error_m`、
  `tangent_heading_error_rad`、`tangent_pitch_error_rad`、
  `lookahead_heading_delta_rad`。对应现有 8D schema 的列 `(0, 1, 2, 3, 7)`。
- 先补 `path_dim=5` 的 schema、BC 数据选择、checkpoint metadata、训练入口和单元测试；
  5D checkpoint 必须记录 feature names/scales/schema hash，不能把 8D 置零或截断伪装成 5D。
- 数据使用完整 DAGGER2
  `.../pc_v0d_r2b_weighted_bc_dagger_20261107/demos_merged_dagger2/demos.npz`，启动前
  校验 SHA-256 `080f4ac3a99ce5f41da119038f0de5a626ea8af1c6b3d9cb7de472d337169d78`。
- BC 训练配方沿用现有 weighted BC 基线（state encoder `[128,128]`、actor
  `[256,128]`、`d_z=64`、batch `4096`、seed `0`），不把网络尺寸混进 5D schema 因子。
- 新 5D BC 先跑训练 smoke（至少 1 epoch 或现有最小可运行预算），再跑完整 early-stop
  训练；之后用 seeds `20–23`（最多 4 seeds）做 BC_ONLY 对照，确认没有行为漂移。
- 如果 5D schema 在今晚前 60–90 分钟无法通过单元测试和最小训练 smoke，则停止 5D 路线，
  记录 `BC5D_SCHEMA_BLOCKED`；不得用旧 8D checkpoint 冒充 5D 继续 PPO。

**C. PPO canonical-profile smoke / screen**

- 只有在 5D BC checkpoint 可加载后才启动 PPO；使用新 BC 作为 base/prior，不使用旧 8D 结果
  作为同协议对照。
- 新训练配置必须绑定 `training_profile="PPO-FLAT-4096x48-E5-MB4"`，即 `4096 envs × 48
  steps`、`5 epochs`、`4 mini-batches`，并在启动前通过
  `总协调/scripts/check_training_profile.py`；没有该字段或仍为 `256 steps / 1 epoch /
  128 mini-batches` 的旧配置只能标 `historical_only`，不得当新协议训练。
- PPO 首轮使用 M 档 MLP `[512,256,128]`（actor/critic mirror）、`path_dim=5`。先把配置
  检查、1–2 iteration smoke、checkpoint round-trip 跑通，再跑 5-iteration screen；screen
  通过且资源允许时，启动 600-iteration 单 seed 训练并监控前若干 iteration。
- 若 Goal182 的 route/heading/speed reward 字段尚未实现，不得临时改旧配置绕过；只能跑
  `PPO_CONTRACT_SMOKE` 并记录 `PPO_REWARD_PENDING`，不能把结果写成 Goal182 科学结论。
- 完整 PPO 微调组仍是必做实验；今晚的 smoke/screen 不能替代它，也不能因为显存/时间不足
  取消它。

**D. Actor 结构整改（并入 PPO 前置，不再后置）**

今晚不能只把旧 actor 的输入宽度从 8 改成 5 就称为整改。以下项目必须在 PPO smoke 前
落地并留下可执行审计：

- **actor 输入**：只保留 5-D non-privileged `actor_path_safe`；progress、remaining、
  curvature、endpoint context、真实 `eta`、真实 current 和任何 critic 特权字段不得进入
  actor。目标字段为 cross、vertical、tangent heading error、tangent pitch error、
  lookahead heading delta。
- **actor 容量**：MLP actor 使用 M 档 `[512,256,128]`；旧 `[256,128]` 只作
  `historical_only`，不再作为今晚主配置。
- **critic 配对**：critic 同样使用 `[512,256,128]`，并通过独立 privileged path encoder
  消费真值路径误差、actuator、disturbance、sensor/safety 特权；不得把特权字段直接拼进旧
  15-D critic state，也不得让 actor/critic 共享同一个 path encoder。
- **actor/critic 分离**：actor 和 critic 的 observation partition、normalizer、path
  encoder 和参数组必须分离；训练 manifest 记录 `actor_path_dim=5`、
  `critic_path_schema`、actor kind、hidden dims、参数量和 schema hash。
- **checkpoint 兼容**：checkpoint metadata 必须记录 actor kind 和 path schema；缺 actor
  kind 的旧 checkpoint 只能按 MLP 加载；5D 与 8D checkpoint 不得静默互转。
- **审计与测试**：至少覆盖 `PATH_PRIVILEGE_LEAK`、`MLP_CRITIC_CAPACITY_MISMATCH`、
  `SILENT_ACTOR_METADATA` 和 5D checkpoint round-trip；记录 actor/critic 参数量和
  actor forward latency。
- **Transformer 顺序**：Goal181 的 M 档 Transformer 仍是后续架构轴
  `A3M_TA_HISTORY_M`（`d_model=112`、2 layers、4 heads、FFN=224，actor/critic 容量
  配对），今晚只有在 MLP actor 结构、5D schema 和 PPO smoke 都通过后，才允许做
  Transformer 的实现/配置 smoke；不得用 Transformer 抢跑或掩盖 MLP 控制契约问题。

PPO 启动前的最低验收命令：

```bash
python /home/bubble/水下机器人/总协调/scripts/check_training_profile.py \
  --profile-id PPO-FLAT-4096x48-E5-MB4 \
  --training-config <new_config> \
  --require-mlp-critic-mirror
```

同时运行 5D/schema/privilege 相关单元测试。任一失败，PPO 只能停在
`PPO_CONTRACT_SMOKE`，不能进入 600-iteration 训练。

**今晚最低交付**

1. ALOS C3/C4 分表结果（至少 C3 六臂完成）。
2. 5D BC 的训练命令、数据 SHA、checkpoint SHA 和 4-seed BC_ONLY 结果；若被 schema 阻塞，
   写阻塞证据和下一步最小修复。
3. PPO 的 actor 结构整改验收、canonical-profile 配置检查结果和 smoke 产物；若进入
   screen/600-iteration，报告 run 目录、GPU、iteration 进度和 checkpoint 位置。

### 4.12 评价矩阵、命名与成果汇报契约（顶级要求）

2026-11-16 用户确认：评价矩阵、成果汇报方式和实验组命名与 PPO 参数同等级，属于
Goal182 的顶级要求。任何新 config、training run、evaluation 或 result 都必须先满足
本节；不满足时记 `INVALID_EXPERIMENT_CONTRACT`，不得进入 Gate、不得写成结论。

本节只定义“如何评价一个已经训练出的 run”。训练什么策略组合由 §4.13 的
`G182-CORE-COMP` 定义；`stage / route / pressure / radius` 是评价轴，不能写成训练
组合轴。

本节的对象层级固定为：

```text
factor      一个科学问题或唯一变化量
arm         factor 的一个 treatment
training_run arm × training_seed
eval_cell   training_run × stage × route_profile × pressure_profile × radius
report_row  一个 eval_cell 的预注册聚合，或一个逐 eval seed 的原始记录
```

#### 4.12.1 arm 与 run 命名

新 arm 一律使用：

```text
G182_<PHASE>_<DOMAIN>_<TREATMENT>
```

`PHASE` 使用 `P0`–`P6`、`ALOS`、`BC`、`INT`；`DOMAIN` 使用
`AUTH / CHAN / NORM / TR / EXPL / CRIT / CURR / REW / ROUTE / STACK`；
`TREATMENT` 使用短大写标识。示例如下：

| 旧名 | canonical arm_id |
|---|---|
| `A0_HISTORICAL` | `G182_P0_AUTH_HIST` |
| `A3_SEP_3D` | `G182_P0_AUTH_SEP_3D` |
| `P3_3D` | `G182_P1_CHAN_3D` |
| `N1_FIXED_EMPIRICAL_BOTH` | `G182_P2_NORM_FIXED_EMP_BOTH` |
| `T1_REFERENCE_TRUST_REGION` | `G182_P3_TR_REFERENCE` |
| `E1_REFERENCE_STD` | `G182_P4_EXPL_REF_STD` |
| `C1_DUAL_PATH_PRIVILEGED` | `G182_P5_CRIT_DUAL_PRIV` |
| `G1_REFERENCE_EARLY_STOP` | `G182_P6_CURR_EARLY_STOP` |
| `A3_VERTICAL_GAIN_ONLY`（ALOS） | `G182_ALOS_GUIDE_VGAIN_ONLY` |
| `I1_BEST_STACK` | `G182_INT_STACK_BEST` |

以下裸名不得再出现在新 matrix、run 目录或结果表中：

- `A0/A3/P0/P3/C0/G0/I0` 这类与 curriculum stage、path family 或历史 Goal 冲突的短名；
- 同一 Goal 内复用的 phase-local 名，例如 Phase 0 与 ALOS screen 同时使用 `A3`；
- 只靠目录上下文才能判断含义的 `best`、`new`、`final`、`v2`。

历史名称允许保留在 `legacy_alias` 字段，但报告主键必须使用 canonical `arm_id`。path
family（`P0`–`P4`）和 curriculum stage（`C0`–`C6`）必须写入独立字段，不能嵌入 arm 短名
冒充实验组。

训练 run 的稳定标识为：

```text
run_id = <arm_id>_s<training_seed>_<training_profile>_<gitsha7>_<UTC timestamp>
```

例如 `G182_P1_CHAN_3D_s86201_PPO-FLAT-4096x48-E5-MB4_c186c74_20261116T120000Z`。
实际目录可分层为 `<root>/<phase>/<arm_id>/s<training_seed>/`，但 manifest 必须记录完整
`run_id`。一个 run 只对应一个 train seed；不得把三个 seed 混写进一个 run 目录。

评价 cell 的稳定标识为：

```text
<run_id>__<stage>__<route_id>__<pressure_id>__r<radius_m>
```

其中 `<stage>` 是课程 stage，`<route_id>` 是显式路线 profile，`<pressure_id>` 是
current/noise/plant 等压力 profile。不得继续用 `level` 这个含糊词同时表示课程难度、
环境压力和路线形状。

每个 cell 内包含预注册的 eval seed 集合；单个评价 episode 的标识才是：

```text
<cell_id>__e<eval_seed>
```

因此 eval seed 不进入 cell 计数。`48 cells / 144 episodes` 表示 48 个 cell 各有 3 个
eval seed，而不是 144 个 cell。

#### 4.12.2 matrix schema v2

**状态：M2 已于 2026-11-16 确认。** Goal182 不再沿用 `auv_goal_matrix.v1` 的
`arms × levels × radii` 隐式笛卡尔积。新 matrix 使用 `auv_goal_matrix.v2`，把
`stage / route / pressure / radius` 拆成显式轴，并要求每个 arm 通过命名
`evaluation_set` 声明自己的覆盖范围。不得依赖全局数组给 arm 静默扩增 cell。

每个 tier 独立生成一张 matrix，因为 `tier` 是单值，且各 tier 的冻结时间不同：
`QUICK_SCREEN` 先于训练、`MAIN_RANKING`/`SPEED_ENVELOPE` 在训练后、`ROUTE_GENERALIZATION` 要等
formal 前三名产生后才能填 arms。tier 文件由报告索引链接，不能把不同 tier 混成一张
可冻结 matrix。

最小结构如下：

```json
{
  "schema_version": "auv_goal_matrix.v2",
  "goal_id": "Goal182",
  "repo": "settled_局部规划器_反应式",
  "matrix_id": "G182-EVAL-FORMAL-CORE",
  "tier": "MAIN_RANKING",
  "matrix_status": "DRAFT | FROZEN | RETIRED",
  "evaluation_sets": {
    "ES_MAIN_RANKING": {
      "stages": ["EVS_C2", "EVS_C3", "EVS_C4", "EVS_C6"],
      "route_profiles": ["EVR_STAGE_MATCH"],
      "pressure_profiles": ["EVP_NOMINAL", "EVP_CURRENT10", "EVP_NOISE_N2", "EVP_PLANT20"],
      "radii_m": [2.0],
      "eval_seeds": [20, 21, 22]
    }
  },
  "arms": [
    {
      "arm_id": "G182_P1_CHAN_3D",
      "phase": "P1",
      "domain": "CHAN",
      "factor": "channel_scope",
      "treatment": "3d",
      "role": "baseline | treatment | diagnostic",
      "program_group_id": "G-ALOS-BODY-RL",
      "control_contract": "CC_ALOS_BODY",
      "train_seeds": [86201, 86202, 86203],
      "evaluation_set": "ES_MAIN_RANKING",
      "config": "configs/...",
      "legacy_alias": ["P3_3D"]
    }
  ],
  "zero_baselines": [
    {
      "baseline_id": "ZERO_BC_FROZEN",
      "kind": "frozen_bc",
      "program_group_id": "G-ALOS-BODY-RL",
      "control_contract": "CC_ALOS_BODY",
      "zero_seed": 86201,
      "evaluation_set": "ES_MAIN_RANKING"
    }
  ],
  "training_profile": "PPO-FLAT-4096x48-E5-MB4",
  "expected_cells": {"policy": 0, "zero": 0, "total": 0},
  "expected_episodes": 0,
  "metrics": [],
  "gates": [],
  "heldout": {"enabled": false, "seeds": []}
}
```

`evaluation_set` 是唯一允许的覆盖声明入口。一个 arm 只引用一个 set；若两个 arm 的
stage/route 覆盖不同，必须定义两个 set，不能靠代码分支或运行参数临时覆盖。
`eval_seeds` 使用显式整数列表，不使用 `start/count`，避免边界和重复歧义；
`radii_m` 使用显式数值列表，当前只允许 `[2.0]`。

`expected_cells` 与 `expected_episodes` 必须由 checker 展开 `evaluation_set` 后计算，
不能手填后不校验：

```text
policy_cells =
  Σ_arm len(train_seeds)
      × len(stages)
      × len(route_profiles)
      × len(pressure_profiles)
      × len(radii_m)

zero_cells =
  Σ_zero len(zero_seeds)
      × len(stages)
      × len(route_profiles)
      × len(pressure_profiles)
      × len(radii_m)

total_cells = policy_cells + zero_cells

expected_episodes = Σ_cell len(eval_seeds(cell))
```

每个 cell 内包含预注册 eval seed 集合；eval seed 是 cell 的内容，不再额外乘一次 cell
数量。`48 cells / 144 episodes per arm` 的分解固定为：

```text
3 train seeds × 4 stages × 1 route × 4 pressures × 1 radius = 48 cells
48 cells × 3 eval seeds = 144 episodes
```

matrix 完整性检查至少包括：

- `schema_version` 必须为 `auv_goal_matrix.v2`，tier 必须属于 §4.12.3 的枚举；
- `matrix_id` 唯一，`arm_id`/`baseline_id` 不重复；
- `program_group_id` 与 `control_contract` 必须匹配，不能把 PE2E 和 ALOSB 先混算；
- 每个 `evaluation_set` 引用必须存在，且 stage/route/pressure/radius/eval seed 都在
  axis registry 中登记；
- 某个 arm 少跑任一预注册 cell 判 `INVALID_INCOMPLETE_EVAL`；
- 某个 coverage 未声明却被实际产物创建，判 `INVALID_UNREGISTERED_CELL`；
- `expected_cells` 或 `expected_episodes` 与展开结果不一致，判
  `INVALID_MATRIX_ARITHMETIC`；
- held-out 继续关闭；出现 held-out 产物判 `INVALID_HELDOUT_LEAK`。

#### 4.12.3 评价层级

| tier | 训练 seed | eval seeds | 用途 | 可写入结论 |
|---|---:|---:|---|---|
| `SMOKE` | 1 | 1 | 只验 schema、checkpoint、输出和 finite | 否 |
| `QUICK_SCREEN`（快速筛查） | 1 | 3 | 方向筛选、instrumentation 检查；`C3` 必选，和其他 stage 分开汇报 | 否，只能 `NOT_FORMAL` / `QUICK_SCREEN` |
| `MAIN_RANKING`（主排名比较） | 3 | 3 | `C2/C3/C4/C6` + 四个压力档 + stage-matched route + `2.0 m` 半径 | 是，按 Gate；唯一主排名集 |
| `SPEED_ENVELOPE`（速度包线） | 3 | 3 | 所有正式 arm：`C2/C3/C4/C6` × `1.0/1.5/2.0/2.5 m/s` 四档 × `nominal` | 是，仅速度包线结论 |
| `ROUTE_GENERALIZATION`（路线泛化测评） | 3 | 3 | 只给 `MAIN_RANKING` 前三名；C6 + 22 条路线（19 条二维 + 3 条三维） | 是，仅前三名 |
| `DEPTH_GENERALIZATION`（深度泛化审计） | 3 | 3 | 只给 `MAIN_RANKING` 前三名；C6 × `R_STRAIGHT + R_3D_M` × 四个绝对深度 × `nominal` | 仅诊断，不参与 winner |
| `DIAGNOSTIC` | 任意 | 任意 | 单 cell、trace、失败归因 | 否 |

Goal182 的 `strict125` 已冻结移除，因此 active protocol 的 radius 轴固定为
`[2.0]`。保留 `radii_m` 字段是为了显式记录，而不是继续偷偷跑两个半径。历史 Goal173/175
的 `1.25 m` 只作历史对照。

**测评层级命名（2026-11-16 统一）**：canonical 名称只使用右列新名；旧名仅作为
`legacy_alias` 读取，不得再写入 matrix、manifest、报告标题或新 config。

| 新 canonical 名称 | 含义 | 旧名 / alias |
|---|---|---|
| `QUICK_SCREEN` | 快速筛查，不是正式结论 | `SCREEN` |
| `MAIN_RANKING` | 主排名比较，决定 winner / 前三名 | `FORMAL_CORE` |
| `SPEED_ENVELOPE` | 固定速度包线测评 | `SPEED_EVAL` |
| `ROUTE_GENERALIZATION` | 22 条路线的泛化测评 | `COLLECTION_EVAL`、`FORMAL_GEOMETRY` |
| `DEPTH_GENERALIZATION` | 绝对深度敏感性审计 | `DEPTH_AUDIT` |
| `EXTERNAL_GEOMETRY_SUITE` | 外部 standing suite D 的最终体检 | `suite D` |

`EXTERNAL_GEOMETRY_SUITE` 只记录外部几何诊断，不参与 Gate、winner、前三名选择或主表
排名；其规模远大于 `ROUTE_GENERALIZATION`，不得为了“统一命名”把所有 arm 都送进去。

正式结论必须同时满足：

- `tier=MAIN_RANKING` 或 `ROUTE_GENERALIZATION`；
- matrix 状态为 `FROZEN`，且有 SHA-256；
- 所有 cells 完整，无 `INVALID_*`；
- 训练 run 的 curriculum provenance 完整；
- 评价与报告都使用同一 canonical arm/run/cell 主键。

#### 4.12.4 路线 profile 与 3D 轴

当前 `level` 只表示 current/noise/plant 压力，不表示路线形状。Goal182 的 active matrix
必须把路线形状拆成显式 `route_profile`。最低定义如下：

| route_profile | 含义 | 用途 |
|---|---|---|
| `R_STAGE_MATCH` | 使用当前 curriculum stage 的 `path_family_mix`、深度、长度和速度 | 每个 stage 的主评价，与训练分布对齐 |
| `R_STRAIGHT` | 直线基线；`depth/lateral` 振幅与 cycles 均为 0，长度 `70 m` | 几何轴的原点 |
| `R_VERT_A<amp>_C<cycles>` | 垂向正弦；振幅 `0.5 / 2 / 5 m` × cycles `2 / 3 / 5`，共 9 条，长度 `70 m` | 垂向难度主轴 |
| `R_LAT_A<amp>_C<cycles>` | 横向正弦；振幅 `0.5 / 2 / 5 m` × cycles `2 / 3 / 5`，共 9 条，长度 `70 m` | 横向难度主轴 |
| `R_3D_<level>` | 垂向 + 横向同时变化：`S=0.5/0.5 m, C2`、`M=2/2 m, C3`、`L=5/5 m, C5`，长度 `70 m` | 三维耦合审计 |
| `R_P0`–`R_P4` | 固定一个 path family，其他几何参数冻结 | 跨 family 泛化审计 |
| `R_LEGACY_SINE` | 现有 `sinusoidal_3d_arclength`：约 `2 m` 总深度漂移、`0.4 m` 垂向振幅、2 cycles | 更强的 legacy 垂向/3D 路线审计 |
| `R_LENGTH_<m>` | 固定长度，其他几何参数一致 | 长度泛化审计 |
| `R_SPEED_<mps>` | 固定路线速度，其他几何参数一致 | 速度包线审计 |

Goal182 的 active geometry 轴固定为
`1 + 2 × 3 × 3 = 19` 条二维路线：`R_STRAIGHT` 加
`R_VERT_{A0P5,A2,A5}_{C2,C3,C5}` 与
`R_LAT_{A0P5,A2,A5}_{C2,C3,C5}`；再加 `R_3D_S/M/L` 三条三维耦合路线，共
`22` 条。二维和三维主轴的 amplitude/cycles 均已冻结。路线 override 必须
通过 evaluator 的 `DIVERSITY_REFERENCE_OVERRIDE_FIELDS` 显式登记，并记录 reference
config hash；不得在脚本里临时改 `depth_amplitude_m` 等字段而不进入 matrix。

`depth_slope_m` 的当前实现含义是“沿归一化路径参数 `[0, 1]` 的总深度漂移”，不是每米
斜率；例如 `depth_slope_m=2.0` 表示全程约漂移 `2 m`。

##### 4.12.4.1 测评速度规则与速度包线评测

训练和评价的速度变化必须分开理解：

- 所有测评 episode 内速度固定，`command_speed_resample_period_s=null`，不得沿用训练的
  每 `5 s` 重采样；
- 速度差异来自不同 evaluation cell 的固定 `speed_mps`，不是同一个 episode 内随时间
  变化。这样才能把速度响应和训练时变命令分开归因；
- `R_STAGE_MATCH` 的 `speed_mps` 必须在生成 frozen matrix 时显式冻结为一个数值，不得
  留成训练 speed interval，也不得在 evaluator 中重新随机采样；
- `MAIN_RANKING` 主比较固定为 `C2/C3/C4/C6`，每个 stage 使用一个冻结 speed；完整速度
  包线放到独立的 `SPEED_ENVELOPE`，不能把速度轴乘进 `MAIN_RANKING` 的所有 cell；
- `SPEED_ENVELOPE` 覆盖所有正式 stage `C2/C3/C4/C6`，不是只针对 C3；固定速度只保留四组
  `1.0/1.5/2.0/2.5 m/s`，压力固定为 `nominal`，每个 cell 含 3 个 eval seeds；速度和
  current 不得同时变化，海流压力仍由 `MAIN_RANKING` 负责；
- `0.9 m/s` 只作额外 OOD 诊断，不进入四组正式速度包线。每个 stage、速度、pressure
  分开报告速度跟踪、heading/yaw、material saturation、cross/vertical RMSE、safety 和
  timeout；禁止把四个速度先 pooled 成一个平均值；
- 快速筛查 `QUICK_SCREEN` 仍强制包含 `C3`，并使用同样的四组固定速度做快速检查；C3 初评是
  `QUICK_SCREEN / NOT_FORMAL`，不得与 C3 formal 或 `SPEED_ENVELOPE` 结果混表；
- 每个正式 arm 的 `SPEED_ENVELOPE` 规模为
  `3 train × 4 stage × 4 speed × 1 pressure = 48 cells / 144 episodes`。该结果只回答
  速度包线问题，不替代 `MAIN_RANKING` 的 `C2/C3/C4/C6` 主比较。

#### 4.12.5 成果汇报契约

**EV9-A 已确认（2026-11-16）**：canonical 产物名称固定为
`matrix_manifest.json`、`report_rows.jsonl`、`report.md`、`summary.json`、
`episodes.jsonl`、`trace.npz`。不改成 `eval_*` 或 `trajectory_*` 等第二套名字；
需要人读解释时在 `report.md` 写清楚，不靠重复命名表达。

| 产物 | 粒度 | 作用 | 能否直接支撑主结论 |
|---|---|---|---|
| `matrix_manifest.json` | 每张 frozen matrix 一份 | 记录 arm、evaluation set、config/checkpoint/BC base、路径和 SHA，以及预期/实际 cell 数 | 不排名，但缺它结果无效 |
| `report_rows.jsonl` | 每个 eval cell 一行 | 机器可读的正式结果；用于 Gate、主排名和前三名选择 | 是 |
| `report.md` | 每个 evaluation set 一份 | 人读报告；四个压力分开汇报，引用可追溯数据，不手填结论数字 | 是 |
| `summary.json` | 每个 run 一份 | 记录 run 状态、iteration/stage、停止原因、最终 checkpoint、产物路径 | 否，用于运行审计 |
| `episodes.jsonl` | 每个实际 episode 一行 | 记录 eval seed、结果、指标、失败标签和 episode id | 否，用于统计、归因和复核 |
| `trace.npz` | 每个 run 一份，内部按 episode id 存时序数组 | 保存 state/action/guidance/sensor/reward/termination 等时序，用于复现、画图和排障 | 否，只作证据 |

每个正式报告至少包含：

1. `matrix_manifest.json`：matrix/training/env/checkpoint/BC base 的路径和 SHA；
2. `report_rows.jsonl`：每个 eval cell 一行，机器可读；
3. `report.md`：只展示预注册聚合和 Gate 判读，不手写无法回溯的数字；
4. 完整性计数：`expected_cells`、`observed_cells`、`valid_cells`、`invalid_cells`；
5. 训练和评价的 protocol/tier/held-out 状态。

`report_rows.jsonl` 必含字段：

```text
goal_id, tier, phase, arm_id, role, training_seed, checkpoint_sha256,
stage, route_profile, path_family, depth_band_m, vertical_relief_m,
length_m, speed_mps, pressure_profile, radius_m, eval_seeds, n_eval_episodes,
success, timeout, safety, plane_miss,
path_three_d_rmse_m, path_horizontal_rmse_m, path_vertical_rmse_m,
pre_goal_3d_rmse_m, path_completion_ratio, material_saturation_rate,
valid, failure_codes
```

`report_rows.jsonl` 的一行是一个 cell，指标按该 cell 预注册的 `eval_seeds` 聚合；
`eval_seed` 只出现在 `episodes.jsonl` 的单 episode 行中。`episodes.jsonl` 与
`trace.npz` 必须使用同一套 `episode_id`，且 `report_rows.jsonl` 的聚合值必须能从
`episodes.jsonl` 重新计算。缺少任一 canonical 产物判 `INVALID_ARTIFACT_SET`；
报告数字无法从明细复算判 `INVALID_REPORT_PROVENANCE`。`trace.npz` 不得用于
挑 checkpoint、挑 winner 或反向修改主排名；它只解释结果为什么发生。

**铁律：不同评价压力必须分开汇报。** `nominal / current10 / noise_n2 / plant20`
（以及后续 `current05/current15/combined20`）不得先合并成一个 pooled 数字再作为主
结论。正式报告必须按 `pressure_profile` 分行或分表；每个 arm × stage × route ×
radius 都要保留各压力自己的 mean、median、p90、`n`、无效 cell 数和 paired delta。
跨压力 pooled mean 只能作为明确标注 `DIAGNOSTIC` 的辅助行，不能进入 Gate、winner
判定或摘要主表。

聚合报告必须同时给出：

- 每个 arm 的 mean、median、p90、`n` 和无效 cell 数；
- 与同一 `training_seed × stage × route × pressure × radius × eval_seed` 的 paired
  baseline delta；
- training-seed 间均值和标准差，不能只看 pooled mean；
- seed 数和不确定性；4 seeds 以下只作诊断，不作显著性声明；
- 失败标签和“为什么不能写 PASS/FAIL”的明确说明。

报告必须分开写以下四类结果，禁止混表：

| 类别 | 例子 | 可写成 |
|---|---|---|
| 固定 stage 微调 | 本轮 600-it C3 PPO vs BC | `NOT_FORMAL` / `C3_FIXED_STAGE` |
| 完整课程 run | C0→C6 晋级训练 | `FORMAL` 或 `INVALID` |
| 速度包线 | `C2/C3/C4/C6` × 四档固定速度 | `SPEED_ENVELOPE` |
| ALOS/BC eval-only | guidance screen、BC_ONLY | `BASELINE` / `QUICK_SCREEN` |
| residual / fine-tune 机制 | authority、residual screen | `DIAGNOSTIC` / `QUICK_SCREEN` |

当前 `goal182_result_20261116.md` 的 600-it 结果必须始终标为
`C3_FIXED_STAGE / NOT_FORMAL`，不得因为 success 4/4 就升级为正式课程结论。

#### 4.12.6 当前矩阵审计结论

- Goal173/175/177/179 已使用 `arms × levels × radii × seeds + zero baselines` 的
  `auv_goal_matrix.v1`，但字段语义不完全统一，且 `level` 同时承载压力档；
- Goal182 的 `goal182_axis_registry.json`、`goal182_core_composition_index.json`、
  `goal182_core_pe2e_matrix.json`、`goal182_core_alosb_matrix.json` 和
  `goal182_eval_matrix.json` 已生成 DRAFT；matrix/eval checker 已存在并通过，
  Goal182 artifact auditor 仍未实现；
- 通读时发现 Goal182 §5.1 已冻结只保留 `default 2.0 m`，而旧 §5.3 仍写 `× 2 radii`；
  该内部冲突已在本次 §5.3 修订中改为单一 `2.0 m`；
- 当前 600-it 结果只覆盖 C3、4 个 eval seeds，不是完整 `arms × levels × radii`；
- `pc_standing_eval_suite_v1.json` 已定义 route/length/speed/depth profile 和
  A/B/C/D tier，但 Goal182 尚未把其中任何一个 route 轴接入 active matrix。
- `pc_standing_eval_suite_v1.json` 的 `depth_profiles_m` 目前只是声明，`build_pc_standing_eval_configs.py`
  并未消费它；P0–P4 eval config 仍继承 base env 的 `depth_start_m`（通常约 `10 m`）。
  因此绝对深度轴目前是文档存在、执行缺失。

#### 4.12.7 3D 路线审计

按当前代码，curriculum stage 确实改变 **绝对深度** 和 **水平/曲率难度**，但没有系统性
改变垂向路线起伏：

| stage | path family mix | 深度带/m | 长度/m | 垂向几何 |
|---|---|---:|---:|---|
| C0 | P0 100% | 9.5–10.5 | 35 | P0 仅极小 depth slope，基本平直 |
| C1 | P0 70% / P1 30% | 10–30 | 35–40 | P1 垂向 sine 幅值约 0.1–0.2 m |
| C2 | P1 50% / P2 50% | 30–80 | 35–45 | P2 sine 垂向幅值约 0.15–0.35 m |
| C3 | P2 50% / P3 50% | 80–150 | 40–50 | P2 小幅 sine；P3 只有线性 slope |
| C4 | P3 60% / P2 40% | 150–250 | 40–50 | 同上 |
| C5 | P3 70% / P4 30% | 250–400 | 40–50 | P4 只有线性 slope |
| C6 | P4 70% / P3 30% | 400–500 | 40–50 | 同上 |

另有一个不在 active stage 中的 legacy sine profile：它使用约 `2 m` depth slope、`0.4 m`
depth amplitude 和 2 个 depth cycles，垂向起伏明显强于 P0–P4。它可以作为
`R_LEGACY_SINE` 进入后续 3D 几何审计，但不能冒充当前 C3 stage-matched 路线。

代码证据：

- `P0`：直线，depth slope 约 `±0.005`；
- `P1`：depth sine 幅值 `0.1–0.2 m`；
- `P2`：depth sine 幅值 `0.15–0.35 m`，或带全程 `±0.01 m` 深度漂移的折线；
- `P3/P4`：水平曲率增大，但 depth 只使用全程 `±0.01 m` 的线性漂移，没有垂向 sine；
- C0–C6 的 length profile 只在 `35→50 m` 内缓慢变化，speed profile 也只在约
  `1.1–1.4 m/s` 内轻微变化。训练速度目标已改为 §4.12.7.2 的中心向外课程；上述
  `1.1–1.4 m/s` 只描述当前代码的旧 v1 状态。

因此当前结论是：

> **训练和评价路线不是“没有 3D”，但 3D 难度主要来自绝对深度、水平曲率和 path family；
> 垂向起伏基本固定在亚米级，pitch 只有几度。当前评测中的 `3D RMSE` 是三维误差指标，
> 不是三维路线难度扫描。**

当前 `C3 + eval seeds 20–23` 的三组 12-cell 评价，只覆盖 C3 的 4 个 path/current/noise
realization；它不能回答“换垂向起伏、换路径 family、换长度后是否仍好”。在加入
`R_VERT_*` / `R_P*` 轴之前，不得把当前结果写成 3D 泛化结论。

##### 4.12.7.1 已确认的垂直三维课程（目标状态）

2026-11-16 用户确认：现有 C0–C6 的垂向起伏太小，不能只靠改变起始深度形成 3D
课程。新的目标配置如下；C0/C1 保持现状，C2 改成单轴隔离，C3–C6 增加真实垂向
正弦。

| 阶段 | 路线构成 | 垂向振幅 | 横向振幅 | cycles | 40–50 m 路线最大俯仰/航向角约 |
|---|---|---:|---:|---:|---:|
| `C2-V` | 纯垂向正弦 | `1.75 m` | `0` | `1` | `15–17°` |
| `C2-H` | 纯横向正弦 | `0` | `1.75 m` | `1` | `15–17°` |
| `C3` | 保留 P2/P3 水平混合 + 垂向 | `2.0 m` | 现有水平几何 | `2` | `29–33°` |
| `C4` | 保留 P3/P2 水平混合 + 垂向 | `2.0 m` | 现有水平几何 | `2` | `29–33°` |
| `C5` | 保留 P3/P4 水平混合 + 垂向 | `3.0 m` | 现有水平几何 | `1.5` | `31–35°` |
| `C6` | 保留 P4/P3 水平混合 + 垂向 | `5.0 m` | 现有水平几何 | `1` | `35–38°` |

约束和解释：

- `C2-V` 与 `C2-H` 各占 C2 的 50%；C2-V 的横向振幅必须为 0，C2-H 的垂向振幅必须
  为 0，二者不得再混入对方的轴，否则无法隔离单轴课程效果；
- C2 的深度带 `30–80 m`、长度 `35–45 m` 保持不变；速度改用
  §4.12.7.2 的 `1.20–1.80 m/s` 中心向外区间；
- C3–C6 的起始深度、水平 path family mix 和长度沿用现有表；速度改用
  §4.12.7.2 的中心向外区间，新增的是垂向正弦振幅和 cycles；
- C5 使用 `3.0 m / 1.5 cycles`，而不是 `3.0 m / 2 cycles`：后者在 40 m 路线上最大
  俯仰约 `44°`，贴近 `phi_term_deg=45°` 终止边界，容易把课程难度误判成安全问题；
- C6 的 `5.0 m / 1 cycle` 是训练上限；路线泛化测评的 `R_3D_L = 5 m / 5 cycles / 70 m`
  仍作为外推压力，不要求训练阶段原样覆盖；
- 路线长度仍为 `35–50 m`，因此这些垂向振幅是“真实 3D 起伏”，但不是长时域深潜；
  若要训练更高频的 5 m 垂向，必须另行延长路线或新开课程版本。

实现约束：

- `CurriculumStageSpec` / `CurriculumEpisodeSpec` 必须新增
  `vertical_amplitude_m`、`vertical_cycles`，以及 C2 的 route-axis/mode 字段；
- `CurriculumSampler` 必须逐 episode 采样并写入 `curriculum_episodes.jsonl`；
- `_build_curriculum_path_set` 必须把采样值注入 `ReferenceConfig`；
- P2/P3/P4 生成器必须真正消费 `depth_amplitude_m` / `depth_cycles`；当前 P3/P4
  只用 `depth_slope_m`，这是本次实现的主要缺口；
- 旧 C0–C6、尤其旧 C3 的 600-it 结果统一标 `historical_only`，不得与新垂向课程
  结果直接混算；
- 必须增加路径长度、垂向振幅/cycles、最大 pitch 和 manifest provenance 的回归测试。

##### 4.12.7.2 已确认的中心向外速度课程（训练目标状态）

2026-11-16 用户确认：速度会改变舵效，训练不能只在一个窄速度带内采样，也不能先从低速
或高速单边扫到另一边。新的 Goal182 速度课程以 `1.5 m/s` 为适中中心，先扩高速侧，再扩
低速侧，交替向外覆盖到 REMUS100 物理包络。稳态舵效近似随动压 `V²` 变化；以
`1.5 m/s` 为 1.00，各速度的相对舵效约为：`1.0=0.44`、`1.2=0.64`、`1.4=0.87`、
`1.6=1.14`、`1.8=1.44`、`2.0=1.78`、`2.2=2.15`、`2.5=2.78`。

| 阶段 | 训练速度区间/m/s | 本阶段相对上一阶段的变化 | 路线侧重点 |
|---|---:|---|---|
| `C0` | `1.40–1.60` | 以 `1.5` 为中心，建立基线 | 直线；先确认速度跟踪和低速舵效正常 |
| `C1` | `1.40–1.80` | 高速侧 `+0.20` | 简单路线；检查高速舵效偏强、偏航振荡和饱和 |
| `C2` | `1.20–1.80` | 低速侧 `-0.20` | 开始有横向/垂向单轴运动；检查弱舵效下 cross/vertical |
| `C3` | `1.20–2.00` | 高速侧 `+0.20` | 有转弯段；检查高速下转向过冲与执行器饱和 |
| `C4` | `1.00–2.00` | 低速侧 `-0.20` 到物理下限 | 更复杂转弯；检查低速转向能力和跟踪迟滞 |
| `C5` | `1.00–2.20` | 高速侧 `+0.20` | 强扰动前的高速度包络 |
| `C6` | `1.00–2.50` | 高速侧 `+0.30` 到物理上限 | 最终包络；高速侧覆盖到 REMUS100 上限 |

采样和重采样约束：

- 训练协议 ID 固定为 `SPEED_V3_CENTER_OUT_5S`；现有 v1 窄速度带和 v2 单调高速表
  都不能冒充本协议；
- 每个 active stage 的速度目标从该 stage 的 `speed_range_mps` 内均匀采样；
  `C0` 的窄区间是基线，不是最终能力边界；
- episode 起点采样一次，之后在 `t=5, 10, …, 55 s` 重新采样；同一 5 s 段内保持目标
  不变。`60 s` episode 不额外在终点重采样，因此最多 `12` 个速度段；
- 重采样只改变速度目标；heading/pitch 仍由 path/ALOS 正常生成，不另建一套路线时钟。
  下游命令限幅和速率限制继续生效，因此“目标重采样”不是执行器阶跃；
- `alos` 用它驱动 desired cruise speed；`path_*` 用它驱动外部速度指令。actor 只能看到
  当前可部署观测和当前参考，不得看到未来的速度段序列；
- 这是 **training-only** 协议。所有正式评价和路线泛化测评仍按 route profile 固定速度，
  `command_speed_resample_period_s=null`；训练时变速度不得用于评价比较；
- `0.9 m/s` 只保留为 eval/OOD 压力，不进入训练下限；训练下限始终是
  `vehicle.min_speed_mps`。

实现和 provenance：

- `CurriculumStageSpec` / `CurriculumEpisodeSpec` 必须表达速度区间、重采样周期和段索引；
  `CurriculumSampler` 必须逐 episode 写 `speed_range_mps`、初始速度、实际访问的
  min/max 和完整 speed-segment 计划；
- 运行时 `info` 每步至少记录 `speed_profile_id`、`stage`、当前段索引、
  `speed_segment_start_s`、`speed_resample_period_s`、`speed_target_mps`、
  `speed_target_min_mps`、`speed_target_max_mps`。原 `episode_command_speed_mps`
  若保留，只能作初始值/兼容 alias，不能继续表示一个 episode 内恒定速度；
- 速度目标必须同时通过 `vehicle.min_speed_mps`、`vehicle.max_speed_mps` 和静态
  `speed + current_variation + variation_amplitude <= max_speed` 可行性检查。C6 名义
  上限是 `2.5 m/s`；若当前 variation reserve 为 `0.1 m/s`，实际可执行上限是
  `2.4 m/s`，精确 `2.5` 只出现在 reserve 为零的标定 cell。不得按瞬时真实海流反向修改
  速度命令；
- 所有速度边界必须引用 `vehicle.min_speed_mps` / `vehicle.max_speed_mps` 或本协议的
  命名常量，禁止在运行时复制 `1.0` / `2.5` 字面量；
- 正式训练前先做 eval-only 速度标定：`R_STRAIGHT` 加一条中等转弯路线，
  `nominal` 单压力，速度
  `1.0/1.5/2.0/2.5 m/s` 四组，3 个 eval seeds。只记录速度跟踪、
  heading/yaw 跟踪、`material_saturation`、cross/vertical RMSE、safety 和 timeout；
  该标定不参与 winner 选择，也不允许据结果静默缩表；
- 若高速端出现持续饱和或安全回归，先报告并讨论，不得擅自把 C6 改回窄速度带或改评价门限。

#### 4.12.8 待用户确认

- [x] M1：采用 `G182_<PHASE>_<DOMAIN>_<TREATMENT>` 作为唯一新 arm 命名，并把旧名放进
  `legacy_alias`。**已确认（2026-11-16）**：核心组合臂使用
  `G182_INT_STACK_<ACTOR>_<CONTROL>_<LEARN>_<MODULE>`；草稿模板 `G182C_...` 退役，
  只允许作历史 alias；详见 §4.13.1。
- [x] M2：采用 `auv_goal_matrix.v2` 和显式 coverage/completeness 公式。
  **已确认（2026-11-16）**：每个 tier 独立一张 matrix；arm/zero baseline 通过
  `evaluation_set` 声明 stage/route/pressure/radius/eval-seed 覆盖；eval seed 使用
  显式列表且属于 cell 内容，不进入 cell 计数；checker 必须展开 coverage 并同时重算
  `expected_cells` 与 `expected_episodes`；详见 §4.12.2。
- [x] M3：`MAIN_RANKING` 固定 `3 train seeds × 4 pressures × 1 radius × 3 eval seeds`，
  route 使用 stage-matched；已按 EV1 确认；
- [x] M4：路线泛化测评（`ROUTE_GENERALIZATION`）固定 `19 条二维 + 3 条三维 = 22 条路线`，
  只给 `MAIN_RANKING` 前三名；二维为 `R_STRAIGHT + 2 方向 × 3 振幅 × 3 cycles`，三维为
  `R_3D_S/M/L` 垂向+横向耦合；已按 EV3/EV3b 确认；
- [x] M5：完整课程 run 的正式评价范围。**已由 EV2 与 D8/CU1 关闭**：所有正式 arm
  的主排名统一评价 `C2/C3/C4/C6`；C0/C1 只作训练热身和诊断；完整 C0→C6 stage
  sweep 只给最终 winner 做一次曲线，不进入所有 arm 的正式排名。
- [x] M6：报告以 `report_rows.jsonl + report.md` 为 canonical 产物。
  **已由 EV9-A 确认（2026-11-16）**：禁止手工散表作为主结论；`report_rows.jsonl`
  必须能从 `episodes.jsonl` 复算，`trace.npz` 只作诊断；详见 §4.12.5。
- [x] M7：是否把绝对深度作为正式轴；若要，必须让 builder 真正消费
  `depth_profiles_m`，并与 `route_profile` 分开命名和计数。当前它是未接线的声明。
  **已由 EV4-A 确认（2026-11-16）**：接线四个深度档位，作为
  `ROUTE_GENERALIZATION` 前三名的独立 `DEPTH_GENERALIZATION`，不把四个深度乘进现有 22 路线 ×
  4 压力的路线泛化测评；`depth` 与 `route_profile` 必须分开计数和汇报。

#### 4.12.9 默认导航传感器：`unbiased_small_noise_v3`（2026-11-18 更新；v2 为历史基线）

**2026-11-18 更新（生效）**：新训练/评测默认改为 `unbiased_small_noise_v3` =
`unbiased_small_noise_v2` + `current_est_tau_s=(0.05, 0.5) s`；v2 保持不变，
只作历史兼容。C5/C6 另加 `0.05–0.3 s` short current-estimate outage（规划见
Goal184 §4.1），outage 是 curriculum 事件 override，不是 profile 切换。
以下 v2 描述作为历史基线保留。

**状态：默认不锁真值。** 新默认 `unbiased_small_noise_v3` 继承 v2 的 `p_nav`
与零均值噪声契约，并把 `current_est_tau_s` 缩短为 `(0.05, 0.5) s`。v2 基于
`unbiased_noise_v1`，只把 `nav_noise_m` 从 `(0.05, 0.20)` 改为 `(0.001, 0.1)`；
v2/v3 默认都没有 bias、drift、delay、outage；
`p_nav = eta[:, :3] + zero_mean_nav_noise`，每步噪声幅度 `0.001–0.1 m`。
`p_nav_truth_lock` 保留为 optional diagnostic，默认 `false`；只有显式打开时才
`p_nav := eta[:, :3]` 并执行 `1e-4 m` 断言。

| 项目 | 契约 |
|---|---|
| 默认 profile | 新协议 `unbiased_small_noise_v3`，`nav_noise_m=(0.001, 0.1)`，`current_est_tau_s=(0.05, 0.5)`；`unbiased_small_noise_v2` 仅历史兼容 |
| 默认 `p_nav` | `eta[:, :3] + zero_mean_nav_noise`；无 bias/drift/delay/outage |
| optional lock | `pc.p_nav_truth_lock=false` 默认；显式 `true` 时 `p_nav := eta[:, :3]`，`nav_age_s=0` |
| lock 断言 | 仅当 `p_nav_truth_lock=true` 时每步 `max(abs(p_nav - eta[:, :3])) <= 1e-4 m`；超限判 `P_NAV_TRUTH_LOCK_VIOLATION` |
| 语义标签 | 默认 `semantics=SYNTHETIC_UNBIASED_NAV`；lock 时 `semantics=ORACLE_NAVIGATION` |
| 终止影响 | Goal182 的 `termination.position_source=eta` 已使终点终止使用真值；默认低噪声 `p_nav` 不影响 termination。若改成 `legacy`/`p_nav`，termination 会读低噪声导航；只有 lock 时才会读真值。终点捕获仍只是描述性统计，不进入评价指标（见 §5.2） |
| 结果可比性 | 旧 `unbiased_noise_v1` 的 `nav_noise_m=0.05–0.20` 和 `unbiased_small_noise_v2`（tau `0.5–1.5 s`）结果标 `historical_only`；`realistic_calibrated` 仍是 stress profile，不与默认主表混比 |
| 部署诊断 | 默认 `success_pnav_tight` 是“低噪声 synthetic navigation”口径，不是真实部署；真实部署仍需 `realistic_calibrated` 或实机标定 |

**范围边界**

- 只改 `p_nav` 的噪声范围；current estimate、DVL、IMU、raw USBL、actuator 等仍按
  基础 `unbiased_noise_v1` 的噪声参数运行，不得被这个决定静默改写；
- `p_nav` 是三维位置，所以默认低噪声也影响 actor 看到的 z / 深度；PC v1 没有独立
  pressure/depth 传感器通道，depth-limit termination 在 `pc.py` 里本来就直接读 true
  `eta[:, 2]`；
- `p_nav_truth_lock` 只用于 oracle diagnostic，不默认打开；打开时 raw USBL 的
  `usbl_age_s`、`usbl_valid`、`usbl_innovation` 仍可作 telemetry，但不得回灌
  `p_nav`；
- `0.001–0.1 m` 是 zero-mean noise，不是 bias，也不是对真实导航精度的声明；不得把
  默认低噪声结果写成“真实系统精度”或部署结论；
- 新增 `unbiased_small_noise_v3`，v2 与更早 profile 保持不变，避免追溯改变历史
  config。manifest 记录 `profile=unbiased_small_noise_v3`、`nav_noise_m`、
  `current_est_tau_s` 和 config SHA；v2 及 `unbiased_noise_v1` 0.05–0.20 结果标
  `historical_only`；
- C5/C6 的 short outage 由 curriculum episode override 提供，默认 profile 本身仍
  不含 outage；不得把 outage 写进全局 profile 或扩展到 C0–C4；
- `realistic_calibrated` 64-seed C3 的导航误差（mean 约 `0.61 m`、RMSE `0.68 m`、
  p95 `1.14–1.16 m`）保留为 stress / 历史证据；
- 迁移到真实部署或 sim-to-real 时，必须重新启用 `realistic_calibrated` 或标定后的
  等价导航模型并单独报告，不得用默认低噪声结果代替部署结论。

**其他传感器读取（默认低噪声和 optional lock 都不改变）**

`unbiased_small_noise_v3` 不是 `clean`：没有 bias、drift、delay，默认没有 outage，但保留每个控制步
的零均值测量噪声。`p_nav_truth_lock` 只替换 `p_nav`，下列通道仍按传感器链读取：

| 通道 | 读取来源 | `unbiased_small_noise_v3` 残余噪声 | 主要消费者 |
|---|---|---|---|
| `p_nav` | synthetic navigation | 默认 `0.001–0.1 m` zero-mean；lock 时 `0` | actor state、actor path、guidance 投影；若 termination 用 `p_nav` 也会读它 |
| `attitude` | IMU | roll/pitch `0.002–0.01 rad`，heading `0.003–0.015 rad` | actor state、actor path 的 heading/pitch error、guidance 的 current body→NED |
| `angular_rate` | gyro | `0.001–0.005 rad/s` | actor state |
| `dvl_ground_body` | bottom DVL | `0.005–0.02 m/s` | actor state（`speed_feedback=ground`）、guidance route-speed floor |
| `dvl_water_body` | water DVL | `0.005–0.02 m/s` | 当前 actor state 不直接消费；保留诊断/未来实验 |
| `current_estimate_body` | current estimator LPF | `0.005–0.02 m/s`，`tau=0.05–0.5 s`；C5/C6 另有 `0.05–0.3 s` short outage | guidance 的 ALOS current compensation；actor 仅在 current encoder 启用时消费 |
| `usbl_position` / raw USBL | USBL | `0.1–0.5 m` | 只作融合 innovation / telemetry；默认低噪声不改变 raw USBL，lock 下也不得回灌 `p_nav` |
| `critic_state` | true `eta`、true `nu`、command error | 无 | critic |
| `critic_path` 30D | true `eta` route、actuator、true current、sensor age/validity | sensor age/validity 仍来自仿真；`nav_innovation` 默认非零（≤`0.1 m`），lock 下为 0 | critic |
| depth / roll / pitch termination | `pc.py` 直接读 true `eta`/`nu` | 无 | termination |

因此：

- actor 仍然会看到带噪声的 attitude、gyro 和 DVL；actor path 的位置误差是低噪声
  `p_nav`（lock 时才是 `eta`）相对路线的误差，但 heading/pitch error 仍受 IMU
  噪声影响；
- guidance 的 current compensation 和 route-speed floor 仍消费带噪声的 current
  estimate / DVL；
- critic 仍然读 true `eta`/`nu`/current 和 privileged 30D；默认 `nav_innovation`
  非零，只有 optional lock 打开时才为 0；
- 若要把 actor 的所有传感器都切成真值，应显式使用
  `observation.ideal_actor_observations=true`（actor 直接读 true `eta`/`nu`/current）；
  若要把 synthetic sensor 链全部切成无噪声，应显式使用 `sensors.profile=clean`。
  这两个都是独立开关，不得被 `p_nav_truth_lock` 静默打开。

#### 4.12.10 Episode 初始状态契约（2026-11-16 已确认）

本节只定义每个 episode reset 时的物理初态和策略历史初态，不改变 current、noise、
plant 压力轴，也不改变奖励或 termination。参考工程的可用思路是：位置/速度/姿态只做
可控的小扰动，机器人执行器从一致的初始条件开始；路径任务不能照搬“原点 + 随机
`yaw ±1 rad`”，因为那会把起点恢复能力混进路径跟踪主指标。

| 项目 | 训练 reset | 正式评价 reset |
|---|---|---|
| 路径进度 | `start_progress_m=0`，从路线起点开始 | 同左；禁止 evaluation override 改进度 |
| 位置抖动 | `x/y ±0.5 m`，`z ±0.5 m`，各轴独立均匀 | 全部为 0，精确落在路线起点 |
| roll | 固定 0 | 固定 0 |
| pitch | 路线切向 pitch `±10°` | 精确等于路线切向 pitch |
| yaw | 路线切向 yaw `±10°` | 精确等于路线切向 yaw |
| 初始前向速度 | 本 episode 命令速度的 `0.8–1.2×` | 本 episode 命令速度的 `1.0×` |
| 初始侧向/垂向速度 | 0 | 0 |
| 初始三轴角速度 | `roll/pitch/yaw rate` 各 `±0.05 rad/s` | 0 |
| 初始推进器 | RPM 比例仍随机 `U(0,1)`；基准 RPM 按“初始前向速度 × speed feedforward”计算 | RPM 比例固定 1.0，基准仍按命令速度前馈计算 |
| PPO 上一动作 | 清零 | 清零 |
| 动作平滑历史 | 清零；前两步不计算二阶动作平滑惩罚，避免把初始物理前馈误判成 policy 跳变 | 同左 |
| 策略历史缓存 | 用 reset 后第一帧 token 填满整个 history window，禁止补零 | 同左 |
| current 估计器 | 第一条有效 current 测量到达时，用它初始化低通状态和历史队列，不从零爬升；禁止读真值 | 同左 |

**“推进器前馈按初始速度设置”的含义**：不是把 RPM 固定死，而是先按当前 episode 的
初始前向速度计算稳态前馈 RPM，再乘以训练时的随机比例 `U(0,1)`。因此同一个比例在
不同速度下对应的绝对 RPM 不同，但执行器尺度一致。正式评价取比例 1.0。

**“current 估计热启动”的含义**：reset 后传感器并非立刻有有效 current 测量；一旦
第一条有效测量出现，就把低通状态和历史队列直接置为该测量值，而不是从零慢慢爬升。
它只使用传感器测量，不使用 true current；真正的传感器噪声、延迟和失效压力仍保留。

训练与评价的边界固定为：

- 训练 reset 使用上表随机初态；正式 `MAIN_RANKING` / `ROUTE_GENERALIZATION` /
  `SPEED_ENVELOPE` / `DEPTH_GENERALIZATION` reset 使用无扰动名义初态；
- “名义初态”只去掉初始位姿、速度和执行器随机；`pressure_profile` 的 current/noise/
  plant 仍然生效，不得把正式评价误写成无海流、无噪声；
- 大起点误差（例如 `x/y ±1–2 m`、`yaw ±20–30°`）以后只作为独立
  `START_STRESS` 诊断，不进入首轮主排名，也不能反向挑 winner；
- manifest 必须记录 `start_jitter_xy_m`、`start_jitter_z_m`、`start_jitter_yaw_rad`、
  `initial_speed_ratio_range`、`initial_propeller_rpm_ratio_range` 和
  `initial_angular_rate_range`；正式评价若出现非零起点抖动判
  `INVALID_EVAL_START_CONTRACT`；
- 当前 config 的 `yaw ±5°`、speed `0.75–1.25×` 和未显式记录的角速度初态都属于
  `historical_only`，实现本契约前不得把它们当作新协议结果。

### 4.13 核心实验矩阵 `G182-CORE-COMP`（训练组合，顶级要求）

§4.12 的评价矩阵回答“一个策略怎么评”；本节回答“到底训练哪些策略组合”。两者不得
混表：`stage / route / pressure / radius` 只能出现在评价矩阵，不能进入核心 arm 的
身份定义。

#### 4.13.1 名称与文件

**状态：M1 + CM1 已于 2026-11-16 确认。** `G182-CORE-COMP` 是训练组合的
family/human index，不再冒充唯一可注册 matrix。实际训练矩阵按 program group 拆成
PE2E 与 ALOSB 两张，避免同一路径被 `auv_control_programs.json` 的两个 group 重复登记。

```text
human_id:            G182-CORE-COMP
index_file:          configs/governance/goal182_core_composition_index.json
index_schema:        auv_core_composition_index.v1
program_matrix_pe2e: configs/governance/goal182_core_pe2e_matrix.json
program_matrix_alosb: configs/governance/goal182_core_alosb_matrix.json
matrix_schema:       auv_core_composition_matrix.v1
axis_file:           configs/governance/goal182_axis_registry.json
```

文件职责固定如下：

| 文件 | program group | `control_contract` | checker 登记 | 作用 |
|---|---|---|---|---|
| `goal182_core_composition_index.json` | 无 | 两者索引 | 否 | 人读/机器汇总索引；只链接两张 program matrix、axis registry 和评价矩阵，不承载 arm |
| `goal182_core_pe2e_matrix.json` | `G-PATH-CONDITIONED-POLICY` | `CC_PE2E` | 是，且只登记一次 | 路径条件端到端训练 arm |
| `goal182_core_alosb_matrix.json` | `G-ALOS-BODY-RL` | `CC_ALOS_BODY` | 是，且只登记一次 | 外部 ALOS + 本体学习 arm |
| `goal182_axis_registry.json` | 无 | 两者共享 | 否 | 核心轴、档位、合法性、legacy alias 的唯一登记处 |

评价矩阵不属于 CM1 的落盘对象；它由 M2/EV9 单独冻结。评价矩阵可以同时包含两类
`control_contract`，但每个 cell/report row 必须带 `program_group_id` 和
`control_contract`，禁止把 PE2E 与 ALOSB 先 pooled 再排名或汇报。

所有新 arm 的 canonical ID 统一使用：

```text
G182_<PHASE>_<DOMAIN>_<TREATMENT>
```

其中：

```text
PHASE        = P0 | P1 | P2 | P3 | P4 | P5 | P6 | ALOS | BC | INT
DOMAIN       = AUTH | CHAN | NORM | TR | EXPL | CRIT | CURR | REW | ROUTE | STACK
TREATMENT    = [A-Z0-9]+(_[A-Z0-9]+)*
```

核心组合臂固定放在 `INT/STACK`，treatment 按四个核心轴展开：

```text
G182_INT_STACK_<ACTOR>_<CONTROL>_<LEARN>_<MODULE>

ACTOR   = MLP | TA
CONTROL = PE2E | ALOSB
LEARN   = BCFT | BCRES | TRRES_CMD | TRRES_ACT
MODULE  = MOD0 | MOD1 | MOD2 | MOD3 | MOD4
```

首批优先 `MOD4 × MLP` 的 6 个 canonical arm 固定为：

```text
G182_INT_STACK_MLP_PE2E_BCFT_MOD4
G182_INT_STACK_MLP_PE2E_BCRES_MOD4
G182_INT_STACK_MLP_ALOSB_BCFT_MOD4
G182_INT_STACK_MLP_ALOSB_BCRES_MOD4
G182_INT_STACK_MLP_ALOSB_TRRES_CMD_MOD4
G182_INT_STACK_MLP_ALOSB_TRRES_ACT_MOD4
```

`MOD4` 改变 actor 输入，启动前必须先完成兼容的 current-encoder BC；紧接同 family 的
6 个 `MOD3` arm 作为配对对照。随后跑 `MOD2 × MLP` 的 6 个 arm，并紧接 `MOD1`
对照。`MOD0` 是最低优先级 no-history baseline，不再作为首波主 treatment。
Transformer 后置。

草稿 registry 中曾出现的 `G182C_<ARCH>_<CONTROL>_<LEARN>_<MODULE>` 模板退役；它只能
作为 `legacy_alias` 出现在历史记录中，不得再用于新 arm、run 目录或结果表。

两张 program matrix 的 `arm_id` 必须满足上述语法，并至少记录：

```text
schema_version:      auv_core_composition_matrix.v1
goal_id:             Goal182
matrix_family_id:    G182-CORE-COMP
matrix_id:           G182-CORE-PE2E | G182-CORE-ALOSB
program_group_id:    G-PATH-CONDITIONED-POLICY | G-ALOS-BODY-RL
control_contract:    CC_PE2E | CC_ALOS_BODY
matrix_status:       DRAFT | FROZEN | RETIRED
axis_registry:       configs/governance/goal182_axis_registry.json
axis_registry_sha256: <生成 matrix 时计算>
training_profile:    PPO-FLAT-4096x48-E5-MB4
cross_program_pooling: false
```

`matrix_id` 在一张 program matrix 内唯一；同一个 `arm_id` 只能属于一个 program
matrix。索引文件不得被登记为 program matrix，也不得被用作 checker 的 canonical
aggregate。matrix 生成后先保持 `DRAFT`，只有 §4.12.8、§4.13.7 全部确认且 checker
通过后才能改为 `FROZEN` 并写 SHA-256。

核心矩阵的基础设计空间写成：

```text
2 actor_backend × 2 control_contract × 3 learning_composition × N module_set
= 12N defined combinations
```

残差接口不是任意第五个乘法因子，而是只展开 residual learning composition 的条件轴。
按当前实现，合法臂数为 `2 actor × (PE2E 2 + ALOSB 4) × N = 12N`：

- `PE2E`：`BCFT`、`BCRES` 两个合法组合；
- `ALOSB`：`BCFT`、`BCRES`、`TRRES_CMD`、`TRRES_ACT` 四个合法组合；
- `N=5` 时完整合法设计空间为 `60` 个 arm；`MS_FULL` 下两个 actor backend
  共 `12` 个可运行 arm（含 Transformer）。首批只跑其中 `AB_MLP_M` 的 `6` 个，
  即 priority treatment #1 `MOD4`；先完成 current-encoder BC，Transformer、其余
  module set 与 `MOD0` baseline 后置。

其中 `N` 不是随手勾选模块的开关数，而是预注册的 `module_set` 档位数。全笛卡尔积是
设计空间，不是首批训练清单。

#### 4.13.2 四个核心轴与残差条件轴

| 轴 ID | 档位 ID | 短码 | 含义 |
|---|---|---|---|
| `AX_ACTOR_BACKEND` | `AB_MLP_M` | `MLP` | M 档 MLP-family actor；`mlp` 或 `token_mlp`，无 actor 级 attention |
| `AX_ACTOR_BACKEND` | `AB_TA_M` | `TA` | M 档 Transformer actor，按 Goal181 的 token/capacity 契约 |
| `AX_CONTROL_CONTRACT` | `CC_PE2E` | `PE2E` | 路径条件端到端；策略内部隐式包含制导；无外部 ALOS |
| `AX_CONTROL_CONTRACT` | `CC_ALOS_BODY` | `ALOSB` | 外部 ALOS 负责制导；学习器负责本体控制或残差 |
| `AX_LEARNING_COMPOSITION` | `LC_BC_FINETUNE` | `BCFT` | BC actor warm start，随后全量 PPO 微调 |
| `AX_LEARNING_COMPOSITION` | `LC_BC_RESIDUAL` | `BCRES` | 冻结 BC actor，PPO 只训练残差 |
| `AX_LEARNING_COMPOSITION` | `LC_TRAD_RESIDUAL` | `TRRES` | 无 BC；传统 ALOS/PD-PI base + PPO 残差；展开为 `TRRES_CMD/TRRES_ACT` |
| `AX_MODULE_SET` | `MS_NONE` | `MOD0` | 不额外启用时序、current 或 privileged-critic 模块 |
| `AX_MODULE_SET` | `MS_TEMP` | `MOD1` | 时序/历史输入模块（DADE/dual-horizon） |
| `AX_MODULE_SET` | `MS_TEMP_PRIV` | `MOD2` | `MOD1` + privileged critic |
| `AX_MODULE_SET` | `MS_TEMP_CURRENT` | `MOD3` | `MOD1` + 直接 current side-channel encoder |
| `AX_MODULE_SET` | `MS_FULL` | `MOD4` | `MOD1` + 直接 current encoder + privileged critic |

`MOD0–MOD4` 只是 module-set 短码，不是 actor/critic 容量档位；容量档位仍写 `M` 档。
`DADE/dual-horizon` 属于 `MOD1` 的 history 实现，`MOD3` 不再重复开启 DADE；它额外
开启的是 `actor_current_encoder_enabled`，把 estimated current 直接编码后加到 `z_t`。

**术语边界（2026-11-16）**：这里的 `history` 指固定窗口观测布局
`[current, short=10, long=50]`，不是某个网络架构本身。当前 `MOD1–MOD4` 的 history
统一由 `DualHorizonEncoder` 消费，canonical `encoder.kind=dual_horizon_transformer`，
也就是 DADE；`MOD1` 只是相对 `MOD0` 打开 history 的对照。代码还支持
`dual_horizon_mlp` 和旧的显式 `history_mlp` 历史栈，但二者都不是 GRU/LSTM。
PC PPO 当前 `is_recurrent=False`；GRU 只存在于旧 DDPG/BC 栈，LSTM 尚未实现。
若以后比较 GRU/LSTM，必须作为独立的 recurrent temporal backend/actor 轴，处理
per-env hidden state、episode reset、sequence minibatch 和 checkpoint metadata，
不能直接塞进 `MOD*`。

DADE 的序列输入只有 `short` 与 `long` 两个窗口；`current` 不会被额外拼成第三个
token。canonical `dual_horizon_transformer` 使用 mean pooling，`current` 参数会被忽略；
attention-pooling 变体只用它作为 query。短窗的最后一帧已经是当前观测，而独立的
`current` 仍通过 actor 的 `z_t` 路径输入，因此它不是“current 序列”，也不是第三组
history token。

**History token 来源（2026-11-16）**：`BatchTorchPCVecEnv._tokens` 生成的 actor token
是 `[normalized actor_state, optional normalized path, side_channel]`。当前
`dade_history_mode=encoded` 时 `history_path_dim=path_dim=5`，所以 path 会进入每个
历史 token；`side_channel` 只有在 `actor_current_encoder_enabled=true`（`MOD3/MOD4`）
时才是 normalized estimated current，否则显式置零，避免海流估计从 history 泄漏。
command 不进入 history token，只作为顶层 `c_t` 输入。critic token 则是
`[normalized critic_state, normalized true current NED]`。每步 token 进入 rolling
buffer；episode reset 用 post-reset token 填满 buffer；短窗取最新 10 帧，长窗取最新
50 帧，stride 都为 1，因此短窗是长窗最近的 10 帧子集。

**CM4 已确认（2026-11-16）：固定 `MOD0–MOD4` 五档，`N=5`。** privileged critic
的正式开关是 `pc.critic_path_dim`：`0` 为关，`30` 为开；`training.privileged_current_critic`
不是 PC 策略的 module-set 开关，不能用来推断 MOD 档位。D7 的“一步到位”表示
privileged critic 一旦开启，`route_truth / actuator / disturbance / sensor_safety`
四组字段必须全开，不允许只开一半；它不表示每个 MOD 都必须强制开启 privileged critic。
因此 `MOD0/MOD1` 使用 `critic_path_dim=0`，`MOD2/MOD4` 使用 `critic_path_dim=30`。

**module-set 执行优先级（用户 2026-11-16 最终确认）：**privileged treatment 的顺序是
`MOD4` 先于 `MOD2`；不得脱离配对对照：

```text
MOD4（priority treatment #1） -> MOD3（同 current encoder 的 no-privilege control）
MOD2（priority treatment #2） -> MOD1（同 history 的 no-privilege control）
MOD0（最低优先级 no-history baseline）
```

`MOD4` 改变 actor 输入，必须先完成兼容的 current-encoder BC；因此 `BC-CURRENT-PREP`
是首波前置任务，而不是第二波工作。`MOD2` 不改变 actor 输入，可共用 `MOD1` 的 history
BC，但排在 `MOD4/MOD3` 之后。`MOD0` 需要独立的 no-history BC，放在最低优先级。
Transformer 仍然后置，不进入这一轮。

**CM2 已确认（2026-11-16）：`AX_ACTOR_BACKEND` 只保留 `AB_MLP_M / AB_TA_M` 两档；
`token_mlp` 不新增为第三档，而是 `AB_MLP_M` 内由 `actor_kind` 标识的公平对照。**

| backend | 允许的 `actor_kind` | 输入契约 | 作用 |
|---|---|---|---|
| `AB_MLP_M` | `mlp` | `fused` | 普通 M 档 MLP；作为 fused-token control |
| `AB_MLP_M` | `token_mlp` | `token_mean` | 与 Transformer 使用同一 token，仅把 attention 换成 mean-pool；公平对照 |
| `AB_TA_M` | `transformer` | `token_attention` | M 档 Transformer；token 间可交互 |

`actor_kind`、`actor_input_contract`、`token_contract` 和 `pooling` 必须显式写入
config、checkpoint metadata、matrix 和 run manifest。`token_mlp` 与 `transformer` 的
对照必须使用相同 token 集合、token 顺序、宽度和 pooling；否则判
`TOKENIZATION_CONFOUND`。普通 `mlp` 的 `fused` 输入只作 fused-token control，不能
冒充 `token_mlp` 去证明/否定 attention 收益。

**Actor metadata 缺失不再有隐式默认（2026-11-16 确认）**：新 Goal182 config、matrix
和 manifest 必须显式记录 `actor_kind`、`actor_input_contract`、`token_contract` 和
`pooling`；缺失任一项判 `ACTOR_METADATA_MISSING`，不得按普通 MLP 静默补默认值。
第一波正式基线使用 `AB_MLP_M + actor_kind=mlp + actor_input_contract=fused`；
`token_mlp` 只作为单独登记的 token-mean 对照，不能复用 `mlp` 的 arm_id。历史 config
或 checkpoint 没有该字段时统一标 `historical_only`，不追溯改写，也不进入新正式主表。

现有 `encoder.kind=dual_horizon_transformer` 只表示历史编码器，不能当作
`AB_TA_M`；DADE/dual-horizon 属于 module set，不是 actor backend。Goal182 首轮只
启用 `M` 档；S/L 档不进入 active core。M 档 MLP actor 与 critic 都是
`[512,256,128]`；M 档 Transformer actor 使用 Goal181 的
`d_model=112, 2 layers, 4 heads, FFN=224`，critic 仍为 `[512,256,128]`。配对 MLP
与 Transformer 的参数量和 actor forward p95 必须各在 `±20%` 内，否则判
`CAPACITY_MISMATCH` 或 `ACTOR_DEPLOYMENT_BUDGET_FAIL`。

`CC_PE2E` 的首选运行配置是 `pc_control_mode=path_fixed_speed`，`actor_command_dim=0`，
外部 ALOS 不进入 canonical runtime；`path_learned_speed` 只作次要速度子实验。
`CC_ALOS_BODY` 使用 `pc_control_mode=alos`，外部 ALOS 提供制导命令，actor 不能偷偷
替换成路径条件端到端策略。

`LC_BC_FINETUNE` 当前代码用 `base_policy.mode=none` 加 `--warmstart-actor` 实现；该
warmstart checkpoint、SHA 和可训练参数范围必须写进 matrix，不能只藏在启动命令行。
`LC_BC_RESIDUAL` 使用 `base_policy.mode=frozen_pc_bc_residual`，当前短码 `BCRES`
就是执行器残差版本。`LC_TRAD_RESIDUAL` 没有 BC 底座，使用传统 ALOS/PD-PI base
加学习残差，并按 `AX_RESIDUAL_INTERFACE` 展开为命令残差和执行器残差。

**TRRES encoder 初始化已确认（2026-11-16）：正式 TRRES 只复用编码器架构，不借用
任何已训练权重。** 同一 `(actor_backend, module_set)` 的 TRRES arm 可以复用
`actor_encoder`、DADE/dual-horizon、current encoder 的类型和隐藏层配置，但不得从
BC checkpoint、其他 TRRES arm 或任何已完成 arm 加载编码器权重。正式 TRRES run 必须
使用 `base_policy.mode=none`、无 `--warmstart-actor`、fresh optimizer、iteration 0
和独立 fresh initialization；manifest 必须记录
`encoder_init=standard_fresh`、`encoder_init_checkpoint=null`、`encoder_seed=<train seed>`
以及编码器 schema/architecture SHA。`MOD4/MOD3` 配对只能共享架构、seed 和输入
schema，不能通过加载外部权重实现配对。若未来要检验 BC encoder 初始化收益，必须
另立预注册诊断族 `TRRES_BCENC_INIT_CMD/ACT`，不得进入 `G182-CORE-COMP`，也不得
把该诊断结果写成正式 `TRRES_CMD/ACT` 结果。

**CM3 已确认（2026-11-16）：正式 learning composition 固定为
`BCFT / BCRES / TRRES` 三个 family。** 每个 `actor_backend × module_set` 的正式
arm 数固定为 6：PE2E 为 `BCFT / BCRES`，ALOSB 为
`BCFT / BCRES / TRRES_CMD / TRRES_ACT`。`TRRES` 只是 family 别名，不能生成无接口
后缀的 arm。完整 PPO 微调 `BCFT` 始终是必做组，不因 `BCRES` 或 `TRRES` 成功而取消。

`SCRATCH` 只保留为 `optional_diagnostic`：无 BC、无传统 base、直接从头 PPO。它不进入
`G182-CORE-COMP`、不进入第一波、不计入 60 个正式组合、不参加 winner/前三名/
`ROUTE_GENERALIZATION`，也不能替代 `BCFT`。只有主实验无法判断 BC/base 的必要性时，才另立
diagnostic matrix；其 run 必须标 `NOT_FORMAL`。

#### 4.13.3 残差接口条件轴与动作接口

残差组新增条件轴 `AX_RESIDUAL_INTERFACE`。它只在 `learning_composition` 为残差组时
生效；`BCFT` 仍直接输出绝对动作，不乘这条轴。

| 条件轴档位 | 动作接口 | 语义 | 当前合法组合 |
|---|---|---|---|
| `RI_ACT` | `AI_ACT_RES` | 在 BC/传统执行器动作上叠加残差 | `BCRES`；`ALOSB × TRRES_ACT` |
| `RI_CMD` | `AI_CMD_RES` | 在 ALOS heading/pitch/speed 指令上叠加残差 | `ALOSB × TRRES_CMD` |

合法性约束：

- `PE2E × RI_CMD` 非法：没有外部 ALOS 指令层可叠残差；
- `BCRES × RI_CMD` 暂不进入矩阵：当前 `frozen_pc_bc_residual` 只支持绝对或执行器
  residual，尚没有命令输出的 BC 底座；若以后要做，必须新增命令输出 BC checkpoint
  和单独 arm 族，不能把现有 `BCRES` 改个名字冒充；
- `ALOSB × TRRES` 的两个接口都是正式 arm，分别命名 `TRRES_CMD` 与
  `TRRES_ACT`；`TRRES` 只作 family 别名，不得生成无接口后缀的歧义 arm；
- `BCRES` 默认就是执行器残差；以后若实现命令残差，短码使用 `BCRES_CMD`。

| 动作接口 ID | 含义 | 当前典型组合 |
|---|---|---|
| `AI_ABS` | actor 直接输出归一化执行器动作 | `BCFT` |
| `AI_ACT_RES` | 在传统/BC 执行器动作上叠加残差 | `BCRES`、`TRRES_ACT` |
| `AI_CMD_RES` | 在 ALOS heading/pitch/speed 指令上叠加残差 | `TRRES_CMD` |
| `AI_GAIN_RES` | 修改传统控制器增益 | 诊断；当前 PPO 未接线 |
| `AI_RESOLVED_ENV` | 由 `action_interface=environment` 解析出的实际接口 | 只能出现在 resolved metadata |

`AX_ACTION_INTERFACE` 必须由 `control_contract × learning_composition ×
AX_RESIDUAL_INTERFACE` 推导并写入 manifest；它不能成为隐藏的自由度。
`CC_PE2E × LC_TRAD_RESIDUAL` 没有传统制导 base，因此仍是 `N/A`，不得为了凑满
2×2×3 强行生成 arm。

#### 4.13.4 合法组合、arm 命名与数量

核心 arm 使用 §4.12.1 的统一命名空间，并由 `INT/STACK` 标识其训练组合身份：

```text
G182_INT_STACK_<ACTOR>_<CONTROL>_<LEARN>_<MODULE>
```

示例：`G182_INT_STACK_TA_ALOSB_BCRES_MOD1` 表示 Transformer actor、ALOS+本体控制、
冻结 BC 残差、启用时序模块；`G182_INT_STACK_TA_ALOSB_TRRES_CMD_MOD1` 表示传统
ALOS + 命令层残差。完整的 `run_id` 和 `eval_cell` 继续使用 §4.12.1 的格式。

每个 `module_set` 的合法基础组合如下：

| control | learning | 合法 | 动作接口 | 说明 |
|---|---|---|---|---|
| `PE2E` | `BCFT` | 是 | `AI_ABS` | 路径条件 BC 全量微调 |
| `PE2E` | `BCRES` | 是 | `AI_ACT_RES` | 冻结路径条件 BC + 残差 |
| `PE2E` | `TRRES` | 否 | `N/A` | 没有传统制导 base |
| `ALOSB` | `BCFT` | 是 | `AI_ABS` | ALOS 输入下的 BC 全量微调 |
| `ALOSB` | `BCRES` | 是 | `AI_ACT_RES` | 冻结 ALOS-BC + 残差 |
| `ALOSB` | `TRRES_CMD` | 是 | `AI_CMD_RES` | 传统 ALOS + 命令层学习残差 |
| `ALOSB` | `TRRES_ACT` | 是 | `AI_ACT_RES` | 传统 ALOS + 执行器学习残差 |

因此每个 `module_set` 有 `2 actor × 6 合法学习组合 = 12` 个 arm；`N=5` 时完整
合法设计空间是 60 个 arm，但不应一次性全部启动。

#### 4.13.5 课程、预算与评价协议（不进入核心轴）

课程学习不作为核心矩阵的第五个乘法轴；它是所有核心 arm 必须共用的训练协议。

- `FORMAL` 核心比较必须使用同一套完整 C0→C6 课程、同一 `training_profile`、同一
  iteration 预算和同一 held-out 关闭规则；
- 所有新训练统一使用 `SPEED_V3_CENTER_OUT_5S` 速度课程：C0 从
  `1.40–1.60 m/s` 起步，按 §4.12.7.2 交替向两侧扩档到 C6 的 `1.00–2.50 m/s`，
  episode 内每 `5 s` 重采样一次；正式评价固定 route speed，不重采样。旧 v1/v2 速度
  协议结果一律 `historical_only`，不得与新协议直接比较；
- 固定 C3 的 600-iteration run 只能标 `C3_FIXED_STAGE / NOT_FORMAL`，不能用于比较
  actor、控制契约或 BC 策略；
- `QUICK_SCREEN` 可用 1 train seed、5 iterations、固定 stage，只用于排除实现错误和明显
  无信号方向；C3 必须作为初评 stage，并跑 §4.12.4.1 的四档固定速度子表；
  C3 初评标 `QUICK_SCREEN / NOT_FORMAL`，不写 formal winner；
- `MAIN_RANKING` 使用 §4.12 的 `3 train seeds × 4 pressures × 1 stage-matched
  route × 1 radius × 3 eval seeds`，并评 `C2/C3/C4/C6`；每个 arm 为
  `48 cells / 144 episodes`。核心 arm 的身份不包含 pressure/route/radius；
- `SPEED_ENVELOPE` 对所有正式 arm 使用 `C2/C3/C4/C6 × 四档固定速度 ×
  nominal × 3 eval seeds`；每个 arm 为 `48 cells / 144 episodes`，
  只写速度包线结论，不替代 `MAIN_RANKING`；
- 完整 C0→C6 正式训练前，必须先修复当前 `curriculum.py` 的 transition hard cap
  与 iteration 预算不一致问题；否则只能跑固定 stage 的诊断。

建议执行阶梯：

0. `BC-CURRENT-PREP`：先训练并冻结与 `MS_FULL/MOD4`、`MS_TEMP_CURRENT/MOD3`
   兼容的 current-encoder BC；没有这份 BC，`MOD4` 不得启动。这里不是第二波，
   而是当前首波的前置；
1. `CORE-SCREEN-PRIV-FULL-A`：current-encoder BC 通过后，优先跑
   `MS_FULL/MOD4 × AB_MLP_M` 的 **6 个 MLP 合法 arm**（`PE2E` 的
   `BCFT/BCRES`，`ALOSB` 的 `BCFT/BCRES/TRRES_CMD/TRRES_ACT`），1 seed、
   5 iterations，确认 privileged critic、current encoder 与基础轴没有实现混因；
   不跑 Transformer；
2. `CORE-SCREEN-PRIV-FULL-B`：紧接 `MOD4` 跑同 family 的
   `MS_TEMP_CURRENT/MOD3` 6 个对照 arm；`MOD4` 与 `MOD3` 使用同一
   current-encoder BC，只改变 `critic_path_dim=30/0`；
3. `CORE-SCREEN-PRIV-TEMP-A`：再跑 `MS_TEMP_PRIV/MOD2 × AB_MLP_M` 的 6 个 MLP
   arm；`MOD2` 与 `MOD4` 的差异是 actor 不使用 current encoder，但 privileged
   critic 仍开启；
4. `CORE-SCREEN-PRIV-TEMP-B`：紧接 `MOD2` 跑同 family 的 `MS_TEMP/MOD1` 6 个
   对照 arm；`MOD2` 与 `MOD1` 使用同一 history BC，只改变
   `critic_path_dim=30/0`。每个 module-set pair 最多 12 个 MLP arm，两对合计最多
   24 个；不得为了凑满数量而保留失败 family；
5. `CORE-SCREEN-BASE`：`MS_NONE/MOD0` 6 个 arm 作最低优先级 no-history baseline；
   `AB_TA_M` 等 Transformer BC 就绪后另起一波；
6. `CORE-FORMAL`：只让通过 screen 的少数 arm 进入 3 seeds、完整 C0→C6 和
   `MAIN_RANKING` 评价；
7. `SPEED_ENVELOPE`：对所有通过 screen 的正式 arm 跑跨 stage 四档固定速度包线
   （`nominal` 单压力）；按 stage 和速度分开报告；
8. `MAIN_RANKING` 前三名进入路线泛化测评 `ROUTE_GENERALIZATION`：C6 + 22 条路线
   （19 条二维 + 3 条三维），`3 train × 4 pressure × 22 route × 3 eval =
   264 cells / 792 episodes per arm`；前三名合计 `2,376 episodes`。绝对深度按
   EV4-A 单独做 `DEPTH_GENERALIZATION`：四个深度、两条代表路线、`nominal` 压力，
   `4 × 2 × 1 × 3 × 3 = 72 episodes per arm`；不乘进上面的 22-route 集合。

#### 4.13.6 当前代码对照与缺口

- `AX_ACTOR_BACKEND`：**schema 和 MLP/token 契约已实现**。`actor_kind=mlp|token_mlp|
  transformer`、`actor_input_contract`、token contract、pooling 已写入 config /
  checkpoint / manifest / matrix，并由 parser、launcher 和 checker 强制；`AB_TA_M`
  的 Transformer 实现与 M 档容量配对仍后置。
- `AX_CONTROL_CONTRACT`：**已实现**。`pc_control_mode` 已有 `alos`、
  `path_fixed_speed`、`path_learned_speed`。
- `LC_BC_FINETUNE`：**已实现**。`warmstart_actor`、SHA 和 `trainable_scope` 已是
  config/manifest 一等字段；BCFT 强制 `base_policy.mode=none`。
- `LC_BC_RESIDUAL`：**已实现执行器残差契约**。BCRES 强制
  `frozen_pc_bc_residual` + checkpoint/SHA，并禁止额外的 warmstart；当前仍只支持
  `absolute/environment/actuator`，不支持命令层残差，`BCRES_CMD` 需另做。
- `LC_TRAD_RESIDUAL`：**代码组件存在但未接入 Goal182**。PPO 支持
  `BatchedPCCommandResidualEnv` 和 `BatchedPCActuatorResidualEnv`；
  `controller_gain` 残差目前主要在 SAC 路径，PPO 路径未接线。
- `AX_MODULE_SET`：**registry 已命名成组，生成器待展开**。history/dual-horizon
  encoder、DADE、actor current encoder、privileged critic 已有 `MOD0–MOD4` 映射；
  训练 config 生成器仍需按该映射展开。
- `FORMAL_C0_C6`：**已实现**。C0–C6 hard cap 已按 iteration 计；transition 只作
  遥测。

#### 4.13.7 待确认 CM1–CM9

- [x] CM1：采用 `G182-CORE-COMP`、核心 matrix schema 和显式文件拆分。
  **已确认（2026-11-16）**：`G182-CORE-COMP` 只作 family/human index；实际拆成
  `goal182_core_pe2e_matrix.json`（`G-PATH-CONDITIONED-POLICY`）与
  `goal182_core_alosb_matrix.json`（`G-ALOS-BODY-RL`），两者 schema 为
  `auv_core_composition_matrix.v1`；索引文件为
  `goal182_core_composition_index.json`，不登记到 program checker；详见 §4.13.1。
- [x] CM2：`AX_ACTOR_BACKEND` 只保留 `AB_MLP_M` / `AB_TA_M` 两档。
  **已确认（2026-11-16）**：`token_mlp` 不新增第三档，而是 `AB_MLP_M` 内由
  `actor_kind=token_mlp` 标识的公平对照；`actor_kind`、输入契约、token 契约和
  pooling 必须写入 config/checkpoint/matrix/manifest。`AB_TA_M` 必须是
  `actor_kind=transformer`，不得用历史 encoder 冒充；详见 §4.13.2。
- [x] CM3：`AX_LEARNING_COMPOSITION` 固定 `BCFT / BCRES / TRRES` 三个 family。
  **已确认（2026-11-16）**：`TRRES` 由 `AX_RESIDUAL_INTERFACE` 展开成
  `TRRES_CMD / TRRES_ACT` 两个正式 arm；每 `actor_backend × module_set` 共 6 个
  正式 arm。`SCRATCH` 只作非正式 optional diagnostic，不进核心矩阵、第一波和 60 个
  正式组合，不得替代 `BCFT`；详见 §4.13.2。
- [x] CM4：`AX_MODULE_SET` 固定 `MS_NONE / MS_TEMP / MS_TEMP_PRIV /
  MS_TEMP_CURRENT / MS_FULL` 五档，`N=5`。
  **已确认（2026-11-16）**：privileged critic 由 `pc.critic_path_dim=0/30` 控制，
  `MOD0/MOD1` 关闭、`MOD2/MOD4` 开启；D7 四组字段全开或全关。执行优先级为
  `MOD4 → MOD3 → MOD2 → MOD1 → MOD0`，但必须按配对对照执行；详见 §4.13.2。
- [x] CM5：每个首波 screen block 仍使用 6 个 MLP 合法 arm 的骨架，但 module-set
  顺序按 CM4 优先级调整，不一次启动 60 个 arm。
  **已确认并更新（2026-11-16）**：原 `MOD0-first` 顺序被用户“优先跑有 privileged
  critic 的 MOD 组”覆盖；随后用户进一步明确 privileged treatment 的内部顺序为
  `MOD4` 先于 `MOD2`。首批主 treatment 为 `MOD4 × AB_MLP_M` 的 6 个 arm，
  `MOD3 × AB_MLP_M` 的 6 个 arm 作为紧邻对照；随后跑 `MOD2 × AB_MLP_M` 与
  `MOD1` 对照，`MOD0` 作为最低优先级 no-history baseline。`MOD4` 前必须先完成
  compatible current-encoder BC。Transformer 仍后置；第一批不把模块轴和基础轴
  一起做全笛卡尔。
- [x] CM6：Transformer 是否必须覆盖三种 learning composition，还是只覆盖
  `BCRES` 和 `BCFT`，`TRRES` 只保留 MLP。
  **已确认覆盖全部（2026-11-16，见 MX5 A）**：`AB_TA_M` 覆盖
  `BCFT / BCRES / TRRES_CMD / TRRES_ACT`；其中 TA `BCFT` 必须有兼容 TA BC。
- [x] CM7：`ALOSB × TRRES` 的动作接口是 `AI_CMD_RES` 还是 `AI_ACT_RES`；
  另一个是否作为独立诊断，不与主 arm 混跑。
  **已确认两者都做（2026-11-16）**：`TRRES_CMD` 与 `TRRES_ACT` 是两个独立正式
  arm，共享同一训练 seed、课程、reward 和预算，只改变残差接口；见 MX6。
- [x] CM8：`BCFT` 与 `BCRES` 的 5D BC checkpoint 关系。**由 D16 与 5D BC canonical
  补关（2026-11-16）**：同一 `actor_backend × module_set` family 内，`BCFT` 与
  `BCRES` 共用同一 5D BC checkpoint、同一 SHA 和同一 path schema；不同 module set
  仍使用各自兼容的 BC，不得拿别的 family 的 BC 冒充。
- [x] CM9：正式核心比较范围。**由 D8/CU1 与 EV2 补关（2026-11-16）**：正式比较统一
  使用完整 C0→C6 训练；固定 C3 只保留为 `QUICK_SCREEN` / diagnostic，不进入核心
  winner 判定。

#### 4.13.8 待商讨清单（2026-11-16）

本节只记录“代码已核对、尚需用户拍板”的问题，不改变 §4.13.7 的 CM1–CM9，
也不把推荐项写成已确认结论。每项确认后再回写对应 CM、§5 或 §6。

**代码核对结果**

- `curriculum.py` 的 C0–C6 hard cap 仍是 transition 计数，合计约 `5.0M`
  transitions；canonical profile 每 iteration `196,608` samples，完整课程只有约
  `25.4` iterations。文档要求 `2600` iterations。见
  `src/auv_lowlevel/curriculum.py:490`、`src/auv_lowlevel/curriculum.py:980`。
- StageGate 仍按 `success / strict_success / timeout / plane_miss / safety` 判定；
  §4.9.1 已定的 route-centric gate 尚未实现。见
  `src/auv_lowlevel/curriculum_trainer.py:38`、`src/auv_lowlevel/curriculum.py:1010`。
- 现有 goal182 env 为 `max_episode_steps=2500`、`dt_s=0.02`，即 `50 s`；D8.1 写的是
  `3000` 步、`60 s`。见 `configs/lowlevel/goal182/env_path_fixed_speed_c3.json`。
- 代码中尚无 training-only route append、C6 early-stop 20/3、显式
  `route_profile` 评测入口；`depth_profiles_m` 只是声明，builder 未消费。
- 现有 goal182 训练配置默认 `dade_enabled=true` 且
  `encoder.kind=dual_horizon_transformer`，因此 `MS_NONE` 不是真正的“无时序模块”。
- 现有 goal182 训练配置是固定 C3、`promotion_enabled=false`，只能作 screen /
  diagnostic，不是正式 C0→C6。
- `goal182_core_composition_index.json`、两张 program core matrix 和
  `goal182_eval_matrix.json` 已生成 DRAFT 并通过 checker；Goal182 artifact auditor
  仍缺失，正式评价前必须补齐。
- evaluator summary 的 `success_rate` 仍是 p_nav tight；standing suite 的 primary 是
  `success_eta_tight_at_done`。§4.12.9 默认是低噪声 `unbiased_small_noise_v3`
  （`nav_noise_m=0.001–0.1 m`，`current_est_tau_s=0.05–0.5 s`），`p_nav` 不再等于 `eta`；只有 optional
  `p_nav_truth_lock=true` 时才需要 `1e-4 m` 断言。实现仍须把三口径对齐。见
  `src/auv_lowlevel/cli/evaluate_pc_ppo/summary.py:61`、
  `src/auv_lowlevel/outcome_contract.py:56`。
- program checker 不允许同一个 matrix 路径登记到两个 group。见
  `总协调/scripts/check_experiment_program.py:101`。
- §4.13.5 的旧“每个 arm 96 个 eval cells”表述已过时：按 EV1/EV2 现在是
  `4 stages × 12 cells = 48 cells / 144 episodes per arm`
  （每个 stage `3 train × 4 pressure × 1 route × 1 radius = 12 cells`，每个 cell 内含
  3 个 eval seed）。

**矩阵 MX1–MX8**

- [x] MX1：`G182-CORE-COMP` 同时覆盖 PE2E 与 ALOSB，但 checker 不允许一张 matrix
  挂两个 program group。A 拆成 `G182-CORE-PE2E` / `G182-CORE-ALOSB`，再用人读的
  cross-index 合并；B 新增 umbrella group `G-CORE-COMPOSITION`，允许两个 group 引用
  同一张表；C 放宽 checker 支持 `cross_program=true`。推荐 A，治理改动最小。
  **已选 A（2026-11-16）**：拆成 PE2E / ALOSB 两张 program-scoped matrix，
  `G182-CORE-COMP` 保留为人读索引；不修改 program checker。
- [x] MX2：`MS_NONE` 的定义与现有 DADE / dual-horizon history 冲突。A 把
  dual-horizon history 与 DADE 当作 PC 基础契约，模块轴改为
  `MS_NONE / MS_PRIV / MS_CURRENT / MS_CURRENT_PRIV`，删除重复的 `MS_TEMP`；B 实现
  single-frame MOD0、MOD1 再开 history，但 actor 输入与 BC checkpoint 都要重做；C 保留
  `MS_TEMP`，把 MOD0 定义为“只关额外模块，不关 history”。推荐 A，除非要把
  “history 开/关”作为独立科学问题。
  **已选 B（2026-11-16）**：`MS_NONE` 是真正的 single-frame / no-history，
  `MS_TEMP` 才开启 dual-horizon history；`MS_NONE` 还必须关闭 DADE/current encoder。
  因此现有带 history 的 5D BC checkpoint 不能直接用于 `MS_NONE`；MX3 必须为
  `MS_NONE` 单独准备 BC checkpoint，或规定 `MS_NONE` 只允许
  `TRRES_CMD / TRRES_ACT / SCRATCH`。
- [x] MX3：BC checkpoint 的兼容边界。当前 BC checkpoint 绑定现有 MLP head +
  dual-horizon encoder + DADE 栈，`AB_TA_M × BCFT/BCRES` 以及改变 encoder 的
  module set 不能直接共用。A 每个 `(actor_backend, module_set)` 各训一份兼容 BC；
  B BC 组只保留 MLP/base module，Transformer 与其他 module 只走
  `TRRES_CMD / TRRES_ACT / scratch`；
  C 先只做 MLP 的 BC 两组，Transformer 第二波再补。推荐 C。
  **已选 C（2026-11-16，MOD4 置顶后修订）**：由于 `MOD4` 是 priority treatment #1，
  第一波必须先准备 `MS_TEMP_CURRENT / MS_FULL` 的 current-encoder BC；它不再放到
  第二波，也不得用现有 history BC 冒充。`MS_TEMP_PRIV` 只改 critic，共用
  `MS_TEMP` 的 actor BC；`MS_TEMP` 的现有 history BC 供随后 `MOD2/MOD1` 使用。
  `MS_NONE` 的 no-history BC 与 `AB_TA_M` 的 Transformer BC 仍后置。
- [x] MX4：learning composition 的排期。A `BCRES` 与 `BCFT` 并行、`TRRES`
  作第三组；B 先 `BCRES`，`BCFT` 第二波；C 先 `BCRES + TRRES`，`BCFT` 第二波。
  推荐 A 如果有算力，否则 C；`SCRATCH` 只作 diagnostic，不进正式三档。
  **已选 A（2026-11-16，残差接口扩展后计数更新）**：`BCFT / BCRES /
  TRRES_CMD / TRRES_ACT` 全部进入首轮 MLP；但执行顺序按
  `CORE-SCREEN-PRIV-FULL-A/B` 与 `CORE-SCREEN-PRIV-TEMP-A/B` 拆开：先跑 `MOD4`
  的 6 个 MLP arm，再给同 family 增开 `MOD3` 对照；随后跑 `MOD2` 的 6 个 MLP
  arm，再增开 `MOD1` 对照。每个 module-set pair 最多 12 个 MLP arm（每个 module
  set 为 PE2E 的 `BCFT/BCRES`，ALOSB 的
  `BCFT/BCRES/TRRES_CMD/TRRES_ACT`，共 6 个），两对合计最多 24 个。
- [x] MX5：Transformer 覆盖范围。A 覆盖 `BCFT/BCRES/TRRES` 三组；B 只覆盖
  `BCRES/TRRES`；C 只覆盖 `BCRES`。推荐 B，前提是 MX3 不为 Transformer 单独训 BC。
  **已选 A（2026-11-16，残差接口扩展后计数更新）**：`AB_TA_M` 最终覆盖
  `BCFT / BCRES / TRRES_CMD / TRRES_ACT`。Transformer 波次之前只跑 MLP：先
  `MOD4/MOD3` 两组合计最多 12 个 MLP arm，再跑 `MOD2/MOD1` 两组合计最多 12 个；
  `MOD0` 最后作 no-history baseline。进入 Transformer 波次前，先为 actor-visible
  module set 训练并冻结兼容的 Transformer BC checkpoint。`BCFT` 与 `BCRES` 共用
  同一 `(actor_backend, control_contract, module_set)` 的 BC SHA；`MS_TEMP_PRIV`
  共用 `MS_TEMP` actor BC。若兼容 TA BC 尚未产出，此时不得用 MLP BC
  冒充 warmstart，也不得把 `TA × BCFT` 偷换成 scratch。
- [x] MX6：`ALOSB × TRRES` 接口。A `AI_CMD_RES`，在 ALOS heading/pitch/speed
  上叠残差；B `AI_ACT_RES`，在 rudder/stern/rpm 上叠残差。
  **已选“两者都做”（2026-11-16）**：不再二选一，`TRRES_CMD` 与 `TRRES_ACT`
  都是正式 arm，使用同一 train seed、课程、reward、预算和评价协议，只改变残差
  接口；结果按 arm 分开报告。当前 `BCRES` 只保留执行器残差，`BCRES_CMD` 等命令
  输出 BC 落地后再作为新 arm 族加入。
- [x] MX7：首批 arm 范围。**已确认并更新（2026-11-16）**：先跑 `MOD4 × AB_MLP_M`
  的 6 个合法 arm：PE2E 的 `BCFT/BCRES`，ALOSB 的
  `BCFT/BCRES/TRRES_CMD/TRRES_ACT`。这 6 个跑完后，立即给同 family 增开
  `MOD3` 对照；随后跑 `MOD2 × AB_MLP_M` 并增开 `MOD1` 对照。每个 module-set
  pair 最多 12 个 MLP arm，两对合计最多 24 个。`MOD4` 前必须先完成兼容的
  current-encoder BC；`MOD0` 最后作 no-history baseline。Transformer 不进入首批。
- [x] MX8：`M0–M4` 与“M 档容量”冲突。**已确认（2026-11-16）**：module-set 短码
  改为 `MOD0–MOD4`，档位 ID 保留 `MS_*`；actor/critic 容量档位继续写 `M` 档。
  axis registry、matrix、文档和 run 命名必须使用 `MOD*`，不得再用裸 `M0–M4`
  表示 module set。
- [x] MX9：TRRES encoder 权重复用边界。**已确认（2026-11-16）**：正式
  `TRRES_CMD/ACT` 只复用同一 `(actor_backend, module_set)` 的编码器架构和配置，
  不从 BC 或其他 arm 加载编码器权重；fresh initialization、fresh optimizer、
  iteration 0。`MOD4/MOD3` 配对共享架构、seed 和输入 schema，不共享训练权重。
  BC encoder 初始化若要做，只能另立预注册 `TRRES_BCENC_INIT_CMD/ACT` 诊断族，
  不进入正式矩阵。

**课程 CU1–CU8**

- [x] CU1：预算单位。A 直接实现 iteration caps
  `120/180/260/360/420/560/700`，合计 `2600`；B 换算为 transition caps（合计约
  `511M`）；C 保留现有 transition caps，但只有约 `25` iterations。推荐 A，
  transition 只作遥测与审计。
  **已确认 A（2026-11-16）**：参考工程
  `/home/bubble/港中深暑研/Wheel-Legged-Lab/STAGED_TRAINING.md` §4 与
  `scripts/rsl_rl/staged_training_config.json` 均使用 per-stage
  `min_iterations / max_iterations`，并统一用 `window=20`、`consecutive=3`；
  其 `flat` 阶段为 `min=600 / max=2600`。Goal182 §8.1 已冻结“单次 C0→C6 run、
  caps 合计 2600”的适配方案，不照搬参考工程的十个独立 stage 数字。当前唯一缺口是
  `curriculum.py` 仍按 transition 计 hard cap，需改为 iteration；transition 只保留
  为遥测与审计。
- [x] CU2：episode 长度。A 保持 `2500 / 50 s`；B 改为 `3000 / 60 s`；C 按是否启用
  续路二选一。推荐若 CU3 现在做则用 B，否则先用 A，但必须让 doc 与 config 单一来源。
  **已选 B（2026-11-16）**：Goal182 正式协议统一 `max_episode_steps=3000`、
  `dt_s=0.02`，即 `60 s`。现有 `2500 / 50 s` 的 600-it 结果保留为
  `historical_only`，不得与新 60 s 结果直接混算；所有新 config、matrix manifest 和
  evaluation metadata 必须记录 `episode_steps=3000` / `episode_seconds=60`。
- [x] CU3：续路现在做还是后做。A 现在实现 training-only append（同 family、最多
  3 次、末端连续、逐次 manifest）；B 正式 core 先固定路线，续路作为第二波独立因子；
  C 用现有 `start_progress_m/full_path_fraction` 近似，但那不是真 append。推荐 B
  先保进度；若“路线不断延长”是核心假设，则 A。
  **已选 B（2026-11-16）**：第一波固定路线，不做 append；`3000 / 60 s` 上限仍生效，
  但固定 35 m 路线通常会在到点后提前结束，timeout 继续只作诊断。续路作为第二波独立
  因子，禁止与第一波 actor/control/learning 轴混跑；评价路线始终固定。
- [x] CU4：`stage_mix`。当前 trainer 的 `preview_enabled=True`，会混入 5% 下一
  stage。A 保留现有 mix（含 preview）；B 当前 stage 100%；C 当前 + 上一 stage，
  不要下一 stage preview。推荐 A，并把 mix 写入 manifest。
  **已选 A（2026-11-16）**：保留 `CURRICULUM_STAGE_MIX_TARGETS` 现有混样和
  `preview_enabled=True`；不重新基线，只把预算从 transition 改成 iteration。
  每个 run 的 `curriculum_episodes.jsonl` / state 必须记录实际生效的 stage_mix。
- [x] CU5：PE2E 与 ALOSB 的 C0/C1 不可比。代码中
  `_curriculum_direct_mask = direct & (pc_control_mode == "alos")`，C0/C1 的 direct
  command 对 PE2E 是空操作。A 两种 control 用各自 warmup，正式比较从 C2 开始；
  B 两边都关 direct command，但 PE2E 仍不消费 command；C 坚持全 C0–C6 共用，但把
  C0/C1 标成 non-comparable。推荐 A，正式 winner 只看 C2–C6。
  **已选 A（2026-11-16）**：C0/C1 保留 control-specific warmup，标为
  `control_specific_warmup`，不进入 PE2E vs ALOSB 的正式比较；正式 stage sweep
  从 C2 开始，C0/C1 只作训练热身与诊断。
- [x] CU6：StageGate shadow 切换。口径已定，代码未实现。需要确认 shadow 至少覆盖
  一个完整 stage 边界 + 一次最终 formal；阈值用 P0 training rollout 的 20-iteration
  分布标定，frozen-dev 只在 stage 边界使用；继续不设 success floor。推荐按此执行。
  **已确认（2026-11-16）**：采用先 shadow 旁路观察、后 enforce；至少覆盖一个完整
  stage 边界 + 一次最终 formal 评价，阈值用 P0 training rollout 的 20-iteration
  分布标定，frozen-dev 只在 stage 边界使用，不设 success/到达终点硬门槛。shadow
  期间只记录 `route_gate_would_pass`、分量、阈值和失败原因，不改变晋级；阈值表冻结
  后再切换 enforce，具体切换 iteration/transition 仍待标定。
- [x] CU7：C6 early-stop 20/3 现在做吗。代码没有 early-stop。A 现在实现
  `min=600 / max=2600 / window=20 / consecutive=3`；B 先跑到 2600，第二波再加。
  **已选 B（2026-11-16）**：首波不实现 C6 early stop，正式 run 按 hard cap 跑满
  2600；首波 checkpoint 使用 final。第二波再启用
  `min=600 / max=2600 / window=20 / consecutive=3`；启用后 early-stop counter
  必须与 routine StageGate counter 独立，不能互相代替。
- [x] CU8：课程几何。**已确认（2026-11-16）**：C0/C1 保持不变；C2 改为 50%
  `C2-V` 纯垂向 `1.75 m / 1 cycle` + 50% `C2-H` 纯水平 `1.75 m / 1 cycle`；
  C3/C4 增加 `2.0 m / 2 cycles` 垂向；C5 增加 `3.0 m / 1.5 cycles`；C6 增加
  `5.0 m / 1 cycle`。C3–C6 的起始深度、水平 path family mix、长度和速度继续沿用
  现有表的起始深度、水平 path family mix 和长度；速度改由 CU9 的中心向外课程控制；
  详见 §4.12.7.1。当前代码未实现，旧结果只能标 `historical_only`。
- [x] CU9：速度课程与重采样。**已确认（2026-11-16）**：采用
  `SPEED_V3_CENTER_OUT_5S`，以 `1.5 m/s` 为中心，C0–C6 区间依次为
  `1.40–1.60 / 1.40–1.80 / 1.20–1.80 / 1.20–2.00 / 1.00–2.00 /
  1.00–2.20 / 1.00–2.50 m/s`；训练 episode 从起点开始每 `5 s` 在 active stage 区间内
  重采样速度目标，正式评价固定 route speed。速度变化用于覆盖弱/强舵效，不得把训练
  时变速度带进评价，也不得按瞬时真实海流改速度命令；详见 §4.12.7.2。

**测评 EV1–EV9**

- [x] EV1：`MAIN_RANKING` 规模。A 保持 `3 train seeds × 4 pressures × 1
  stage-matched route × 1 radius × 3 eval seeds = 12 cells / 36 episodes per arm`；
  B eval seeds 改成 40；C train seeds 改成 4。推荐 A，40 只用于 diagnostic /
  geometry。
  **已选 A（2026-11-16，后按 EV3 修正为 3 eval seeds）**：正式 core 固定
  `3 train seeds × 3 eval seeds`，每个 stage/arm 为 `12 cells / 36 episodes`；
  `4 stages total = 48 cells / 144 episodes per arm`。eval seed 从 8 降到 3，是为了
  把预算让给 EV3 的 22 条几何路线；3 个 eval seed 只做描述性不确定性和配对
  train-seed 比较，不作强显著性声明。40 eval seeds 不进入 Goal182 active protocol；
  只有需要接外部 standing suite 时才作单独诊断，且必须另标 tier。
  **铁律：四个压力必须分开汇报，禁止 pooled 后当主结论。**
- [x] EV2：评哪些 stage。用户确认正式 core 为 `C2/C3/C4/C6` 四个 stage；每个
  arm 为 `4 stages × 12 cells = 48 cells / 144 episodes`。C0/C1 只保留训练热身与
  诊断；全 C0–C6 sweep 只给最终 winner 做一次完整曲线，不进入所有 arm 的 formal
  core。速度包线由独立 `SPEED_ENVELOPE` 覆盖 `C2/C3/C4/C6` 四个 stage；C3 另保留
  `QUICK_SCREEN` 的四档 fixed-speed 初评，但初评结果与 C3 formal / `SPEED_ENVELOPE` 必须分开。
- [x] EV3：路线泛化测评（`ROUTE_GENERALIZATION`）路线轴。用户确认改为因子化设计：
  `1 条直线 + 2 方向 × 3 振幅 × 3 波浪次数 = 19 条二维路线`。
  具体为 `R_STRAIGHT`；垂向 `R_VERT_A{0P5,2,5}_C{2,3,5}` 共 9 条；横向
  `R_LAT_A{0P5,2,5}_C{2,3,5}` 共 9 条。振幅 `0.5 / 2 / 5 m`，cycles `2 / 3 / 5`，
  长度统一 `70 m`，速度 `1.2 m/s`，`depth_slope_m=0`，只让振幅/cycles 变化。
  `P0–P4` 和 `legacy_sine` 继续作 path-family / legacy 诊断，不混入这 19 条主轴。
- [x] EV3b：`R_3D_<level>` 已确认采用“垂向+横向同时变化”三档：
  `R_3D_S = depth 0.5 m / lateral 0.5 m / cycles 2`、
  `R_3D_M = depth 2.0 m / lateral 2.0 m / cycles 3`、
  `R_3D_L = depth 5.0 m / lateral 5.0 m / cycles 5`，长度仍 `70 m`。
- [x] EV3 评价规模（2026-11-16）：路线泛化测评只给 `MAIN_RANKING` 前三名，评 C6；
  `3 train × 4 pressure × 22 route × 3 eval = 264 cells / 792 episodes per arm`。
  前三名合计 `2,376 episodes`。19 条二维路线为 `R_STRAIGHT + 9 VERT + 9 LAT`；
  3 条三维为 `R_3D_S/M/L`。
- [x] EV3 前三名选择规则（2026-11-16，用户选 B 并要求压力加权）：`MAIN_RANKING`
  结束后，先按预注册 Gate 筛掉不合格 arm；合格 arm 按“压力加权胜场”排序，取前 3。
  每个压力先单独计算 C6 route 3D RMSE 并单独排名，禁止把四个压力的 RMSE 先平均；
  加权只作用于“排名结果”，不作为对外汇报的性能均值。
  - 压力权重固定为 `nominal=1.0`、`current10=1.5`、`noise_n2=1.25`、
    `plant20=1.5`，启动 `MAIN_RANKING` 前冻结；
  - `weighted_wins(a) = Σ_p w_p × 1[arm a 在压力 p 的 C6 route 3D RMSE 最低]`，
    并列最低时平分该压力的胜场；
  - 第一破同分键为加权名次分 `weighted_rank_score(a) = Σ_p w_p × (N - rank_p(a) + 1)`，
    其中 `N` 是当次合格 arm 数，`rank_p=1` 表示该压力最好；
  - 第二破同分键为 `worst_pressure_rmse(a) = max_p J_p(a)`，越小越好；仍并列时并列保留，
    进入路线泛化测评后再记录，不得引入 nominal-only 的隐式偏好；
  - 若合格 arm 不足 3 个，只跑实际合格的 arm，不用失败 arm 凑数。路线泛化测评不得反向
    参与前三名选择。
- [x] EV4：绝对深度轴。`depth_profiles_m` 现在未接线。A 真正接进 builder，只用于
  `ROUTE_GENERALIZATION` 前三名 audit；B 从 suite 删除，不写进结论。推荐 A，但只放在
  路线泛化测评。**已选 A（2026-11-16，四个档位）**：计划使用
  `DEPTH_GENERALIZATION = [10, 80, 250, 500] m`，只给 `MAIN_RANKING` 前三名；固定 C6、
  `R_STRAIGHT + R_3D_M` 两条代表路线、`nominal` 压力、3 train seeds × 3 eval seeds，
  每个 arm 为 `4 × 2 × 1 × 3 × 3 = 72 episodes`，前三名合计 `216 episodes`。
  这是独立审计集，不改变现有 `ROUTE_GENERALIZATION` 的 22 路线计数，也不参与前三名选择；
  若要把它扩成全路线/全压力交叉，必须另立决定。
- [x] EV5：success 口径。A 三个口径都报，`eta_tight_at_done` 作研究 primary，
  `p_nav` 作低噪声 synthetic navigation 诊断，route RMSE 作性能 primary；
  B 只报 p_nav；C 只报 eta。推荐 A，
  并把 summary 的 `success_rate` 改名为 `success_rate_pnav_tight`。
  **已选 A（2026-11-16）**：`success_pnav_tight`、`success_eta_tight_ever`、
  `success_eta_tight_at_done` 三口径都输出；研究 primary 固定为
  `success_eta_tight_at_done`，`success_eta_tight_ever` 只作宽松辅助。
  §4.12.9 默认是低噪声 `unbiased_small_noise_v3`（`nav_noise_m=0.001–0.1 m`，`current_est_tau_s=0.05–0.5 s`），所以
  `success_pnav_tight` 是“低噪声 synthetic navigation”口径，不是真实部署；只有
  optional `p_nav_truth_lock=true` 时才变成 oracle-navigation 一致性检查。真实部署
  诊断必须另开 `realistic_calibrated` 评价。canonical summary 字段使用
  `success_rate_pnav_tight`，旧 `success_rate` 仅作兼容别名；默认 manifest 标
  `semantics=SYNTHETIC_UNBIASED_NAV`，lock 时才标 `ORACLE_NAVIGATION`。旧
  `unbiased_noise_v1` 0.05–0.20 m 结果和 `realistic_calibrated` 64-seed C3 的导航
  误差（mean 约 `0.61 m`、RMSE `0.68 m`、p95 `1.14–1.16 m`）只作历史/stress
  证据，不与新默认主表混比。当前 `src/auv_lowlevel/outcome_contract.py` 的 docstring
  仍写 `eta_tight_ever` 为 research primary；实现时只改评价字段和文档语义，改为
  `eta_tight_at_done` primary，禁止让 eta 口径影响 rollout、reward、termination
  或 checkpoint 选择。
- [x] EV6：zero baselines。**已选 A（2026-11-16，PE2E 部分）**：按 control
  contract 配对，不硬造跨架构基线：
  - `BCRES`：用该 arm 自己的 checkpoint（`base_policy.mode=frozen_pc_bc_residual`），
    `--action-mode frozen-base`；内部冻结 BC 输出 base action，残差恒为 0。
    `base_policy.checkpoint`/SHA 必须与该 arm 匹配；不得拿一个全局 BC 文件替代。
  - `BCFT`：用训练时 `--warmstart-actor` 指向的初始 BC checkpoint，PPO 更新 0 次。
    BC-only 评测必须和 BCFT 使用同一 stage/route/pressure/radius/eval seed，并输出
    同一套 `report_rows.jsonl` 字段；无法统一时判 `INVALID_ZERO_BASELINE_PROTOCOL`。
  - `TRRES_CMD / TRRES_ACT`：使用同一 env/guidance 配置，`--action-mode alos`，残差
    恒为 0，得到传统 ALOS + PD/PI；checkpoint 只作 schema/provenance，actor 不参与
    动作。两个 wrapper 的零残差结果必须实际核对；若数值不同，分别命名
    `zero_alos_cmd` / `zero_alos_act`，不得合成一个基线。
  - `PE2E`：没有同底座 zero baseline；采用 A，报告标
    `ZERO_BASELINE_UNPAIRED`，只和其他学习组、BC/传统基线做外部参照。禁止拿冻结
    BC 或零动作冒充 PE2E 的 paired zero baseline。
  - 所有 zero baseline 必须使用与对应 arm 相同的 `evaluation_set`；缺失 paired
    zero 判 `INVALID_INCOMPLETE_EVAL`，只有 PE2E 的显式 `UNPAIRED` 例外。
- [x] EV7：checkpoint 选择。**已选 A（2026-11-16）**：首波没有 early stop，正式
  评价固定使用 `final`；第二波开启 C6 early stop 后，使用 early-stop 触发的
  checkpoint。所有 arm 同一规则，禁止用 dev 选 best、禁止在多 checkpoint 中挑最好。
  manifest 记录 `checkpoint_selection=final|early_stop`、checkpoint path/SHA、训练
  iteration 和 early-stop 触发信息；违反规则判 `CHECKPOINT_SELECTION_LEAK`。
- [x] EV8：外部 standing suite 接入方式。**已选 A（2026-11-16）**：
  Goal182 的主排名只使用 `MAIN_RANKING`；前三名只追加
  `ROUTE_GENERALIZATION`。外部 `EXTERNAL_GEOMETRY_SUITE`（当前 suite D）只给最终
  winner 作一次外部 geometry 体检，不参与 Gate、winner 排名、前三名选择、early-stop
  或 checkpoint 选择。suite D 为
  `756 scenarios × 40 seeds ≈ 30,240 episodes/arm/train seed`，因此禁止所有 arm
  都跑。最终 winner 按 EV7 的 `final|early_stop` checkpoint 规则，对其正式训练 seeds
  逐一运行，并单独标 `EXTERNAL_GEOMETRY_ONLY`；结果不得反向改写主排名结论。
  B（所有 arm 都跑 suite D）已否决，代价过高且会形成第二套选型集。
- [x] EV9：canonical 产物。**已选 A（2026-11-16）**：固定
  `matrix_manifest.json + report_rows.jsonl + report.md`；每个 run 另带
  `summary.json`、`episodes.jsonl`、`trace.npz`。`matrix_manifest` 每张 frozen
  matrix 一份，`report_rows` 每个 eval cell 一行并在 cell 内聚合 `eval_seeds`，
  `episodes.jsonl` 每个 episode 一行，`trace.npz` 与 episodes 共用 `episode_id`。
  `report_rows` 必须能从 `episodes.jsonl` 重新聚合；缺产物判
  `INVALID_ARTIFACT_SET`，无法复算判 `INVALID_REPORT_PROVENANCE`。`trace.npz`
  只作诊断，不得用于挑选 checkpoint、winner 或前三名。详见 §4.12.5。

**无争议的文档/契约修正**

- 把 §4.13.5 的旧“96 eval cells”改成“48 cells / 144 episodes per arm”
  （4 个 stage × 12 cells，每个 cell 3 个 eval seed）。
- 把 `总协调/doc/governance/EXPERIMENT_PROGRAMS.md` 里 Transformer critic 固定
  `[256,128,64]` 改成 M 档 `[512,256,128]`，与已确认的 critic 容量一致。
- 所有 formal config 补 `training_profile`，启动前跑 `check_training_profile.py`。
- 把 `warmstart_actor`、checkpoint SHA、可训练范围从 CLI 提升到 config/matrix 一
  等字段。
- 现有 `check_experiment_matrix.py` 只支持 v1，需要补
  `auv_core_composition_matrix.v1` 与 `auv_goal_matrix.v2` 的 checker。
- Goal182 launcher 的 eval seed 默认值对齐 8；通用 CLI 的 40 可以保留。
- evaluator / manifest 增加显式 `route_profile` 字段；按 EV4-A 接线
  `depth_profiles_m=[10, 80, 250, 500]`，并新增独立的 `DEPTH_GENERALIZATION` provenance；
  `depth` 与 `route_profile` 分开计数。
- 实现 route-centric StageGate 的 shadow 计算；实现 iteration-based curriculum
  caps；实现 routine training-rollout 20/3。C6 early-stop 按 CU7-B 放到第二波，
  不阻塞首波；首波不得留下一个静默启用的 early-stop counter。

**建议讨论顺序**

`MX1 → MX2 → MX3 → MX4 → CU1 → CU2 → CU3 → CU5 → EV1 → EV3 → EV5`。
这 11 个决定会直接改变 matrix 的行数、训练 run 数和评测 episode 数；其余随后补。

#### 4.13.9 Current 语义与实现清理待核对（CQ1–CQ8）

状态：`DONE`。CQ1–CQ8 已确认。以下问题只登记结论，不代表已经修改训练代码；原问题
描述保留供追溯。结论已回填本表和 §13 决策记录。当前 `goal182`
训练配置是 `MOD2-like`：
`actor_current_encoder_enabled=false`、`critic_path_dim=30`、`dade_enabled=true`。

| ID | 待核对问题 | 选项 | 当前代码事实 | 用户结论 |
|---|---|---|---|---|
| CQ1 | 海流估计器精度是否随 `C0–C6` 课程变化？ | A 固定一套 `sensors.profile`；B 每个 stage 映射不同 profile；C 另开独立 `ESTIMATOR_PROFILE` 实验轴 | 当前为 A。`current_estimator_level` 在 curriculum 中生成和记录，但 `core.py` / `sensors.py` 没有消费它来切换估计器；实际仍由 `sensors.profile` 决定 | **A（2026-11-16 已确认）**：Goal182 固定一套 profile；`current_estimator_level` 只作 metadata，不作为 profile 切换开关 |
| CQ2 | actor 使用原始海流估计，还是使用 guidance 的过滤估计？ | A 原始 `current_estimate_body`；B 与 guidance 相同的 valid/age/speed EMA 过滤；C 原始为主，过滤版做消融 | 当前为 A。actor 直接用 `sensor_output.current_estimate_body`；guidance 另有 robust filter，二者不是同一个值 | **A（2026-11-16 已确认）**：actor 使用 raw estimate；filtered / valid-age-aware 版本作为后续独立消融 |
| CQ3 | `config.current` 与 curriculum 冲突时以谁为准？ | A curriculum 生效并在启动时打印 effective value；B config 为准；C effective mismatch 直接报错 | 当前为 A。`env_path_fixed_speed_c3.json` 的 `current.speed_mps=0.25–0.45`，但 C3 curriculum 实际采样 `0.30–0.55`；当前没有自动打印 effective value | **A（2026-11-16 已确认）**：curriculum 生效；config 只作 fallback；启动时必须打印 effective stage/current/period/shear/profile/noise scale |
| CQ4 | MOD 是否可以顺带改变海流估计准确度？ | A 不改变，MOD 只控制 actor 是否消费；B MOD3/4 使用更准的估计；C 精度变化全部移到独立传感器实验轴 | 当前为 A：MOD 只控制 `actor_current_encoder_enabled` 等消费接口，不改变 `sensors.profile` | **A（2026-11-16 已确认）**：MOD 只控制信息消费；精度变化另开独立传感器轴 |
| CQ5 | `MOD1/MOD2` 不启用 current 时，history token 是否保留 3 个零维 side channel？ | A 保留固定 3 个零维，schema 不变；B 改成变长、直接删除 side channel；C 保留零维但启动日志显式写 `actor_current_disabled` | 当前为 A：layout 仍保留 3 维，`actor_current_encoder_enabled=false` 时显式置零 | **A+C（2026-11-16 已确认）**：保留固定 3 个零维；metadata/启动日志/manifest 显式标记 `actor_current_enabled=false` 与 `actor_current_side_channel=zero` |
| CQ6 | `current_estimator_level` 与 `training.privileged_current_critic` 两个旧/未接线开关如何处理？ | A 现在删除；B 标记 deprecated，非默认值报错；C 真正实现它们 | `current_estimator_level` 未接入传感器；`training.privileged_current_critic` 不是 PC privileged critic 的正式开关，PC 看 `pc.critic_path_dim` | **B（2026-11-16 已确认）**：两个字段都标记 deprecated；`current_estimator_level` 只作 provenance / metadata，不切换估计器；PC 配置中 `training.privileged_current_critic=true` 直接报错；启动日志只打印 `pc.critic_path_dim` |
| CQ7 | DADE 的 `current` query 参数如何表达？ | A mean pooling 时显式传 `None`，只有 attention pooling 要求 query；B 删除该参数；C 把 attention pooling 作为独立实验 | 当前接口仍传 `current`，但 canonical `dual_horizon_transformer` 使用 mean pooling，`_WindowEncoder.forward` 会忽略它 | **A（2026-11-16 已确认）**：mean pooling 时不构造、不传 `current` query；只有 attention 类 pooling 才要求 query。attention pooling 作为独立 optional diagnostic，不混入 canonical DADE、核心矩阵或 60 个正式组合 |
| CQ8 | reset 后 current estimate 从零爬升的初始滞后是否保留？ | A 保留，视为真实估计器启动；B 用第一次测量热启动；C 只在诊断中用真值初始化 | 当前为 A。传感器低通初始为零；只读 sanity check 中 `0.42 m/s` 恒定流在 `0 s` 误差约 `0.41 m/s`，`1 s` 约 `0.16`，`5 s` 后约 `0.007`，主要是启动滞后 | **B（2026-11-16 已确认）**：第一条有效 current 测量到达时，用它初始化低通状态与历史队列，不从零爬升；不得使用真值初始化 |

**CQ1 后续项（已记录，不在本轮执行）**：后续单独降低 current estimate 的延迟；不得把
这一改动混入当前 MOD 实验或 `current_estimator_level` 的课程映射。降低延迟应作为
独立传感器/估计器实验，保留 profile、延迟、噪声和 valid/age 的 provenance。

**CQ2 后续项（已记录，不在本轮执行）**：把 actor 的 filtered / valid-age-aware current
作为独立消融；本轮主线保持 raw `current_estimate_body`，不把 guidance EMA 接入 actor，
也不改变 MOD 语义。

**CQ3 后续项**：训练/评价启动器和 manifest 必须记录 effective current 参数，至少包含
stage、current speed range、period、shear scale、sensor profile 和 noise scale；不得只
记录 config 中的 fallback 值。

**CQ4 后续项（已记录，不在本轮执行）**：若未来研究 current estimate 不准或传感器退化，
单独开 `ESTIMATOR_PROFILE` / `SENSOR_STRESS` 轴；不得通过 MOD3/4 顺带改 profile。

**CQ5 后续项**：所有 arm 的 metadata/manifest 必须记录 `actor_current_enabled`、
`actor_current_dim`、`actor_current_side_channel` 和 `side_channel_dim`；MOD1/MOD2 必须
显示 `actor_current_enabled=false` / `actor_current_side_channel=zero`，不得只靠读者
从输入维度推断。

**CQ6 后续项**：`current_estimator_level` 和 `training.privileged_current_critic` 不再作为
行为开关。前者允许为了 curriculum provenance 继续记录，但必须标记
`semantics=metadata_only`；任何试图用它切换 `sensors.profile` 的实现都视为回归。后者在
PC 训练配置中标记 deprecated，默认值只能是 `false`；设为 `true` 时启动器必须直接报错，
并提示改用 `pc.critic_path_dim`。manifest 只记录正式字段，不把 deprecated 字段当成实验
因子，也不允许在 `CQ1=A` 下重新接线 `current_estimator_level`。

**CQ7 后续项**：DADE 的 `current` query 只属于 attention 类 pooling。canonical mean
pooling 路径必须传 `None`，不得为了“保持接口形状”而构造一个被忽略的 query；attention
pooling 若作为后续实验启用，必须显式声明 pooling 类型，并单独验证收益、延迟和跨 seed
稳定性。该清理是接口语义修正，预期不改变当前 mean-pooling 策略的数值行为。

**CQ7 实验边界**：`DADE_ATTN_POOL` 只作为独立 optional diagnostic，不进入 canonical
DADE、核心组合矩阵或首轮正式 arm。它和 `AB_TA_M` 不是同一个因子：`AB_TA_M` 是 actor
backend 的 Transformer token 交互，`DADE_ATTN_POOL` 是历史窗口的聚合方式；两者必须分开
登记、分开命名、分开解释。若后续要跑，必须保持 actor backend、容量、token 集合、窗口、
训练数据、seed 和评价协议不变，只改变 pooling，否则判 `ATTENTION_CONFOUND`，不得把收益
归因于 attention。

**CQ8 后续项**：current estimate 的首次有效测量热启动必须同时作用于滤波器状态和
current delay queue；只改当前帧而 history 仍是零，不算热启动。manifest 必须记录
`current_est_start=hot_first_valid_sample`。重放历史 checkpoint 时保留旧语义，不得追溯
改写历史结果。

核对顺序固定为 `CQ1 → CQ2 → CQ3 → CQ4 → CQ5 → CQ6 → CQ7 → CQ8`。CQ1–CQ8 已全部
确认；后续任何 current 语义改动都必须另立 decision，不得静默修改本组结论。

#### 4.13.10 DADE 预训练 optional diagnostic（未启动）

状态：`BLOCKED_ON_TRIGGER`，不属于核心矩阵、首轮 60 个正式组合或 winner 选择。调研
记录见 `../../../research/paper_reading/2026-11-16-dade-pretraining-survey.md`。

**触发条件（全部满足才启动）**

- [ ] `MOD1–MOD4` 已在足够训练预算下与对应 no-history control `MOD0` 完成配对比较；
- [ ] history-enabled family 没有超过 `MOD0` 的实质收益，或收益小于 seed 噪声；
- [ ] DADE telemetry 显示 `c_hat` 输出范数/门控长期接近初始化值，且训练中没有明显增长；
- [ ] 已排除 reward、curriculum、训练不足、checkpoint 选择、配置错误和评价协议问题；
- [ ] 预训练仍使用同 actor backend、同容量、同 token/window、同数据和同评价协议；任何
  encoder 或 tokenization 变化都必须另立 arm，不能混入“预训练有没有用”的结论。

**目的**

不是训练一个完整 world model，而是让 DADE 学到对控制有用的变化规律：短时动力学、
路径演化、海流估计漂移，以及噪声/短时丢测下的历史补全能力。预训练 loss 只用于诊断，
不能作为晋级指标；最终必须看闭环 tracking error、health 和 current/noise 压力下的表现。

**最小实验序列**

按 `D1 → D2 → D3` 顺序做 optional diagnostic，每一步先 1 seed screen，只有超过 `D0`
且差距大于 seed 噪声才扩到 3 seeds；不并行铺开，不进入正式矩阵：

- `D0`：不预训练，当前基线；
- `D1`：预测未来 `1/2/5` 步的 actor-visible state delta、path delta、current estimate
  side-channel delta；优先多步/平滑目标，避免单步噪声；
- `D2`：随机 mask 历史帧、current side channel 或传感器通道，做 masked prediction；
- `D3`：`D1 + D2` 组合，确认预测与鲁棒性目标是否互补。

可选 `D4`：用 BC/teacher action 训练临时 action head，预训练后丢弃；只在 `D1–D3`
出现正信号后考虑，不作为第一轮。

**训练契约**

- 使用现有 BC demonstration 数据，按 `episode_id + step` 重建 short/long history；
- 训练/验证按 episode 划分，禁止按帧随机划分造成历史窗口泄漏；
- 冻结 BC 的 `actor_encoder` / `path_encoder`，只训练 DADE 和临时预测头；
- checkpoint 只保存 DADE state dict、token/window/normalization 契约；临时预测头不进入
  正式 policy；
- PPO 加载预训练 DADE 时不能继续把 DADE 输出强制清零；保留权重，用初始接近零的可学习
  gate/scalar 让 DADE 贡献逐步接入，既保护 BC 先验，又允许 PPO 使用预训练表征；
- 目标只使用 actor-visible 量，禁止使用 true current、progress、curvature 或 privileged
  critic 字段；否则判 `PRIVILEGED_PRETRAIN_LEAK`；
- 任何预训练结果若不满足相同的 actor 输入、窗口、normalization 和评价协议，判
  `INVALID_PRETRAIN_CONTRACT`，不得与 D0 比较。

**晋级 Gate**

只有 `D1/D2/D3` 在 nominal + current/noise 压力下相对 `D0` 有稳定收益，且没有以 health、
安全或延迟为代价，才允许把预训练作为后续正式 treatment 候选。单 seed 正结果只能记为
`QUICK_SCREEN / NOT_FORMAL`；没有跨 seed 复现就停止，不进入路线泛化或 winner 选择。

## 5. 评价矩阵与指标

### 5.1 数据与隔离

- checkpoint：Goal175 有效根 `P_RUD_075_COMP`，或 Phase 0 校准后的等价 checkpoint；
- 训练 seeds：`86201–86203`；Phase 0/1 screen 可只用 `86201`；
- 评价 dev seeds：`20–22`（3 seeds；EV1 已确认从 8 降为 3）；
- held-out `100–149`：关闭；
- pressure profiles：`nominal / current10 / noise_n2 / plant20`；
- radii：只保留 `default 2.0`；`strict125 1.25` 已从 Goal182 active protocol 移除；
- navigation：Goal182 默认 `sensors.profile=unbiased_small_noise_v3`，
  `nav_noise_m=(0.001, 0.1)`，`current_est_tau_s=(0.05, 0.5)`；`p_nav = eta[:, :3] + zero_mean_noise`，标签
  `SYNTHETIC_UNBIASED_NAV`。`p_nav_truth_lock` 是 optional diagnostic，默认
  `false`；只有显式打开时才 `p_nav := eta[:, :3]` 并标 `ORACLE_NAVIGATION`。
  旧 `unbiased_noise_v1` 0.05–0.20 结果和 `realistic_calibrated` 结果都不得与
  新默认主表混比；详见 §4.12.9；
- zero baselines：`BCRES → zero_bc50`（同一 checkpoint 内嵌冻结 BC，残差 0）、
  `BCFT → warmstart BC`（PPO 更新 0 次）、`TRRES_CMD/ACT → zero_alos_comp`
  （传统 ALOS + PD/PI，残差 0）；`PE2E → UNPAIRED`，标
  `ZERO_BASELINE_UNPAIRED`，不算缺失，也不得用跨架构基线替代。

### 5.2 指标

**终点捕获策略（2026-11-16 用户确认）**

- `terminal_capture`、`goal_reached`、`at_goal` 或任何等价事件可以记录为描述性统计
  和调试 telemetry，但不能作为评价指标；
- 它们不得进入 Gate、winner 排名、前三名选择、early-stop 判定、路线泛化入选、
  matrix completeness 或 primary/secondary KPI；
- 如果报告中保留，只能放在明确标注 `DESCRIPTIVE_ONLY_NOT_A_METRIC` 的诊断附录，
  并同时注明样本数和对应 cell；
- 后续 success 指标必须由 EV5 单独定义，不能把终点捕获事件换个名字当成 success；
- EV5-A 的 success 定义：同时报告 `success_pnav_tight`、`success_eta_tight_ever`、
  `success_eta_tight_at_done`；研究 primary 使用 `eta_tight_at_done`，
  `eta_tight_ever` 只作宽松辅助。Goal182 默认是低噪声 `unbiased_small_noise_v3`
  （`nav_noise_m=0.001–0.1 m`，`current_est_tau_s=0.05–0.5 s`），所以 `success_pnav_tight` 是“低噪声 synthetic
  navigation”口径，不是真实部署；只有 optional `p_nav_truth_lock=true` 时才变成
  oracle-navigation 一致性检查，并须满足 §4.12.9 的 `1e-4 m` 断言。真实部署导航
  诊断必须另开 `realistic_calibrated` 评价，不能混入主表。summary 的 canonical
  名称仍为 `success_rate_pnav_tight`，旧 `success_rate` 只能作兼容别名；默认
  manifest 标 `semantics=SYNTHETIC_UNBIASED_NAV`，lock 时才标
  `ORACLE_NAVIGATION`。
- 该限制不禁止记录终点捕获次数、触发时刻或 terminal longitudinal error；记录本身是
  允许的，禁止的是把它用于评价、排序或晋级。

主指标：

- route 3D RMSE；
- route cross abs p90；
- route vertical abs p90；
- preterminal 3D RMSE；
- path completion / progress，只作防停滞 guard 和过程诊断，不重新引入 actor 的
  progress 观测，也不把到达终点当主要优化目标。

`route 3D` 在这里只作为评价/StageGate 候选指标；按 §4.9，reward 不再包含 3D 合成项。

credit / 训练指标：

- `residual_abs / action_std`；
- 每通道 applied share；
- KL、clip fraction、grad norm、update cosine；
- entropy；raw policy std；per-channel `sigma_eff`、applied delta mean/std/p95、
  `applied_ppo_snr`；启用 bounds/anneal 时记录命中率；
- value loss、explained variance、advantage 相关性。

安全与效率：

- success / timeout / safety；
- speed max、overspeed steps；
- magnitude/rate clip rate；
- transitions-to-gate、wall-clock、forward p95。

### 5.3 评价单元

```text
cell = arm × training_seed × stage × route_profile × pressure_profile × radius
episode = cell × eval_seed
```

每个 cell 内评价预注册 dev seed 集合；cell 计数不含 eval seed，episode 计数才含。
`level` 一词退出 active matrix：压力档使用 `pressure_profile`，课程难度使用
`stage`，路线形状使用 `route_profile`，成功半径使用 `radius`。四者不得互相替代。

**铁律：pressure_profile 是报告主键之一。** 报告必须先按
`arm × training_seed × stage × route_profile × pressure_profile × radius` 聚合并
分行输出；禁止把 `nominal / current10 / noise_n2 / plant20` 先平均成一个数再写主表。
跨压力 pooled 值只允许出现在明确标注 `DIAGNOSTIC` 的附录，不能用于 Gate、winner
或摘要结论。

- `SMOKE`：1 arm × 1 train seed × 1 stage × 1 route × 1 pressure × 1 radius；
- `QUICK_SCREEN`：1 train seed × 4 pressure profiles × 1 stage-matched route × `2.0 m`，
  只作 `NOT_FORMAL`；其中 C3 初评子表固定 `nominal`、四档固定速度、
  3 eval seeds，并单独汇报；
- `MAIN_RANKING`：3 train seeds × `C2/C3/C4/C6` × 4 pressure profiles ×
  1 stage-matched route × `2.0 m` × 3 dev seeds；
- `SPEED_ENVELOPE`：3 train seeds × `C2/C3/C4/C6` × 4 个固定 speed profile ×
  `nominal` × `2.0 m` × 3 dev seeds；
- `ROUTE_GENERALIZATION`（路线泛化测评）：只给 `MAIN_RANKING` 前三名；3 train seeds × C6 ×
  4 pressure profiles × 22 routes（19 条二维 + 3 条三维）× `2.0 m` × 3 dev seeds。
- `DEPTH_GENERALIZATION`（深度泛化审计）：只给 `MAIN_RANKING` 前三名；3 train seeds × C6 ×
  `R_STRAIGHT + R_3D_M` × `depth_m=[10, 80, 250, 500]` × `nominal` × `2.0 m` ×
  3 dev seeds；每 arm `4 × 2 × 1 × 3 × 3 = 72 episodes`，只审计绝对深度敏感性，
  不进入 winner、前三名选择或核心结论。

`strict125` 已从 Goal182 active protocol 移除，因此 radius 轴只允许 `2.0 m`。缺任一
预注册 cell 判 `INVALID_INCOMPLETE_EVAL`；cell 数完整但 route profile 仍是隐式 seed
采样时，判 `INVALID_ROUTE_AXIS_UNCONTROLLED`。

## 6. 执行计划

### 6.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 因子隔离 | 每个 Phase 只允许一个因子变化 | `MULTI_FACTOR_CONFOUND` |
| 核心组合矩阵 | 训练组合只用 `G182-CORE-COMP` 的四个核心轴；评价轴不得进入 arm 身份 | `INVALID_CORE_MATRIX` |
| 训练课程 | `curriculum.enabled=true`；逐 episode `curriculum_episodes.jsonl` | `INVALID_NONCURRICULUM` |
| 评价矩阵 | 预注册 arms × stages × routes × pressures × radii × eval seeds 全跑；`level` 不得再作为混合轴 | `INVALID_INCOMPLETE_EVAL` / `INVALID_ROUTE_AXIS_UNCONTROLLED` |
| 零基线 | `BCRES/BCFT/TRRES` 必须有同底座 paired zero；`PE2E` 显式 `UNPAIRED`；不得跨架构替代 | `INVALID_INCOMPLETE_EVAL` / `INVALID_ZERO_BASELINE_PROTOCOL` |
| seeds | 训练 `86201–86203`；评价 dev `20–22`（3 seeds）；held-out 关闭 | `INVALID_SEED_SPLIT` |
| 启动与结束 | 先跑协议检查器，再跑实验，最后跑 artifact auditor；失败非零退出 | `INVALID_PROTOCOL` |
| 速度 | 训练使用 `SPEED_V3_CENTER_OUT_5S`，每 `5 s` 重采样；评价固定 route speed；`speed + current + variation <= 2.5 m/s`；speed provenance 完整 | `INVALID_SPEED_CONTRACT` / `INVALID_SPEED_CURRICULUM` |
| 导航位置 | Goal182 默认 `unbiased_small_noise_v3` + `nav_noise_m=(0.001, 0.1)` + `current_est_tau_s=(0.05, 0.5)`，`p_nav = eta + zero-mean noise`；`p_nav_truth_lock` 默认 `false`，只有显式 `true` 时 `p_nav := eta[:, :3]` 且每步 `max(abs(p_nav - eta)) <= 1e-4 m` | `NAV_NOISE_PROFILE_INVALID` / `P_NAV_TRUTH_LOCK_VIOLATION` |
| 主指标 | route 3D/cross/vertical、preterminal、completion/progress | 不得只看 success |
| 训练路线 | D8 续路只允许 training-only、最多 4 段、逐 append manifest；评价路线固定 | `INVALID_ROUTE_PROTOCOL` |
| 命名与汇报 | canonical `arm_id`/`run_id`/cell ID；matrix v2；`report_rows.jsonl` 与完整性计数 | `INVALID_EXPERIMENT_CONTRACT` |
| 压力汇报 | `nominal/current10/noise_n2/plant20` 必须按 `pressure_profile` 分行/分表；禁止 pooled 主结论 | `INVALID_PRESSURE_POOLING` |
| 训练早停 | `min=600`、`max=2600`、`window=20`、`consecutive=3`；只用于 arm，不替代 StageGate | `EARLY_STOP_PROTOCOL_INVALID` |
| 失败裁决 | 不超过 seed 噪声不得宣称 winner | `INCONCLUSIVE_SEED_NOISE` |

- [ ] Step 0：建立参考快照和来源清单（§4.0）；确认 §4.12.8 的 M1–M7、§4.13.7 的
  CM1–CM9 和 §9 的待定项；先生成并冻结 `goal182_axis_registry.json`、
  `auv_core_composition_index.v1` 的 `goal182_core_composition_index.json`、
  `auv_core_composition_matrix.v1` 的 `goal182_core_pe2e_matrix.json` /
  `goal182_core_alosb_matrix.json`，再生成 `auv_goal_matrix.v2` 的
  `goal182_eval_matrix.json`；跑 core/eval/naming/report checker。
- [ ] Step 1：实现 authority / normalization / exploration / critic / gate 的配置 schema 和单元测试。
- [ ] Step 1b：实现 §4.12.7.1 的垂直三维课程：stage/episode vertical profile 字段、
  C2 单轴隔离、P2/P3/P4 的 depth amplitude/cycles、manifest provenance 与路径测试；
  旧 C3 结果标 `historical_only`。
- [ ] Step 1c：实现 §4.12.7.2 的 `SPEED_V3_CENTER_OUT_5S` 速度课程：C0–C6 中心向外
  区间、episode 内每 `5 s` 重采样、当前/初始速度 provenance、静态速度可行性检查和
  训练/评价隔离；跑 eval-only 速度-舵效标定。旧 v1/v2 速度协议结果标
  `historical_only`，评价配置必须保持固定 route speed。
- [ ] Step 1d：实现 §4.12.9 的默认导航噪声：新增 `unbiased_small_noise_v2`
  profile，`nav_noise_m=(0.001, 0.1)`；
  manifest 记录 effective profile、`nav_noise_m` 和 config SHA。`p_nav_truth_lock`
  默认 `false`；只有显式 `true` 时才启用 `p_nav=eta`、`1e-4 m` 断言、
  `ORACLE_NAVIGATION` 标签和 `P_NAV_TRUTH_LOCK_VIOLATION` 失败路径。旧
  `unbiased_noise_v1` 0.05–0.20 m 结果和 `realistic_calibrated` 结果标
  `historical_only`，不追溯改写。
- [ ] Step 2：实现 Goal182 runner、artifact auditor、credit/telemetry 采集。
- [ ] Step 3：Phase 0 eval-only smoke（1 arm × 1 level × 1 seed），确认 schema 和 trace。
- [ ] Step 4：Phase 0 全量校准；按 Gate 0 判读。
- [ ] Step 5：Phase 1 channel scope（rudder/stern/3D）screen；通过后跑 formal（3 seeds），再决定 Phase 1b 3D schedule。
- [ ] Step 6：Phase 2 screen + formal；只让 winner 进入 Phase 3。
- [ ] Step 7：Phase 3 screen + formal；复核 Goal92 的 clip-only 结论。
- [ ] Step 8：Phase 4 screen + formal；记录有效 std、entropy、safety。
- [ ] Step 9：Phase 5 screen + formal；确认 actor 输入不变。
- [ ] Step 10：Phase 6 离线 gate 回放 + screen + formal；审计 curriculum provenance。
- [ ] Step 11：Phase 7 integration，3 seeds，完整评价矩阵。
- [ ] Step 12：写 result md、ledger decision、STATE；运行 research ledger checker。
- [ ] Step 13：经用户批准后只提交本 goal 相关文件；不自动 commit/push。

### 6.2 运行命令（脚本名在 Step 1–2 固定）

```bash
python scripts/run_goal182_ppo_stack.py \
  --root artifacts/lowlevel/goal182_ppo_stack_governance_20261114 \
  --phase 0 --smoke

python scripts/run_goal182_ppo_stack.py \
  --root artifacts/lowlevel/goal182_ppo_stack_governance_20261114 \
  --phase 2 --seed 86201 --iterations 5

python scripts/audit_goal182_ppo_stack.py \
  --root artifacts/lowlevel/goal182_ppo_stack_governance_20261114
```

## 7. Gate 与判读

### 7.1 分阶段 Gate

| Gate | check | 阈值 | 失败标签 |
|---|---|---|---|
| G182-0 | authority + channel 校准 | 三组 channel scope 都有可测 applied share；vertical/cross 达到 `>=1e-2` 或 `>=10x` Goal176 同方向基线，forward 达到 `>=1e-3` 或 `>=10x` Goal176 同方向基线；逐通道记录 `applied_ppo_delta_norm_mean_abs`、`sigma_eff`、`applied_ppo_snr` 和 clip/safety；raw `residual/action_std` 只作辅助，不单独决定 PASS；3D 无 safety 回归 | `AUTHORITY_CALIBRATION_FAILED` / `THREE_D_SAFETY_BLOCKED` |
| G182-1a | channel scope training | rudder/stern/3D 三组都保持 applied share；3D 无 safety/timeout 回归；3D 至少在 vertical 或 cross 方向出现 route 剂量响应。若 3D 安全但短期 route 输给 rudder-only，标 `THREE_D_INTERIM_TRADEOFF`，保留 3D 继续 Phase 1b，不判失败 | `CHANNEL_SCOPE_NO_GAIN` / `THREE_D_SAFETY_BLOCKED` / `THREE_D_INTERIM_TRADEOFF` |
| G182-1b | 3D authority schedule | `P3_3D_SCHED` 的 route 或 safety 不差于 `P3_3D_FIXED`，且至少一项优于固定 authority | `THREE_D_SCHEDULE_NO_GAIN` |
| G182-2 | normalization | norm arm 的同预算 route 或 residual share 改善大于 seed 噪声 | `NORMALIZATION_NO_GAIN` |
| G182-3 | trust region | `T1_REFERENCE_TRUST_REGION` 的 route per transition 优于 `T0`，且无 safety 回归；KL/clip/learning-rate/grad-norm telemetry 完整；`T3_CLIP_ONLY` 不进首轮，若后续运行则不应优于 `T1` | `TRUST_REGION_NO_GAIN` / `CLIP_ONLY_CONTRADICTS_GOAL92` |
| G182-4 | exploration | `E1_REFERENCE_STD` 相对 `E0_CURRENT` 提高 residual share 或 route，且 entropy/clip/safety telemetry 完整；逐通道记录实际 applied std，但首轮不设 `0.05–0.10` 目标，也不因实际 std 偏离参考值而静默调参；safety/timeout 不恶化 | `EXPLORATION_NO_GAIN` / `EXPLORATION_UNSTABLE` |
| G182-5 | privileged critic | value loss / explained variance / advantage 相关性 / route 至少一项改善大于 seed 噪声；actor 只使用 5-D deployable path block；暴露路径不含 progress/remaining/curvature/endpoint；critic path 与字段来源审计完整 | `PRIVILEGED_CRITIC_NO_GAIN` / `PATH_PRIVILEGE_LEAK` |
| G182-6 | reference early stop | 首波所有 formal run 必须记录 `min_iterations=600`、`max_iterations=2600`、`window=20`、`consecutive=3` 的协议 provenance，并显式记录 `c6_early_stop_enabled=false`、跑满 hard cap；routine StageGate 使用独立的 training-rollout 20/3 counter；第二波启用 C6 early stop 后，才要求逐 iteration rule status、C6-only 作用域和两套独立 counter；无提前晋级/阶段回退 | `EARLY_STOP_PROTOCOL_INVALID` / `METRIC_GATE_UNSTABLE` |
| G182-7 | integration | `I1` 在 3 seeds、完整矩阵上优于 `I0`，且无 safety 回归 | `INTEGRATION_NO_GAIN` |
| G182-8 | magic-number provenance | `alos`、`path_*` 的速度包络全部直接引用 `vehicle.min_speed_mps` / `vehicle.max_speed_mps`；运行时代码无与 registry 匹配的速度字面量；registry、baseline 和 checker 测试齐全；Goal182 touched files 不得新增 `MAGIC_NUMBER_WITHOUT_PROVENANCE` | `MAGIC_NUMBER_WITHOUT_PROVENANCE` / `MAGIC_NUMBER_GUARD_MISSING` |
| G182-9 | speed-curriculum protocol | C0–C6 精确匹配 `SPEED_V3_CENTER_OUT_5S`；训练每 `5 s` 重采样并有逐段 provenance；评价无重采样、使用固定 route speed；速度目标不依赖瞬时真实海流；静态 `speed + current + variation <= 2.5 m/s` 检查通过；速度评测覆盖 `C2/C3/C4/C6` 四 stage 和 `1.0/1.5/2.0/2.5 m/s` 四档 | `SPEED_CURRICULUM_PROTOCOL_INVALID` |

### 7.2 seed 噪声规则

每个 formal 比较使用 3 个训练 seed 的配对差：

```text
delta_seed = metric(treatment, seed) - metric(baseline, seed)
winner 条件 = mean(delta_seed) > max(2 * std(delta_seed), minimum_effect)
```

`minimum_effect` 的具体数值在 P0 baseline 完成后、首次 formal 前冻结，冻结后不得因
结果改值。判读方向固定为 route 3D、cross p90、vertical p90；completion 只作
“不得下降超过 `0.01`”的防停滞 guard，不作为到达终点奖励或 winner 主指标。若
`mean` 为正但小于噪声，判 `INCONCLUSIVE_SEED_NOISE`，不得写成 PASS。

### 7.3 判读表

| 观察 | 判读 | 下一步 |
|---|---|---|
| Phase 0 无可用 authority | 权限不是可调参数能解决的 | 停止训练，转 plant/动作方向诊断 |
| Phase 1 有剂量响应但 route 不动 | authority 可见但 credit/plant 仍有问题 | 转 Goal179 Phase B/C，不继续加 epochs |
| Phase 1 的 3D 安全但短期 route 输给 rudder-only | 3D 不是短期最优，但仍是最终目标 | 标 `THREE_D_INTERIM_TRADEOFF`，保留 3D 进 Phase 1b，不删 arm |
| Phase 2 norm 有效 | 输入尺度是独立瓶颈 | 保留 norm，进入 Phase 3 |
| Phase 2 norm 无效 | 输入尺度不是瓶颈 | 删除 norm，保持简单 |
| Phase 3 `T3` 最好 | 与 Goal92 冲突 | 重审 Goal92 协议和 seed 噪声，不直接采纳 |
| Phase 4 只增加 safety 问题 | 探索过大 | 回退 `E0`，先修 credit |
| Phase 5 critic 有效 | critic 信息是独立收益 | 保留 dual-path privileged critic |
| Phase 5 actor path 出现 progress/remaining/curvature/endpoint | 部署观测泄漏 | 立即判 `PATH_PRIVILEGE_LEAK`，不执行正式评价 |
| Phase 6 early-stop 永远不触发 | 规则过严或训练仍不稳 | 先用离线回放诊断，不扫 iteration |
| Phase 6 早停改变课程晋级 | 两套时钟混淆 | 立即判 `EARLY_STOP_PROTOCOL_INVALID` |
| Phase 7 无集成收益 | 因子收益不叠加 | 只保留单因子 winner，不强行组合 |

## 8. 预算与停止条件

### 8.1 预算

- screen：1 seed × 5 iterations = `983,040` transitions / arm；
- canonical profile：`4096 × 48 = 196,608` transitions / iteration；
- formal：3 seeds × 整次 run `min_iterations=600`、`max_iterations=2600` =
  `117.96M–511.18M` transitions / seed，`353.9M–1.53B` transitions / arm；
- 只有 screen 方向正确且 instrumentation 完整的 arm 进入 formal；
- Phase 1 channel scope 允许 3 个 treatment arm（rudder/stern/3D）进入 formal；
  其他 Phase 最多 2 个 treatment arm 进入 formal；
- 不按每个 C 阶段各跑 600–2600；整次 run 的 C0–C6 hard caps 合计 2600
  iterations。默认硬上限为 `C0 120 / C1 180 / C2 260 / C3 360 / C4 420 /
  C5 560 / C6 700`，soft cap 默认取各自 60%；该分配在首次 formal 前冻结，不扫参；
- 20/3 早停只在课程进入 C6、`iteration >= 600` 且 window 填满后判断；未通过则继续到
  2600 或课程 hard cap 先到者；
- `5 epochs / 4 minibatches` 已由 canonical profile 固定；

### 8.2 停止条件

- Phase 0 Gate 失败：停止 Goal182，不启动训练；
- 同一 Phase 连续两次 Gate 失败：进入 autoresearch loop；
- 连续三次修复后指标仍不移动：写 blocker，decision 只能是 `pivot` 或 `stop`；
- 任何 safety 回归、held-out 泄漏，或未经 D13 冻结方案批准的 reward/termination
  改动：立即停止并标 `INVALID_PROTOCOL`；
- 结果落在 seed 噪声内：判 `INCONCLUSIVE_SEED_NOISE`，不选 winner。

## 9. 待用户定夺

未标“已定”的项仍为 `PENDING_USER_DECISION`；未确认前不生成最终 matrix，不启动训练。

| # | 决策 | 建议默认 | 影响 |
|---|---|---|---|
| D1 | authority 组成 | **已定：B**。独立 `bc_authority` + per-channel `ppo_authority`；`bc_authority=0.75` 仅作 Phase 0/1 控制变量；BC authority 另立因子 | 决定 Phase 0/1 的 arm |
| D2 | 通道范围 | **已定：三组 `stern-only` / `rudder-only` / `3D(all)`；3D 是最终目标**，但 Phase 1 训练仍按三组对照，不跳过 rudder/stern | 决定 Phase 0/1 的 arm 和 3D 安全门 |
| D3 | observation normalization | **已定：A+E**。保留 `PCNormalization` 固定前端，再叠加 rsl_rl empirical running mean/std（actor+critic）；物理 clip 默认不做 | 决定 Phase 2 主对照为 `N0` vs `N1`；`N3` 冻结策略另测 |
| D4 | advantage / return normalization | **已定：full-rollout，与参考工程默认一致**；minibatch 只作未来 cheap ablation，不进入首批 matrix；return/value 不做 | 避免把“已启用”误写成“待启用”，也避免把 minibatch 当成参考做法 |
| D5 | 有效探索目标 | **已定：仿照参考工程**：`init_std=0.3`、`noise_std_type=scalar`、`clip_actions=1.0`；首轮不加 min/max、不做 anneal；只记录 per-channel applied normalized delta 作为 telemetry | 决定 Phase 4 的 `E0_CURRENT` vs `E1_REFERENCE_STD`；bounds/anneal 留作后续 |
| D6 | trust region | **已定：仿照参考工程整包**：`clip=0.20`、`desired_kl=0.01`、`schedule=adaptive`、初始共享 `lr=3e-4`；沿用它内置的 `1/1.5×` adaptive LR 规则 | 决定 Phase 3 的 `T0` vs `T1`；`T3_CLIP_ONLY` 只作后续 Goal92 诊断；epochs/minibatches 已由 canonical profile 固定 |
| D7 | actor/critic path partition | **已定：一步到位**。actor 只保留 5-D non-privileged path block；删除 progress/remaining/curvature/endpoint；critic 增加独立 privileged path encoder、actuator/disturbance/sensor-safety 特权；不拆 C1–C3 训练臂 | 决定 Phase 5 schema 和 field audit；不能归因到单组特权，只保留 mask 诊断 |
| D8 | formal 预算与路线 | **已定：training-only 有界续路 + 整次 run 600–2600 iterations**。60 s episode 不变，接近末端时同族追加路线，最多 3–4 段；评价路线固定；课程 hard caps 合计 2600 | 决定训练分布和总计算量；续路不得进入评价环境 |
| D9 | 晋级 / 早停 | **已定：仿照参考代码的窗口参数；CU7-B 限定首波不启用 C6 early stop。** 首波 C6 跑满 2600；第二波再做 `min=600`、`max=2600`、`window=20`、`consecutive=3`；C0–C6 晋级继续使用 StageGate；winner 规则独立 | 首波只保留 routine StageGate 20/3；第二波 early-stop 与 StageGate counter 独立；`timeout` 仅诊断，不作为 early-stop 条件 |
| D10 | 与 Goal181 的顺序 | **已定：旧 `[256,128]` 退役。** Goal182 固定使用 M 档 MLP actor/critic `[512,256,128]` 选择 PPO profile；随后 Goal181 比较 M 档 Transformer 与 M 档 MLP | 避免小网络结论误导架构选择；旧网络只作 `historical_only` |
| D11 | 是否重跑历史 baseline | **已定：重跑 `P0_HISTORICAL`。** 先 1 seed screen，再补到 3 seeds formal，并作为所有 Phase 共用的新协议 baseline | 避免把 runner、epoch、minibatch、LR、entropy 等协议差异混进 treatment delta |
| D12 | 是否允许参考式 staged curriculum | **已定：不照搬。** 保持单 run 的 C0–C6 curriculum/StageGate；网络每轮用当前 stage 采样数据的 return 更新，晋级不重置网络或 optimizer | 避免训练组织与 PPO stack 混因；参考式多 stage runner 若需要，另立 goal |
| D13 | reward / StageGate 是否改为 route-centric | **已选 C；两套 reward 修改目标表已确认（§4.9），StageGate v1 口径已确认（§4.9.1），阈值待定。** Goal182 内同时重构 route-centric reward 和 StageGate；训练奖励以 cross/vertical 及对应的 heading/pitch/speed 跟踪为主，不再用 progress、到达奖励或 3D 合成奖励项 | 标注“审计起点”的权重先按表实现；StageGate v1 用 safety 硬否决、completion 防停滞、cross/vertical 主门控，timeout/plane_miss/success/3D/overshoot 仅诊断；具体阈值未冻结前不生成 matrix、不训练 |
| D14 | gamma / lambda | **已定：`gamma=0.995`、`lam=0.95`。** 保留 AUV 较长时程的信用跨度，不照搬参考工程的 `gamma=0.99` | 作为所有新协议 arm 的公共配置，不单独作为 treatment 因子 |
| D15 | entropy coefficient | **已定：`entropy_coef=0.005`。** 作为所有新协议 arm 的公共配置；Phase 4 只改变 `init_std`，不再同时改变 entropy | 与参考工程一致；避免把探索幅度和熵正则混成一个因子 |
| D16 | warmstart / optimizer / checkpoint | **已定：所有 arm 使用同一 BC 底座 checkpoint 和同一残差初始化规则；fresh optimizer；iteration 从 0 开始；使用 final/early-stop checkpoint；禁止 dev best 选择；禁止跨 arm resume** | 保证 arm 公平；防止验证集参与 checkpoint 选择 |
| D17 | actor / critic capacity | **已并入 D10：固定 M 档 MLP `[512,256,128]`，actor/critic 同容量** | 不再单独讨论旧容量 |
| D18 | early-stop 作用域 | **已定：首波 C6 不早停，第二波只在 C6 允许 arm early stop；C0–C5 只由 StageGate 或 hard cap 结束；StageGate 自身按 R5-D 使用 training-rollout 20/3** | 首波 `c6_early_stop_enabled=false`；第二波 arm early stop 与 StageGate 使用独立 counter；每次 stage 晋级/回退都清空对应 counter |

**BC path schema（2026-11-16 已确认）**：采用重训 `path_dim=5` 的 BC，不用 8D 置零
补丁。5D 只含 cross、vertical、tangent heading error、tangent pitch error 和
lookahead heading delta；progress/remaining/curvature 从 actor schema 中删除。新 5D
BC 是 Goal182 canonical 底座；旧 8D DAGGER2 只作历史对照，不得与新 BC 混在同一矩阵
比较。重训前需冻结演示数据来源、episode split 和 checkpoint SHA，并先跑 BC_ONLY
对照确认没有行为漂移。

**5D BC 重训数据与执行位置（2026-11-16 已确认）**：使用完整 DAGGER2 演示数据，不用
本地缩减版；数据为服务器
`.../pc_v0d_r2b_weighted_bc_dagger_20261107/demos_merged_dagger2/demos.npz`，期望
SHA-256 `080f4ac3a99ce5f41da119038f0de5a626ea8af1c6b3d9cb7de472d337169d78`。重训优先
在 8×4090 服务器执行：本地完成 5D schema/入口改动并 commit，服务器用独立 worktree
检出已知 commit，复用现有 GPU venv，产物写入
`/data/yhx/underwater-robotics/runs/auv-local/goal182/<run_id>/bc5d/`。不得原地修改旧
服务器副本，不用 `rsync` 覆盖源码；只同步 manifest、summary、小 checkpoint 和评估
摘要回本地。启动前必须记录数据 SHA、代码 commit、环境版本；结束后用同一协议与旧 8D
DAGGER2 做 BC_ONLY paired 对照。

**残差→微调顺序与必做规则（2026-11-16 已确认）**：保留“先 residual screen、后
fine-tune”的串行顺序，但完整 PPO 微调组是**必做实验**，不因 residual 成功而取消。以下
三条只用于判断 residual 是否提前收缩、优先排微调或补归因：1）完成 authority 校准后，
residual/action_std 仍无法进入 `[0.20, 0.35]`，或 applied share 仍接近零；2）
authority 已可见，但 screen/Phase 1 的 route 改善不超过 seed 噪声，且 completion/safety
没有改善；3）BC/PPO 明显抵消，或 reward/update 梯度方向错误。第三条先修 credit。
residual 失败时记录 `RESIDUAL_AUTHORITY_FAIL` / `RESIDUAL_ROUTE_NO_GAIN` /
`RESIDUAL_CREDIT_FAIL` 并在 `research/LEDGER.jsonl` 写 `pivot`，用于解释顺序；但微调组
matrix 始终生成，且使用同一 BC、reward 和评价协议。依赖缺失、路径错误、配置 typo、
一次性 smoke 崩溃或单 seed 波动不触发提前收缩。

**StageGate 评估节奏（2026-11-16 已确认）**：采用 R5-D，照参考工程。C0–C6 的 routine
晋级使用训练 rollout 的 20-iteration 滑动平均，连续 3 次满足该 stage 门槛才晋级；
每个 stage 保留自己的 min/max iteration。frozen-dev 只在 stage 边界和最终 formal
评价运行，固定 dev seeds `20–22`（3 seeds，EV1 已修正），不再每 20 iterations
另开一次考试。
首波不启用 C6 的 20/3 early stop；C6 跑满 hard cap。第二波启用后，C6 的 20/3 early
stop 同样使用训练 rollout 滑窗；边界评价可因 safety 或协议问题否决，但不能反向调
routine 阈值。该口径在首次 formal 前冻结。

D3 仍需确认的子项：

- **D3.1 `until` 策略（已确认）**：`N1_FIXED_EMPIRICAL_BOTH` 按参考实现锁定
  `until=1e8`、`eps=1e-2`、actor+critic 全开；物理 clip 默认不做。由于本项目
  formal 为 `117.96M–511.18M` transitions/seed，所有 formal run 都会在约
  `1e8 / 196,608 ≈ 509 iterations` 后停止更新 running stats；这符合参考的
  `until=1e8`，不是按 stage 或 wall-clock 冻结；若 running stats
  漂移明显，再加 `N3_EMPIRICAL_FREEZE`。冻结不能按绝对早期预算拍脑袋，应等
  curriculum 最终 stage 的 path block 稳定后再累计固定 transitions（例如
  `0.5M–1M`），否则后面的 block 会拿到过时统计量。
- **D3.2 `eps`**：`N1` 先按参考 `eps=1e-2`。若审计发现大量 post-PC std `<1e-2`，
  另立 `eps` 因子（例如 `1e-6/1e-3`），不要在 `N1` 内静默改。
- **D3.3 actor/critic 范围**：`N1` 同时开 actor+critic；只有在 `N1` 有增益后才跑
  `N2` critic-only，用来判断收益来自 actor 还是 critic。
- **D3.4 固定 scale 是否保留**：保留 `PCNormalization` 原位置和现有尺度；只做审计，
  不把 empirical 当成修正坏尺度的替代品。若某个 block 的 post-PC std 极端异常，
  另立 `N0b` 固定尺度修正因子。
- **D3.5 物理 clip**：默认不做；只记录 outlier/clip-rate。若 empirical 的 running
  std 被极端值显著抬高，再把它作为安全因子，而不是归一化因子。

D5 已定子项：

- **D5.1 测量口径**：记录 per-channel `applied_ppo_delta_norm` 的 mean/std/p95/SNR，
  raw policy std 作辅助；修正旧 `head residual/action_std` 混合缩放的问题，但不改变
  参考工程的 `init_std=0.3`。
- **D5.2 bounds / anneal**：`E2_BOUNDED` 和 `E3_ANNEAL` 不在首轮执行，等
  `E1_REFERENCE_STD` 的 screen/formal 结果后再单独定夺。

D6 已定子项：

- **D6.1 参考 profile**：`clip=0.20`、`desired_kl=0.01`、`schedule=adaptive`、
  初始 `learning_rate=3e-4`；与 canonical profile 的 `5 epochs / 4 minibatches`
  一起使用。
- **D6.2 adaptive 规则**：完全使用 rsl_rl 2.3.3 内置规则，每个 minibatch 检查
  KL；`KL > 2*desired_kl` 时 `lr=max(1e-5, lr/1.5)`，`0 < KL < desired_kl/2`
  时 `lr=min(1e-2, lr*1.5)`，否则不变。参考 cfg 未覆盖该规则。
- **D6.3 单一 LR**：参考实现是单一 optimizer/单一 LR，因此 actor 与 critic 初始
  LR 同为 `3e-4`；首轮不允许 separate actor/critic LR、DADE-head 独立 LR 或
  critic warmup。若将来要拆速率，另立实现与消融因子。
- **D6.4 不拆参考组合**：首轮只有 `T0_CURRENT_PROFILE` 与
  `T1_REFERENCE_TRUST_REGION`；`T3_CLIP_ONLY` 不进入首轮 matrix，只在需要复核
  Goal92 时单独运行。

D7 已定子项：

- **D7.1 两套 path schema**：新增 `actor_path_safe` 与 `critic_path_priv`。前者由
  `pnav`/导航估计和参考路线计算，后者由真实 `eta` 和真实路线投影计算；两者不得共享
  encoder 输出，也不得把 privileged 字段混入 actor history、DADE 或 command。
- **D7.2 actor 字段**：只允许 cross、vertical、tangent heading、tangent pitch、
  lookahead heading 共 5 列。progress、remaining、curvature、endpoint context 一律
  mask 为 0，并写入 schema hash 和 field audit。
- **D7.3 critic 字段组**：`route_truth`、`actuator`、`disturbance`、
  `sensor_safety` 四组一次实现。`route_truth` 不含 progress/remaining/goal distance；
  `actuator` 记录上一实际执行器、residual、rate、saturation；`disturbance` 记录真实
  current 与 innovation；`sensor_safety` 记录 USBL/DVL age/validity、导航创新和
  safety margin。
- **D7.4 归因边界**：首轮 `C1_DUAL_PATH_PRIVILEGED` 不拆字段组；若 Gate 失败，才用
  mask 做离线/训练诊断，不把诊断结果直接写成新 winner。
- **D7.5 truth-lock / 低噪声导航下的解释**：§4.12.9 默认是
  `unbiased_small_noise_v3` 低噪声（`nav_noise_m=0.001–0.1 m`，`current_est_tau_s=0.05–0.5 s`），actor path 的 position
  源与 critic `route_truth` 的 `eta` 不完全相同，`nav_innovation` 默认非零；只有
  optional `p_nav_truth_lock=true` 时两者才相同且 `nav_innovation=0`。若仍运行 raw
  USBL 仿真，`usbl_innovation` 仍可能是非零 telemetry，但不得回灌 `p_nav`。
  `ORACLE_NAVIGATION` 只属于 optional lock，不得据此合并两套 schema、删掉 privilege
  audit，或把结果当成真实导航鲁棒性证据；真实导航差异必须另开
  `realistic_calibrated` 评价。

D8 已定子项：

- **D8.1 training-only 有界续路**：保持 `max_episode_steps=3000` 和 60 s episode；
  当当前路线剩余 `<=5 m` 时，从同一 path family 追加一段，最多追加 3 次（总计最多
  4 段）。追加段必须与当前路线末端位置和切向连续，不 reset agent、不重置 episode
  时间、不改变 action/reward 语义。
- **D8.2 有限且可审计**：每次 append 写 manifest：episode、stage、原路线长度、新增长度、
  segment 数、触发 progress、path family、schema hash。无法安全追加时回退为不延长，
  写 `ROUTE_EXTENSION_FALLBACK` telemetry；不得静默改成无限路线。
- **D8.3 评价隔离**：续路只存在于训练环境。dev/formal 评价使用固定路线和原 termination；
  若评价 summary 出现 append 计数或训练 length range，判 `INVALID_ROUTE_PROTOCOL`。
- **D8.4 预算**：整次 run 600–2600 iterations；C0–C6 hard caps 默认合计 2600，
  soft caps 取 60%。`600` 是全局最早早停边界，不是每个 stage 的最小预算。
- **D8.5 reward 边界**：D8 不改 reward/terminal 权重；terminal bonus 因续路自然变稀，
  但删除或重配到达奖励必须另立 reward 因子，不与 D8 混跑。

D9 已定子项：

- **D9.0 首波边界（CU7-B）**：首波 `c6_early_stop_enabled=false`，C6 不判断 early
  stop，按 hard cap 跑满 2600；首波 checkpoint 使用 final。第二波才启用 D9.2–D9.5。
- **D9.1 三套时钟分开**：
  1. curriculum promotion：C0–C6 每个 stage 用训练 rollout 的 `window=20`、
     `consecutive=3`，并按该 stage 的 min/max iteration 结束；
  2. arm early stop：第二波启用，C6 使用同一 20/3 滑窗，`min_iterations=600`、
     `max_iterations=2600`；首波不存在这项时钟；
  3. winner judgement：§7.2 的 seed 噪声和最小效应规则。
- **D9.2 early-stop 作用域（第二波）**：只允许在课程进入 C6 后触发最终早停；之前阶段只由
  StageGate 或课程 hard cap 结束。每次 stage 晋级/回退时清空 20-iteration window 和
  consecutive 计数，防止混入旧 stage 指标。
- **D9.3 early-stop 规则（第二波）**：window 填满且 `iteration >= 600` 后，每个 iteration 检查
  route 3D/cross/vertical、completion guard、safety、KL/clip/telemetry 完整性；全部
  规则连续 3 个 iteration 通过才停。任一 miss 将 consecutive 归零。`timeout` 只记录
  和诊断，不作为 early-stop 通过条件，避免 D8 续路后路线长度变化污染 20/3 规则。
- **D9.4 阈值冻结（第二波）**：route 最小效应和 early-stop 目标值在 P0 baseline 完成后、首次
  formal 前冻结；冻结后不得根据 treatment 结果改阈值。参考代码提供结构和 `20/3`
  规则，不复制它的机器人速度阈值。
- **D9.5 到点即停**：首波按 2600 iteration 或课程 hard cap 结束；第二波启用 early-stop
  后，未通过 early-stop 只写 `EARLY_STOP_NOT_REACHED`，不把跑满 2600 写成通过。

D10 已定子项：

- **D10.1 旧容量退役**：Goal182 的新 arm 一律不使用旧 `[256,128]`；旧配置/结果只作
  历史证据，标 `historical_only`，不参与新协议 winner 判定。
- **D10.2 M 档 MLP 平台**：Goal182 固定 M 档 MLP，`actor_hidden_dims` 和
  `critic_hidden_dims` 都是 `[512,256,128]`；不得一边扩 actor、一边保留旧 critic。
- **D10.3 与 Goal181 的顺序**：先完成 Goal182 的 M 档 MLP PPO profile 选择，再进入
  Goal181 的 Transformer M vs MLP M 架构比较；Goal181 必须复用同一 PPO stack。

D11 已定子项：

- **D11.1 baseline 重跑**：`P0_HISTORICAL` 在 Goal182 canonical protocol 下重新运行，
  不允许直接把旧协议的 success/route 数值当作新 baseline。
- **D11.2 两阶段预算**：先跑 1 seed × screen；协议和 telemetry 通过后，再补到 3 seeds
  formal。screen 结果不得写成 formal 结论。
- **D11.3 共用 baseline**：3-seed `P0_HISTORICAL` 作为各 Phase 共用的协议基线；每个
  treatment 只在同一 baseline 上比较一个因子。

D12 已定子项：

- **D12.1 单 run 语义**：C0–C6 是同一个训练 run 内的难度课程，不是六次独立训练。
  网络参数、optimizer 和 iteration 在晋级时连续保留。
- **D12.2 更新依据**：每个 PPO iteration 使用当前 rollout 的 reward/return 更新；奖励
  来自当前采样到的 curriculum stage，不存在“C0–C6 全部跑完后的总奖励”再统一更新。
- **D12.3 不复制 staged runner**：首轮不使用参考工程的独立 stage 重启、
  `load_weights_only`、fresh optimizer/iteration；若后续要研究训练组织，另立独立 goal。

D13 已选方向（reward 目标已确认，StageGate 待定）：

- **D13.1 同时改 reward 与 StageGate**：采用 C，不把 reward 和晋级标准拆成两个不同
  goal；训练奖励使用 §4.9 的 route-centric 目标，课程晋级以 route cross/vertical
  为主，3D 只作综合报告；不能继续让 progress/到达终点主导。
- **D13.2 reward 目标表已确认**：§4.9 是后续 reward 修改的唯一目标表；纯端到端与
  ALOS+本体分别使用各自一列。表中“删除”均为权重置 0、接口保留；标“审计起点”的权重
  先按表实现，只读 scale audit 后若需改值，必须回填本文件并再次确认。
- **D13.3 `pc_ppo_scale` 单独处理**：暂保留 `0.01`，不与 reward 形状一起改；后续作为
  独立尺度因子比较 `0.01 / 0.1 / 1.0`，并观察 value loss、explained variance、grad norm。
- **D13.4 StageGate v1 口径已确认**：硬否决只看 `safety`；防停滞用
  `path_completion_ratio`；主门控为 cross/vertical RMSE；pre-goal 3D 仅作 telemetry；
  `timeout`、`plane_miss`、`success`、3D RMSE、overshoot 仅作诊断；routine 晋级用
  训练 rollout 的 20/3，frozen-dev 只在 stage 边界和最终 formal 运行。详见 §4.9.1。
- **D13.5 阈值标定政策已确认**：纯端到端与 ALOS 分开生成 stage threshold 表，但使用
  同一相对规则；首轮只做总体 route gate，per-stratum 仅报告 worst stratum；不设
  到达/success floor，completion + safety 是唯一防病态硬条件；先 shadow 后 enforce。
  详见 §4.9.2。
- **D13.6 StageGate 数值仍待定**：具体阈值、最终 margin 和 shadow 切换点必须由
  reference 标定和 P0 baseline 产生，不得按 treatment 结果调整。routine 晋级使用
  训练 rollout 的 20/3；边界/final 评价使用 frozen-dev；两者不得互相替代。
- **D13.7 冻结边界**：在 reward 目标表、scale audit 和 StageGate 阈值全部写入并冻结前，
  不修改 reward/StageGate，不生成 `goal182_core_pe2e_matrix.json` /
  `goal182_core_alosb_matrix.json` / `goal182_eval_matrix.json`，不启动训练；不得用
  “已经选了 C”或“目标表已确认”作为提前改配置的理由。

D14 已定子项：

- **D14.1**：所有新协议 arm 固定 `gamma=0.995`、`lam=0.95`；不把 gamma/lambda 作为
  Goal182 的扫描因子。
- **D14.2**：若后续怀疑信用跨度是瓶颈，另立 cheap ablation，不在 PPO stack 首轮中
  边跑边改。

D15 已定子项：

- **D15.1**：所有新协议 arm 固定 `entropy_coef=0.005`，与参考工程一致。
- **D15.2**：Phase 4 的 `E0_CURRENT` 与 `E1_REFERENCE_STD` 共用 `entropy_coef=0.005`，
  只改变 `init_std`；不得把 entropy 和 init std 一起写成 treatment。

D16 已定子项：

- **D16.1**：所有 arm 共用同一个 BC 底座 checkpoint，PPO 残差头使用同一初始化规则；
  初始化差异只能来自训练 seed，不得给某个 arm 单独换初始化。
- **D16.2**：每个 arm 都 fresh optimizer、`iteration=0`；不从另一个 arm 的
  checkpoint 直接续跑，也不恢复旧 optimizer 状态。
- **D16.3**：评价使用 final 或 early-stop checkpoint；禁止在 dev 上挑 best
  checkpoint，禁止按验证集结果回选模型。

D17 已定子项：

- **D17.1**：D10 已冻结 M 档 MLP actor/critic `[512,256,128]`；旧 `[256,128]` 只作
  `historical_only`，不再作为 Goal182 的可选容量。

D18 已定子项：

- **D18.0**：按 CU7-B，首波 `c6_early_stop_enabled=false`；C6 不判断 early stop，
  正式 run 按 hard cap 跑满 2600，首波 checkpoint 使用 final。第二波才启用 D18.1–D18.3。
- **D18.1（第二波）**：C6 arm early stop 只在课程进入 C6、window 填满且
  `iteration >= 600` 后判断。
- **D18.2**：C0–C5 由 routine StageGate 或课程 hard cap 结束；routine StageGate 按
  R5-D 使用 training-rollout 20/3。第二波启用 C6 early stop 后，它不得与 C6 arm
  early-stop counter 混用；首波不存在 C6 early-stop counter。
- **D18.3（第二波）**：每次 stage 晋级/回退都清空 routine gate 和 early-stop 两套
  window 与 consecutive 计数；首波只清空 routine gate，并在 manifest 显式记录
  `c6_early_stop_enabled=false`。

## 10. Git 与提交边界

禁止 `git add -A`。只 add 本 goal 文档、matrix、脚本、测试、result/decision/ledger、
来源清单和必要索引；不提交 checkpoint、trace、日志或 cache。参考快照位于被忽略的
`third_party/`，不进入常规 Git；只提交来源清单、许可证和治理记录。

提交前：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal182 \
  --goal-doc doc/goals/goal182_ppo_stack_governance_20261114.md \
  --strict
python ../总协调/scripts/check_magic_number_provenance.py \
  --repo . \
  --registry ../总协调/configs/governance/magic_number_registry.json \
  --changed --base origin/main --strict
git diff --cached --check
```

是否 commit/push 必须由用户明确批准；不自动提交。

## 11. Goal 提示词

```text
你现在执行 Goal182：PPO 训练栈与 residual authority 治理实验。

先读：
- doc/goals/goal182_ppo_stack_governance_20261114.md
- doc/goals/goal179_head_only_credit_diagnostics_20261114.md
- doc/goals/goal176_result_20261114.md
- doc/goals/goal177_result_20261114.md
- doc/goals/goal178_result_20261114.md
- configs/governance/goal179_matrix.json

硬约束：
1. 每个 Phase 只允许一个因子变化；禁止 authority + norm + clip + entropy 混跑；
2. Goal179 Gate A 未通过前，不启动 Phase 1 训练；
3. D13 的 reward 目标表已确认（§4.9），但 StageGate 与 scale audit 未定稿；在 D13
   完整冻结前，不改 reward、termination、success、goal radius、BC checkpoint、
   held-out，也不得生成 matrix 或启动训练；
4. 所有训练启用 curriculum，并保留逐 episode provenance；
5. 先 screen（1 seed × 5 iterations），方向正确且 instrumentation 完整才进 formal；
6. formal 至少 3 seeds；winner 必须超过 seed 噪声，否则判 INCONCLUSIVE_SEED_NOISE；
7. 不打开 held-out；不把 success 单项当主指标；
8. 失败先写 blocker、至少两个可证伪 hypothesis 和 decision；
9. 不自动 commit/push。
10. 通道范围必须保留 `stern-only / rudder-only / 3D(all)` 三组；3D 是最终目标，
    不得因为 rudder-only 短期更好就删掉 3D。
11. 参考代码只能通过 pinned snapshot + 来源清单使用；禁止把参考仓库直接复制进
    `src/`，禁止生产代码直接 import `third_party`。许可证和复用规则见
    `总协调/doc/governance/REFERENCE_CODE_POLICY.md`。
12. 遵守 §4.12：arm/run/cell 必须使用 canonical 命名，matrix 使用
    `auv_goal_matrix.v2`，正式结果必须同时产出 `report_rows.jsonl`、`report.md` 和
    完整性计数；禁止把 `level` 同时当作 stage、pressure、route 或 radius。
13. 评价必须显式列出 `stage × route_profile × pressure_profile × radius`；当前
    `strict125` 已移除，radius 只用 `2.0 m`。没有 route 轴的 `3D RMSE` 不能写成
    3D 几何泛化结论。

执行顺序：
- Step 0：建立参考快照和来源清单（§4.0），确认 §9 待定项，生成 matrix；实现并运行
  protocol、matrix/naming/report、magic-number provenance 和 research-ledger checker；
- Step 1–2：实现配置 schema、单元测试、runner、auditor；
- Step 3–4：Phase 0 authority calibration；
- Step 5：Phase 1 channel scope（rudder/stern/3D）+ Phase 1b 3D schedule；
- Step 6–10：Phase 2–6 单因子实验；
- Step 11：Phase 7 integration；
- Step 12–13：写 result/ledger/STATE，等待用户批准后提交。
```

## 12. 自主迭代契约

本 goal 必须执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每个 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时先查论文、查代码并写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue / pivot / stop`；
5. `pivot` 时写新 goal 和新 prompt，不修改失败 Gate 的阈值；
6. 改变任务成功定义、打开 held-out、改主算法族或 commit/push 前必须停止并询问用户；
7. 不把 `INVALID` 写成 `FAIL`，也不把 `FAIL` 写成 `PASS`。

## 13. 决策记录

| 日期 | 决策 | 结论 | 备注 |
|---|---|---|---|
| 2026-11-14 | D1 authority 组成 | 采用 B：独立 `bc_authority` + per-channel `ppo_authority` | `bc_authority=0.75` 只作历史锚点；BC authority 另立因子 |
| 2026-11-15 | D2 通道范围 | 采用三组：`stern-only`、`rudder-only`、`3D(all)` | 3D 是最终目标；Phase 0/1 必须保留三组对照 |
| 2026-11-15 | D3 归一化 | 采用 A+E：保留 `PCNormalization` 固定前端，再叠加 empirical actor+critic 归一化 | `until=1e8`、`eps=1e-2`；物理 clip 默认不做 |
| 2026-11-15 | D4 advantage 归一化 | 采用 full-rollout，与参考工程默认一致 | minibatch advantage 只作未来 cheap ablation |
| 2026-11-15 | D5 探索幅度 | 仿照参考工程：`init_std=0.3`、scalar std、`clip_actions=1.0` | 首轮不加 bounds/anneal；有效 applied std 仅作 telemetry |
| 2026-11-15 | D6 trust region | 仿照参考工程整包：`clip=0.20`、`desired_kl=0.01`、`schedule=adaptive`、初始共享 `lr=3e-4` | 使用 rsl_rl 2.3.3 内置 `1/1.5×` LR 规则；actor/critic 不拆 LR；`T3_CLIP_ONLY` 后置 |
| 2026-11-17 | D6.3 warmstart LR 例外 | 用户确认：BCFT warmstart arm 允许拆 actor/critic LR，使用不同学习率 | BCFT finetune 使用 actor `1e-5`、critic `3e-4`、`schedule=fixed`；该例外不改变 fresh/TRRES 的参考 profile |
| 2026-11-17 | D3/D5 warmstart 作用域 | BC warmstart arm 不启用 empirical actor normalization，`init_std=0.05` | 仅作用于 BCFT finetune；BCRES/TRRES 暂保留 reference recipe，待 C0 诊断后再定；不得把 empirical + `std=0.3` 重新带回 warmstart |
| 2026-11-15 | D7 path partition | 一步到位：actor 5-D 非特权 path block；critic 独立 privileged path/actuator/disturbance/sensor-safety；删除 progress/remaining/curvature/endpoint | 不拆 C1–C3 训练臂；字段 mask 只用于诊断；泄漏判 `PATH_PRIVILEGE_LEAK` |
| 2026-11-15 | D8 预算与续路 | training-only 有界续路（最多 4 段）；整次 run `600–2600` iterations；C0–C6 hard caps 合计 2600 | 60 s episode 和评价路线不变；续路不得进入评价；reward 另立因子 |
| 2026-11-15 | D9 晋级/早停 | 仿照参考 `min=600`、`max=2600`、`window=20`、`consecutive=3`；仅用于 arm early stop；C0–C6 保留现有连续 2 次 StageGate | 20/3 不用于课程晋级；阈值在 P0 baseline 后冻结；`timeout` 只诊断，不作为 early-stop 条件 |
| 2026-11-15 | D10 网络容量与 Goal181 顺序 | 旧 `[256,128]` 退役；Goal182 固定 M 档 MLP `[512,256,128]` 选择 PPO profile，随后 Goal181 比较 M 档 Transformer vs M 档 MLP | 旧网络只作 `historical_only`；actor/critic 必须同容量 |
| 2026-11-15 | D11 历史 baseline | 重跑 `P0_HISTORICAL`；先 1 seed screen，再补 3 seeds formal，作为共用新协议 baseline | 禁止跨协议直接比较历史结果 |
| 2026-11-15 | D12 训练组织 | 不照搬参考式多 stage runner；保持单 run C0–C6 StageGate，晋级不重置网络/optimizer | 每轮使用当前 stage rollout 的 reward/return 更新，不存在 C0–C6 总奖励 |
| 2026-11-15 | D13 reward/StageGate | 选择 C；两套 route-centric reward 修改目标表已确认（§4.9）；StageGate v1 口径已确认（§4.9.1）；阈值标定政策已确认（§4.9.2） | StageGate v1：`safety` 硬否决、completion 防停滞、cross/vertical 主门控；`timeout/plane_miss/success/3D/pre-goal 3D/overshoot` 仅诊断。纯端到端/ALOS 分开标定、首轮只做总体 gate、不设 success floor、先 shadow 后 enforce |
| 2026-11-17 | D13.6 参数冻结 | 用户确认 D13.6a–d 按推荐冻结；cross/vertical margin `0.10`，不设 `pre_goal_3d_rmse_m` margin；completion floor `0.80–0.90`；首波 shadow | `pre_goal_3d_rmse_m` 仍记录但不进入 route gate；soft cap `72/108/156/216/252/336/420`；阈值分开标定 |
| 2026-11-17 | INFRA-D11=B 覆盖 D13.6d | 首波 MOD4 在 PE2E/ALOSB 阈值标定并冻结后直接 `enforce`，不再先跑一整轮 shadow；标定只用 reference/P0，不得用 treatment 结果回调阈值 | D13.6d 的“首波 shadow”保留为历史；launcher 已切 `--curriculum-route-gate-mode enforce`，缺阈值直接失败；标定后先跑 20/3 smoke 再启动 18-run；`INFRA-D11` 是基础设施编号，不是本文历史 baseline 的 `D11` |
| 2026-11-15 | D14 gamma/lambda | `gamma=0.995`、`lam=0.95` | 保留 AUV 长时程信用跨度，不照搬 `0.99` |
| 2026-11-15 | D15 entropy | `entropy_coef=0.005` 作为所有新 arm 公共配置 | Phase 4 只改变 `init_std`，不把 entropy 混入探索因子 |
| 2026-11-15 | D16 warmstart/checkpoint | 同一 BC 底座与残差初始化；fresh optimizer；iteration=0；final/early-stop checkpoint；禁止 dev best 和跨 arm resume | 保证 arm 公平并隔离验证集 |
| 2026-11-16 | TRRES encoder 初始化 | 正式 TRRES 只复用 encoder 架构，不借用 BC/其他 arm 的已训练权重；fresh init、fresh optimizer、iteration=0 | `MOD4/MOD3` 仅共享架构、seed 和输入 schema；manifest 记录 `encoder_init=standard_fresh`、`encoder_init_checkpoint=null`；BC encoder 初始化另立诊断族 `TRRES_BCENC_INIT_CMD/ACT`，不进入正式矩阵 |
| 2026-11-15 | D17 capacity | 并入 D10：固定 M 档 MLP `[512,256,128]`，actor/critic 同容量 | 旧 `[256,128]` 仅 `historical_only` |
| 2026-11-16 | 残差接口轴 | 用户确认残差组两种接口都有价值；新增条件轴 `AX_RESIDUAL_INTERFACE`，`ALOSB × TRRES` 拆成 `TRRES_CMD` 与 `TRRES_ACT` 两个正式 arm | `BCRES` 当前只支持执行器残差；`BCRES_CMD` 需兼容命令输出 BC，另立 arm 族 |
| 2026-11-15 | D18 early-stop 作用域 | C6 arm early stop 只在 C6 使用 20/3；C0–C5 由 StageGate/hard cap 结束；StageGate 自身按 R5-D 使用 training-rollout 20/3 | arm early stop 与 StageGate 使用独立 counter；stage 变化时两套 window 和 consecutive 都清空 |
| 2026-11-15 | R6 heading/speed 核 | heading 与速度跟踪改用零基线指数核 | `path_heading_std_rad=cmd_heading_std_rad=0.5`、`cmd_speed_std_mps=0.25`；速度地板与 stall 口径见 Q1/Q11 行 |
| 2026-11-15 | R6 数值筛选 | E2E：`w_path_cross=3.5`、`w_path_vertical=1.5`、`w_yaw=w_path_pitch=1.0`；ALOS：cross/vertical 为 `3.5/1.5`，`w_cmd_yaw=w_cmd_pitch=1.0`；两种模式 `w_phi=3.0`、`w_p=0.02` | `w_cmd_speed=2.5`；横滚角项加强到 3.0，角速率项保持 0.02；需 scale audit |
| 2026-11-15 | R6 speed 语义 | 选 `S-DEF-G`：`w_cmd_speed` 跟踪 `ground_velocity_body[:,0]` | 不复用 `privileged_surge`；reward 使用独立 `reward_ground_surge`，hold/critic 若同步修改需另记录 |
| 2026-11-15 | Q1/Q11 reward speed 口径 | Q1-E + Q11-A：首轮移除奖励侧速度地板，guidance 保底加速保留；新增 stall 接口但权重为 0；reward 参考使用补偿前地速向量 | `w_route_speed_pc=0`、`w_route_ground_speed_stall=0`；`w_cmd_speed` 使用 `reference_ground_velocity_ned` 转机体系后的 x 分量；旧 floor 不得作为 stall 别名 |
| 2026-11-15 | Q2 goal-plane miss | 采用 Q2-A：训练 reward 的 `goal_plane_miss_penalty` 与 `goal_plane_miss_distance_weight` 都置 0 | `pc_goal_plane_miss_termination_enabled` 保持；终止和诊断仍计数，但不参与 reward |
| 2026-11-15 | Q3 饱和奖励 | 采用 Q3-A：`w_action_clip_attempt_pc=0`，只保留 `w_actuator_material_sat_pc=0.02` 的实际材料饱和事件 | 新名为 canonical，旧 `w_sat_pc` / `w_material_sat_pc` 只作兼容别名；尝试项若未来启用，必须使用任何 clip 之前的 raw action |
| 2026-11-15 | Q4 二阶动作平滑 | 采用 Q4-C：`w_action_smooth_pc=0.005`，两种控制模式相同 | 只罚舵和艉翼的二阶差分，RPM 不参与；需要两步历史并在 reset 时清零 |
| 2026-11-15 | Q5 heading/pitch 互斥 | 采用 Q5-A：path_* 禁用 `w_cmd_yaw/w_cmd_pitch`，ALOS 禁用 `w_yaw/w_path_pitch` | 错误码 `DUPLICATE_PATH_COMMAND_TRACKING`；speed 项不受影响；要用 parser 和 matrix checker 可执行拦截 |
| 2026-11-16 | Q5.1/Q5.2 speed reward 与统一上下限 | `alos`、`path_fixed_speed`、`path_learned_speed` 都使用 `w_cmd_speed=2.5`；`w_u=0` 仅作 legacy 兼容。三个模式统一采用 `v_min=vehicle.min_speed_mps`、`v_max=vehicle.max_speed_mps` | ALOS 最终指令与终点制动都不再使用 `guidance.minimum_speed_ratio`；path 模式分别用 `v_ref=clamp(v_cmd_episode, v_min, v_max)` 和 `command.speed_mps=clamp(solution.command_speed_mps, v_min, v_max)`；`path_learned_speed` 仅允许速度参考在物理上下限内变化且仍是次要实验；旧 `w_u` 非零、或与 `w_cmd_speed` 同时非零，均须由 config guard 拒绝 |
| 2026-11-16 | ALOS terminal envelope | robust terminal alignment 固定为 `5 m`（`terminal_alignment_min_m=terminal_alignment_max_m=5.0`） | 修复旧 `50 m` 在 35–50 m 路线上从起点进入 terminal、覆盖路径切向制导的问题；`terminal_capture_distance_m` 保持独立 legacy 语义，后续单独确认 |
| 2026-11-16 | ALOS 在线路线契约 | 训练 D8 用持续增长 path，terminal 只认真实 mission goal；部署可用 rolling buffer，但 buffer 末端不等于 mission goal | 追加同 family 段时保留 projection hint；路径替换/重规划需全路径投影并重置 hint；buffer 前方至少覆盖 `max_lookahead_m + terminal margin` |
| 2026-11-16 | magic-number provenance | 速度上下限必须引用 `vehicle.min_speed_mps` / `vehicle.max_speed_mps`，不得复制 `1.0` / `2.5`；新增 `MAGIC_NUMBER_WITHOUT_PROVENANCE` 静态检查目标 | 跨仓库规范见 `总协调/doc/governance/MAGIC_NUMBER_PROVENANCE.md`；先建立 registry、baseline 和 checker 测试，再接入 G182-8；Goal182 的 touched files 不得新增违规，也不得用 inline 豁免绕过速度包络 |
| 2026-11-15 | 评价半径 | 取消 `strict125 1.25 m`，Goal182 只保留 default `2.0 m` | 历史 Goal 文档不回溯改写；active matrix、训练和评价不得出现 strict125 |
| 2026-11-15 | 深度终止 | `max_depth_m=510.0`，真实 `z>510 m` 触发 `DEPTH_LIMIT`；z bounds 上界设为 `520 m` | 保持 `min_depth_m=0.0`，x/y bounds 不变；z bounds 必须高于 depth threshold，否则会被 `OUT_OF_BOUNDS` 遮蔽 |
| 2026-11-15 | 参考代码治理 | 采用只读 pinned snapshot；逐文件许可证审计；生产代码禁止直接 import `third_party` | 参考 `Wheel-Legged-Lab@e61bfe1...`；规则见 `总协调/doc/governance/REFERENCE_CODE_POLICY.md` |
| 2026-11-16 | 训练路线来源 | C0–C6 的水平 P0–P4 路径族、长度和起始深度沿用现有；第一波不续路，D8 append 只放第二波。速度部分已由同日“中心向外速度课程”覆盖 | 评价固定路线且不续路；路线末端扩展只在 training-only D8 生效；速度以 §4.12.7.2 / CU9 为准 |
| 2026-11-16 | 垂直三维课程 | C2 改为 50% 纯垂向 `1.75 m/1 cycle` + 50% 纯水平 `1.75 m/1 cycle`；C3/C4 加 `2.0 m/2 cycles`，C5 加 `3.0 m/1.5 cycles`，C6 加 `5.0 m/1 cycle` | 见 §4.12.7.1；旧 C3 600-it 结果统一 `historical_only`；P3/P4 当前不消费 depth amplitude/cycles，需代码实现 |
| 2026-11-16 | ALOS guidance pre-screen | 先做 eval-only 六臂筛选：`A0 3.0/2.0/2.5/1.0`、`A1 1.5/1.5/2.5/2.5`、`A2 1.5/1.5/2.5/1.0`、`A3 3.0/2.0/2.5/2.5`、`A4 1.0/1.0/2.5/2.5`、`A5 2.0/2.0/2.5/2.5` | 字段顺序为 horizontal lookahead / vertical lookahead / horizontal gain / vertical gain；统一 `terminal_alignment=5 m`；今晚 C3/C4 六个 arm 各 4 seeds（seeds 20–23），共 48 个 seed-episode；seed 不再扩；C3/C4 分开汇总；不训练、不改默认 config、不生成 formal matrix；执行细节见 §4.10 与 review R9 |
| 2026-11-16 | 今晚联合训练授权 | 用户允许今晚并行做 ALOS screen、BC 5D 重训和 canonical-profile PPO smoke/screen | 顺序为 ALOS → 5D BC → PPO；5D schema 卡住则记 `BC5D_SCHEMA_BLOCKED`，不得用旧 8D 冒充；PPO 必须绑定 `PPO-FLAT-4096x48-E5-MB4`，reward 未实现时只能标 `PPO_CONTRACT_SMOKE` / `PPO_REWARD_PENDING`；完整微调仍必做；详见 §4.11 |
| 2026-11-16 | Actor 结构整改前置 | 今晚把 actor 结构整改并入 PPO 前置：5-D non-privileged path、删除 progress/remaining/curvature/endpoint、M 档 actor/critic `[512,256,128]`、privileged critic 独立 encoder、actor kind/schema metadata | 审计 `PATH_PRIVILEGE_LEAK` / `MLP_CRITIC_CAPACITY_MISMATCH` / `SILENT_ACTOR_METADATA`；MLP 整改通过后才允许 Goal181 M Transformer 实现/smoke，不能抢跑或掩盖控制契约问题 |
| 2026-11-16 | BC 5D 重训 | 采用 R4b-B：重训 `path_dim=5` BC，删除 progress/remaining/curvature；旧 8D DAGGER2 只作历史对照 | 数据使用完整 DAGGER2 `demos_merged_dagger2/demos.npz` 并校验 SHA；5D schema/入口先本地 commit；训练优先在 8×4090 独立 worktree 执行；与旧 8D 做 BC_ONLY paired 对照后再进入 PPO |
| 2026-11-16 | 残差→微调失败判据 | 采用三条判据：权限失败、因果失败、信用失败；满足任意一条才允许切微调 | 切换前记录 `RESIDUAL_AUTHORITY_FAIL` / `RESIDUAL_ROUTE_NO_GAIN` / `RESIDUAL_CREDIT_FAIL` 并在 ledger 写 `pivot`；smoke 崩溃、配置 typo、单 seed 波动不触发 |
| 2026-11-16 | StageGate 评估节奏 | 采用 R5-D：照参考工程，routine 晋级用训练 rollout 的 20/3；frozen-dev 只在 stage 边界和最终 formal 使用 | 每 stage 保留 min/max iteration；dev seeds `20–22`（3 seeds，EV1 已修正）仅用于边界/final；首波 C6 early stop 关闭，第二波启用后同样用训练 rollout；首次 formal 前冻结 |
| 2026-11-16 | Goal180/181 边界 | 采用 R8-A：Goal182 是唯一 canonical 主栈；Goal180 降级为历史诊断，Goal181 只保留架构轴 | 两者都必须复用 Goal182 的 5D actor、reward、PPO profile 和 600–2600 预算；旧 8D/16-iteration 结果只作历史参考；Goal182 profile 锁定前不启动 Goal181 formal |
| 2026-11-16 | Q7 reward 路径基准 | 采用 Q7-A：reward 侧路径误差全部使用 true `eta` 重新投影 | actor observation 仍只用 sensor/projection；cross、vertical、route speed、pre-goal 3D 必须共用同一投影入口，不能混用 sensor/true state |
| 2026-11-16 | Q8 速度平滑归一化 | 采用 Q8-B：公式改为 `-w*((Δv)/0.25)²`，权重由 `0.1` 改为 `0.025` | 与旧 `0.1*(Δv²/0.25)` 等强度；breakdown/telemetry 记录公式版本，旧 0.1 config 不得直接混用 |
| 2026-11-16 | Q9 能耗奖励范围 | 采用 Q9-A：`w_path_energy=0.01` 保持不变，只罚螺旋桨能耗 | 命名与 telemetry 标注 propeller-only proxy；舵/艉翼能耗不加进该项，后续需要时另立字段 |
| 2026-11-16 | Q10 全局尺度与安全终止 | 采用 Q10-A：`pc_ppo_scale=0.01` 与 `safety_terminal_penalty=-50` 都冻结 | safety 只在实际安全终止步计一次，乘 scale 后 `-0.5`；两者都不进入首轮扫参，尺度审计另立因子 |
| 2026-11-16 | Q12 动作平滑语义 | 采用 Q12-A：`w_action_pc` 继续罚实际执行器差分 | telemetry 同时记录 raw、clipped-requested、actual 三种 delta；若后续发现抖动来自 policy 侧，再另立 requested 项 |
| 2026-11-16 | Q13 pitch-rate 惩罚 | 采用 Q13-A：首轮不新增 pitch-rate 惩罚 | 只记录 `pitch_rate`、p95 和与指令 pitch 的关系；确认振荡不是指令本身造成后，才另立 `w_q_track` |
| 2026-11-16 | M1 矩阵/命名/汇报 | 用户将实验矩阵、成果汇报和实验组命名提升为 Goal182 顶级要求；原则已确认，M1–M7 具体形式待逐项确认 | 新增 §4.12：canonical arm/run/cell ID、`auv_goal_matrix.v2`、`report_rows.jsonl`、route profile 与 3D 审计；补充 `depth_profiles_m` 未接线审计；未满足记 `INVALID_EXPERIMENT_CONTRACT`。当前 600-it 结果固定标 `C3_FIXED_STAGE / NOT_FORMAL` |
| 2026-11-16 | CM 核心组合矩阵与轴命名 | 用户确认核心矩阵是训练组合矩阵，不是评价矩阵；定义 `G182-CORE-COMP`、四个核心轴及全部档位短码，并把评价轴单独命名 | 核心轴：`AX_ACTOR_BACKEND`、`AX_CONTROL_CONTRACT`、`AX_LEARNING_COMPOSITION`、`AX_MODULE_SET`；新增 `goal182_axis_registry.json`；`goal182_core_composition_matrix.json` 与 `goal182_eval_matrix.json` 分开生成；CM1–CM9 待逐项确认 |
| 2026-11-16 | 首批 core arm 范围 | 首批优先跑 `MOD4 × AB_MLP_M` 的 6 个合法 arm；紧接同 family 的 `MOD3` 配对对照，随后跑 `MOD2 × AB_MLP_M` 并紧接 `MOD1` 对照 | 用户明确要求 privileged treatment 顺序为 `MOD4` 先于 `MOD2`；`MOD4` 是 priority treatment #1，`MOD2` 是 #2；`MOD4` 前必须先完成 compatible current-encoder BC；每个 module-set pair 最多 12 个 MLP arm，两对合计最多 24 个；`MOD0` 最后作 no-history baseline；Transformer 后置 |
| 2026-11-16 | module-set 短码 | `M0–M4` 与容量 M 档重名，module set 统一改为 `MOD0–MOD4`；档位 ID 仍用 `MS_*` | axis registry、matrix、run 命名和汇报必须同步；裸 `M0–M4` 只作 deprecated alias |
| 2026-11-16 | 中心向外速度课程 | 采用 `SPEED_V3_CENTER_OUT_5S`：以 `1.5 m/s` 为中心，C0–C6 依次为 `1.40–1.60 / 1.40–1.80 / 1.20–1.80 / 1.20–2.00 / 1.00–2.00 / 1.00–2.20 / 1.00–2.50 m/s`；训练每 `5 s` 重采样速度目标 | 覆盖速度引起的舵效变化；先扩高速侧、再扩低速侧，交替到物理包络。正式评价固定 route speed、不重采样；`0.9 m/s` 仅 eval/OOD；新增逐段 speed provenance、静态速度可行性检查和 G182-9。见 §4.12.7.2 / CU9 |
| 2026-11-16 | 测评速度与 C3 主比较 | 所有测评 episode 内速度固定；速度评价固定四组 `1.0/1.5/2.0/2.5 m/s`，覆盖 `C2/C3/C4/C6` 全部正式 stage；为隔离速度因子，`SPEED_ENVELOPE` 只用 `nominal` 压力。正式 core 同为 `C2/C3/C4/C6`，每 arm `48 cells / 144 episodes`；`SPEED_ENVELOPE` 每 arm `48 cells / 144 episodes` | `0.9 m/s` 仅 OOD；C3 另保留四档初评并标 `QUICK_SCREEN / NOT_FORMAL`；速度评价不是 C3 专属；current/noise/plant 压力不进入速度轴，避免多因子混淆；固定速度按 stage/speed 分开汇报，不得先 pooled。见 §4.12.4.1 |
| 2026-11-16 | M1 + CM1 矩阵命名与落盘 | 确认 `G182_<PHASE>_<DOMAIN>_<TREATMENT>` 为唯一 arm 命名；核心组合使用 `G182_INT_STACK_<ACTOR>_<CONTROL>_<LEARN>_<MODULE>`；`G182-CORE-COMP` 只作 family/human index，实际拆成 PE2E/ALOSB 两张 program matrix | 索引为 `goal182_core_composition_index.json`；训练矩阵为 `goal182_core_pe2e_matrix.json` / `goal182_core_alosb_matrix.json`；schema 为 `auv_core_composition_matrix.v1`。旧单文件 `goal182_core_composition_matrix.json` 与 `G182C_...` 模板退役。索引不登记 checker，禁止跨 program pooling。见 §4.13.1 |
| 2026-11-16 | M2 评价矩阵 v2 与 cell 语义 | 采用 `auv_goal_matrix.v2`；每个 tier 独立 matrix；arm/baseline 通过 `evaluation_set` 声明覆盖；eval seeds 用显式列表；cell 不含 eval seed，episode 才含 eval seed；checker 自动重算 cells/episodes | `policy_cells` 和 `zero_cells` 由 coverage 展开；`expected_episodes` 另算；缺 cell/未登记 cell/算术不一致/held-out 泄漏分别判 `INVALID_INCOMPLETE_EVAL` / `INVALID_UNREGISTERED_CELL` / `INVALID_MATRIX_ARITHMETIC` / `INVALID_HELDOUT_LEAK`。见 §4.12.2 |
| 2026-11-16 | 终点捕获统计边界 | 终点捕获/`goal_reached` 可以记录为描述性统计，但绝对不能作为评价指标 | 禁止进入 Gate、winner 排名、前三名选择、early-stop、路线泛化入选、matrix completeness 或 primary/secondary KPI；如报告保留，只能标 `DESCRIPTIVE_ONLY_NOT_A_METRIC`。success 必须由 EV5 单独定义，不能把终点捕获改名成 success。见 §5.2 |
| 2026-11-16 | CM2 actor backend | `AX_ACTOR_BACKEND` 只保留 `AB_MLP_M / AB_TA_M` 两档；`token_mlp` 不新增第三档，而是 `AB_MLP_M` 内由 `actor_kind` 标识的公平对照 | `actor_kind=mlp|token_mlp|transformer` 必须写入 config/checkpoint/matrix/manifest；`AB_TA_M` 只允许 `transformer`，不得用历史 encoder 冒充；token-MLP 与 Transformer 必须共享 token 集合、顺序、宽度和 pooling，否则判 `TOKENIZATION_CONFOUND`；M 档容量配对与 ±20% 参数/latency 规则见 §4.13.2 |
| 2026-11-16 | Actor metadata 显式化 | 用户确认后续基础设施必须把 `actor_kind` 显式写入 config/manifest；扩为 `actor_kind / actor_input_contract / token_contract / pooling` 四项一起显式 | 新 profile-bound config 缺失即 `ACTOR_METADATA_MISSING`；首波基线固定 `mlp/fused/none/none`；历史 config/checkpoint 标 `historical_only`。实现见 `goal182_infra_audit_20261116.md` |
| 2026-11-16 | CM3 learning composition | 正式固定 `BCFT / BCRES / TRRES` 三个 family；`TRRES` 展开为 `TRRES_CMD / TRRES_ACT`；每 `actor_backend × module_set` 为 6 个正式 arm | `SCRATCH` 只作非正式 optional diagnostic，不进核心矩阵、第一波和 60 个正式组合，不参加 winner/路线泛化，不能替代 `BCFT`；完整 PPO 微调始终必做。见 §4.13.2 |
| 2026-11-16 | CQ1 current estimator profile | 采用 A：Goal182 固定一套 `sensors.profile`；`current_estimator_level` 只作 metadata，不切换估计器 | 后续单独立项降低 current estimate 延迟；不得把估计器精度变化混入 MOD 或 curriculum；其余 CQ2–CQ8 待核对。见 §4.13.9 |
| 2026-11-16 | CQ2 actor current view | 采用 A：actor 使用 raw `current_estimate_body` | guidance 的 filtered / valid-age-aware current 作为后续独立消融；本轮不改 actor current 语义，不把 EMA 过滤接入 actor。见 §4.13.9 |
| 2026-11-16 | CQ3 effective current source | 采用 A：curriculum 生效，config.current 只作 fallback | 启动器和 manifest 必须打印/记录 effective stage、current speed range、period、shear、profile 和 noise scale；不得静默使用 fallback 值。见 §4.13.9 |
| 2026-11-16 | CQ4 MOD vs sensor accuracy | 采用 A：MOD 只控制 actor/critic 是否消费 current，不改变 `sensors.profile` 或估计器精度 | 未来若研究传感器退化，单独开 `ESTIMATOR_PROFILE` / `SENSOR_STRESS` 轴；不得把精度变化混入 MOD。见 §4.13.9 |
| 2026-11-16 | CQ5 disabled-current token schema | 采用 A+C：MOD1/MOD2 保留固定 3 个零维 side channel，并在 metadata/日志/manifest 显式标记 current disabled | schema 宽度保持稳定；必须记录 `actor_current_enabled`、`actor_current_dim`、`actor_current_side_channel`、`side_channel_dim`，不得靠输入维度推断。见 §4.13.9 |
| 2026-11-16 | CQ6 deprecated current switches | 采用 B：`current_estimator_level` 与 `training.privileged_current_critic` 标记 deprecated，非默认行为直接报错 | `current_estimator_level` 只作 provenance / metadata，不切换 profile；PC 中 `training.privileged_current_critic=true` 报错并提示改用 `pc.critic_path_dim`；manifest 不把旧字段当实验因子。见 §4.13.9 |
| 2026-11-16 | CQ7 DADE current query + attention boundary | 采用 A：mean pooling 时显式不传 `current` query，只有 attention 类 pooling 才要求 query；attention pooling 作为独立 optional diagnostic | 当前 canonical DADE 是 mean pooling，query 被忽略；清理接口语义不改变现有数值行为。`DADE_ATTN_POOL` 不进核心矩阵/首轮正式 arm，且必须与 `AB_TA_M` 的分开登记；后续若跑须固定其他因子并单独验证收益、延迟和跨 seed 稳定性。见 §4.13.9 |
| 2026-11-16 | DADE 预训练触发与最小筛选 | 暂不预训练。若 `MOD1–MOD4` 相对 `MOD0` 仍无实质收益，或 DADE 输出/门控长期接近零，再启动 `D1 → D2 → D3` optional diagnostic | 目的是学 actor-visible 变化规律与噪声/丢测鲁棒，不做完整 world model；D1 预测未来 state/path/current delta，D2 masked history/side-channel，D3 组合。预训练不进入核心矩阵/60 个正式组合；先 1 seed screen，需跨 seed 复现且闭环优于 D0。见 §4.13.10 与 `research/paper_reading/2026-11-16-dade-pretraining-survey.md` |
| 2026-11-16 | CQ8 current estimate cold start | 采用 B：第一条有效 current 测量到达时热启动低通状态和历史队列，不从零爬升 | 不使用真值；manifest 记录 `current_est_start=hot_first_valid_sample`；历史 checkpoint 保留旧语义，不追溯改写。CQ1–CQ8 至此全部确认。见 §4.13.9 |
| 2026-11-16 | CU7 C6 early-stop 时机 | 采用 B：首波不实现 C6 early stop，正式 run 按 hard cap 跑满 2600；第二波再启用 `min=600 / max=2600 / window=20 / consecutive=3` | 首波 checkpoint 使用 final；一次 run 的 routine StageGate 20/3 继续保留；第二波 early-stop counter 必须与 routine StageGate counter 独立，stage 变化时分别清空。见 §4.13.7 与 D18 |
| 2026-11-16 | EV4 绝对深度轴 | 采用 A + 四档：`DEPTH_GENERALIZATION = [10, 80, 250, 500] m`，只给 `MAIN_RANKING` 前三名；固定 C6、`R_STRAIGHT + R_3D_M`、`nominal` 压力、3 train × 3 eval | 每 arm `4 × 2 × 1 × 3 × 3 = 72 episodes`，前三名 `216 episodes`；不改变 `ROUTE_GENERALIZATION` 的 792 episodes/arm，不参与 winner/前三名选择。builder 必须真正消费 `depth_profiles_m`，`depth` 与 `route_profile` 分开计数。见 §4.12.3、§4.13.5、§5.3 |
| 2026-11-16 | EV5 success 口径 | 采用 A：同时报告 `success_pnav_tight`、`success_eta_tight_ever`、`success_eta_tight_at_done`；研究 primary 用 `eta_tight_at_done` | `eta_ever` 只作宽松辅助；默认低噪声 `unbiased_small_noise_v2` 下 `success_pnav_tight` 是“低噪声 synthetic navigation”诊断，不是真实部署；只有 optional `p_nav_truth_lock=true` 时才变成 oracle-navigation 一致性检查。summary canonical 字段为 `success_rate_pnav_tight`，旧 `success_rate` 仅兼容。见 §4.12.9、§5.2 |
| 2026-11-16 | 默认导航噪声与 p_nav lock | 默认 `unbiased_small_noise_v2` + `nav_noise_m=(0.001, 0.1)`，`p_nav = eta + zero-mean noise`；`p_nav_truth_lock` 默认 `false`，只有显式 `true` 时才 `p_nav := eta[:, :3]` 并断言 `max(abs(p_nav - eta)) <= 1e-4 m` | 只改 `p_nav` 噪声；current/DVL/IMU/raw USBL/actuator 不变。新增 `unbiased_small_noise_v2`，全局 `unbiased_noise_v1` 保持不变，避免追溯改历史；manifest 记录 `profile`、`nav_noise_m`、config SHA。旧 `unbiased_noise_v1` 0.05–0.20 和 `realistic_calibrated` 结果标 `historical_only`；真实部署或 sim-to-real 必须另开 `realistic_calibrated`。见 §4.12.9、§5.1、§6.1 |
| 2026-11-16 | EV6 zero baselines | `BCRES` 用同一 checkpoint 内嵌冻结 BC + 残差 0；`BCFT` 用 `warmstart_actor` 的初始 BC checkpoint、PPO 更新 0 次；`TRRES_CMD/ACT` 用同 env/guidance 的零残差 ALOS+PD/PI；`PE2E` 采用 A：不设同底座 zero，标 `ZERO_BASELINE_UNPAIRED` | `BCRES/BCFT/TRRES` 缺 paired zero 判 `INVALID_INCOMPLETE_EVAL`；BCFT 的 BC-only 评测无法统一字段判 `INVALID_ZERO_BASELINE_PROTOCOL`；TRRES 两个 wrapper 若零残差结果不同，分别命名 `zero_alos_cmd` / `zero_alos_act`。PE2E 禁止拿冻结 BC 或零动作冒充 paired zero。见 §4.12.8 EV6、§5.1、§6.1 |
| 2026-11-16 | EV7 checkpoint 选择 | 采用 A：首波固定 `final`；第二波 early stop 后用 early-stop checkpoint；所有 arm 同一规则 | 禁止 dev best-by-eval、禁止多 checkpoint 挑最好；manifest 记录 `checkpoint_selection=final|early_stop`、path/SHA、iteration 和 early-stop 触发信息；违反判 `CHECKPOINT_SELECTION_LEAK`。见 §4.12.8 EV7 |
| 2026-11-16 | EV8 外部 suite 接入与测评命名 | 采用 A：主排名只用 `MAIN_RANKING`，前三名追加 `ROUTE_GENERALIZATION`；`EXTERNAL_GEOMETRY_SUITE`（suite D）只给最终 winner 做外部体检 | suite D 不参与 Gate、winner、前三名、early-stop 或 checkpoint 选择；只按 EV7 的正式 checkpoint 对 winner 的正式训练 seeds 逐一运行，并标 `EXTERNAL_GEOMETRY_ONLY`。命名统一为 `QUICK_SCREEN / MAIN_RANKING / SPEED_ENVELOPE / ROUTE_GENERALIZATION / DEPTH_GENERALIZATION`；旧名 `SCREEN / FORMAL_CORE / SPEED_EVAL / COLLECTION_EVAL / DEPTH_AUDIT / FORMAL_GEOMETRY` 只作 `legacy_alias`。见 §4.12.3、EV8 |
| 2026-11-16 | EV9 canonical 产物 | 采用 A：固定 `matrix_manifest.json`、`report_rows.jsonl`、`report.md`；每个 run 使用 `summary.json`、`episodes.jsonl`、`trace.npz` | `matrix_manifest` 每张 frozen matrix 一份；`report_rows` 每个 cell 一行并聚合 `eval_seeds`；`episodes.jsonl` 每 episode 一行；两者用同一 `episode_id`，且主表必须能从明细复算。缺产物判 `INVALID_ARTIFACT_SET`，无法复算判 `INVALID_REPORT_PROVENANCE`；`trace.npz` 只作诊断，不参与 checkpoint/winner/前三名选择。见 §4.12.5、EV9 |
| 2026-11-16 | Episode 初始状态 | 训练从路线起点 reset：`x/y ±0.5 m`、`z ±0.5 m`、roll 0、pitch/yaw 切向 `±10°`、前向速度 `0.8–1.2×` 命令、侧向/垂向 0、三轴角速度 `±0.05 rad/s`；正式评价使用无扰动名义起点 | 训练推进器 RPM 比例仍取 `U(0,1)`，但基准按初始速度前馈；动作/平滑历史清零，前两步不罚二阶平滑；history 用首帧填满。正式评价 `x/y/z/yaw/pitch` 抖动、初始角速度和速度比例扰动均为 0，RPM 比例 1.0；current/noise/plant 压力仍生效。current 估计用第一次有效测量热启动，禁止读真值。见 §4.12.10 |
| 2026-11-18 | C5/C6 传感器退化范围 | 用户确认：C5/C6 去掉 current estimate 的 transport delay；保留 outage 和噪声。新训练/评测 profile 为 `unbiased_small_noise_v3`（v2 + `current_est_tau_s=0.05–0.5 s`），v2 仅历史兼容；C5/C6 `noise_scale=2.0–3.0`；BC demo、PPO 训练和后续实验都遵循同一契约 | `current_estimator_level` 仍只是 legacy metadata/provenance，`delayed_noisy` / `degraded_outage` 不得作为接线依据。真实 outage 规划见 Goal184 §4.1（`PLANNED / NOT_IMPLEMENTED`）：C5 hazard `0.002–0.008 /s`、duration `0.05–0.3 s`；C6 hazard `0.008–0.020 /s`、duration `0.05–0.3 s`；delay 恒为 0。不得用 `realistic_calibrated` 顺带引入 delay/bias |

## 14. 基础设施决策索引（INFRA-D1…INFRA-D12）

**归属**：本表是 Goal182 内部的决策与实施项，不是 goal 外前置，也不是额外实验。
恢复 Goal182 就是继续执行本表及其后续标定、smoke 和训练。

**编号**：本表统一使用 `INFRA-Dx`。`goal182_infra_audit_20261116.md` §13.3 中的短号
`Dx` 均指本表；跨文档引用必须写 `INFRA-Dx`，避免与 §13 的 `D1–D18`
（架构/训练决策）混淆。尤其注意：`INFRA-D8`（训练续路）对应本文 `D8`；
`INFRA-D11`（20/3 晋级）不是本文 `D11`（历史 baseline）。

| ID | 决策 | 状态 | 证据 / 下一步 |
|---|---|---|---|
| `INFRA-D1` | stage hard cap 逐 iteration 检查，达到即 fail-closed；launcher 拒绝 completed 但 final stage != C6 的 run；`CURRICULUM_STALLED` 暂不终止 | `FIXED` | trainer/launcher/定向测试 |
| `INFRA-D2` | 正式训练前用冻结 5D BC + ALOS reference 标定并冻结 cross/vertical 阈值；未冻结只能 shadow/screen | `FIXED`（注入路径）/ `BLOCKER`（数值） | 缺阈值 launcher 直接失败；阈值待标定 |
| `INFRA-D3` | 先只读诊断 C0 completion 低和 timeout 高的原因；不直接降低 `completion_min` | `DIAGNOSED / USER_DECISION_REQUIRED` | 旧 `speedv3-5d80de2` 最后 20 个 training-rollout：finetune completion `0.595`、residual `0.957`，两者 success `0`；residual vertical RMSE `31.96 m`；旧 run 在 iteration 72 记录 hard-cap 后仍继续。待决定首波前是否调整 C0 |
| `INFRA-D4` | active 评价固定 `2.0 m`，不保留 radius 轴；缺失/非 2.0 判 `INVALID_GOAL182_RADIUS`；success/timeout/plane_miss 只作 telemetry | `FIXED` | launcher/report fail-closed，intent-to-fail 测试通过 |
| `INFRA-D5` | 深度在训练和测评两侧都落实：训练侧 C0–C6 逐 episode `depth_m`；测评侧 `10/80/250/500 m × R_STRAIGHT/R_3D_M`，depth 与 route 分开计数 | `FIXED` | 24 个 depth cell；训练侧深度覆盖测试 |
| `INFRA-D6` | BCRES/BCFT/TRRES 必须有同 evaluation set、同 scenario、同 seed 的 paired zero；PE2E 显式 `ZERO_BASELINE_UNPAIRED` | `FIXED` | manifest 级 pairing + report 级 seed 校验；arms builder 生成 4 个 ALOSB zero |
| `INFRA-D7` | active formal 统一 `3000 / 60 s`；`2500 / 50 s` 只作历史/screen，禁止混表 | `FIXED`（active formal）/ `LEGACY_ONLY`（2500 env） | 两份 formal env 固定 `3000×0.02 s` |
| `INFRA-D8` | PPO evaluator 的 `episodes.jsonl` 逐行写 `route_profile` 与 `depth_m`；`ROUTE_GENERALIZATION` 由同一 builder 生成 22 条路线 × 4 个压力档 | `FIXED`（metadata + 22-route builder） | 88 scenarios / 264 cells / 792 episodes per arm；压力映射 `nominal/current/noise/plant → nominal/current10/noise_n2/plant20`；首波不阻塞 |
| `INFRA-D9` | 首波不做 `pc_ppo_scale=0.01/0.1/1.0` 对比，记录 `SCALE_AUDIT_DEFERRED` | 已定（效率规则） | 若要做 scale 实验，先补只读 audit |
| `INFRA-D10` | 首波关闭 C6 early stop；第二波启动前实现并预注册 | 已定（第二波） | 不阻塞首波 |
| `INFRA-D11` | 首波按 D18.2 20/3 training-rollout 晋级：先用 reference/P0 rollout 标定并冻结 PE2E/ALOSB cross/vertical 阈值，launcher 切 `enforce`，跑 20/3 smoke，通过后启动 18-run | `FIXED`（实现）/ `BLOCKER`（阈值与 smoke） | 当前唯一首波启动阻断；标定不得扩成矩阵外扫参 |
| `INFRA-D12` | 首波固定路线不续路；第二波必须实现 training-only 有界续路（剩余 `<=5 m` 同 family 追加，最多 3 次/4 段，不 reset，逐次 append manifest，无法追加写 `ROUTE_EXTENSION_FALLBACK`） | 已定 A，待第二波实现 | 评价侧固定路线；append 泄漏判 `INVALID_ROUTE_PROTOCOL` |

**当前结论**：`INFRA-D1/D4/D5/D6/D7/D8` 已落地；`INFRA-D3` 只读诊断；
`INFRA-D9/D10` 按效率规则延后；`INFRA-D12` 第二波硬交付；首波唯一启动阻断是
`INFRA-D11` 的真实阈值、20/3 smoke 和 `INFRA-D2` 的冻结数值。训练完成后由
`scripts/build_goal182_eval_arms.py` 生成正式 arms manifest，再进入 `MAIN_RANKING`。
`D13.6b` 另有覆盖率阻断：现有 60-iteration reference calibration 最多覆盖 C0/C1，
旧实现用单张阈值表套到 C0–C6。2026-11-18 已补逐 stage 生产/消费：calibrator 在每个
stage 内独立滑窗并输出 `thresholds_by_stage`，trainer 按 stage 选择，缺 stage 或多阶段
旧单表会 fail-closed。最终阈值前仍必须补 C0–C6 覆盖，或由用户显式接受
`PARTIAL_STAGE_COVERAGE`；不得把 C0 单段表写成完整冻结表。
最小 stage sweep 已落地：7 个单 stage env + 14 个 frozen-reference training config，
每 stage 100 iterations、`promotion_enabled=false`、单 stage mix；launcher 按
PE2E/ALOSB 两条 control 各串行跑 C0–C6，再在每个 stage 内独立滑窗并冻结逐 stage 表；
每个 stage 至少要求 20 条完整 iteration record。该 sweep 只补覆盖，不是参数/seed/route
扫参；尚未在服务器运行，因此数值仍是 `BLOCKER`。
