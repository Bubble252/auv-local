# Goal17 V-0D：Actor 海流估计 × ALOS Crab 2×2 因子实验计划

日期：2026-11-07

状态：**已收口（负结果）。** 用户已确认进入执行；Goal18 和 Goal19 已收口。当前进度
（2026-11-07）：阶段 0–5 完成；encoded-current 分支已实现并通过
目标回归（115 passed、ruff 通过），教师筛选已选定 `T1`（crab on），`T0/T1` 示范已用
同一批训练 seed 采集。阶段 4 的四格单训练 seed screen 已完成，现按预注册规则扩到
`91430/91431/91432` 三个训练 seed；多 seed dev 评估已完成。主门禁判定为
`FAIL_BC_FACTOR_GATE`：`A10/A11` 均未达到 default RMSE `<=0.95× A00`，因此按预注册
规则停止在 BC 因子实验，不进入阶段 6 残差 PPO，也不评估 held-out `100–149`。

关联文档：

- `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_report_20261107.md`
- `doc/goals/goal16_v0d_crab_rate_limited_teacher_prompt_20261107.md`
- `doc/goals/goal18_v0d_sign_magnitude_bc_prompt_20261107.md`
- `doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`
- `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`
- `doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`
- `doc/goals/goal9_execution_state.md`

本文件不修改 Goal16 的限速教师、BC 配方、门禁或停止条件。Goal16 先回答“限速能否让
同一个 BC 过门禁”；Goal17 再回答“actor 是否应该看到经过编码的海流估计，以及它与
ALOS crab 的关系”。

## 0. 执行前已确认的决策

这些决策在结果产生前冻结，后续不得因为 strict125 结果而临时改写：

- **default 是主目标和硬门禁；**
- **strict125 是软目标**：不单独一票否决，但要求不出现灾难性退化，并作为
  default 接近时的排序指标；
- strict125 的灾难性护栏暂定为：`success >= 0.10`，且 RMSE 不超过 A00 的 `1.20×`；
- Goal16 没有选出可用教师，因此共同教师底座采用现有 R2B baseline 配置
  `odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json` 及其
  `current_crab_enabled` 开关版本；
- actor current 只使用 3D `current_estimate_body`，不额外拼接 valid/age 位；
- 主矩阵关闭 DADE；
- BC screen 先做单训练 seed、20 个评估 seed；只有主效应接近门禁时才扩到多训练 seed。
- 评估 seed 从本文件开始执行三分离，后续任何 BC、PPO 或消融都不得混用：
  - `0–19`：历史筛查集。只用于已经完成的教师筛选和旧 BC 回归，不再用于新 BC/PPO
    的模型、checkpoint 或 epoch 选择；
  - `20–39`：开发集（dev）。四格 BC screen、`current-zero/current-shuffle/path-zero`
    消融和单训练 seed 选择固定使用该集合；
  - `100–149`：最终 held-out。只在候选 BC 和 PPO 配方完全冻结后评估一次，不能用于
    选择格子、checkpoint、epoch、学习率、奖励或任何超参数。
- 已完成的 T0/T1 教师筛选使用历史 seed `0–19`，这是当时预注册的口径；该结果只用于
  选出 `T1`，不能作为后续四格 BC 的比较集，也不为了 seed 规则改变而重跑。

## 1. 要回答的问题

后续必须做一个严格的 2×2 因子实验，分开回答：

1. 把 `current_estimate_body` 作为新的观测块编码进 `z_t^a`，是否改善路径跟踪？
2. ALOS 教师使用 `current_crab_enabled`，是否改善教师和最终策略？
3. 两者是独立相加、冗余，还是存在正向/负向交互？
4. 如果 actor 看到了海流估计，闭环动作是否真的依赖它，而不是只用它作为无关输入？

这里讨论的是**可部署的 `current_estimate_body`**，不是真实海流。真实海流仍然只能进入
环境动力学、奖励、特权 critic 或诊断，不能进入 actor 部署路径。

## 2. 标准 PC 结构（本计划保持不变）

本计划沿用仓库现有的 PC 结构：原始传感器块只进入各自的编码器，编码后融合成固定
宽度的 `z_t^a`，actor MLP 不接收任何原始观测的裸拼接。

这里的 `o_t^a` 是 actor 侧可观测块的统称，包括基础传感器状态、路径特征和可选的
海流估计。它们的关系是：

```text
z_t^a = E_base(o_base,t^a) + alpha_path * E_path(path_t)
        + alpha_cur * E_cur(current_estimate_t^b)

其中：
o_base,t^a        = 当前 12D sensor state
path_t            = 当前 8D path feature
current_estimate_t^b = 当前 3D current estimate

actor 顶层输入 = [z_t^a, c_t, c_hat_t^a]
```

约束：

- `dim(z_t^a) = d_z_total`，四格固定为同一宽度（当前配方为 `64`）；
- `E_cur` 是独立的 3D -> `d_z_total` 编码分支，不是把 3D 原值拼给 actor MLP；
- `E_path` 是独立的 8D -> `d_z_total` 编码分支；`E_base` 是 12D -> `d_z_total`
  编码分支；三个分支最终都作用于 `z_t^a`，而不是扩宽 actor 顶层；
- actor 顶层宽度不变；`A00/A01` 关闭 current 分支，
  `A10/A11` 打开 current 分支；
- `c_t=[]`，保持 V-0D 纯路径契约；
- DADE 槽位 `c_hat_t^a` 的宽度保持不变；主矩阵关闭 DADE 时该槽填零，而不是删除槽位；
- 这里的 12D 只是 `E_base` 的输入宽度，不是 actor MLP 的输入宽度；actor 从不接收
  原始传感器观测，也不把 current estimate 直接拼到 actor 最后一层。

因此，本实验比较的是“有没有经过编码的海流分支”，不是改变 actor 的顶层输入宽度。
所有四格的 actor MLP 输入宽度、`z_t^a` 宽度和 DADE 槽位宽度必须完全一致。

## 3. 2×2 实验矩阵

记法：第一个下标表示 encoded-current 分支是否开启，第二个下标表示 ALOS crab 是否开启。

| cell | encoded-current 分支 | actor 顶层输入 | z_t 宽度 | ALOS crab | 作用 |
|---|---|---|---:|---|---|
| `A00` | 关闭 | `[z_t, c_t, c_hat_t]` | 64 | 关闭 | 纯 ALOS 基线 |
| `A10` | 开启，编码 `current_estimate_body` | `[z_t, c_t, c_hat_t]` | 64 | 关闭 | actor current 主效应 |
| `A01` | 关闭 | `[z_t, c_t, c_hat_t]` | 64 | 开启 | crab 主效应 |
| `A11` | 开启，编码 `current_estimate_body` | `[z_t, c_t, c_hat_t]` | 64 | 开启 | 两者同时开启及交互 |

固定条件：

- `actor_command_dim=0`；
- `path_dim=8`；
- actor 路径仍为现有 8D 路径特征；
- 教师 `compensate_current=false`；
- 不使用真实海流作为 actor 输入；
- `A10/A11` 的海流输入使用与 crab 相同的带延迟、噪声、bias 和失效窗口的估计；
- 主矩阵中 **关闭 DADE**（`c_hat_t=0`，但保留固定宽度槽位），避免海流同时从 history
  side channel 进入，污染“encoded-current 分支”的归因；
- critic 设置、reward、plant、sensors、current、evaluation seeds 在四格中保持一致。
- 四格 BC 的开发评估固定使用 `20–39`；最终 held-out `100–149` 只在所有选择冻结后
  使用。教师采集 seed `90000–90063` 与评估 seed 完全独立。

说明：如果后续要研究 DADE 侧通道与直接输入的组合，另开 Goal17b；不要把 DADE 侧通道
混进本 2×2 主矩阵。

## 4. 为什么主矩阵关闭 DADE

当前 PC actor 的 history token 可以携带 `current_estimate_body` 侧通道。若主矩阵同时
打开 DADE，可能出现两条海流通路：

- `A10/A11` 的 current encoder 分支读取 current estimate；
- DADE 又从 history side channel 读取 current estimate。

这样即使 `A10` 变好，也无法判断收益来自即时输入还是历史侧通道。因此本矩阵固定
`pc.dade_enabled=false`，但保留 `c_hat_t` 的固定宽度槽位。DADE 的增量应在本矩阵
得出结论后单独验证；这不会把 PC 顶层结构改成另一种接口。

## 5. 教师、数据和 BC 的复用关系

教师只由 crab 决定，actor current 不改变教师动作标签。因此：

| 教师条件 | 生成的数据 | 可训练的 actor cell |
|---|---|---|
| `T0`：ALOS 无 crab | 一份示范数据 | `A00`、`A10` |
| `T1`：ALOS 有 crab | 一份示范数据 | `A01`、`A11` |

关键要求：

- `T0` 和 `T1` 必须使用同一批环境 seed、同一路径 bank、同一参考轨迹；
- `A00/A10` 共享完全相同的示范状态和动作标签；
- `A01/A11` 共享完全相同的示范状态和动作标签；
- 训练 BC 时只改变 current encoder 分支是否存在，不改变 base state、路径输入、数据、
  标签、epoch、学习率、`z_t` 宽度或 actor MLP 顶层宽度；
- 示范数据必须保存 `current_estimate_body`，否则 `A10/A11` 无法复现。

## 6. 技术栈与代码边界

执行栈：

- Python；
- PyTorch batch PC 环境和 BC/PPO 训练器；
- `scripts/evaluate_pc_bc.py`、`scripts/evaluate_pc_teacher.py`；
- 服务器 CPU float64 做固定口径评估，GPU 只用于 BC/PPO 训练；
- 服务器仓库：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`；
- 服务器 Python：
  `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`。

预期需要非破坏性修改的代码边界：

- `src/auv_lowlevel/ppo_config.py`：新增可选字段
  `actor_current_encoder_enabled: bool = False`，以及该分支的 `hidden_dims` 和
  `fusion_alpha`；默认关闭；
- `src/auv_lowlevel/batch/pc_env.py`：新增独立的 `actor_current` 观测块，
  其值来自 `current_estimate_body`；**不改变 `actor_state_dim=12`**，不把 current
  追加到 base state；
- `src/auv_lowlevel/batch/pc_normalization.py`：新增 `current_encoder_input_scale`，
  对 3D current estimate 使用 `current_scale_mps` 归一化；不改现有 12D base-state
  的归一化尺度；
- `src/auv_lowlevel/batch/pc_bc.py`：新增 current encoder 分支、融合和 checkpoint
  契约校验；`A00/A01` 不创建可训练 current encoder，`A10/A11` 创建分支；
- `src/auv_lowlevel/batch/pc_policy.py`：新增 `E_cur`，按
  `z_t = E_base(state) + alpha_path*E_path(path) + alpha_cur*E_cur(current)` 融合；
  actor MLP 输入仍为固定宽度 `[z_t, c_t, c_hat_t]`；
- `PCPolicyLayout`/flatten：允许携带原始 `actor_current` 编码器输入块，但 actor 顶层
  flat 宽度和 `z_t` 宽度不能随 current 开关变化；
- checkpoint metadata：新增 `actor_current_encoder="none"` 或
  `"encoded"`、`z_t_dim`、`actor_top_level_input_dim`，禁止静默加载错误契约；
- 评估器：新增 `current-zero` 与 `current-shuffle` 消融，作用于编码前的 3D current
  estimate，用于证明 actor 编码器确实使用了海流估计；
- 所有新增字段必须默认关闭，旧配置、旧 checkpoint 和旧评估结果不变。

## 7. 执行阶段

### 阶段 0：依赖与基线审计

- [x] Goal16 已完成，且用户已确认进入 Goal17。
- [x] Goal16 的教师选择、BC 门禁和 PPO 门禁已冻结；本 Goal 不另改这些门禁。
- [x] 记录服务器没有残留 teacher/BC/PPO/eval runner。
- [x] 记录当前 R2B/Goal16 教师配置、BC 配方和所有 baseline SHA256。
- [x] 确认 `guidance.compensate_current=false`。
- [x] 确认 `current_estimate_body` 在示范数据中存在，且不是真实海流。
- [x] 确认主矩阵 `pc.dade_enabled=false`，并记录该选择。

### 阶段 1：实现 encoded-current 分支（保持 actor 顶层维度不变）

- [x] 新增 `actor_current_encoder_enabled`，默认 `false`。
- [x] `A00/A01` 不启用 current encoder；`A10/A11` 启用独立的 3D current encoder。
- [x] `E_base` 的输入仍为现有 12D sensor state；current 是独立观测块，不拼进
  `E_base` 的输入。这里的 12D 仅描述编码器输入，不描述 actor 输入。
- [x] 四格 `z_t` 维度固定为 `d_z_total`，actor MLP 顶层输入宽度固定为
  `d_z_total + command_dim + PC_DADE_FEATURE_DIM`。
- [x] `A10/A11` 的 current 值严格来自 `current_estimate_body`，不是 `current_ned_mps`。
- [x] 新 checkpoint 记录 current encoder、`z_t` 宽度和顶层输入宽度。
- [x] 旧 checkpoint（其 `E_base` 输入为标准 12D schema）在 current 分支关闭时的
  加载、评估和动作输出逐位不变。
- [x] `current-zero`：把编码前的 3D current estimate 置零，闭环评估。
- [x] `current-shuffle`：在 batch 内打乱编码前的 3D current estimate，闭环评估。
- [x] `path-zero` 仍作为路径输入归因，不因新增 current 列而失效。
- [x] 新增/更新测试，覆盖 current 分支开关、`z_t` 宽度不变、actor MLP 顶层宽度不变、
  checkpoint 拒绝错误契约和旧行为兼容。

阶段 1 的完成标准不是“代码能跑”，而是：旧 checkpoint 在默认关闭新分支时动作输出
逐位不变，新 checkpoint 的 metadata 能拒绝 current 宽度或契约不匹配，且关闭 DADE
时 actor 顶层宽度仍与四格完全一致。

验证命令示例（以实际仓库脚本为准）：

```bash
PYTHONPATH=src:../settled_全局规划器/src .venv/bin/pytest -q \
  tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_pc_bc.py \
  tests/test_lowlevel_pc_policy.py \
  tests/test_lowlevel_pc_contract.py \
  tests/test_lowlevel_pc_eval_contract.py
.venv/bin/ruff check src/auv_lowlevel tests
```

本地实测（2026-11-07）：`115 passed`；上述改动文件 `ruff check` 全部通过。验证范围
额外包含 `tests/test_lowlevel_pc_bc_eval_records.py`、`test_lowlevel_pc_bc_merge.py`、
`test_lowlevel_pc_vec_env.py`、`test_audit_pc_bc_action_saturation.py` 和
`test_lowlevel_pc_teacher.py`。

### 阶段 2：教师 2 组筛选

- [x] 以 R2B baseline 教师参数为共同底座，只切换 `current_crab_enabled`；没有采用
  Goal16 未通过门禁的限速参数。
- [x] 生成 `T0`（crab off）与 `T1`（crab on）两个 default 配置。
- [x] 生成对应 strict125 配置，只改 `goal_radius_m=1.25`。
- [x] 固定历史 seed `0–19`、CPU、float64，分别跑纯教师评估；该集合此后不再用于
  新模型选择。
- [x] 记录 3D RMSE、success、timeout、safety、material saturation。
- [x] 记录 current estimate valid rate 与 current estimate age。
- [x] 记录 crab angle。现有 `teacher_summary.json` 没有该字段，原始 trace 也未保留；
  按非阻塞诊断项记录为“当前产物不可得”。
- [x] 两组教师都能稳定生成示范，停止条件未触发，进入 BC 数据阶段。

教师筛选结果（历史 seed `0–19`，20 episodes/组）：

| 教师 | default RMSE | default success | strict125 RMSE | strict125 success |
|---|---:|---:|---:|---:|
| `T0` crab off | `4.7379 m` | `0.80` | `10.3278 m` | `0.10` |
| `T1` crab on | `4.3531 m` | `0.85` | `9.3916 m` | `0.15` |

crab 的配对效果：default RMSE `-0.3848 m`、success `+0.05`；strict125 RMSE
`-0.9361 m`、success `+0.05`。因此选定 **`T1`（crab on）**作为后续两份示范和四格
BC 的教师底座。

决策产物（服务器）：
`artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/teacher/stage2_teacher_selection.json`，
SHA256 `7da5fe5e8ccdd3b6253f02b329b0eb5d1940d9e4c84fd0b466d3b1b2995da1d4`。

教师配置原始文件 SHA256（当前服务器与本地一致）：

| 配置 | 原始文件 SHA256 |
|---|---|
| `T0` default `..._r2b_current_nocrab.json` | `798b82c3a655a4b11fee94324ce033efe4d94cdaa0b18bb65e5c2b77ed9a4f09` |
| `T1` default `..._r2b_current.json` | `8e4ac934a305f93ae5bc2737a9eec1cf9725fe17adb4e4a9f96f348610464b23` |
| `T0` strict125 `..._r2b_current_nocrab_strict125.json` | `7739068b7b8d4b2682c05fb12d45116881d01f8bf1c7a8d3d1f128c3e900b29a` |
| `T1` strict125 `..._r2b_current_strict125.json` | `398dd219e5674ec793a417262af8dbcaf518e6584c815df82557fc9a2bd688ae` |

注意：`stage2_teacher_selection.json` 中的 `config_sha256` 是 `load_config` 对解析后
配置做 canonical 序列化得到的哈希，不是原始文件 SHA；两者必须分别记录。

教师阶段不把“actor current”作为变量，因为教师动作只由 crab 和 ALOS 决定。

### 阶段 3：采集两份共享示范

- [x] `T0` 采集 262,144 transitions：
  `64 env × 4096 step`，seed `90000–90063`。
- [x] `T1` 用同一批 seed 再采集 262,144 transitions。
- [x] 固定 `path_dim=8`、`actor_command_dim=0`、不添加动作噪声。
- [x] 记录 dataset SHA256、episode 数、safety events、current estimate 统计。
- [x] 检查 `T0/T1` 的示范状态、路径和 current estimate 在相同 seed 下可配对。

### 阶段 4：四格 BC screen

- [x] `A00`：用 `T0` 示范，current encoder 关闭，crab off。
- [x] `A10`：用 `T0` 示范，current encoder 开启，crab off。
- [x] `A01`：用 `T1` 示范，current encoder 关闭，crab on。
- [x] `A11`：用 `T1` 示范，current encoder 开启，crab on。
- [x] 四格使用同一 BC 网络、同一训练 seed `91430`、同一 epoch/patience/学习率和同一
  评估 seed。
- [x] 每格固定开发评估 seed `20–39`、CPU、float64、default + strict125。
- [x] held-out `100–149` 不在阶段 4 使用；只有四格比较和 checkpoint 选择冻结后才允许
  做最终评估。
- [x] `A10/A11` 追加 `current-zero` 和 `current-shuffle` 评估。
- [x] 四格都追加 `path-zero`，确认路径输入仍然有效。

四格 BC 的正式比较不以单个 success 最高点选模型；主排序指标是 3D RMSE，success、
timeout、safety 只作护栏。

#### 阶段 4b：多训练 seed 扩展（结果读取前冻结）

单训练 seed `91430` 中 `A11` 的 default RMSE ratio 为 `0.96864`，接近但未达到
`<=0.95` 的硬门禁，因此只允许按以下规则扩展，不允许根据扩展结果改写规则：

- 训练 seed 固定为 `91430/91431/91432`，不增加、不替换；
- 每个训练 seed 必须与同一训练 seed 的 `A00` 配对，不使用跨 seed 的 `A00` 作对照；
- 评估 seed 固定为 dev `20–39`，CPU、float64、`--path-dim 8`；
- 四格都报告 default、strict125、path-zero；`A10/A11` 追加 current-zero、
  current-shuffle；
- 聚合时报告三个训练 seed 的 cell 均值、训练 seed 间波动、逐评估 seed 配对差值的
  均值和 95% CI；
- default 硬门禁保持不变：RMSE ratio `<=0.95`、success 绝对下降不超过 `0.05`、
  safety 绝对增加不超过 `0.02`；
- strict125 仍是软目标，不因某一训练 seed 的 strict125 排名而改变 default 门禁；
- 只有 `A10` 或 `A11` 在三个训练 seed 的聚合口径下通过 default 硬门禁，才允许冻结
  对应 BC 并进入阶段 6；
- 未通过时停止 BC 因子实验，不把 PPO 当作动作表示问题的补救手段。

阶段 4b 结果（dev `20–39`，三个训练 seed 聚合）：

| cell | default RMSE | 相对 A00 | success | success Δ | safety Δ | 门禁 |
|---|---:|---:|---:|---:|---:|---|
| `A00` | `8.0067 m` | `1.0000×` | `0.500` | `0` | `0` | baseline |
| `A10` | `9.6232 m` | `1.2019×` | `0.450` | `-0.050` | `0` | **FAIL** |
| `A01` | `8.7582 m` | `1.0939×` | `0.533` | `+0.033` | `0` | 非门禁格 |
| `A11` | `8.1176 m` | `1.0139×` | `0.550` | `+0.050` | `0` | **FAIL** |

配对 episode RMSE 差值（同训练 seed 对同训练 seed 的 `A00`）：

| cell | 均值 ΔRMSE | 95% CI | 解释 |
|---|---:|---:|---|
| `A10` | `+1.2257 m` | `[+0.0741, +2.3773]` | 显著变差 |
| `A11` | `-0.2663 m` | `[-1.2012, +0.6686]` | 无显著改善 |

因子效应（default RMSE，负值才表示改善）：

| 对比 | 均值 | 三个训练 seed |
|---|---:|---|
| `A10 - A00`（current，crab off） | `+1.6165 m` | `+2.1048, +0.8797, +1.8651` |
| `A11 - A01`（current，crab on） | `-0.6406 m` | `-1.9506, +0.3587, -0.3298` |
| `A01 - A00`（crab，无 current） | `+0.7515 m` | `+1.6985, -0.3458, +0.9018` |
| `A11 - A10`（crab，有 current） | `-1.5056 m` | `-2.3568, -0.8669, -1.2930` |
| 交互项 | `-2.2571 m` | `-4.0553, -0.5210, -2.1949` |

解释：current 和 crab 同时开启时存在较强互补交互，且 `A11` 相对 `A10` 有稳定收益；
但 `A11` 相对真正的 baseline `A00` 仍为 `+0.1109 m`，没有净改善。`A11` 的
current-zero/current-shuffle 分别使 RMSE 变差 `+1.5446 m`（CI `[+0.3626, +2.7266]`）
和 `+1.1759 m`（CI `[+0.0016, +2.3503]`），说明策略确实使用了 current estimate，
但“被使用”不等于“有收益”。

### 阶段 5：BC 进入 PPO 的门槛

Goal16 没有选出可用教师，因此使用以下已冻结门槛：

- `A10/A11` 的 **default** 3D RMSE 不超过 `A00` 的 `0.95×`；
- default success 相对 `A00` 的绝对下降不超过 `0.05`；
- safety 绝对增加不超过 `0.02`；
- strict125 作为软目标：success 不低于 `0.10`，RMSE 不超过 `A00` 的 `1.20×`；
- 若多个格子同时通过，优先选择 default RMSE 更低者；default 接近时再看 strict125；
- `current-zero/current-shuffle` 应当造成可解释的动作或闭环退化；若完全不变，标记
  `current_estimate_not_effectively_used`，不得宣称 actor 使用了海流。

只要四格 BC 中没有任何一格达到进入 PPO 的门槛，停止在 BC 因子实验，不用 PPO 掩盖
BC 的动作表示问题。

阶段 5 判定：**没有任何格子通过。** `A11` 是唯一接近者，但 default RMSE ratio
`1.0139 > 0.95`，且配对 CI 跨 0；`A10` 明显变差。按冻结规则执行
`stop_goal17_bc_gate_failed`，不冻结 BC、不进入残差 PPO、不使用 held-out。

### 阶段 6：残差 PPO 确认（未触发，不适用）

未触发；阶段 5 已判定没有通过门禁的格子。以下条目保留为协议原文，不属于本轮的未完成
工作。

- [ ] 冻结对应 BC，记录 SHA256。
- [ ] 使用 `pc.dade_enabled=false`、但保持标准 `[z_t, c_t, c_hat_t]` 顶层宽度的同一残差
  PPO 配方。
- [ ] 先跑 zero-residual 基线，再跑 PPO screen。
- [ ] screen 固定 `4096 env × 256 step × 40 iteration`，训练 seed 至少 1 个，先看
  dev seed `20–39` 的配对 RMSE ratio。
- [ ] screen 通过后才扩到 3 个训练 seed；正式评估使用 held-out `100–149`，且只在配方
  冻结后评估一次。
- [ ] 不得用 held-out `100–149` 反选 PPO checkpoint、残差尺度、学习率或 epoch。
- [ ] 主终点仍是 3D RMSE ratio；success/timeout/safety 作护栏。
- [ ] 不对失败格子继续调学习率、残差尺度、reward 或 seed。

## 8. 评估与因子分析

基础指标：

- default 3D RMSE；
- strict125 3D RMSE；
- success、timeout、safety；
- path-zero 恶化幅度；
- `current-zero` 与 `current-shuffle` 的动作/闭环变化；
- crab angle 与 current estimate valid/age 统计。

配对分析固定使用相同评估 seed，报告每 seed 差值、均值、样本标准差和 95% 配对置信
区间，不只报告四个格子的均值。四格主分析一律使用 dev `20–39`；held-out `100–149`
只用于最终确认，必须单独报告，不能和 dev 合并，也不能用 held-out 结果回头改四格。

若 dev 上 RMSE 改善只来自“成功集重新分配”，但逐 seed 配对均值或置信区间没有改善，
不得把该格判定为有效改进；success/timeout 变化必须与逐 seed trace 一起报告。

四个对比量：

```text
actor_current 主效应（crab off） = A10 - A00
actor_current 主效应（crab on）  = A11 - A01
crab 主效应（无 current）        = A01 - A00
crab 主效应（有 current）        = A11 - A10
交互项 = (A11 - A01) - (A10 - A00)
```

解释规则：

- 两个主效应都为负且交互接近 0：current 和 crab 近似独立、可叠加；
- 只有 crab 主效应稳定：收益主要来自 ALOS crab，actor 直接看海流没有额外价值；
- 只有 actor current 主效应稳定：可以把估计编码进 `z_t^a`，crab 不是必要条件；
- 只有同时开启才改善，即交互项明显为负：两者存在互补关系，需要联合设计；
- 交互项明显为正：两者冲突或冗余，不能简单叠加。

## 9. 产物命名

建议根目录：

```text
artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/
```

子目录：

```text
teacher_T0/
teacher_T1/
demos_T0/
demos_T1/
bc_A00/
bc_A10/
bc_A01/
bc_A11/
eval/
ppo/
```

日志：

```text
logs/pc_v0d_current_crab_factorial_20261107/
```

每个 run 的 manifest 必须记录：

- git HEAD；服务器没有 `.git` 时记录本地 HEAD 并注明；
- 完整命令行；
- config、dataset、checkpoint 的 SHA256；config 同时记录原始文件 SHA256 和
  `load_config` 的 canonical `config_sha256`，两者不可混写；
- `actor_current_encoder`（`none` 或 `encoded`）；
- `z_t_dim` 和 `actor_top_level_input_dim`；
- `pc.dade_enabled`；
- 教师 crab 开关；
- 教师选择 JSON 路径与 SHA256、选定教师 `T1`；
- 训练 seed、评估 seed、seed role（`dev` 或 `heldout`）、device、dtype、num_envs；
- 开始和结束时间。

## 10. Git 操作规范

每一步完成后只提交该步明确列出的文件，不使用 `git add -A`，不提交 artifacts、checkpoint
或日志。

推荐流程：

```bash
git status --short
git diff -- <要提交的文件>
git add <明确文件列表>
git commit -m "docs(goal17): preregister current-crab factorial plan"
git push
```

只有在用户明确要求时才执行 `git commit` 或 `git push`。本文件创建阶段默认只写文档，
不自动提交。

## 11. 已确认事项与 seed 契约

- [x] Goal16 未选出可用教师；Goal17 采用现有 R2B baseline 作为共同底座。
- [x] 不采用 Goal16 的限速参数，避免把未过门禁的教师作为新基线。
- [x] actor current 只加入 3D 估计值，不加入 valid/age。
- [x] 主矩阵关闭 DADE。
- [x] BC screen 先做单训练 seed。
- [x] 先使用 20 个开发评估 seed `20–39`；最终 held-out 使用 `100–149`。
- [x] 历史 `0–19` 不再参与新模型选择。
- [x] 教师阶段已完成并选择 `T1`；阶段 2 的 crab angle 诊断当前产物不可得，已按
  非阻塞诊断项记录，不影响 encoded-current 实现、示范采集和 BC 因子结论。

## 12. 停止条件

- 旧 checkpoint 在 current encoder 默认关闭时不能保持逐位兼容；
- 无法在不扩宽 actor 顶层输入的前提下实现 encoded-current；
- dev `20–39` 与 held-out `100–149` 被混用，或 held-out 被用于选择模型；
- T0/T1 教师无法稳定生成示范；
- BC screen 没有任何一格达到进入 PPO 的门槛；
- 需要打开真实海流、删除 crab、改变 actor/critic 契约或修改已冻结门禁才能继续；
- 出现 safety 事件或评估进程异常，且无法用完全相同的配置重试。

## 13. 最终输出表

| 问题 | 需要回答 |
|---|---|
| 教师底座是什么？ | R2B baseline，选定 `T1`（crab on） |
| T0/T1 教师的 3D RMSE、success、crab 指标分别是什么？ | default `4.7379/0.80` vs `4.3531/0.85`；strict125 `10.3278/0.10` vs `9.3916/0.15`；crab angle 当前产物不可得 |
| 四格 BC 的 default/strict125/path-zero 指标是什么？ | default 见阶段 4b 表；详细 strict125/path-zero/current 消融见 `goal17_v0d_current_crab_factorial_report_20261107.md` |
| `current-zero` 和 `current-shuffle` 是否使动作/闭环退化？ | `A11` 两者都显著变差；`A10` 不稳定，不能据此宣称有效利用 |
| actor current 主效应、crab 主效应和交互项分别是多少？ | current（crab off）`+1.6165 m`；current（crab on）`-0.6406 m`；crab（无 current）`+0.7515 m`；crab（有 current）`-1.5056 m`；交互 `-2.2571 m` |
| 哪一格达到 PPO 门槛？是否进入残差 PPO？ | 无；未进入残差 PPO |
| 是否能证明 actor 使用了 current estimate，而不是偶然相关？ | `A11` 可以证明“被使用”；不能证明“带来净收益” |
| 最终建议：保留 current、保留 crab、两者都保留，还是都不保留？ | 当前配方下都不作为收益来源保留；若继续，应另开 Goal 优先解决 BC/动作表示，而不是用 PPO 掩盖 |

