# Goal22：nominal 传感器域 BC 重采与重训提示词

日期：2026-11-07

状态：**已执行；Goal22 预注册门禁 FAIL（2026-11-07），未进入 P3/P5/理想臂**

执行前必读：

- `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`
- `doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_report_20261107.md`
- `doc/goals/goal21_sensor_profiles_nominal_stress_20261107.md`
- `doc/goals/goal9_execution_state.md`

## 0. 一句话目标

在 `realistic_calibrated` nominal 传感器域、`endpoint_stop` 终止协议和 12D 终点
上下文观测下，重新采集 PC 教师示教数据、重训 BC，并建立一个可复现的新 BC 底座。

本 Goal 只做 BC 重采、重训和新底座评估，**不启动 P3 DAgger、不启动 P5 残差 PPO、
不评估 held-out**。

## 1. 原计划与本 Goal 的位置

Goal9 的原始后续路线是：

```text
P2A 终点上下文观测 + P2B endpoint_stop
  → P3 default + strict125 定向 DAgger
  → P5 残差 PPO（仅在 BC 门禁通过后）
```

Goal20 已完成 P2A/P2B；Goal21 又把 nominal 传感器 profile 收紧。因此不能直接沿用
旧 BC 数据进入 P3：旧数据是在旧的 `realistic_calibrated` 长尾域和 8D path 上采集的。

新增顺序固定为：

```text
Goal20 P2A/P2B（已完成）
  → Goal22 nominal BC 重采 + 重训 + 新底座门禁
  → P3 default + strict125 定向 DAgger
  → P5 残差 PPO
```

如果 Goal22 失败，保留负结果并停止，不用扩大训练步数、扫 seed 或放宽门禁去“救”它。

## 2. 非破坏式实现约束

- 不删除、不重命名现有配置字段；新行为通过新配置文件和显式开关启用。
- 不覆盖、搬迁或重写旧 `g17/g18/g19` artifacts、旧 checkpoint、旧 summary 和旧数据集。
- 不直接修改当前正在被任务使用的配置；先检查进程和输出目录。
- 旧 8D checkpoint 必须继续原样加载；12D 只能作为新增能力。
- 默认值保持旧语义：所有 12D/endpoint 能力默认关闭，只有新配置显式开启。
- 不改 actor 顶层 MLP 输入宽度；终点特征只能在 path encoder 内扩为 12D，再编码进
  `z_t`。
- 不把真实海流、真实位置或其他仿真特权量写进 actor 观测。
- 不使用 `git add -A`；不自动 commit/push，除非用户明确要求。
- 启动任何长任务前先记录本地/服务器已有进程，禁止杀进程、抢占 GPU 或覆盖运行目录。

开始前执行并保存证据：

```bash
ps -eo pid,etime,pcpu,pmem,args | rg -i \
  'supermimic|pc_.*train|train_pc_bc|collect_pc_bc|evaluate_pc_bc|ppo|isaac' \
  | rg -v 'rg -i' || true
nvidia-smi || true
git status --short
```

只有确认资源空闲后才启动训练。

## 3. 当前代码事实与 Phase 0 审计

已知但必须在执行时复核：

- `pc_path.py` 已有 `PC_PATH_DIM_V1=8`、`PC_PATH_DIM_ENDPOINT=12` 和
  `pc_path_feature_names_for_dim()`；
- `evaluate_pc_bc.py` 已支持 `--path-dim 12` 与 `--termination-mode endpoint_stop`；
- `pc_bc.py` 的 `PCBCTrainingConfig`、`PCBCPolicy` 和 checkpoint loader 目前仍把
  path_dim 限制为 `0/8`；
- `train_pc_bc.py` 的 `--path-dim` 仍只接受 `0/8`；
- `collect_pc_bc_demos.py` 仍使用固定的 `PC_PATH_DIM=8` 和固定的 8D feature names；
- `collect_pc_bc_dagger_demos.py` 当前按 8D 自动推断，尚不能按 12D checkpoint 采集。

Phase 0 必须先审计这些事实；若代码已部分更新，沿用现状，不回滚已验证的 Goal20
兼容逻辑。

### 3.1 必须补齐的最小非破坏改动

1. 将 BC 训练配置、BC policy、checkpoint metadata/loader 的 path_dim 支持从 `0/8`
   扩展为 `0/8/12`；
2. 使用 `pc_path_feature_names_for_dim(path_dim)` 生成 feature names，不能把 8D
   names 写死到 12D checkpoint；
3. `train_pc_bc.py --path-dim` 接受 `0/8/12`，默认仍按数据集 metadata；
4. `collect_pc_bc_demos.py --path-dim` 接受 `0/8/12`，默认保持旧值 `0`；
5. `--path-dim 12` 要求配置显式开启
   `observation.path_endpoint_context_enabled=true`；
6. `--path-dim 8` 要求该字段为 `false`；
7. 旧 8D checkpoint 的 SHA256、结构、加载和输出必须不变；
8. 12D checkpoint metadata 必须记录 schema、feature names、scales、path encoder
   输入宽度和顶层 actor 输入宽度；
9. `collect_pc_bc_dagger_demos.py` 若本 Goal 不实际采集 DAgger，也必须至少能根据
   checkpoint 正确识别 8D/12D，禁止把 12D checkpoint 静默喂成 8D。

### 3.2 Phase 0 测试

新增或补充测试，至少覆盖：

- 8D checkpoint 旧行为逐位可加载；
- 12D path encoder 可训练、可保存、可重新加载；
- 8D/12D feature names 和 schema hash 正确；
- `path_dim=12` 但 endpoint flag 关闭时拒绝启动；
- `path_dim=8` 但 endpoint flag 开启时拒绝启动；
- 12D actor 顶层输入宽度与 8D 相同；
- path-zero 时 endpoint 特征不泄漏；
- 错误 path_dim 或 metadata 不匹配时明确报错，不静默截断。

Phase 0 未通过，不允许采集数据或训练。

## 4. 固定实验矩阵

使用两套 BC 路径输入，固定相同的教师、传感器域、终止协议和训练配方：

| Arm | path_dim | endpoint 观测 | 终止模式 | default 采集 | strict125 采集 |
|---|---:|---|---|---:|---:|
| `N8` | 8 | 关闭 | `endpoint_stop` | 64×4096 | 32×4096 |
| `N12` | 12 | 开启 | `endpoint_stop` | 64×4096 | 32×4096 |

固定条件：

- sensor profile：`realistic_calibrated`；
- `guidance.current_estimate_max_age_s=0.9`；
- default goal radius：`2.0 m`；
- strict125 goal radius：`1.25 m`；
- 教师控制逻辑沿用当前 `r2b current` 教师，不改 crab/current 研究因子；
- default 训练 seed：`90000–90063`；
- strict125 训练 seed：`90100–90131`；
- 两套 arm 使用相同 seed，确保传感器和教师轨迹可比；
- 评估 dev seed 固定为 `20–39`；held-out `100–149` 本 Goal 不碰；
- 评估统一 CPU、float64、`endpoint_stop`。

禁止：

- 不让 `N8` 使用 legacy 终止、`N12` 使用 endpoint_stop；
- 不把 8D 旧数据直接混入 12D 数据集；
- 不把 strict125 和 default 的评估结果混成单一主指标；
- 不根据 dev 结果回头增删 seed 或训练 arm。

## 5. 新配置

从当前 `r2b current` 教师配置复制，新建而不是修改旧配置：

```text
configs/lowlevel/
  odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_default_20261107.json
  odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_strict125_20261107.json
  odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_default_20261107.json
  odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_strict125_20261107.json
```

四份配置的共同字段：

```json
{
  "sensors": {
    "profile": "realistic_calibrated",
    "seed": 81300
  },
  "guidance": {
    "current_estimate_max_age_s": 0.9
  },
  "termination": {
    "pc_goal_plane_miss_termination_enabled": true,
    "goal_plane_miss_penalty": 0.0,
    "goal_plane_miss_distance_weight": 0.0
  }
}
```

N8 额外要求：

```json
{"observation": {"path_endpoint_context_enabled": false}}
```

N12 额外要求：

```json
{"observation": {"path_endpoint_context_enabled": true}}
```

strict125 版本只改 `plant.goal_radius_m=1.25`，不要同时改变传感器域、教师或奖励逻辑。

## 6. 数据采集预算与顺序

产物根目录固定为：

```text
artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/
```

顺序：

1. `N8/default`：64 env × 4096 steps，seeds `90000–90063`；
2. `N8/strict125`：32 env × 4096 steps，seeds `90100–90131`；
3. `N12/default`：64 env × 4096 steps，seeds `90000–90063`；
4. `N12/strict125`：32 env × 4096 steps，seeds `90100–90131`；
5. 分别合并 `N8/default + N8/strict125` 和
   `N12/default + N12/strict125`；
6. 合并后检查 episode id 唯一、样本数、finite、path dim、sensor metadata 和 SHA256。

建议命令形态如下；实际 Python 路径以当前机器配置为准：

```bash
PY="${PY:-python}"
ROOT=artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107

"$PY" scripts/collect_pc_bc_demos.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_default_20261107.json \
  --output "$ROOT/demos_N8_default" \
  --num-envs 64 --steps-per-env 4096 --seed-start 90000 \
  --device cuda --dtype float64 --path-dim 8 --actor-command-dim 0

"$PY" scripts/collect_pc_bc_demos.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_strict125_20261107.json \
  --output "$ROOT/demos_N8_strict125" \
  --num-envs 32 --steps-per-env 4096 --seed-start 90100 \
  --device cuda --dtype float64 --path-dim 8 --actor-command-dim 0

"$PY" scripts/collect_pc_bc_demos.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_default_20261107.json \
  --output "$ROOT/demos_N12_default" \
  --num-envs 64 --steps-per-env 4096 --seed-start 90000 \
  --device cuda --dtype float64 --path-dim 12 --actor-command-dim 0

"$PY" scripts/collect_pc_bc_demos.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_strict125_20261107.json \
  --output "$ROOT/demos_N12_strict125" \
  --num-envs 32 --steps-per-env 4096 --seed-start 90100 \
  --device cuda --dtype float64 --path-dim 12 --actor-command-dim 0

"$PY" scripts/merge_pc_bc_demos.py \
  --inputs "$ROOT/demos_N8_default/demos.npz" "$ROOT/demos_N8_strict125/demos.npz" \
  --output "$ROOT/demos_N8_merged"

"$PY" scripts/merge_pc_bc_demos.py \
  --inputs "$ROOT/demos_N12_default/demos.npz" "$ROOT/demos_N12_strict125/demos.npz" \
  --output "$ROOT/demos_N12_merged"
```

本 Goal 的“BC 重采”只指教师示教数据重采；P3 的定向 DAgger 仍留到 Goal22 之后。

## 7. BC 训练配方

两个 arm 使用完全相同的配方，只允许 path_dim 不同：

```text
state encoder: 128,128
actor: 256,128
d_z_total: 64
n_in_reserved: 8
n_z_reserved: 4
activation: elu
batch size: 4096
max epochs: 300
patience: 30
learning rate: 1e-3
weight decay: 1e-5
validation fraction: 0.15
seed: 0
action head: continuous
saturation weight: 2.0
saturation threshold: 0.9
path encoder: 64
path encoder alpha: 1.0
device: cuda（不可用时明确记录并切 CPU）
```

输出：

```text
$ROOT/bc_N8/pc_bc_actor.pt
$ROOT/bc_N12/pc_bc_actor.pt
```

建议命令形态：

```bash
"$PY" scripts/train_pc_bc.py \
  --demos "$ROOT/demos_N8_merged/demos.npz" \
  --output "$ROOT/bc_N8" \
  --state-encoder-hidden-sizes 128,128 --actor-hidden-sizes 256,128 \
  --d-z-total 64 --n-in-reserved 8 --n-z-reserved 4 --activation elu \
  --batch-size 4096 --max-epochs 300 --patience 30 \
  --learning-rate 1e-3 --weight-decay 1e-5 --validation-fraction 0.15 \
  --seed 0 --device cuda --actor-command-dim 0 \
  --path-dim 8 --path-encoder-hidden-sizes 64 --path-encoder-alpha 1.0 \
  --saturation-weight 2.0 --saturation-threshold 0.9 --action-head continuous

"$PY" scripts/train_pc_bc.py \
  --demos "$ROOT/demos_N12_merged/demos.npz" \
  --output "$ROOT/bc_N12" \
  --state-encoder-hidden-sizes 128,128 --actor-hidden-sizes 256,128 \
  --d-z-total 64 --n-in-reserved 8 --n-z-reserved 4 --activation elu \
  --batch-size 4096 --max-epochs 300 --patience 30 \
  --learning-rate 1e-3 --weight-decay 1e-5 --validation-fraction 0.15 \
  --seed 0 --device cuda --actor-command-dim 0 \
  --path-dim 12 --path-encoder-hidden-sizes 64 --path-encoder-alpha 1.0 \
  --saturation-weight 2.0 --saturation-threshold 0.9 --action-head continuous
```

训练完成后立即记录：

- checkpoint SHA256；
- best epoch 和 validation RMSE；
- 数据集 SHA256 和 manifest SHA256；
- 训练命令、Python 环境、git HEAD/dirty 状态；
- 8D/12D path encoder 输入宽度和 actor 顶层输入宽度。

## 8. 离线审计

对两个 checkpoint 分别执行：

1. action saturation audit；
2. path-zero audit；
3. 数据/模型 metadata 一致性检查；
4. 旧 8D checkpoint 兼容性回归；
5. 12D endpoint 特征只依赖 `p_nav`、路径终点和姿态，不读取真值。

任何 NaN/Inf、维度不一致、静默截断或旧 checkpoint 加载失败都直接失败，不继续闭环评估。

## 9. dev 闭环评估

统一使用：

- seeds `20–39`；
- CPU、float64；
- `--termination-mode endpoint_stop`；
- default 用 default 配置，strict125 用 strict125 配置；
- N8 用 `--path-dim 8`，N12 用 `--path-dim 12`；
- 额外跑 path-zero 作为因果检查。

至少评估：

| 模型 | default | strict125 | path-zero |
|---|---|---:|---:|
| 旧 `g19_dagger` 参考 | 新 nominal 配置、path_dim 8 | 是 | 是 |
| `N8` | 是 | 是 | 是 |
| `N12` | 是 | 是 | 是 |

旧 `g19_dagger` 只作为参考，不参与超参数选择，也不覆盖旧结果。所有模型必须在同一个
nominal 传感器域和同一个 `endpoint_stop` 口径下评估。

### 9.1 主指标与诊断指标

主排序指标：

- `pre_goal_3d_rmse_m`；
- default 和 strict125 分开报告。

必须同时报告：

- `success` / `goal_capture_rate`；
- `goal_plane_miss_rate`；
- `timeout_rate`；
- `safety_rate`；
- `legacy_full_3d_rmse_m`（诊断，不与 pre-goal 混列）；
- `success_only_3d_rmse_m`；
- `post_goal_overshoot_m`；
- `terminal_longitudinal_rmse_m`；
- action saturation 和 reversal 指标。

### 9.2 预注册门禁

`N12` 必须相对 `N8` 满足：

- default `pre_goal_3d_rmse_m <= 0.95 × N8`；
- default `legacy_full_3d_rmse_m <= 1.05 × N8`；
- default success 下降不超过 `0.05`；
- goal capture 不下降，plane miss 不增加超过 `0.05`；
- strict125 `pre_goal_3d_rmse_m <= 1.10 × N8`；
- strict125 success 不低于 `0.10`；
- safety 增加不超过 `0.02`；
- path-zero 应出现可解释退化，不能与正常 path 输入几乎相同。

附加护栏：

- `N8` 若相对旧 `g19_dagger` 在 default pre-goal RMSE 上退化超过 `10%`，必须明确记录
  为 nominal 传感器域带来的分布迁移，不得掩盖或改口径；
- 任一 arm 的 strict125 表现变差时，不得通过牺牲 default 主指标来换取 strict125；
- 未通过 `N12` 门禁时，保留 `N8` 作为下一阶段候选，并写负结果，不强行把 12D 接入 P3。

## 10. 产物与报告

固定产物目录：

```text
artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/
  demos_N8_default/
  demos_N8_strict125/
  demos_N8_merged/
  demos_N12_default/
  demos_N12_strict125/
  demos_N12_merged/
  bc_N8/
  bc_N12/
  eval/
  comparison.json
  comparison.md
  contract_checks.json
  run_manifest.json
```

报告必须包含：

- 固定矩阵、seeds、预算和终止模式；
- 数据集/checkpoint 哈希和训练命令；
- 8D/12D 对照表；
- 旧 `g19_dagger` 的 nominal 参考表；
- 门禁逐项 PASS/FAIL；
- 失败时的根因分类：传感器分布迁移、观测别名、动作饱和、训练不足或实现错误；
- 下一阶段建议：使用 `N8`、使用 `N12`，或停止。

报告写入：

```text
doc/goals/goal22_nominal_bc_rebase_report_20261107.md
```

并在 `doc/goals/goal9_execution_state.md` 顶部加入一行本 Goal 状态摘要。

## 11. 停止条件

出现以下任一条件立即停止，不进入 P3/P5：

- 12D BC 代码需要破坏旧 8D checkpoint；
- 数据 manifest 不能证明 nominal profile、seed 和 path_dim；
- 训练或评估出现 NaN/Inf；
- 没有完成 default/strict125/path-zero 的固定矩阵；
- `N12` 未通过预注册门禁；
- 资源冲突、覆盖旧 artifact 或无法记录完整哈希。

## 12. 完成勾选

- [x] Phase 0：确认当前 12D BC 缺口并完成非破坏式支持；
- [x] Phase 0：8D 旧 checkpoint 兼容测试通过；
- [x] Phase 0：12D 保存/加载/metadata 测试通过；
- [x] 新建 4 份 nominal BC 配置，未修改旧配置；
- [x] 采集 N8 default/strict125 教师数据；
- [x] 采集 N12 default/strict125 教师数据；
- [x] 合并 N8/N12 数据集并校验 manifest；
- [x] 训练 N8/N12 BC；
- [x] 离线饱和、path-zero、finite、哈希审计通过；
- [x] dev `20–39` 的 default/strict125/path-zero 评估完成；
- [x] 旧 `g19_dagger` nominal 参考评估完成；
- [x] 预注册门禁逐项判定（结果：**FAIL**，见报告）；
- [x] 报告和 execution state 更新；
- [x] 未启动 P3、P5、held-out 或超参数扫参。

## 13. Git 记录建议

完成后先检查：

```bash
git status --short
git diff --stat
```

只暂存本次明确修改的文件，不使用 `git add -A`。示例：

```bash
git add \
  src/auv_lowlevel/batch/pc_bc.py \
  scripts/train_pc_bc.py \
  scripts/collect_pc_bc_demos.py \
  scripts/collect_pc_bc_dagger_demos.py \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_default_20261107.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n8_strict125_20261107.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_default_20261107.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_n12_strict125_20261107.json \
  tests/test_lowlevel_pc_bc.py \
  doc/goals/goal22_nominal_bc_rebase_report_20261107.md \
  doc/goals/goal9_execution_state.md

git commit -m "feat(pc): rebase BC under nominal sensor profile"
```

推送只能在用户明确要求时执行：

```bash
git push
```

如果服务器仓库不是受控 git 工作区，只记录源码哈希、文件清单和运行命令，不伪造 commit。

## 14. 最终回复格式

必须用中文给出：

1. 原计划位置和本次插入点；
2. 是否完成 8D/12D 非破坏式支持；
3. 四个数据集的样本数、seed、sensor profile、path_dim 和 SHA256；
4. N8/N12 的训练与评估表；
5. 预注册门禁 PASS/FAIL；
6. 是否选择 N8 或 N12 作为 P3 输入；
7. 未启动 P3/P5/held-out 的明确声明；
8. 完整产物路径和下一步建议。
