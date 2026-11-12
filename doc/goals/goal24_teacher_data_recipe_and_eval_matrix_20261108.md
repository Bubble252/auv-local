# Goal24：教师一致性审计、数据配方对照与统一评估矩阵

日期：2026-11-08

状态：**提示词已就绪，待执行；默认不自动启动训练**

## 1. 一句话目标

在不扩大无关数据量的前提下，先修正新 nominal 配置与旧 R2B current 教师的控制差异，
再通过受控数据配方实验判断“纯教师 BC、DAgger、stress 增广”各自贡献多少；同时把
低层 PC 的评估矩阵集中到一个显式配置中，避免 seed、终止协议和 path_dim 继续散落
在命令行里。

## 2. 背景与已确认事实

Goal22 已完成 N8/N12 的 nominal BC 重采、训练和 dev `20–39` 评估，但预注册门禁
FAIL：N12 default 的 pre-goal RMSE、success、capture 和 plane-miss 四项未通过，
因此没有进入 P3/P5/理想 actor 臂。报告见：

- `doc/goals/goal22_nominal_bc_rebase_report_20261107.md`

2026-11-08 追加审计发现两个必须先处理的混杂因素：

1. 新生成的 `v1_nominal_bc_*` 配置没有显式继承旧 R2B current 的
   `current_crab_enabled=true`、`robust_terminal_enabled=true`、
   `route_speed_floor_enabled=true`；配置加载器对三者默认取 `false`。
2. 旧 `g19_dagger` 的训练数据为 `421,888` samples / `330` episodes，其中
   `159,744` 个是 DAgger 样本（约 `37.9%`）；新 N8/N12 是 `393,216` samples /
   `317` episodes 的纯教师 BC，DAgger 为 `0`。旧数据的传感器训练域也更严苛
   （nav RMSE `0.716 m` vs 新数据 `0.390 m`，current age `1.37 s` vs `0.41 s`）。

因此当前不能把“新 N8 不如旧 g19”简单归因于网络或 12D 终点块。下一阶段必须先把
教师控制路径对齐，再用受控数据配方把“教师质量、on-policy 覆盖、stress 鲁棒性”
分开。

## 3. 前置约束

- 不删除、不重命名旧配置、旧 checkpoint、旧 dataset、旧 summary 和旧 artifacts；
- 所有新配置、新数据、新训练和新评估使用新名称和新目录；
- 不修改正在被其他任务使用的配置或输出目录；
- 不碰 held-out seeds `100–149`，除非进入最后冻结候选的一次性评估；
- 不扩 seed、不扫超参、不根据 dev 结果事后放宽门禁；
- 不使用 `git add -A`；不自动 commit/push，除非用户明确要求；
- 若服务器不可用而改在本地运行，所有产物必须标记 `local_only`，不能在报告中
  冒充服务器正式结果；服务器恢复后按同一协议重跑并替换结论；
- 启动任何长任务前先记录进程、GPU、磁盘和已有输出。

## 4. Phase 0：统一低层 PC 评估矩阵

### 4.1 目标

新增一个显式评估矩阵配置，把当前散落在 shell/CLI 中的评估条件集中写死。该配置只
描述“怎么评估”，不改变训练环境语义。

新增文件：

```text
configs/lowlevel/eval/pc_v1_nominal_eval_matrix_20261108.json
```

建议 schema：

```json
{
  "schema_version": "auv_pc_eval_matrix.v1",
  "base_environment_configs": {
    "default": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json",
    "strict125": "configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_strict125_20261108.json"
  },
  "seeds": {
    "dev": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "dagger_pool": {"start": 200, "end": 239},
    "held_out": {"start": 100, "end": 149}
  },
  "scenarios": ["default", "strict125"],
  "path_modes": ["normal", "path_zero"],
  "termination_mode": "endpoint_stop",
  "path_dim": 8,
  "device": "cpu",
  "dtype": "float64",
  "sensor_profile": "realistic_calibrated",
  "metrics": [
    "success",
    "pre_goal_3d_rmse_m",
    "legacy_full_3d_rmse_m",
    "success_only_3d_rmse_m",
    "goal_capture_rate_pnav",
    "goal_plane_miss_rate",
    "timeout_rate",
    "safety_rate",
    "material_saturation_rate"
  ],
  "output_root": "artifacts/lowlevel/pc_v1_nominal_eval_matrix_20261108"
}
```

说明：

- `held_out` 只作为边界声明，本 Goal 不运行；
- 40 个 dev seeds 用于降低 20-seed 下 2 个 episode 翻转造成的统计噪声；
- `dagger_pool` 只用于筛选失败状态和采集 DAgger，不参与 D0/D1/D2 的最终比较；
- Phase 0 首次跑通矩阵时，`base_environment_configs` 暂时指向已有的
  `v1_nominal_*_20261107` 配置，用于 N8 复现校验；Phase 1 产出 teacher-fixed
  配置后，只替换这两个路径，不改变矩阵 schema 或评估口径；
- checkpoint 仍可由命令行传入，避免矩阵绑定某一个模型。

### 4.2 评估器修改

新增一个非破坏式包装脚本或给现有评估器增加可选参数：

```text
scripts/run_pc_eval_matrix.py
```

职责：

- 读取 matrix JSON 并校验 seed 不重叠、held-out 不进入运行集；
- 对每个 checkpoint × scenario × path_mode 调 `scripts/evaluate_pc_bc.py`；
- 强制传入矩阵中的 `--termination-mode`、`--path-dim`、device、dtype；
- 生成 `eval_matrix_manifest.json`、`comparison.json` 和 `comparison.md`；
- 不改变 `evaluate_pc_bc.py` 的默认行为。

必须新增测试：

- 缺字段、重复 seed、held-out 混入 dev 时明确报错；
- `strict125` 的 `goal_radius_m` 必须为 `1.25`，default 必须为 `2.0`；
- path_dim 与 checkpoint metadata 不一致时拒绝启动；
- matrix 只影响评估，不写入训练配置。

### 4.3 Phase 0 完成条件

- [ ] 新 matrix JSON 和包装脚本已提交；
- [ ] 不修改旧 shell runner 的默认行为；
- [ ] 新 matrix 在同一 40-seed 列表下与直接调用 `evaluate_pc_bc.py` 的结果一致；
- [ ] held-out 未被运行；
- [ ] 测试和 ruff 通过。

注意：`_seed_command_generator` 会按整个 seed 列表计算 generator 偏移，因此
40-seed 矩阵不能与 Goal22 的 20-seed 历史数字逐项相等。复现校验必须固定同一
seed 列表；若跨 seed 数量比较，只能作为协议差异记录，不能当作模型回归。

## 5. Phase 1：教师一致性审计（不训练）

### 5.1 修正配置

以旧 R2B current 教师为控制基准，新建（不覆盖）两份 nominal 教师配置：

```text
configs/lowlevel/
  odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json
  odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_strict125_20261108.json
```

在旧 R2B current 的基础上只做必要替换：

- 传感器 profile 使用新的 nominal `realistic_calibrated`；
- 在 `termination` 中显式写入 `pc_goal_plane_miss_termination_enabled=true`，
  不依赖 CLI 或加载器默认值；
- 显式写入：

```json
{
  "guidance": {
    "current_crab_enabled": true,
    "robust_terminal_enabled": true,
    "route_speed_floor_enabled": true
  }
}
```

```json
{
  "termination": {
    "pc_goal_plane_miss_termination_enabled": true
  }
}
```

- 不改变 DAgger/PPO 的 actor 契约；
- 不在本轮同时改路径维数、网络、奖励或动作头。

### 5.2 教师评估

只跑 dev `20–59`（40 seeds），同一 CPU/float64/endpoint_stop，分别评估 default 和
strict125。教师审计不训练，也不使用 student 结果：

```bash
uv run python scripts/evaluate_pc_bc.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json \
  --checkpoint /dev/shm/pc_v0d_nominal_bc_rebase_20261107/bc_N8/pc_bc_actor.pt \
  --output artifacts/lowlevel/pc_v1_teacher_audit_20261108/fixed_teacher/default \
  --seed-start 20 --seed-count 40 \
  --device cpu --dtype float64 \
  --path-dim 8 --termination-mode endpoint_stop
```

checkpoint 在这里只是 `evaluate_pc_bc.py` 的契约载体，用来提供 path_dim 和加载
student 分支；所有报告数字必须从 `summary.json` 的 `teacher` 字段读取，禁止把
`student` 指标当作教师结果。运行前把 checkpoint 的实际路径和 SHA256 写入审计
manifest；若该 checkpoint 不存在或 path_dim 不匹配，应先新增只读 teacher-only
模式，不能用随便的 student 模型替代。

同一 seed 范围和同一指标口径，对旧 R2B current 控制配置（只补
`pc_goal_plane_miss_termination_enabled=true`，不改其他字段）跑一次 control；
新旧教师的 success、`pre_goal_3d_rmse_m`、timeout/safety 必须能在同一口径下比较。

### 5.3 通过条件

- default teacher success `>= 0.80`；
- strict125 teacher success `>= 0.45`；
- `pre_goal_3d_rmse_m` 与旧配置教师 control 相差不超过 `0.05 m`；
- timeout/safety 不高于旧配置教师 `+0.02`；
- 记录 `goal_plane_miss_rate`、saturation 和 path-zero 教师诊断。

若 strict125 教师仍低于 `0.45`，停止，不采任何 BC 数据；先检查配置继承、教师代码和
传感器 profile 是否还有差异。

### 5.4 Phase 1 完成条件

- [ ] 两份 teacher-fixed 配置建立；
- [ ] 配置 diff 只包含预期字段；
- [ ] default/strict125 教师评估完成；
- [ ] 门禁逐项判定；
- [ ] 若失败，记录根因并停止。

## 6. Phase 2：数据配方对照（不扩大总量）

### 6.1 固定条件

所有 arm 必须固定：

- Phase 1 的同一教师和同一环境配置；
- state `128,128`、actor `256,128`、path encoder `64`、`d_z=64`；
- batch `4096`、lr `1e-3`、weight decay `1e-5`、seed `0`；
- saturation weight `2.0`、threshold `0.9`；
- default/strict125 的同一训练 seeds；
- 总样本预算 `393,216`；
- dev 评估 seeds `20–59`（40 个），held-out `100–149` 不碰。

### 6.2 三个基础 arm

| arm | 教师数据 | 一般 DAgger | 定向 DAgger | 总样本 |
|---|---:|---:|---:|---:|
| D0 pure-BC | 393,216 | 0 | 0 | 393,216 |
| D1 +general-DAgger | 335,872–352,256 | 40,960–57,344（10.4–14.6%） | 0 | 393,216 |
| D2 +targeted-DAgger | 294,912–311,296 | 0 | 81,920–98,304（20.8–25.0%） | 393,216 |

数据处理规则：

- D1 的 general DAgger 从当前 D0 学生策略在训练 seed 池 `91000–91031` 上采一轮，
  固定选取其中的 10–14 个 seed（每个 4096 steps），不得使用评估 dev seeds；
- D2 的 targeted DAgger 从独立的 `dagger_pool` seeds `200–239` 的失败轨迹中选，
  default 和 strict125 各最多 10–12 个失败 seed，每个 4096 steps；合计最多
  20–24 个失败 seed，即 `81,920–98,304` samples，正好对应 `20–25%` 预算；
- 若失败池不足 20 个 seed，D2 必须记为 `D2_partial`，记录实际 targeted 比例；
  不得用随机教师数据在事后补成 20–25% 后仍宣称是完整 D2 对照；
- `dagger_pool` 与最终评估 dev `20–59` 完全分离，避免在评估 seed 上补数据造成泄漏；
- 失败筛选固定为 `success=false`、plane miss、timeout、safety，或 material
  saturation `> 0.5`；
- 采集时 teacher 必须重新标注 student 访问到的状态，不能只用 student 原动作；
- 每个数据分片必须记录 source（teacher/general-DAgger/targeted-DAgger）、seed、
  sensor profile、config hash、episode id 和 SHA256；
- 三个 arm 的教师样本必须来自同一批 teacher-fixed 数据，禁止混入旧 g19 数据来做
  “看起来更好”的对照。

### 6.3 评估与门禁

使用 Phase 0 的 40-seed 矩阵评估 D0/D1/D2，并报告：

- `success`、`pre_goal_3d_rmse_m`、`legacy_full_3d_rmse_m`；
- `success_only_3d_rmse_m`、`goal_capture_rate_pnav`、`goal_plane_miss_rate`；
- `timeout_rate`、`safety_rate`、`material_saturation_rate`；
- normal vs path-zero 的退化比例；
- default 与 strict125 分开报告，不合并成一个平均分。

预注册门禁：

- D1 或 D2 相对 D0，default pre-goal RMSE 至少改善 `3%`，或 success 至少提高
  `0.05`；
- default success 不得低于 D0；
- strict125 pre-goal RMSE 不得劣于 D0 超过 `5%`；
- strict125 success 不得低于 D0 超过 `0.05`；
- safety 增量不超过 `0.02`；
- path-zero 相对正常 input 至少退化 `5%`，证明路径分支仍被使用；
- 若三个 arm 都没有超过 D0，停止，不进入 stress 增广或 PPO。

最终 BC 候选仍要达到与旧 g19 可比的 dev 水平，建议目标：

- default pre-goal RMSE `<= 2.65 m`、success `>= 0.70`；
- strict125 pre-goal RMSE `<= 2.70 m`、success `>= 0.40`。

这是筛选目标，不是事后放宽的门禁；若只达到部分目标，记录为“部分改善”，不强行
进入 PPO。

### 6.4 Phase 2 完成条件

- [ ] D0/D1/D2 数据 manifest 和哈希完成；
- [ ] 三个 arm 同总预算、同网络、同 seed；
- [ ] 40-seed dev 评估完成；
- [ ] 门禁逐项判定；
- [ ] 未使用 held-out；
- [ ] 写出数据配方报告。

## 7. Phase 3：少量 stress 数据增广

仅在 Phase 2 出现明确胜出 BC 后执行。保持 nominal 为主，不要用 stress 替代主训练域。

建议两组：

| arm | nominal | stress | 目的 |
|---|---:|---:|---|
| S10 | 90% | 10% | 最小鲁棒性增广 |
| S20 | 80% | 20% | 更强鲁棒性诊断 |

stress 数据使用同一 teacher-fixed 控制逻辑和 `realistic_stress` profile；
重点覆盖传感器延迟、失锁、current estimate 退化和 USBL 跳变，不改变 actor 输入维度。

门禁：

- nominal pre-goal RMSE 不得比 Phase 2 胜出 arm 差超过 `2%`；
- nominal success 下降不超过 `0.03`；
- stress 评估的 success 至少提高 `0.05`，或 pre-goal RMSE 至少改善 `0.10 m`；
- safety 不得增加超过 `0.02`；
- 若 nominal 明显退化，回退到纯 nominal 配方，stress 只作为压力测试，不进入主训练。

Phase 3 完成条件：

- [ ] S10/S20 数据 manifest、哈希和传感器 metadata 完整；
- [ ] nominal + stress 双评估完成；
- [ ] 主指标仍以 nominal 为准；
- [ ] 记录 stress 收益与 nominal 代价。

## 8. Phase 4：残差 PPO（最后才做）

只有 Phase 2/3 的 BC 达到门禁后才允许启动。残差 PPO 的前提是“基础策略已经不错”；
不能用 PPO 代替教师/数据修复。

- DADE 保持关闭；
- 残差尺度固定，不扫参；
- 训练预算固定 `4096 env × 128 steps × 200 iterations`；
- 先 1 个训练 seed screen，通过后扩到 3 个训练 seed；
- 评估使用 Phase 0 的 40-seed dev 矩阵；
- 主门禁看 `pre_goal_3d_rmse_m` 和 success，完整 RMSE 仅诊断；
- 若 BC 仍未通过，禁止启动 PPO，并在报告中写明 `ppo_not_started`。

Phase 4 完成条件：

- [ ] zero-residual 基线完成；
- [ ] 1-seed screen 完成；
- [ ] 3-seed 扩展仅在 screen 通过后执行；
- [ ] 与冻结 BC 同 seed 配对评估；
- [ ] 若失败，停止且不扫参。

## 9. 产物目录

```text
artifacts/lowlevel/pc_v1_teacher_audit_20261108/
artifacts/lowlevel/pc_v1_data_recipe_20261108/
artifacts/lowlevel/pc_v1_stress_mix_20261108/
artifacts/lowlevel/pc_v1_eval_matrix_20261108/
configs/lowlevel/eval/pc_v1_nominal_eval_matrix_20261108.json
doc/goals/goal24_data_recipe_report_20261108.md
```

报告必须包含：

- 教师配置 diff 和 teacher audit 表；
- D0/D1/D2 数据组成、总量、哈希和 seed 来源；
- 40-seed dev 主表和逐 seed 配对统计；
- stress 增益与 nominal 代价；
- BC 是否通过以及是否启动 PPO；
- 未使用 held-out、未扩 seed、未扫超参的声明。

## 10. 执行顺序与停止条件

固定顺序：

```text
Phase 0 统一评估矩阵
  → Phase 1 教师一致性审计
  → Phase 2 纯 BC / general-DAgger / targeted-DAgger 对照
  → Phase 3 少量 stress 增广（可选，仅在 Phase 2 胜出后）
  → Phase 4 残差 PPO（仅在 BC 门禁通过后）
```

立即停止条件：

- 教师 strict125 success 未恢复到 `0.45` 左右；
- 训练或评估出现 NaN/Inf；
- 数据 manifest 不能证明 source、seed、sensor profile 和 hash；
- dev 没有改善，或改善只来自单个 seed；
- 需要使用 held-out 才能做决策；
- 资源冲突或可能覆盖旧 artifact。

## 11. 完成勾选

- [ ] Phase 0：统一评估矩阵配置和包装脚本；
- [ ] Phase 0：矩阵复现 Goal22 N8 结果；
- [ ] Phase 1：teacher-fixed 配置建立；
- [ ] Phase 1：default/strict125 教师审计完成；
- [ ] Phase 1：strict125 教师 success 达到 `0.45` 左右；
- [ ] Phase 2：D0/D1/D2 三个数据配方完成；
- [ ] Phase 2：40-seed dev 评估和门禁完成；
- [ ] Phase 3：S10/S20 stress 增广（仅在需要时）；
- [ ] Phase 4：BC 门禁通过后才启动残差 PPO；
- [ ] 报告、manifest、哈希和未启动项声明完成；
- [ ] 未使用 held-out `100–149`；
- [ ] 未扩 seed、未扫超参、未覆盖旧产物。

## 12. Git 记录建议

完成后先检查：

```bash
git status --short
git diff --stat
```

只暂存本 Goal 明确修改的文件，不使用 `git add -A`：

```bash
git add \
  configs/lowlevel/eval/pc_v1_nominal_eval_matrix_20261108.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_default_20261108.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_teacher_fixed_strict125_20261108.json \
  scripts/run_pc_eval_matrix.py \
  tests/test_lowlevel_pc_eval_matrix.py \
  doc/goals/goal24_teacher_data_recipe_and_eval_matrix_20261108.md \
  doc/goals/goal24_data_recipe_report_20261108.md

git commit -m "feat(pc): add teacher audit and data recipe matrix"
```

推送只在用户明确要求时执行；若当前环境不是受控 git 工作区，只保存文件清单、SHA256
和运行命令，不自动 commit/push。
