# Goal23：Goal22 → P3 DAgger → P5 残差 PPO（含理想 actor 观测臂）

日期：2026-11-07

状态：**已于 Goal22 门禁停止；N12 未通过 default 门禁，P3/P5/理想臂均未启动**

## 1. 一句话目标

先完成 Goal22 的 nominal BC 重采、重训和门禁；通过后连续执行 P3 定向 DAgger 与
P5 冻结 BC 残差 PPO，直到输出最终报告。同时增加一个理想 actor 观测对照臂，用于
量化“观测理想化收益”和“部署落差”。

## 2. 前置文档

- `doc/goals/goal22_nominal_bc_rebase_prompt_20261107.md`
- `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`
- `doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_report_20261107.md`
- `doc/goals/goal21_sensor_profiles_nominal_stress_20261107.md`
- `doc/goals/goal9_execution_state.md`

## 3. 固定实验矩阵

主臂与理想臂的所有非观测因素必须一致：网络结构、动作空间、奖励、教师、终止逻辑、
数据预算、训练 seed、评估 seed、传感器与海流场景。

| 臂 | path | 场景 | actor 输入 | 用途 |
|---|---|---|---:|---|
| N8 | 8D | default/strict125 | nominal realistic | 主臂、无终点上下文 |
| N12 | 12D | default/strict125 | nominal realistic | 主臂、含终点上下文 |
| I8/I12 | 与胜出 N 臂一致 | default/strict125 | ideal | 上界诊断，不可部署 |

主臂固定：

- `realistic_calibrated` nominal profile；
- `current_estimate_max_age_s=0.9`；
- `endpoint_stop`；
- default `64×4096`、strict125 `32×4096`；
- default seeds `90000–90063`，strict125 seeds `90100–90131`；
- dev 评估 seeds `20–39`，不碰 held-out `100–149`。

理想臂只替换 actor 输入：真实无噪声位姿、真实角速度、真实速度/地速、真实海流、
真实路径投影和终点量；不改变 actor 顶层输入宽度，不改变 critic、教师、奖励和终止。
manifest 必须写入 `actor_observation_source=ideal|nominal` 和
`actor_current_field=actor_current_body`。

## 4. Phase 0：代码与契约

- [x] BC `path_dim` 支持 `0/8/12`，旧 8D checkpoint 保持可加载；
- [x] 12D feature names、checkpoint metadata 和训练脚本参数支持；
- [x] 环境新增 `observation.ideal_actor_observations`，默认关闭；
- [x] 理想 actor 使用真实位姿、姿态、角速度、地速、海流和终点路径；
- [x] BC/DAgger 采集使用 `observation.actor_current_body` 并记录来源；
- [x] DAgger 在 current-conditioned checkpoint 上传入 `actor_current`；
- [x] 新增 Goal22 nominal/ideal 配置矩阵生成器；
- [x] 新增 `--seed-list`，支持 P3 按失败 seed 定向采集；
- [x] 定向测试与 ruff 通过。

## 5. Goal22：nominal BC 重采与重训

- [x] 生成 N8/N12 default/strict125 配置；
- [x] 采集 N8 default/strict125 教师数据；
- [x] 采集 N12 default/strict125 教师数据；
- [x] 合并 N8/N12 数据集并校验 manifest；
- [x] 训练 N8/N12 BC；
- [x] 离线饱和、path-zero、finite 和哈希审计；
- [x] dev `20–39` 的 default/strict125/path-zero 评估；
- [x] 旧 `g19_dagger` nominal 参考评估；
- [x] 逐项判定 Goal22 门禁：**FAIL（4/13 项失败）**；N12 不进入后续，N8 仅留作未来候选。

Goal22 门禁未通过时停止，不进入 P3/P5，也不扩 seed、不扫超参。门禁定义以
`goal22_nominal_bc_rebase_prompt_20261107.md` 第 9.2 节为准。

本次结果：`doc/goals/goal22_nominal_bc_rebase_report_20261107.md`。由于 Goal22
未通过，以下理想臂、P3 和 P5 均未启动，不能把本节后续的未勾选项描述为已执行。

## 6. 理想 actor 完整对照臂（BC → P3 → P5）

- [ ] 对 Goal22 胜出的 N 臂创建同预算 I 臂；
- [ ] 采集 I 臂 default/strict125 教师数据；
- [ ] 训练 I 臂 BC，架构与对应 N 臂一致；
- [ ] 对 I 臂执行与主臂相同的 P3 定向 DAgger，并用失败 seed 重训；
- [ ] 对 I 臂执行与主臂相同的 P5 冻结 BC 残差 PPO；
- [ ] 在 ideal 输入和 nominal realistic 输入下双评估；
- [ ] 记录 success、`pre_goal_3d_rmse_m`、完整 RMSE、timeout、safety、saturation；
- [ ] 计算 `idealization_gain = nominal_metric - ideal_metric`；
- [ ] 计算 `deployment_gap = nominal_input_metric - ideal_input_metric`。

该臂不允许用于放宽主臂门禁，也不得作为可部署结果。

## 7. P3：default + strict125 定向 DAgger

- [ ] 从 Goal22 胜出 BC 的评估记录中提取失败 seed；
- [ ] 失败条件固定为 success=false，或 plane miss/timeout/safety，或饱和率 > 0.5；
- [ ] 对 default 和 strict125 分别使用 `--seed-list` 采集一轮教师纠正；
- [ ] DAgger 预算固定为每场景最多 16 个失败 seed × 4096 步；
- [ ] 合并原始 BC 数据和 DAgger 数据，保留 episode 来源与 seed；
- [ ] 沿用 Goal22 BC 配方重训；
- [ ] 用同一 dev `20–39` 评估 P3 BC；
- [ ] 按 Goal9 P3 门禁判定；失败即停止，不扩轮数、不放宽门禁。

## 8. P5：冻结 BC 残差 PPO

- [ ] 只使用 P3 通过后的强 BC；
- [ ] 先跑 zero-residual 基线，确认冻结 BC 的动作不被训练器改变；
- [ ] 训练环境启用 `endpoint_stop` 和 12D 终点上下文；
- [ ] DADE 保持关闭；残差尺度固定，不扫参；
- [ ] 先做 1 个训练 seed 的 screen；通过后扩到 3 个训练 seed；
- [ ] 使用独立评估 seeds `20–39`，不碰 held-out；
- [ ] 按 Goal9 P5 门禁判定；失败即停止。

本地 8GB 显存实测 `4096×256×100` OOM；正式预算固定为
`4096 env × 128 steps × 200 iterations`，与 `4096×256×100` 的总环境步数等价。
主臂与理想臂必须使用同一预算，不得针对结果单独增减。

奖励只允许按预注册单变量加入：

- 终端接近奖励；
- 终点通道横向对准惩罚；
- `GOAL_PLANE_MISS` 固定终端惩罚。

不增加时间惩罚、额外速度惩罚、额外动作惩罚或全局过冲惩罚。

## 9. 双输入评估

主臂和理想臂都必须报告两套输入：

1. `ideal`：真实无噪声 actor 输入；
2. `nominal_realistic`：可部署传感器估计输入。

主表至少包含：

| 模型 | 输入 | 场景 | success | pre_goal RMSE | full RMSE | timeout | safety | saturation |
|---|---|---|---|---:|---:|---:|---:|---:|

同时报告 `idealization_gain`、`deployment_gap` 以及相对主臂的变化。主指标仍是
`pre_goal_3d_rmse_m`；完整 RMSE 只作为诊断，不能与 pre-goal 混列。

## 10. 产物目录

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
  eval_goal22/
  p3/
  p5/
  ideal_actor/
  comparison.json
  comparison.md
  contract_checks.json
  run_manifest.json
```

磁盘不足时可把大体积数据放在 `/dev/shm` 临时盘，但报告、manifest、checkpoint 和
最终 comparison 必须回到仓库产物目录；不得覆盖旧 artifacts。

## 11. 运行与资源检查

每次启动前执行：

```bash
ps -eo pid,etime,pcpu,pmem,args | rg -i \
  'supermimic|pc_.*train|train_pc_bc|collect_pc_bc|evaluate_pc_bc|ppo|isaac' \
  | rg -v 'rg -i' || true
nvidia-smi || true
df -h /
git status --short
```

服务器恢复后优先在服务器跑满预算；本地只做 smoke 或临时盘验证。当前服务器
`capri_target` 的 `2237` 端口拒绝连接，不能把本地结果冒充服务器结果。

## 12. Git 记录

完成后先检查：

```bash
git status --short
git diff --stat
```

只暂存本次明确修改的文件，不使用 `git add -A`：

```bash
git add \
  src/auv_lowlevel/config.py \
  src/auv_lowlevel/batch/pc_env.py \
  src/auv_lowlevel/batch/pc_bc.py \
  scripts/train_pc_bc.py \
  scripts/collect_pc_bc_demos.py \
  scripts/collect_pc_bc_dagger_demos.py \
  scripts/generate_pc_goal22_nominal_bc_configs.py \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_bc_*.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_ideal_bc_*.json \
  tests/test_lowlevel_pc_bc.py \
  tests/test_lowlevel_pc_env.py \
  doc/goals/goal23_goal22_to_p5_ideal_actor_20261107.md \
  doc/goals/goal9_execution_state.md

git commit -m "feat(pc): add goal22 nominal and ideal actor BC pipeline"
git push
```

推送只在用户明确要求时执行；若服务器仓库不是受控 git 工作区，只记录文件清单、
SHA256 和运行命令。

## 13. 最终报告

必须用中文给出：

1. Goal22 N8/N12 数据、训练和门禁表；
2. 胜出臂及选择依据；
3. 理想臂 ideal/nominal 双评估表；
4. P3 DAgger 数据量、门禁和结果；
5. P5 zero-residual 与 PPO screen/扩展结果；
6. 是否达到 P3/P5 门禁；
7. 所有产物路径、哈希和未启动项声明；
8. 失败根因和下一步建议。
