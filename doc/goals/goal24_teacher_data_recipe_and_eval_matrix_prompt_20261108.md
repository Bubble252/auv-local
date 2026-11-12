# Goal24 执行提示词：教师审计、数据配方与统一评估矩阵

日期：2026-11-08

状态：**已就绪，待执行**

## 提示词

请按 `doc/goals/goal24_teacher_data_recipe_and_eval_matrix_20261108.md` 完整执行
Goal24，并在每个 Phase 结束后写入报告；除门禁失败、资源冲突或需要用户裁决外，不要
中途停下来等我确认。

先做资源和产物检查：记录当前进程、GPU、`df -h` 和已有输出，确认没有覆盖旧配置、
旧 checkpoint、旧 dataset 或旧 artifacts。保持非破坏式修改，新配置、新数据和新
checkpoint 全部使用新名称。若服务器不可用而只能在本地跑，所有产物标记
`local_only`，并明确不能冒充服务器正式结果；服务器恢复后按同一协议重跑。

执行顺序固定如下：

1. **Phase 0：统一评估矩阵。** 新增
   `configs/lowlevel/eval/pc_v1_nominal_eval_matrix_20261108.json` 和非破坏式的
   `scripts/run_pc_eval_matrix.py`，集中管理 dev seeds `20–59`、default/strict125、
   normal/path-zero、`endpoint_stop`、`path_dim=8`、CPU/float64、指标和输出目录。
   不改变旧评估器默认行为；用 N8 checkpoint 做一次矩阵复现校验。
2. **Phase 1：教师一致性审计，不训练。** 以旧
   `odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json` 为控制基准，
   新建两份 nominal teacher-fixed 配置，显式写回：
   `current_crab_enabled=true`、`robust_terminal_enabled=true`、
   `route_speed_floor_enabled=true`；同时在 `termination` 中显式写
   `pc_goal_plane_miss_termination_enabled=true`。只跑 dev `20–59`（40 seeds）的
   default/strict125 教师评估。教师审计用 `evaluate_pc_bc.py` 时必须明确：checkpoint
   只是评估器载体，所有上报数字只取 `summary.json` 的 `teacher` 字段，不能把
   student 结果当教师结果；并把 checkpoint 路径和 SHA256 写入 manifest。
   strict125 教师 success 必须恢复到约 `0.45`，default 约 `0.80`；否则记录失败原因
   并停止，不采任何 BC 数据。旧 R2B current 控制配置按同一 40-seed 口径跑一次
   control，用于比较 `pre_goal_3d_rmse_m`、timeout 和 safety。
3. **Phase 2：数据配方对照。** Phase 1 通过后，固定教师、网络、seed、总样本
   `393,216` 和 8D path，比较 D0 纯教师 BC、D1 约 10–15% general DAgger、D2 约
   20–25% targeted DAgger。general DAgger 只从训练 seed 池 `91000–91031` 采；
   targeted DAgger 只从独立失败池 `200–239` 采，default 和 strict125 各最多 10–12
   个失败 seed（合计 20–24 个，最多 98,304 samples，约 20–25%）；评估 dev 固定为
   `20–59`，held-out `100–149` 绝对不碰。重点补失败状态、plane miss、timeout 和
   饱和动作附近的纠正数据，不要随机多采教师轨迹。
4. **Phase 3：少量 stress 增广。** 仅在 Phase 2 出现明确胜出 BC 后，做 90/10 和
   80/20 两组 nominal/stress 混合；stress 只增强鲁棒性，不得替代 nominal 主训练。
   nominal 指标明显退化就回退。
5. **Phase 4：残差 PPO。** 只有 BC 通过门禁后才启动。先 zero-residual 基线，再做
   1 个训练 seed screen；通过后才扩到 3 个训练 seed。DADE 关闭，残差尺度固定，
   不扫参；未通过 BC 时必须在报告中写 `ppo_not_started`。

数据和评估必须保留 source、seed、sensor profile、config hash、episode id 和
SHA256；default 与 strict125 分开报告，主指标为 `pre_goal_3d_rmse_m`，完整 RMSE
仅作诊断。20-seed 统计噪声较大，Phase 2 的模型选择使用 40 个 dev seeds，并报告
逐 seed 配对统计。

任何阶段出现以下情况立即停止并写负结果：教师 strict125 未恢复到约 `0.45`；
训练/评估出现 NaN/Inf；manifest 不能证明数据来源和 seed；dev 无改善或收益只来自
单个 seed；必须使用 held-out 才能做决策；发生资源冲突或可能覆盖旧产物。

最终用中文输出：教师审计表、D0/D1/D2 数据构成与门禁表、40-seed dev 主表、
stress 增益与 nominal 代价、BC 是否通过、是否启动 PPO、所有产物路径与哈希，以及
未使用 held-out、未扩 seed、未扫超参的声明。不要自动 commit/push，除非我明确要求。

## 最短版

> 按 `goal24_teacher_data_recipe_and_eval_matrix_20261108.md` 自动执行：先建统一
> 评估矩阵；再做 teacher-fixed 审计，strict125 教师约 `0.45` 才继续；然后固定总预算
> 比较纯 BC、10–15% general DAgger、20–25% targeted DAgger；之后才做少量 stress
> 增广和残差 PPO。评估 dev 用 `20–59`，DAgger 失败池用 `200–239`，不碰 held-out
> `100–149`。非破坏式修改，门禁失败即停止并报告。
