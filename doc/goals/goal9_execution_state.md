# Goal9 跨轮次执行状态

更新时间：2026-11-07 CST

本文件只记录实时运行状态，不替代协议、`summary.json` 或阶段报告。每次接手任务必须先
核对服务器进程、日志和 artifacts；若与本文件冲突，以实际证据为准。

## 2026-11-08 Goal24：教师一致性、数据配方与统一评估矩阵已登记

- 规划：`doc/goals/goal24_teacher_data_recipe_and_eval_matrix_20261108.md`；
- 执行提示词：`doc/goals/goal24_teacher_data_recipe_and_eval_matrix_prompt_20261108.md`；
- 前置结论：Goal22 失败不能只归因于 12D；新 nominal BC 配置缺少旧 R2B current 的
  crab/robust-terminal/route-speed 开关，且新 BC 为纯教师数据，旧 g19 含约 37.9%
  DAgger 数据；
- 顺序固定为：统一评估矩阵 → 教师审计 → 纯 BC / general-DAgger / targeted-DAgger
  数据配方对照 → 少量 stress 增广 → BC 通过后才做残差 PPO；
- 评估矩阵计划使用 dev `20–59` 共 40 seeds，held-out `100–149` 不碰；
- 执行提示词已就绪；教师审计、D2 预算和 checkpoint 契约已统一到同一口径；
- 当前仅登记，尚未启动新训练、数据采集或 held-out 评估。

## 2026-11-07 Goal22 nominal BC 重采与重训：执行完成，预注册门禁 FAIL

- 报告：`doc/goals/goal22_nominal_bc_rebase_report_20261107.md`；
- N8（8D）default `pre-goal 2.8237 m / success 0.65`，strict125
  `2.8044 m / 0.25`；
- N12（12D）default `2.8810 m / 0.55`，strict125 `2.8664 m / 0.35`；
- 门禁 9/13 PASS、4/13 FAIL：N12 default pre-goal、success、capture、plane-miss
  四项不过线；`overall_pass=false`；
- 判定：**stop_goal22_gate_failed**；不进入 P3 DAgger、P5 残差 PPO、理想 actor
  完整对照臂；不扩 seed、不扫超参、不碰 held-out；
- N8 仅保留为未来新 Goal 的候选底座，本次没有继续训练；
- 2026-11-08 追加复核发现新旧 BC 跨代比较有混杂：旧 g19 含 `159,744/421,888
  = 37.9%` DAgger 数据和更严苛的传感器训练分布；新 N8/N12 为纯 BC。另新 nominal
  配置未显式继承 `current_crab_enabled/robust_terminal_enabled/route_speed_floor_enabled`，
  加载器默认均为 false；后续进入 P3/P5 前必须先修正或做配置对照；
- dev `20–39` 全部评估完成，10 份 `summary.json` finite；path-zero 均明显退化，
  路径分支有效；定向测试 `56 passed`，ruff 通过；
- 产物：`artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107/`（指向
  `/dev/shm/pc_v0d_nominal_bc_rebase_20261107/`），`comparison.json/md` 已落盘。

## 2026-11-07 Goal22 nominal BC 重采与重训：历史登记（已被上方结果取代）

- 提示词：`doc/goals/goal22_nominal_bc_rebase_prompt_20261107.md`；
- 插入位置：Goal20 P2A/P2B 已完成之后、P3 定向 DAgger 之前；
- 目标：在 `realistic_calibrated` nominal 传感器域和 `endpoint_stop` 下，重采 PC
  教师示教并重训 BC；同时用 N8/N12 两臂隔离终点上下文观测收益；
- 已知代码缺口：`pc_bc.py`/`train_pc_bc.py`/`collect_pc_bc_demos.py` 仍只支持 8D
  BC；Goal22 必须先做非破坏式 12D 支持，同时保证旧 8D checkpoint 原样可加载；
- 实验预算：default `64×4096` + strict125 `32×4096` 每个 path-dim arm；dev seeds
  `20–39`，不碰 held-out；
- 停止线：本 Goal 不启动 P3 DAgger、P5 残差 PPO、held-out 或超参数扫参；
- 当前未启动数据采集、训练或评估。

## 2026-11-07 Goal21 传感器 profile 拆分：nominal / legacy / stress

- 规划与契约：`doc/goals/goal21_sensor_profiles_nominal_stress_20261107.md`；
- `realistic_calibrated` 重定义为 nominal 实机观测默认域；13 个长尾字段收紧，
  `current_estimate_max_age_s=0.9` 的新入口配置；
- 旧标定语义完整保留为 `realistic_calibrated_legacy`；旧 `realistic` 极端范围新增
  显式别名 `realistic_stress`；`realistic` 和 `strong` 本身均未修改；
- 新入口配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_20261107.json`、
  `..._nominal_strict125_20261107.json`、`..._stress_20261107.json`；
- 修改期间有两个 SUPERMIMIC WM 数据导出进程正在使用旧 v1 配置；未杀进程、未重启，
  也没有修改它们正在读取的配置文件；
- 测试：传感器、配置、PC contract/env/vec-env/teacher 共 `98 passed`；ruff check
  通过；format check 仍报告既有整文件格式漂移，未做无关重排；
- 未启动新的训练、BC 重采或 PPO；后续新实验优先使用 nominal 配置，压力测试显式使用
  stress 配置。

## 2026-11-07 Goal20 P2B/P2A：执行完成，结论 PASS_CONTRACT

- 提示词：`doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_prompt_20261107.md`；
- 报告：`doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_report_20261107.md`；
- 结论：`PASS_CONTRACT`；实现、legacy 兼容、8 组复评、`GOAL_PLANE_MISS` 隔离、
  长尾指标和静态检查均通过；
- 已实现 `PCTerminalKind.GOAL_PLANE_MISS=9`、默认关闭的 `endpoint_stop`、
  `advance_mask` 终止行冻结、12D `endpoint_context` path block 和评估器
  `--path-dim {0,8,12}` / `--termination-mode {legacy,endpoint_stop}`；
- 终止优先级已修正为 endpoint 模式下
  `GOAL > GOAL_PLANE_MISS > timeout/safety/out-of-bounds`；legacy 分支保持历史顺序；
- 8 组 dev `20–39` 已顺序完成；legacy 四个组合与 P1 数值在 `1e-12` 内逐字段一致；
- endpoint 四组 timeout/safety 均为 `0`，原 timeout 主要转为 `GOAL_PLANE_MISS`；
  `pre_goal_3d_rmse_m` 与 legacy 最大差 `0.0096 m`；
- 复核发现初版 evaluator 曾把 legacy 的 `eta` goal 终止误标成 `p_nav`；已新增独立
  `pnav_goal_capture_seen` 并重跑 8 组，修正前产物备份在
  `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107_pre_pnav_metric_fix_20261107/`；
- 用户裁决：`p_nav` 为可部署的主成功口径，真实 `eta` 只作诊断；历史
  `paired_classification_only_plane_miss` 检查降级为 `DIAG`，不再作为硬门禁；
- seed 级仍有 5 个分类迁移（2 个 `success->plane_miss`、3 个 `timeout->success`）；
  它们由 `p_nav` 与真实 `eta` 的估计边界造成，不是终止后继续推进、提前终止或
  safety/timeout 重叠造成的长尾；
- 8 组均同时记录 `capture p_nav` 与 `capture eta`；endpoint 四组的 `p_nav`/`eta`
  成功率分别为 `0.70/0.65`、`0.45/0.45`、`0.70/0.50`、`0.50/0.40`；
- legacy 四组在 `eta` 终止前的 `p_nav` 捕获率分别为
  `0.10/0.05/0.20/0.15`，对应真实 `eta` 捕获率为 `0.70/0.45/0.65/0.50`；legacy
  不等待 `p_nav` 进球，因此该列只能作诊断，不能当作 legacy success；
- 产物：
  `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/`，包含 8 个
  `summary.json`、`comparison.json/md`、`contract_checks.json`、`validation.json`；
- 测试：`136 passed`；ruff check 通过，格式检查 `13 files already formatted`；
- 方案 2（goal capture 用真实 `eta`、plane miss 用 `p_nav`）保留为后续备选臂，本次
  没有重跑，也没有混入主实验；
- 未启动 P3 DAgger、未训练 BC/PPO/残差 PPO；等待用户明确启动下一阶段。

## 2026-11-07 Goal9 P1/P2：统一复评与路径误差审计完成

- 规划：`doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`；
- 报告：`doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md`；
- 下一阶段提示词：
  `doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_prompt_20261107.md`；
  范围固定为 P2B `endpoint_stop`、P2A 终点上下文观测和双模式复评；已于同日执行，
  最终结论见上方 `PASS_CONTRACT`；
- P1：7 个冻结 checkpoint × default/strict125/path-zero，共 21 份 dev `20–39`
  评估完成；没有训练新模型；
- `g19_dagger` 的本地缺失 checkpoint 已从旧训练服务器恢复，SHA256
  `acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3`；
- default 第一：`g18_s0`，3D RMSE `6.3695 m`、success `0.70`、timeout `0.30`；
- strict125 第一：`g19_dagger`，3D RMSE `7.9071 m`、success `0.50`、timeout `0.50`；
- Goal17 A 矩阵统一复评：`A00 8.0384`、`A10 10.1432`、`A01 9.7369`、`A11 7.7864`；
- P2：`g18_s0`、`g19_satw`、`g19_dagger`、`A00`、`A11` 的 default 审计完成；
  `g18_s0`、`g19_dagger`、`A00`、`A11` 的 strict125 审计完成；
- P2 主结论：timeout 回合的沿路径纵向误差主导 3D RMSE；`g18_s0` 纵向 MSE 占比
  `74.1%`、`g19_dagger 72.9%`、`A00 70.9%`；hint/global 投影差异只有 `0.012–0.034 m`；
- P2.1 已完成：从已有 P2 traces 重算四项终点诊断指标，没有重新训练或 rollout；
  `hint_rmse_3d_m` 与旧 summary 逐位一致，`success_only_3d_rmse_m` 与旧
  `success_group.hint_rmse_3d_m` 也逐位一致；
- P2.1 产物：
  `artifacts/lowlevel/pc_v0d_p2_1_diagnostic_metrics_20261107/`；
- P2.1 关键结论：success-only 和 pre-goal RMSE 明显低于完整 RMSE；例如
  `g18_s0/default` 为 `2.6827 / 2.5805` 对 `6.3695 m`，`g19_dagger/strict125` 为
  `1.5872 / 2.4847` 对 `7.9071 m`；post-goal overshoot 和 terminal longitudinal
  才是主要差距来源；
- A11−A00 default：3D RMSE `-0.5193 m`，95% CI `[-2.3682, +1.3297]`，不显著；
  横向 `+0.5057 m`、纵向 `-1.0111 m`；
- 根因：现有 8D 路径特征在终点把 `progress_ratio` 裁到 1、`remaining_ratio` 裁到 0，
  又没有显式终点距离/过冲，actor 难以区分“刚到终点”和“已越过终点”；
- 产物：
  `artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107/`、
  `artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107/`；
- 新增脚本：
  `scripts/audit_pc_bc_path_error_20261107.py`、
  `scripts/summarize_pc_v0d_p1_unified_bc_20261107.py`、
  `scripts/summarize_pc_v0d_p2_path_audit_20261107.py`；
- 当前没有教师、BC、PPO 或评估进程在运行；
- 历史 3D RMSE 主指标保持不变；四项诊断指标已完成并作为 `diagnostic_only` 输出；
- 用户已确认新增协议：PPO 和后续新评估启用版本化 `endpoint_stop`，第一次越过终点
  切平面但未进入 `goal_radius_m` 时立即以 `GOAL_PLANE_MISS` 终止并给固定终端惩罚；
  不再让模型继续飞越终点；
- 新评估改为“任务护栏 + `pre_goal_3d_rmse_m` 主排序 + `legacy_full_3d_rmse_m`
  历史回归”；`post_goal_overshoot_m` 和 `terminal_longitudinal_rmse_m` 只在
  `legacy` 轨迹上报告；
- PPO 稠密奖励不重复添加已有的 progress/3D/yaw/speed-smooth/energy/action/saturation
  项；只登记两个终点附近候选：终端接近奖励和终点通道横向对准惩罚，均需单变量预注册；
- P2A 已登记具体终点观测：`goal_distance_3d_m`、
  `signed_goal_tangent_offset_m`、`goal_bearing_heading_error_rad`、
  `goal_bearing_pitch_error_rad`，可选 `endpoint_active_flag`、
  `terminal_capture_ratio`、`goal_radius_excess_m`；全部编码进 `z_t`，不扩大 actor
  顶层输入；其中两个 bearing error 和切平面偏移在 actor 输入前乘终点平滑门控，
  初版 `d_enter=20 m`、`d_full=5 m`，只在临近终点时增强，远距离不参与路径跟踪；
  终点距离在 actor 侧改为单调的
  `goal_distance_proximity = g_endpoint * (1 - clip(d / d_enter, 0, 1))`，
  不直接把原始米数乘门控；
- 该协议尚未实现；下一步先完成 P2B 终止协议、P2A 终点观测块，再启动 P3 双场景
  DAgger；
- 不自动启动 P2A/P2B/P3，不先做残差 PPO，不继续扫旧 A 矩阵超参数。

## 2026-11-07 Goal17：encoded-current × crab 因子实验完成，BC 门禁 FAIL

- 协议：`doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`；
- 状态：阶段 0–5 完成；阶段 4b 三训练 seed 判定 `FAIL_BC_FACTOR_GATE`；阶段 6
  未触发；
- 报告：
  `doc/goals/goal17_v0d_current_crab_factorial_report_20261107.md`；
- 教师底座为 R2B baseline，只切换 `current_crab_enabled`；T0/T1 均完成
  default、strict125 的 20-seed 纯教师评估；
- T0（crab off）：default `4.7379 m / 0.80`，strict125 `10.3278 m / 0.10`；
- T1（crab on）：default `4.3531 m / 0.85`，strict125 `9.3916 m / 0.15`；
- crab 配对效果：default RMSE `-0.3848 m`、success `+0.05`；strict125 RMSE
  `-0.9361 m`、success `+0.05`；因此选定 **T1（crab on）**；
- 服务器决策产物：
  `artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/teacher/stage2_teacher_selection.json`；
  SHA256 `7da5fe5e8ccdd3b6253f02b329b0eb5d1940d9e4c84fd0b466d3b1b2995da1d4`；
- seed 契约：`0–19` 仅作历史筛查，不再参与新模型选择；四格 BC 和消融使用
  开发集 `20–39`；最终 held-out 使用 `100–149`，只在候选方案冻结后评估一次；
- 阶段 2 的 `teacher_summary.json` 未落盘 crab angle，原始 trace 也未保留；已按非阻塞
  诊断项记录为“当前产物不可得”；
- `T0/T1` 示范已完成：各 `262,144` samples，SHA256 分别为
  `4c9faf126a5d5bec8dc6b3f65766cef39215862789a1564b1c7636510eb1d7d2` 和
  `5e89cb7d790dcef8f2e9e8de1d1b99a1983a54420a7f21185f75b3b9454bd105`；
- 四格 BC 已完成训练 seed `91430/91431/91432` 与 dev `20–39` 评估；
- default 三 seed 聚合：`A00 8.0067 m / 0.50`、`A10 9.6232 m / 0.45`、
  `A01 8.7582 m / 0.533`、`A11 8.1176 m / 0.55`；
- 门禁：`A10` RMSE ratio `1.2019`、`A11` ratio `1.0139`，均未达到 `<=0.95`；
- `A11` 的 current-zero/shuffle 分别变差 `+1.5446 m` 和 `+1.1759 m`，说明 current
  encoder 被使用，但没有带来相对 `A00` 的净收益；
- 决策：`stop_goal17_bc_gate_failed`；不冻结 BC、不进入残差 PPO、不评估 held-out；
- 完成审计：相关测试 `115 passed`、`ruff` 通过、12 个 checkpoint 契约一致、dev 评估
  safety 全为 `0`，未发现 held-out `100–149` 评估产物；
- 机器可读收口：`artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/stage5_closure.json`；
- 当前没有 Goal17 的教师、BC、PPO 或评估 runner 在运行；
- 下一步：Goal17 停止；若继续 current/crab 方向，应另开 Goal 解决 BC 动作表示或数据
  分布问题，不能靠 PPO 掩盖失败 BC。

## 2026-11-07 Goal17 后续：BC 提升路线已登记（仅规划，不执行）

- 规划文档：`doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`；
- 已登记六个候选方向：
  1. 统一复评已有 BC；
  2. 审计 3D RMSE 构成；
  3. default + strict125 双场景 DAgger；
  4. action chunking / 降低控制频率；
  5. 残差 PPO；
  6. 在更强 BC 上复测 current/crab；
- 推荐顺序：先做 P1/P2 诊断，不训练；再根据瓶颈在 P3/P4 中只选一个；P5 只能在 BC
  门禁通过后启动；P6 依赖 P1 选出强底座；
- 当前没有启动教师、BC、PPO 或评估 runner；该文件不改变 Goal17 的负结果和任何已
  冻结门禁。

## 2026-11-07 Goal19：weighted BC 定向 DAgger 完成，FAIL_RMSE

- 协议：`doc/goals/goal19_v0d_weighted_bc_targeted_dagger_prompt_20261107.md`
- 报告：`doc/goals/goal19_v0d_weighted_bc_targeted_dagger_report_20261107.md`
- 状态：阶段 0–6 完成并收口；判定 `FAIL_RMSE`；阶段 7 未触发；
- 服务器产物根目录：
  `artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/`；
- 当前代码重评 weighted BC，test seeds `0–19`：
  default `7.4591 m / 0.65`，strict125 `9.7894 m / 0.45`，
  path-zero `11.6752 m / 0.05`；与历史数值一致；
- dev seeds `20–39` default：`6.8103 m / 0.70`，timeout `0.30`；
- 冻结目标 seed：`37,34,24,20,39,31,35`；前 6 个为 dev timeout，
  seed 35 为最高 RMSE 的 success 补足；
- 采集：7 × 4096 = 28,672 个新样本，观测契约一致，无 safety/NaN；
- 合并后：`421,888` samples、`330` episodes，新样本占比 `6.80%`；
- 新 BC checkpoint SHA256：
  `acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3`；
- test default：`6.6236 m / 0.70`，PASS default 门禁；
- test strict125：`10.5982 m / 0.25`，超过 `10.5 m`，FAIL；
- test path-zero：`12.8765 m / 0.00`，path-zero/default ratio `1.9440`；
- strict125 回退集中在 seed `11,15,16,18,19` 由 success 变 timeout；
- 未冻结新 BC，未进入残差 PPO，未启动正式 `3×50`；
- 当前没有 Goal19 teacher/BC/PPO/eval runner 在运行；
- 下一步：另开 Goal，先分析 default 与 strict125 的分布差异，再决定是否做
  action chunking 或覆盖 strict125 的定向补数据；Goal17 已完成，教师 `T1` 已选定，
  见本文件顶部。

## 2026-11-07 Goal18：sign/magnitude BC-only 完成，FAIL_RMSE

- 协议：`doc/goals/goal18_v0d_sign_magnitude_bc_prompt_20261107.md`
- 报告：`doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`
- S0 continuous：default `8.9129 m / 0.55`，strict125 `10.1828 m / 0.30`，
  path-zero `12.8576 m / 0.10`；相对同配方历史 BC `9.0944 m` 为 `0.9800`，
  S0 复现成立；
- S1 sign/magnitude：default `10.2868 m / 0.50`，strict125 `11.7268 m / 0.30`，
  path-zero `11.9473 m / 0.00`、safety `0.20`；
- default RMSE `10.2868 > 7.0`，strict125 `11.7268 > 10.5`，default success
  `0.50 < 0.60`；
- 动作诊断：sign accuracy `0.9501 / 0.9548` 通过，但饱和恢复率仅
  `0.2689 / 0.1524`，反转率 `0.6241 / 0.1859 次/秒`，rate-limit 命中率
  `71.70%`（教师 `38.69%`）；
- 判定 `FAIL_RMSE`，决策 `stop_goal18_bc_gate_failed`；未进入残差 PPO，
  未启动正式 `3×50`；
- 服务器与本地产物路径：
  `artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/`；
- 日志：`logs/pc_v0d_sign_magnitude_bc_20261107/`；
- 当前没有 Goal18 的 train/eval/PPO runner 在运行；
- 预注册门禁、seed、目标半径、actor 观测和动作接口均未改变。

下一动作：Goal18 停止。下一步另开 Goal，在 action chunking 与 mixture-density 中
只选一个先做；基于本轮反转/rate-limit 证据，建议先做 action chunking。Goal17
已启动，不再等待 Goal18。

## 2026-11-07 V-0D 饱和动作 BC 补救完成

- 协议：`doc/goals/goal9_v0d_teacher_r2b_revalidation_prompt_20261106.md`
- 报告：`doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`
- 阶段 1：4 个平滑教师全部 FAIL，`selected_variant=null`；
- 阶段 2：只训练一组 `w=2.0`、`threshold=0.9` 的 weighted BC；
- weighted BC default `7.4591 m / 0.65`，strict125 `9.7894 m / 0.45`，
  path-zero `11.6752 m / 0.05`；
- default RMSE `7.4591 > 7.0`，新 BC **FAIL**；未进入 PPO，
  未启动正式 `3×50`；
- 服务器与本地产物：
  `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/`；
- 日志：`logs/pc_v0d_r2b_saturation_bc_20261107/`；
- 当前没有教师、BC、PPO 或评估 runner 在运行；
- 预注册门禁、seed、目标半径和 actor 契约均未修改。

下一动作：本 Goal 停止。若继续，需要另开 Goal 研究动作表示
（sign/magnitude、mixture density、action chunking）或
actor 可观测性，不在本 Goal 内继续扫描权重或阈值。

## 2026-11-07 Goal16：crab-preserving 真限速教师完成

- 协议：`doc/goals/goal16_v0d_crab_rate_limited_teacher_prompt_20261107.md`
- 报告：`doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md`
- 修正：`guidance.command_rate_limit_enabled` 已与
  `saturation_governor_enabled` 解耦；旧配置缺省 `None`，行为不变；
- 教师：保留 `current_crab_enabled=true`、`compensate_current=false`，只测
  `rate035_018` 与 `rate020_012`；
- `rate035_018`：default `4.7499 m / 0.80`，rudder/stern sat
  `61.84%/25.48%`；strict125 `8.6849 m / 0.25`；
- `rate020_012`：default `4.9943 m / 0.80`，rudder/stern sat
  `58.97%/21.09%`；strict125 `9.0071 m / 0.30`；
- 两组都通过 RMSE/success/strict125/timeout/safety 门禁，但都未通过 default
  rudder `<=40%`、stern `<=20%` 门禁；
- `selected_variant=null`，决策为 `stop_goal16_teacher_gate_failed`；
- 未重采 262,144 transitions，未训练新 BC，未进入 PPO，未启动正式 `3×50`；
- 服务器与本地产物：
  `artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107/`；
- 日志：`logs/pc_v0d_r2b_rate_teacher_20261107/`；
- 当前没有教师、BC、PPO 或评估 runner 在运行；
- 预注册门禁、seed、目标半径、crab、current 契约和 actor 契约均未修改。

下一动作：Goal16 按预注册停止。若继续，应另开 Goal 研究 sign/magnitude、
mixture-density 动作头，或直接约束 actuator 幅值/切换频率的教师；
不再扫第三组 rate，也不把旧 `h20/gov/rate` 结果当作替代方案。

## 后续计划登记：Goal18（已完成，FAIL_RMSE）

- 计划文档：`doc/goals/goal18_v0d_sign_magnitude_bc_prompt_20261107.md`；
- 目的：只用现有 R2B 合并示范，做 BC-only 的 sign/magnitude 动作头对照；
- 对照：S0 连续 MSE、S1 sign/magnitude，数据、观测、trunk、seed、训练预算完全一致；
- 主门禁：S1 default 3D RMSE `<=7.0 m`、success `>=0.60`，并恢复 rudder/stern
  饱和动作；strict125、safety、path-zero 和反向/rate-limit 诊断同时受检；
- 不重采教师、不改 actor 观测、不混入 current/DADE/世界模型；
- 若 PASS，才冻结 BC 进入残差 PPO screen；若 FAIL，停止并另开 Goal 研究
  action chunking 或 mixture-density；
- 已完成并收口，结果见 `doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`；
- 当前没有 Goal18 的教师、BC、PPO 或评估进程在运行；
- Goal18 已停止；Goal17 已完成并停止，见本文件顶部。

## Goal17 执行登记（已完成，BC 门禁 FAIL）

- 计划文档：`doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`；
- 该计划登记 `actor current estimate × ALOS crab` 的 2×2 因子实验：
  `A00`（无 current、无 crab）、`A10`（有 current、无 crab）、
  `A01`（无 current、有 crab）、`A11`（两者都有）；
- current 不直接拼到 actor 最后一层，而是经独立 `E_cur` 编码后融入固定宽度的
  `z_t^a`；12D 仅是 `E_base` 的输入宽度，不是 actor 输入宽度；actor 顶层仍为
  `[z_t, c_t, c_hat_t]`，四格宽度不变；
- 主矩阵固定 `actor_command_dim=0`、`path_dim=8`、`compensate_current=false`，
  并关闭 DADE，避免 current 从 history side channel 混入；
- 该计划不是 Goal16 的一部分，不改变 Goal16 的限速教师、BC 配方、门禁或停止条件；
- Goal16 和 Goal18 均已收口；Goal17 已获用户确认并启动；
- 阶段 1 encoded-current 已实现；阶段 2 `T1` 已选定；阶段 3 两份示范已完成；
- 阶段 4b 已完成三个训练 seed；阶段 5 判定 `FAIL_BC_FACTOR_GATE`；阶段 6 未触发；
- 主比较使用 dev seeds `20–39`，最终 held-out `100–149` 只评估一次；
- 当前没有 Goal17 的教师、BC、PPO 或评估进程在运行；结果与停止决策见本文件顶部及
  `goal17_v0d_current_crab_factorial_report_20261107.md`。

## 指标优先级（2026-11-06 更新）

用户已明确：**3D RMSE 改善是当前最重要的指标**，success 不是首要排序指标，但仍作为
护栏。后续 P/V/R 阶段统一采用：

- 主指标：3D RMSE；
- 护栏指标：success、strict125、timeout、safety；
- 辅助指标：time-to-goal、速度平滑和执行器风险；
- 风险/诊断指标：material saturation；
- saturation 继续逐 seed 计算、展示和标记 `exceeds_threshold`，但不再参与通过/否决，
  也不再作为候选进入下一阶段的前置条件；
- 默认允许 success/strict125 的小幅波动，但不能超过绝对 `0.02`；timeout/safety 同样
  不得超过 `0.02` 的恶化。若需要在 RMSE 与 success 之间做更大交换，必须提前书面改
  门禁，不能看结果后调整。
- 该规则不改变已完成 A/B/C 的历史结论。

## 项目范围（2026-11-06 确认）

Goal9 的主线是**路径规划/路径跟踪**：在可部署观测和真实执行器约束下完成给定路径的
闭环跟踪；以 3D RMSE 为主指标，success、strict125、time-to-goal、timeout 和 safety
为护栏与辅助指标。
低层动作残差、历史编码、决策频率、DADE 和 ALOS-free 都是服务路径跟踪的组件实验，
不是独立研究主线。世界模型和 SuperMimic 属于 `goal11`，不进入 Goal9 的执行链。

V 系列最终契约（2026-11-06 用户确认）：**直接以 `V-0D` 纯路径输入为主臂，
`V-3D` 只作可选诊断控制。** 不再把 `V-3D → V-2D → V-1D → V-0D` 的命令逐级消融
作为必经主线。权威计划见 `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。

V-0D 的 actor 输入为 `base_state + 8D path`，命令 `c_t=[]`；路径只通过 path
encoder 融合进 `z_t`，不进入 encoder/DADE/history。当前速度 `u/v/w` 仍可留在基础
状态中，删除的是目标命令 `u_d`。旧 V0 的 `[u_d]` 完整策略不是新 V-0D 基线。

**技术边界：** 不能把现有 3D BC checkpoint 的 command 置零来冒充 V-0D，因为冻结
BC 的 `base_action()` 仍读取 command（`src/auv_lowlevel/batch/pc_policy.py:825`）。
真正的 V-0D 必须新增同契约的 0D BC：`state + 8D path -> action`，训练后冻结，再从
零初始化动作残差。当前 BC 侧仍是 3D 命令契约且示范数据没有 path 字段，需要非破坏性
扩展。

## 当前阶段

新 Goal 的单入口启动词：`doc/goals/goal9_run_all_pvabc_prompt_20261105.md`。本文件
仍是跨轮次状态账本；若与服务器 artifacts 冲突，以实际证据为准。

- S15 的 P0/P1/P2/P3、A screen、B10 smoke/screen、B5 smoke/screen 均已完成。
- A screen：`selected_arm=A0_stride1`、`selection_status=baseline_no_signal`。
  A2 虽有 default 正指标，但 `history-zero`/`history-reverse` 无可解释退化，只能算
  相关性，不是历史时间感受野的因果收益。
- B10 正式 screen：`status=completed`、`training_audit_passed=true`、summary
  `pass_non_inferior`；default success/strict 降幅 `0.0`、RMSE ratio `1.0014`；
  latency gate 通过，p95 `0.8907 ms`、compute ratio `0.1186`。B10 是效率候选，
  不是相对 A0 的性能提升。
- B5 正式 screen：`status=completed`、`training_audit_passed=true`、summary
  `fail_non_inferior`；default/strict125 success 均下降 `0.05`、RMSE ratio `1.1153`、
  timeout 增加 `0.05`、saturation 变化 `-0.0037`。5 Hz 降频分支停止，不重跑。
- 当前没有活跃的 A/B/C/V runner。C 已正式记录为 `C_not_triggered`。
- S15 已冻结：`baseline_arm=A0_stride1`、
  `selection_reason=baseline_no_signal_conservative_fallback`；B10 只登记为
  `efficiency_candidate`，不与 A0 混成两个同时使用的 V 基线。
- 旧 S16-V0 正式 screen 已失败，见 `doc/goals/goal9_v0_report.md`；旧
  V1/V1b/V2/V3 按历史协议阻断。V 代码、配置、runner、summarizer 和测试已通过
  targeted pytest/ruff。
- V-0D 主臂已完成 R-BC-0D/R-Z0 门禁、首轮 R-PPO screen、actor-LR 诊断和
  residual-scale 单变量诊断；三次 PPO 诊断均未通过主门禁，当前停在机制方案人工审核。

## P1 优化复核完成（2026-11-05）

- 协议：`doc/goals/goal9_p_optimization_plan.md`；
- 报告：`doc/goals/goal9_p_optimization_report.md`；
- 新产物根目录：
  `artifacts/lowlevel/pc_p1opt_20261105/`（服务器副本同路径）；
- X1 `alpha=1`：default success `+0.05`、RMSE `+0.4376 m`，但 `path-zero`
  success drop `0.00`，`no_path_attribution`；
- X2 `128 iterations`：success delta 均为 `0.00`、RMSE `-0.0246 m`，预算不是充分原因；
- X3 `pc_ppo_scale=1.0`：success delta 均为 `0.00`，奖励尺度不是主因；
- X4lr `actor_lr=1e-4`：default success `+0.10`、strict125 `+0.25`，但
  `path-zero` default success drop 仍为 `0.00`；actor 学习率是真瓶颈，路径归因仍失败；
- X4norm：与 X1 基本一致，逐 mini-batch advantage normalization 不是主因；
- 结论：P1 在当前 ALOS command 输入下仍是 `no_path_attribution`；不再扫 P1 预算、
  奖励尺度或门禁。actor LR `1e-4` 作为后续训练候选，但不得写成路径分支成功。

## V0 smoke 完成（2026-11-05）

- 根目录：`artifacts/lowlevel/pc_alos_free_v_20261105/smoke/V0_fixed_speed/`
- training audit：`passed=true`；actor layout 为 command `1`、critic command `1`、
  path `8`、history path `8`，DADE history 为 encoded，history 为 `10/50` stride `1`；
- evaluation：default/strict125/grid_hs 各 8 个 smoke seed 全部完成；
- summary：`status=smoke_contract_only`、`passed=true`；one-update 的科学预览为
  `fail_v0`，按协议不作为性能结论；
- **基线重评估：** 旧 A_screen summary 缺少 `time_to_goal_s` 和速度包线字段。已用同一
  冻结 checkpoint `A0_stride1_seed91430/final.pt`、同一 seeds/tags 和当前 evaluator
  重新评估到 `artifacts/lowlevel/pc_alos_free_v_20261105/baseline_reval/`；smoke 与
  screen 的 success/timeout/RMSE/saturation 均与旧 summary 完全一致，只新增缺失字段。
  没有重训、没有换 seed、没有重选 S15 基线。
- 下一步：V0 screen（32 iterations × 128 envs × 256 steps；评估 seeds `1000–1019`）。

## V0 pre-opt 诊断与修正（2026-11-05）

- 首次 V0 screen 使用 `path_encoder_alpha=0.01`、`actor_lr=1e-5`，结果三组全崩；
- 第 31 iteration 的 `path_encoder_effective_output_scale` 仅 `1.55e-4`，路径支路基本
  关闭；V0 没有 ALOS heading command，因此这是配置导致的 `preopt_path_closed`，不是
  ALOS-free 的科学负结果；
- 旧 smoke/screen 产物已移动到 `artifacts/lowlevel/pc_alos_free_v_20261105/preopt/`；
- 修正配置：V0/V1 统一 `path_encoder_alpha=1.0`、`learning_rate_actor=1e-4`，两者仍只差
  `u_d` 输入；runner 增加对应配置门禁；
- 详细记录：`doc/goals/goal9_v0_preopt_report.md`。

## V0 正式 screen 失败（2026-11-05）

- 修正配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json`
- 训练审计通过；第 31 iteration 路径有效输出尺度 `0.1380`，路径支路已打开；
- default success `0/20`、strict125 `0/20`、grid_hs `0/20`，三组全部 timeout；
- default success delta `-0.75`、strict125 `-0.35`、grid_hs `-0.90`；
- 3D RMSE ratio 分别为 `2.7928 / 1.6095 / 5.3778`；
- material saturation 只记录，不参与判定；即使忽略它，primary/timeout 仍远低于门禁；
- 结论：`V0=fail_v0`，停止 V 分支，保留 ALOS 外置方案；V1/V1b/V2/V3 不启动。
- 证据：`artifacts/lowlevel/pc_alos_free_v_20261105/V0_fixed_speed/`

## 架构修正：V0 不是残差实验（2026-11-06）

- V0 配置 `base_policy.mode=none`，actor 直接输出完整执行器动作，BC 只做 warm-start；
- 因此 V0 的 `60/60 timeout` 只能否定“完整策略 + ALOS-free”这一组合，不能否定“冻结 BC + 动作残差”；
- Goal9 S15 的 `A0_stride1/A1_stride5/A2_stride10/B10` 也使用 `base_policy.mode=none`，是 BC 热启动的完整策略，不是残差；
- 使用 `frozen_pc_bc_residual` 的是 PC-DADE grid 的 `frozen_bc_residual` 系列，以及 AUV-SM 的 B0/B2；同名 A0/A1/A2 不能跨工程直接比较；
- V1/V1b/V2/V3 的状态不变，仍为未启动；
- 后续若重做，先按 `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md` 执行
  `R-BC-0D → R-Z0-0D → R-PPO-0D`，不能直接沿用 V1/V2 的完整策略扩训。

## V-0D 主臂确认（2026-11-06）

- 用户确认：不再先走 `V-3D → V-2D → V-1D → V-0D`，直接以 `V-0D` 纯路径为主臂；
- `V-3D` 降级为可选诊断控制，只在 `R-PPO-0D` 失败或需要归因时启动；
- 原始速度轴仍保留：不额外加 speed head，先由 `pc_path_track` 奖励、当前速度和 RPM
  隐式学习速度；只有明确证据表明速度行为不可接受时才另开分支；
- 不能用现有 3D BC + `command_mask` 冒充 0D：冻结 BC 的 `base_action()` 仍读取
  command，会使 BC 基座同时失去方向命令；
- 正确路线：新增 `R-BC-0D`（`state + 8D path -> action`，`command_dim=0`）并冻结，
  然后 `R-Z0-0D`、`R-PPO-0D`；
- 权威计划：
  `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`；
- 当前没有启动新训练；旧 V0 的 `fail_v0` 和 artifacts 不改写。

## P2 完成证据

- stage decision：
  `artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/report/stage_decision.json`
- summary：
  `artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/report/P2b_vs_A0/summary.{json,md}`
- run manifest：
  `artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/run_manifest.json`
- P2b checkpoint：
  `artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/P2b_encoded_screen/training/P2b_encoded_seed91430/final.pt`

## A screen 完成证据

- run manifest：
  `artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/run_manifest.json`
  （`status=completed`、`wall_time_s=705.22`、`training_audit_passed=true`）
- summary：
  `artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/report/summary.{json,md}`
- 结论：`selected_arm=A0_stride1`、`selection_status=baseline_no_signal`；
  `A2_stride10` 的 `pass_without_history_attribution` 只表示 default 相关性，不表示
  历史时间感受野的因果收益。

## B10 完成证据

- 根目录：`artifacts/lowlevel/pc_abc_temporal_20261105/B10/`
- run manifest：`status=completed`、`training_audit_passed=true`、wall `298.84 s`
- summary：`pass_non_inferior`
- default：success/strict delta `0.0`、RMSE ratio `1.0014`、timeout delta `0.0`、
  saturation delta `0.0009`、safety delta `0.0`
- strict125：success/strict delta `+0.05`、RMSE ratio `0.9532`；grid_hs：
  success/strict delta `+0.05`、RMSE ratio `0.8296`
- latency：`gate_passed=true`，decision period `100 ms`，p95 `0.8907 ms`，
  compute ratio `0.1186`

## B5 完成证据

- 根目录：`artifacts/lowlevel/pc_abc_temporal_20261105/B5/`
- run manifest：`status=completed`、`training_audit_passed=true`、wall `297.998 s`
- summary：`fail_non_inferior`
- default：success/strict delta `-0.05`、RMSE ratio `1.1153`、timeout `+0.05`、
  saturation `-0.0037`、safety `0.0`
- strict125：success/strict delta `-0.05`、RMSE ratio `1.0548`、timeout `+0.05`
- 结论：按协议停止 5 Hz 分支；不重跑、不增加训练步数、不换 seed、不改门禁。
- runner PID `2688627` 和 training PID `2688692` 已退出；服务器当前没有 A/B/C/V
  训练或评估进程。

## 最终完成节点

- P0：完成，复用基线 `A0_stride1`。
- P1：正式 screen 完成，`no_path_attribution`，路径分支停止；不重跑。
- P1_curriculum：门禁主动跳过，不是训练失败。
- P2b smoke：完成，链路通过，`no_signal`。
- P2 screen：完成，`no_signal`；保留 `A0_stride1`，停止 DADE-z 重构。
- P3：已正式记录 `not_applicable`。
- A smoke：完成，训练审计通过；1-iteration smoke 不作为正式科学结论。
- A screen：完成；`A0_stride1` 为无因果信号基线，A2 只有相关性。
- B10 smoke/screen：完成；screen 为 `pass_non_inferior` + latency gate 通过。
- B5 smoke/screen：完成；screen 为 `fail_non_inferior`，5 Hz 分支停止。
- C：正式判定 `C_not_triggered`。
- S15：已冻结 `A0_stride1`。
- V0：正式 screen `fail_v0`，V 分支停止。
- 旧 V1/V1b/V2/V3：未启动；不满足历史前置条件。
- V-0D 主臂：R-BC-0D/R-Z0 门禁完成；R-PPO screen、actor-LR、residual-scale 诊断均
  完成且主门禁失败；未进入正式 `3 training seeds × 50 eval seeds`。

## 下一动作

1. 最终报告已写入 `doc/goals/goal9_final_report.md`。
2. 不重跑 P1/V0，不换 seed，不增加训练预算，不降低门禁。
3. V-0D 当前已完成 R-BC-0D/R-Z0 门禁和三次 PPO 单变量诊断；首轮 screen、
   actor-LR、residual-scale 均未通过 `PPO / BC-0D 3D RMSE <= 0.95` 主门禁。
4. 正式 `3 training seeds × 50 eval seeds` 未启动；下一步必须由人工审核新的机制性
   假设，不能靠继续增加预算、扫 LR/residual scale 或降低门禁推进。
5. 本次旧 V0 的 `fail_v0` 结论保持不变；不得把旧 V0 的 `[u_d]` 完整策略结果写成
   V-0D 残差结果。

## V-0D R-Z0 门禁与 R-PPO screen（2026-11-06）

- R-Z0 首次评估暴露冻结 BC path encoder alpha 未从 checkpoint 恢复；已做非破坏性修复，
  保留服务器原文件备份 `.server_backup/v0d_alpha_fix_20261106/`；
- 服务器 targeted suite：`55 passed`；
- `R-Z0-0D` 20-seed default 完成：
  `artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-Z0-0D/default/`；
- `max |residual_action| = 0`、`max |requested_action - base_action| = 0`；
- 冻结 BC 18 个 state entry 最大漂移为 `0`；
- 同一 checkpoint 的 1-seed `policy` / `frozen-base` 对照 summary 指标与全部 trace
  数组逐位相同；
- 新增默认关闭的 `evaluate_pc_ppo.py --action-mode frozen-base`，用于严格同 harness
  BC 基线；默认 `policy` 行为不变；
- 同 harness BC-0D path-zero 完成：
  `artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-BC-0D/path_zero_same_harness/`；
- path-zero 3D RMSE 从 `12.2837 m` 恶化到 `20.7088 m`（`+68.59%`），
  success `0.20 -> 0.00`，17/20 seed 恶化；
- 门禁结论：通过；证据汇总见
  `doc/goals/goal9_v0d_rz0_gate_20261106.md`；
- `R-PPO-0D-screen` 已启动：server GPU4，`4096 envs × 256 steps × 40 iterations`，
  训练 seed `91430`，最终 checkpoint 预先固定，不用 screen seed 选模型；
- screen 输出：`artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/`；
- screen 日志：`logs/pc_v0d_20261106/train_R-PPO-0D-screen.log`；
- 当前没有修改旧 V0、旧 BC 或历史 artifacts。

### R-PPO-0D screen 完成结果

- 训练完成：`4096 envs × 256 steps × 40 iterations = 41,943,040` 环境步；
- wall time `2164.21 s`，所有 loss 有限，冻结 BC 最大漂移 `0`；
- final checkpoint SHA256：
  `eb1fad7fd5791cbdaae6c3d42e2ff05edbd1837e3fcfc71c0a933320a7efe661`；
- 20-seed default：3D RMSE `12.2888 m`，success `0.20`，timeout `0.80`；
- 同 harness frozen-base BC：3D RMSE `12.2837 m`，success `0.20`，timeout `0.80`；
- `PPO / BC-0D` 3D RMSE ratio `1.0004`，主门禁目标 `<= 0.95`，未通过；
- 逐 seed 平均差 `+0.00659 m`，95% CI `[-0.00274, +0.01592] m`，8 改善 / 12 变差；
- 残差 L2 / 基座 L2 均值 `0.007756`，最大残差 `0.002529`，修正量过小；
- path-zero 使 3D RMSE 恶化 `+68.47%`，路径归因仍然通过；
- success/timeout/safety 均未退化，但主目标没有正收益；
- 预注册结论：screen 失败，不进入正式 `3 training seeds × 50 evaluation seeds`；
- 报告：`doc/goals/goal9_v0d_rppo_screen_20261106.md`；
- 若继续，只允许做一次预注册单变量诊断：actor learning rate `1e-4 -> 3e-4`，
  其余训练/评估口径不变；不能把本次和诊断结果混成一个正式比较表。

### R-PPO-0D actor-LR 单变量诊断完成结果

- 预注册与结果：
  `doc/goals/goal9_v0d_rppo_alr3e4_diagnostic_20261106.md`；
- 新配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64_alr3e-4.json`
  （SHA256 `223609c9fe9153fc27576286ab58ada774731d81ffa1f4a448965e307e751224`）；
- 训练：server GPU4，`4096 × 256 × 40 = 41,943,040` 环境步，seed `91430`；
- 训练审计：`completed`，所有 loss 有限，冻结 BC 最大漂移 `0.0`，初始残差 `0.0`；
- wall time `1807.29 s`；最终 checkpoint SHA256：
  `4ca14cbf34d640cc018a2dc202d9b8b31d1ac5865270abfd900e5b38a3f289e2`；
- 20-seed default：3D RMSE `12.2713 m`，success `0.20`，timeout `0.80`；
- 同 harness frozen-base BC：3D RMSE `12.2837 m`，success `0.20`，timeout `0.80`；
- `PPO / BC-0D` 3D RMSE ratio `0.99899`，主门禁 `<= 0.95` 未通过；
- 逐 episode 配对差均值 `-0.00666 m`，95% CI `[-0.02526, +0.01194] m`，
  `11/20` 改善、`9/20` 变差；
- 残差 / 基座动作 L2 从 `0.007756` 升到 `0.011832`，平均绝对残差约翻倍，
  但闭环收益仍不可测；
- path-zero 使 3D RMSE 恶化 `+68.34%`，`17/20` seed 恶化，路径归因通过；
- success/timeout/safety 均未退化；该评估器仍不输出 `strict125`，明确记为不可用；
- 结论：`fail_alr3e4`；停止 actor-LR 分支，不扫 LR、不增加预算、不换 seed、
  不进入正式 3×50；下一步需人工审核新的机制性假设。
- 产物：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4/`；
- 日志：
  `logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4.log`、
  `logs/pc_v0d_20261106/eval_R-PPO-0D-screen-alr3e-4_*.log`。

### R-PPO-0D residual-scale 单变量诊断完成结果

- 预注册与结果：
  `doc/goals/goal9_v0d_rppo_scale1p0_diagnostic_20261106.md`；
- 唯一科学变量：`base_policy.residual_scale: 0.25 -> 1.0`；其余训练和评估配置不变；
- 配置 SHA256：
  `7ec88c7d39431ee77311ce3764dd80b1d385d525402cb3238936895ee8f55b42`；
- 执行硬件：GPU5（启动发现 GPU4 被非本实验任务占用，已在预注册文档记录为基础设施
  变更）；同 harness frozen-base BC 仍为 `12.28370678 m`，与 GPU4 结果一致；
- 训练审计：`completed`，`4096 × 256 × 40 = 41,943,040` 环境步，seed `91430`，
  全部 loss 有限，初始残差 `0.0`，冻结 BC 最大漂移 `0.0`；
- 训练 wall time `1257.65 s`；最终 checkpoint SHA256：
  `a382843b2e22df3f98390758514c385f6edbec73a1296795aae3b4b6ead804b8`；
- 20-seed default：3D RMSE `12.2098 m`，success `0.20`，timeout `0.80`，safety `0`；
- 同 harness frozen-base BC：3D RMSE `12.2837 m`；
- `PPO / BC-0D` 3D RMSE ratio `0.99399`，主门禁 `<= 0.95` 未通过；
- 逐 seed 配对差均值 `-0.061574 m`，95% CI `[-0.086648, -0.036500] m`，
  `17/20` 改善、`3/20` 变差；
- 残差 / 基座动作 L2 从 `0.011832` 升到 `0.018307`，平均绝对残差 `0.002949`，
  最大绝对残差 `0.005358`；动作权限确实增大，但闭环收益只有约 `0.50%`；
- path-zero 使 3D RMSE 恶化 `+68.43%`，路径归因通过；该组按设计出现
  success `0.20 -> 0.00`、timeout `0.80 -> 1.00`，不作为主策略护栏回归；
- `strict125` 当前 evaluator 不输出，明确记为不可用；
- 结论：`fail_scale1p0`；停止 residual-scale 分支，不扫 `0.5/0.75`、不加预算、
  不换 seed、不改门禁、不进入正式 3×50。
- 产物：
  `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen-alr3e-4-scale1p0/`；
- 日志：
  `logs/pc_v0d_20261106/train_R-PPO-0D-screen-alr3e-4-scale1p0.log`、
  `logs/pc_v0d_20261106/eval_R-PPO-0D-screen-alr3e-4-scale1p0_*.log`。

### 监督残差初始化 smoke 完成结果（2026-11-06）

- 预注册：
  `doc/goals/goal9_v0d_supervised_residual_diagnostic_20261106.md`；
- 第一次 smoke 因 path-conditioned history token 缺路径块而失败，旧目录和日志保留；
- 修复 history token 与 encoded-DADE 测试夹具后，本地和服务器 targeted suite 均为
  `18 passed`；
- `R-SR-0D-smoke-v2` 在 GPU5 完成 1 epoch，manifest 为 `completed`；
- 固定布局核验通过：`path_dim=8`、`history_path_dim=8`、`command_dim=0`、
  history token `23`；short/long `[262144, 10, 23]` / `[262144, 50, 23]`；
- 初始残差最大绝对值 `0.0`，冻结 BC 漂移 `0.0`，路径历史特征非零；
- 进入正式 `R-SR-0D-scale1p0` 监督训练；样本数 `262,144`，episode `144`，
  train/validation 按 episode 划分为 `115/29`；
- smoke 仅证明链路可运行，后续仍需完成预注册的 20-seed 三组闭环评估和门禁。
- 正式 v1 的训练完成了 48 epoch（best epoch 17），但评估器因其 `pc.path` 元数据
  缺少 `dim/blocks` 结构而拒绝加载；v1 原样保留，判定为基础设施失败；
- 修复 pretrain 的 path checkpoint 元数据，并用同一配置、seed、数据和 GPU 重跑为
  `R-SR-0D-scale1p0-v2`；重跑前后仍需 smoke 和元数据解析门禁。

### R-SR-0D 监督残差初始化完成结果

- 预注册：
  `doc/goals/goal9_v0d_supervised_residual_diagnostic_20261106.md`；
- 结果报告：`doc/goals/goal9_v0d_supervised_residual_result_20261106.md`；
- v1 因 `pc.path.dim/blocks` 元数据缺失被评估器拒绝，原样保留，不参与最终比较；
- v2 在 GPU5 完成 47 epoch（best epoch 16），loss 全有限，冻结 BC 漂移 `0.0`；
- final checkpoint SHA256：
  `498bb683b30ad6035fa95ef893bfd5df57bed32adccba8c0882274cfbbc4d349`；
- 20-seed default：3D RMSE `12.4160 m`，success `0.15`，timeout `0.85`；
- 同 checkpoint frozen-base BC：3D RMSE `12.2633 m`，success `0.15`，timeout `0.85`；
- `SR / BC-0D` ratio `1.012448`，主门禁 `<= 0.95` 失败；
- `SR / R-BC-3D` ratio `0.986348`，护栏通过；
- path-zero：3D RMSE `21.2390 m`，相对 default 恶化 `+71.06%`，路径归因通过；
- success/timeout/safety 相对 BC-0D 变化均为 `0.0`；
- 逐 episode default vs frozen-base 配对差 `+0.151831 m`，95% t 区间
  `[+0.030777, +0.272884]`，`6/20` 改善、`14/20` 恶化；
- 结论：`fail_sr0d`；停止监督残差分支，不 warm-start PPO，不进入正式 3×50，
  不再扫监督学习率/epoch/output L2。

### V-0D strict125 护栏对照完成结果（2026-11-06）

- 预注册：`doc/goals/goal9_v0d_strict125_guardrail_20261106.md`；
- 结果：`doc/goals/goal9_v0d_strict125_guardrail_result_20261106.md`；
- 环境配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_strict125.json`
  （SHA256 `c47536b7025f60adddedc323050f8f8be58dbf31b566655a8e96c2c087301b15`）；
- 固定 seed `0–19`、CPU float64、统一 `4` 线程；未重新训练、未换 checkpoint；
- 六组评估全部完成：
  `R-BC-3D/default`、`R-BC-0D/default`、`R-BC-0D/path_zero`、
  `R-PPO-0D-screen/default`、`.../frozen_base`、`.../path_zero`；
- `R-BC-3D` 3D RMSE `12.9402 m`，success `0.10`，timeout `0.90`；
- `R-BC-0D` 3D RMSE `12.6950 m`，success `0.05`，timeout `0.95`；
- `R-PPO/frozen_base` 3D RMSE `12.7011 m`，`R-PPO/default` `12.7065 m`；
- 同 harness `PPO / frozen-base` ratio `1.000424`，95% CI
  `[-0.003166, +0.013721] m`，`7/20` 改善、`13/20` 变差；
- 跨 harness `PPO / R-BC-0D` ratio `1.000906`，方向一致；
- 相对 `R-BC-3D` ratio `0.981938`，满足 `<= 1.05` 护栏，但不是 PPO 残差收益；
- success/timeout/safety 相对同 harness 冻结基座均为 `0.00` 变化；
- path-zero 使 PPO 3D RMSE 恶化 `+66.86%`，BC-0D 恶化 `+67.19%`，路径归因通过；
- PPO 残差 L2 / 基座 L2 均值 `0.007578`，最大绝对残差 `0.002540`，仍过小；
- strict125 下 success 只有 `0.05`，存在 floor effect；主判定仍以 3D RMSE 为准；
- 结论：`fail_strict125`；主门禁 `<= 0.95` 失败，正式 `3×50` 继续停止；
- 产物：
  `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/`；
- 日志：`logs/pc_v0d_20261106_strict125/`。

### Goal9 V-0D 收口（2026-11-06）

- 收口报告：`doc/goals/goal9_v0d_closure_20261106.md`；
- V0D-0/1/2/3 已完成并通过；V0D-4 screen、actor-LR、residual-scale 均完成但主门禁失败；
- 监督残差 `R-SR-0D` 完成但失败；strict125 六组对照完成但主门禁失败；
- `R-BC-0D` 在 default 和 strict125 下都不差于 `R-BC-3D`，命令删除不是当前首要瓶颈；
- 可选 `V-3D` 判定跳过：现有信息不支持为此新增 BC-3D+path 契约；
- 正式 `3 training seeds × 50 evaluation seeds` 按预注册未启动，CPU PVS 也未启动；
- 服务器 targeted suite `90 passed`，`ruff check` 通过；
- strict125 六组 summary 均为 seed `0–19`、20 episodes、指标有限；
- checkpoint SHA、strict125 配置 SHA 与预注册一致；没有残留评估进程；
- 结论：V-0D 既定实验轴已收口为科学负结果，不通过加预算、扫参、换 seed 或降门禁改写；
- 后续若继续研究，必须另开 goal 预注册新的机制性假设。

### Goal9 V-0D + R2B 教师复验与 BC 重采（2026-11-06）

- 执行记录：`doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`；
- 技术修正：V-0D 的 `path_learned_speed` 不读取 ALOS guidance；
  教师配置必须使用 `pc_control_mode=alos` + `reward=pc_command_track`，
  actor 契约仍在 BC/PPO 层保持 `command_dim=0`、`path_dim=8`；
- 教师 20-seed：R2B default `4.3531 m / 0.85`，strict125 `9.3916 m / 0.15`；
  R0 default `11.9876 m / 0.15`，strict125 `12.7146 m / 0.05`；
- 归因修正：`NO_TERMINAL` 与 R2B 完全一致，terminal 开关当前参数下无效；
  `ALOS_PLAIN` `4.8663 m / 0.70`，主要收益来自 ALOS 制导本身；
  crab/floor 为增量，不能把收益归因于 terminal；
- BC 新数据：262,144 transitions + 一轮 DAgger 131,072，合并 393,216；
- 新 BC default `9.0944 m / 0.50`，DAgger BC default `8.9129 m / 0.55`；
  old BC default `12.2396 m / 0.15`；default 门禁失败；
- strict125：新 BC `9.9476 m / 0.40`，DAgger `10.1828 m / 0.30`，
  均通过 strict125 门禁，但 default 主门禁失败；
- path-zero 明显恶化，路径输入有效；
- 失败诊断：教师 rudder/stern 饱和率 `55.3%/19.9%`，BC 仅
  `21.5%/8.3%`（DAgger `27.5%/9.1%`）；MSE 行为克隆平滑了 bang-bang 动作；
- 未进入残差 PPO，未启动正式 `3×50`；一轮 DAgger 补救已用尽，
  按预注册停止，不换 seed、不加预算、不改门禁；
- 产物：`artifacts/lowlevel/pc_v0d_teacher_r2b_revalidation_20261106/`、
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/`；
  日志：`logs/pc_v0d_teacher_r2b_revalidation_20261106/`、
  `logs/pc_v0d_r2b_bc_20261106/`。

## 恢复与安全规则

- 同一 run 仍活跃时只等待/接管，禁止重启或覆盖产物。
- screen 若因基础设施失败退出，可最多重试两次，保持 mode、seed、预算、tag、ablation 不变。
- 科学负结果不重跑，不换 seed、不改门禁、不加训练步数、不调 reward 权重。
- 不自动 commit/push，不使用 `git add -A`，不提交 artifacts、checkpoint 或日志。
