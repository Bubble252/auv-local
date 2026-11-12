# 速度×海流网格泛化实验

## 背景

六环境等权混合残差在 H1/H2/H3 的分布内拟合明显变好，但 H1/H2/H3 都参与了训练，
而且执行器饱和接近上限；strict125 和 C4 也仍有退化。因此，这部分结果只能说明
“混合分布内拟合有效”，不能说明跨速度-海流组合的过拟合已经解决。

现有 mixed-domain 筛选已经把 C4 移出训练，并留出了 H0/H1，但训练域仍由历史配置
拼接而来：低速格和高速格的命令速度区间、海流强度、时变方式并不完全同口径。
本实验补齐一个显式的 `2×2` 速度-海流网格，避免把“配置来源不同”误解释成
“速度-海流解耦能力”。

## 实验目标

回答一个明确问题：冻结 BC + 残差策略只在三格训练时，能否在高速度+强海流这一
未参与训练的格子上保持成功率、RMSE、跟踪误差和执行器饱和不退化。

主对比实验不使用 DAgger。DAgger 是后续候选，只在本网格仍失败时再启动，且必须
复用同一个扩大域，不能用 DAgger 代替缺失的速度-海流格子。

## 网格设计

| 格子 | 命令速度 | 海流强度 | 用途 |
|---|---:|---:|---|
| `grid_low_weak` | `1.0–1.4 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_low_strong` | `1.0–1.4 m/s` | `0.30–0.45 m/s` | 训练 |
| `grid_high_weak` | `1.8–2.0 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_high_strong` | `1.8–2.0 m/s` | `0.30–0.45 m/s` | **留出，零样本评测** |

四格使用相同的生成器、传感器 profile、归一化、执行器包线和海流时变参数。海流为
`uniform_horizontal`，速度时变幅度 `±0.05 m/s`、方向时变幅度 `±0.15 rad`、
周期 `8 s`。高强格最坏海流上界为 `0.50 m/s`，与 H1 的标定上界一致；命令速度
上界 `2.0 m/s` 加顶流后仍位于 PC20 静态可达包线内。

C4 继续只作压力测试；H4 继续只作包线压力测试。两者都不参与残差训练，也不进入
主门禁。

## 技术栈

- 仿真与教师：`auv_lowlevel` 的 PC20 batch Torch 环境、PD/PI teacher；
- 残差策略：冻结 BC actor + no-DADE 残差 head；
- 训练入口：`scripts/pretrain_pc_residual.py`；
- 闭环评测：`scripts/evaluate_pc_ppo.py` 与
  `scripts/eval_pc_server_basics_20261103.sh`；
- 汇总与门禁：`scripts/summarize_pc_server_evals_basics_20261103.py` 和新网格报告脚本；
- 固定评测：训练 seed `94030–94034`，每格 `32` 个闭环 seed。

## 产物

- 四格环境配置：`configs/lowlevel/odarrl_control_pc_grid_*.json`；
- 训练配置：`configs/lowlevel/odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_lr3e-4_scale1p0_*.json`；
- 教师数据：`artifacts/lowlevel/pc_speed_current_grid_teacher_demos_20261104/`；
- 训练结果：`artifacts/lowlevel/pc_speed_current_grid_supervised_residual_20261104/`；
- 闭环评测：`artifacts/lowlevel/pc_speed_current_grid_eval_20261104/`；
- 汇总与报告：`artifacts/lowlevel/pc_speed_current_grid_{summary,report}_20261104/`；
- 队列日志：`logs/pc_speed_current_grid_20261104/`。

## 执行计划

- [x] 生成四格环境配置，并用 CPU smoke 检查环境可构造；
- [x] 收集四个格子的 teacher demos，其中高强格只留作诊断，不进入 merge；
- [x] 合并低弱、低强、高弱三格 demos；
- [x] 五个训练 seed 并行训练冻结 BC + no-DADE 残差；
- [x] 审计：只允许三格训练配置，冻结 BC 漂移为零，高强格未泄漏；
- [x] 评测网格臂 `initial/final` 在四格、default、strict125、C4、H4 上的闭环表现；
- [x] 在相同四格上评测 multienv 与 mixed-domain 两个对照臂的 final checkpoint；
- [x] 生成等 seed 汇总与门禁报告；
- [x] 高强留出格未通过后，基于同一扩大域完成并评测一轮 DAgger。

## 门禁

训练格：相对同一残差臂的 pretrain `initial`，final 的成功率下降不超过 `0.02`，
三维 RMSE 不超过 `1.10×`，material saturation 增量不超过 `0.05`。

高强留出格：除满足上述不退化条件外，还必须相对 `initial` 在成功率或 RMSE 上至少
有一项改善；并相对 multienv、mixed-domain 两个 final 对照，成功率下降不超过
`0.05`、RMSE 不超过 `1.25×`、saturation 增量不超过 `0.05`。

default/strict125：相对 multienv final，default 成功率下降不超过 `0.05`、
RMSE 不超过 `1.10×`；strict125 成功率下降不超过 `0.02`、RMSE 不超过 `1.05×`。

C4/H4 只报告，不作为主门禁。门禁未通过时保留完整负结果，不通过延长训练或反向
挑选评估 seed 掩盖。

## DAgger 单轮设计（预注册）

网格门禁失败后只做一轮 DAgger，目标是检验“学生访问到的高强/边界状态是否缺少
教师标签”，而不是继续调监督配方。该轮复用同一个三格训练域，仍严格不使用
`grid_high_strong`。

- 学生：网格 no-DADE 残差每个训练 seed 的 `final.pt`，即 seed `94030–94034` 各一份；
- 采样：在 `grid_low_weak`、`grid_low_strong`、`grid_high_weak` 三格分别跑
  `16 env × 2048 step` 的确定性学生闭环；不加入随机动作噪声，避免把探索噪声和
  DAgger 标签效应混在一起；
- 标签：学生访问的每个状态由冻结 PD/PI teacher 在线计算 `[rudder, stern, rpm]`，
  转成 PC 归一化动作；teacher 的积分状态随 episode reset 清零；
- 数据：每个 seed 把原三格 teacher demos 与新增 DAgger 数据合并，episode id 由
  merge 脚本重映射；然后从头重训同一残差结构，不热启动旧残差，避免把两轮训练
  混成一个不可归因的变量；
- 审计：五个 seed 必须完成，训练 config 仍只含三格，`grid_high_strong` 零样本
  泄漏即失败，冻结 BC 漂移必须为零；
- 评测与门禁：完全相同于本页主门禁，使用相同 `32` 个闭环 seed、相同
  default/strict125 保留、相同 multienv/mixed-domain 对照；C4/H4 仍只作压力报告；
- 停止条件：只做这一轮。无论通过与否都保留结果；若仍失败，不再续做同配方 DAgger，
  转而诊断域覆盖、执行器权限、教师质量或观测通路。

## 结果（2026-11-04）

五个训练 seed `94030–94034` 全部完成，审计无失败：训练配置严格为
`pc_grid_low_weak`、`pc_grid_low_strong`、`pc_grid_high_weak`，高强格
`pc_grid_high_strong` 没有进入训练，五个 run 的冻结 BC 参数漂移均为 `0.0`。
每格使用固定的 32 个闭环评估 seed。

| Tag | Role | Initial success | Final success | Initial RMSE | Final RMSE | Final saturation | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `grid_lw` | in-domain | `1.0000` | `0.9750` | `1.87 m` | `2.28 m` | `0.1833` | **fail** |
| `grid_ls` | in-domain | `0.6250` | `0.6500` | `5.01 m` | `4.88 m` | `0.1437` | pass |
| `grid_hw` | in-domain | `0.9688` | `0.9812` | `4.34 m` | `2.95 m` | `0.4779` | pass |
| `grid_hs` | held-out | `0.8438` | `0.8313` | `6.05 m` | `6.54 m` | `0.4190` | **fail** |
| `default` | retention | `0.6875` | `0.6062` | `4.93 m` | `5.54 m` | `0.1339` | **fail** |
| `strict125` | retention | `0.4688` | `0.4250` | `6.52 m` | `6.56 m` | `0.1253` | pass |
| `c4` | pressure-only | `0.2500` | `0.2062` | `8.49 m` | `8.84 m` | `0.2335` | pressure-only |
| `h4` | pressure-only | `0.4688` | `0.9375` | `18.88 m` | `5.86 m` | `0.4749` | pressure-only |

总体门禁为 **fail**。高强留出格相对 pretrain 没有改善：成功率下降 `0.0125`，
RMSE ratio `1.081×`。相对 multienv 对照，成功率低 `0.04375`、RMSE ratio
`1.271×`；相对 mixed-domain 对照，成功率低 `0.05625`。
因此这组三格训练没有建立可用的高速度+强海流零样本迁移。

训练格也不是全通过：低弱格的 final 比 initial 成功率低 `0.025`，三维 RMSE
从 `1.87 m` 升到 `2.28 m`；高弱格虽然通过，但 saturation 已达到 `0.4779`，
接近 `0.5` 的执行器上限。default 相对 multienv final 的成功率低 `0.0875`、
RMSE ratio `1.135×`，同样失败。strict125 通过，但成功率也只有 `0.4250`。

C4 继续没有改善；H4 的分布内压力结果大幅改善只说明残差在包线边缘仍可工作，
不能替代高强留出证据。

报告与证据：

- 汇总：`artifacts/lowlevel/pc_speed_current_grid_summary_20261104/summary.{json,md}`；
- 门禁报告：`artifacts/lowlevel/pc_speed_current_grid_report_20261104/report.{json,md}`；
- 训练审计：`artifacts/lowlevel/pc_speed_current_grid_supervised_residual_20261104/audit.json`；
- 队列日志：`logs/pc_speed_current_grid_20261104/queue.log`。

## DAgger 单轮结果（2026-11-04）

按预注册完成一轮 DAgger：五个学生 seed 分别采集 `98,304` 个新样本，共
`103–108` 个 episode；teacher 修正 L2 均值为 `0.2251–0.2515`，没有 safety
termination。把新数据与原三格 teacher demos 合并后，每个 seed 从头重训，训练集为
`294,912` 个样本。审计确认训练配置仍严格只有 `pc_grid_low_weak`、
`pc_grid_low_strong`、`pc_grid_high_weak`，高强格没有泄漏，五个 run 的冻结 BC
漂移均为 `0.0`。每格使用相同的 `32` 个闭环评估 seed。

下表同时保留 no-DAGGER 的 final，便于区分 DAgger 的边际作用；“Gate”仍按
DAgger final 相对 pretrain initial 的预注册门禁计算。

| Tag | Role | Initial success | No-DAGGER final | DAgger final | Initial RMSE | No-DAGGER RMSE | DAgger RMSE | DAgger saturation | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `grid_lw` | in-domain | `1.0000` | `0.9750` | `0.9750` | `1.87 m` | `2.28 m` | `2.33 m` | `0.1888` | **fail** |
| `grid_ls` | in-domain | `0.6250` | `0.6500` | `0.6500` | `5.01 m` | `4.88 m` | `4.88 m` | `0.1481` | pass |
| `grid_hw` | in-domain | `0.9688` | `0.9812` | `0.9875` | `4.34 m` | `2.95 m` | `2.56 m` | `0.4879` | **fail** |
| `grid_hs` | held-out | `0.8438` | `0.8313` | `0.8500` | `6.05 m` | `6.54 m` | `5.86 m` | `0.4470` | **fail** |
| `default` | retention | `0.6875` | `0.6062` | `0.6062` | `4.93 m` | `5.54 m` | `5.47 m` | `0.1384` | **fail** |
| `strict125` | retention | `0.4688` | `0.4250` | `0.4313` | `6.52 m` | `6.56 m` | `6.60 m` | `0.1298` | pass |
| `c4` | pressure-only | `0.2500` | `0.2062` | `0.2062` | `8.49 m` | `8.84 m` | `8.82 m` | `0.2336` | pressure-only |
| `h4` | pressure-only | `0.4688` | `0.9375` | `0.9688` | `18.88 m` | `5.86 m` | `4.17 m` | `0.5015` | pressure-only |

总体门禁仍为 **fail**。DAgger 相对 no-DAGGER 在高强留出格上有方向一致的边际
收益：成功率 `0.8313 → 0.8500`（`+0.01875`），三维 RMSE `6.54 → 5.86 m`
（`-10.4%`）；但相对 pretrain initial，成功率只提高 `0.00625`、RMSE 只改善
`3.1%`，没有达到预注册的 `+0.02` 成功率或 `0.95×` RMSE 阈值。因此这只能算
“DAgger 有弱正信号”，不能算 held-out 过拟合已解决。

其余门禁也没有被 DAgger 修复：`grid_lw` 的成功率下降 `0.025`，RMSE 为 initial
的 `1.244×`；`grid_hw` 的成功率和 RMSE 改善，但 saturation 增量为 `+0.052`，
超过 `0.05` 上限；default 仍相对 multienv 下降 `0.0875`、RMSE 为 `1.12×`。
所有格的 saturation 都比 no-DAGGER 上升，`h4` 压力测试达到 `0.5015`，说明
DAgger 增加的是学生分布上的教师标签覆盖，并没有解除执行器权限瓶颈。C4 没有
改善（`0.20625` 成功率、`8.82 m` RMSE），strict125 仍只有 `0.4313`。

结论：单轮 DAgger 值得保留为正信号，但没有通过门禁。按预注册停止条件，不再续做
同配方 DAgger；下一步应诊断域覆盖、教师质量、执行器权限或观测/残差通路，而不是
继续堆叠训练轮次。

DAgger 证据：

- 汇总：`artifacts/lowlevel/pc_grid_dagger_summary_20261104/summary.{json,md}`；
- 门禁报告：`artifacts/lowlevel/pc_grid_dagger_report_20261104/report.{json,md}`；
- 训练审计：`artifacts/lowlevel/pc_grid_dagger_supervised_residual_20261104/audit.json`；
- DAgger 数据清单：`artifacts/lowlevel/pc_grid_dagger_demos_20261104/seed*/{new,combined}/demos.manifest.json`。

## 运行与 Git

启动队列：

```bash
bash scripts/run_pc_speed_current_grid_20261104.sh
bash scripts/run_pc_grid_dagger_20261104.sh
```

查看进度：

```bash
tail -f logs/pc_speed_current_grid_20261104/queue.log
tail -f logs/pc_grid_dagger_20261104/queue.log
```

本地验证：

```bash
../.venv/bin/pytest -q tests/test_pc_speed_current_grid_report.py \
  tests/test_pc_server_eval_summary.py tests/test_m4_bc.py
../.venv/bin/ruff check scripts/summarize_pc_speed_current_grid_20261104.py \
  scripts/collect_pc_dagger_demos.py scripts/summarize_pc_server_evals_basics_20261103.py
```

当前代码库有大量历史未提交改动。本实验不自动提交、不自动 push；完成后由用户确认
再执行：

```bash
git add configs/lowlevel/odarrl_control_pc_grid_*.json \
  configs/lowlevel/odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_*.json \
  configs/lowlevel/odarrl_control_ppo_pc_grid_dagger_*.json \
  scripts/run_pc_speed_current_grid_20261104.sh \
  scripts/run_pc_grid_dagger_20261104.sh scripts/collect_pc_dagger_demos.py \
  scripts/summarize_pc_speed_current_grid_20261104.py \
  scripts/summarize_pc_server_evals_basics_20261103.py \
  src/auv_lowlevel/batch/pc_vec_env.py \
  tests/test_pc_speed_current_grid_report.py \
  tests/test_pc_server_eval_summary.py tests/test_m4_bc.py \
  scripts/eval_pc_server_basics_20261103.sh \
  doc/goals/goal12_speed_current_grid_overfit.md
git commit -m "experiments: add speed-current grid and dagger queues"
git push
```

## 当前状态

四格环境、teacher demo、merge、五 seed 监督训练、冻结漂移/泄漏审计、32-seed
闭环评测和一轮 DAgger 均已完成。监督残差队列门禁为 **fail**；DAgger 队列同样为
**fail**，但相对 no-DAGGER 在高强留出格上有弱正信号（成功率 `+0.01875`、RMSE
`-10.4%`），未达到预注册迁移阈值，且 saturation 普遍上升。`grid_lw`、`grid_hw`
和 default 仍失败，因此不能宣称速度-海流解耦或过拟合问题已经解决。

按预注册停止条件，不再续做同配方 DAgger，也不通过延长训练、增加 seed 或反向
挑选评测 seed 来追求门禁通过。后续实验应转向缺失域覆盖、教师质量、执行器权限或
观测/残差通路的系统诊断。
