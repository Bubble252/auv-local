# 训练实验 standing protocol

**日期**：2026-11-11<br>
**状态**：`ACTIVE / APPLIES_TO_ALL_TRAINING_PARADIGMS`<br>
**适用**：传统控制器、BC、BC+残差、ALOS+残差、PPO、SAC，以及后续 REDQ/DroQ 等任何训练架构。

> 本文件规定“训练实验怎么组织”；具体测评指标、路线/长度/速度/噪声矩阵和报告格式以
> `research/EVALUATION_PROTOCOL.md` 为准。以后不再接受只报 success 的 PC 结果。

## 1. 为什么需要这个协议

过去几轮出现过同一类问题：单个场景、单个 seed、单个 reward 或单个 residual 权限设置下的
结果，被误读成“方法有效/无效”。例如：

- T2-SAC 在 nominal 有 success 信号，但 full residual 在全路径引入 `+0.278 m` path 代价；
- C0 的 ALOS 能过 Gate，但 C1 在 residual 介入后仍只有 `0.625–0.65`；
- PPO 的 residual 数值非零，但闭环结果与 zero-residual 完全相同；
- 同一 checkpoint 换一个 current 档位，success 可以从 `37/40` 掉到 `9/40`。

因此以后任何训练型 Goal 都必须同时回答三个问题：

1. **reward 是否把正确的量放进了优化目标？**
2. **课程/权限是否把训练数据放到了正确的状态分布？**
3. **换环境后是否仍然成立，而不是只对一个 nominal 场景过拟合？**

## 2. 强制矩阵

每个训练型 Goal 至少包含以下维度；可以在 screen 阶段减少组合，但不能只跑一个格子。

| 维度 | 最少档位 | 说明 |
|---|---|---|
| 基线 | zero-residual 传统控制器 | K4/ALOS 按当前主线选择；必须与 policy 配对 |
| reward | R0 baseline、R1 `F_z`、R2 `F_z + F_zdot` | potential-based，不直接把绝对误差当 raw dense reward |
| 课程 | C-fixed（直接最终分布）、C-full（分阶段 promotion/rollback） | 容量匹配；C-full 与 C-fixed 同预算 |
| 残差权限 | always-on、near10_2、shell10_3、staged 0→1 | 先 eval-only 或小预算 screen，再进正式训练 |
| 环境 | nominal、current05/10/15、noise N2、plant20、combined20 | 至少覆盖 nominal + 一个 current + 一个 noise/plant + combined |
| success 口径 | default 2.0 m、strict 1.25 m | `eta_tight_at_done`，不按 p_nav 改写 |

## 3. 指标

每个 arm 必须报告：

- success、plane miss、timeout、safety；
- path 3D/horizontal/vertical RMSE（同时区分 episode-mean 和 pooled active-step）；
- `eta_done_distance`、near-goal vertical/horizontal error、终点附近分箱分布；
- residual L2 / 通道 RMS，按 `<=2 m`、`2–10 m`、`>10 m` 分箱；
- policy-head saturation、effective residual saturation、actuator/material saturation、residual clipping；
- 方向性失败分解（terminal kind、signed vertical/N/E bias）；
- 路线族、长度、速度和压力档位标签；凡是“多环境成立”，必须列出具体 scenario；
- 训练吞吐、实际 transitions、checkpoint SHA。

## 4. 晋级规则

- 单 seed、单环境、单 reward 的结果只能叫 `screen`，不能叫 formal。
- screen 通过后必须扩到 3 seeds 和至少 4 个环境档位；否则不能写“方法有效”。
- 任何“换算法”必须另开预注册；不能在同一 Gate 内把 PPO 换成 SAC，或把 SAC 换成 REDQ/DroQ。
- 纵向误差主导时才启动 Goal140 的 reward pilot；横向/terminal 权限主导时先查动作参数化。
- 不打开 held-out `100–149`；不因失败放宽 Gate；不通过增加单 arm 预算救结果。

## 5. 文档与检查

每个训练型 Goal 的文档必须：

1. 写清 reward / curriculum / authority / eval 四个维度；
2. 声明 `eval_suite: configs/lowlevel/eval/pc_standing_eval_suite_v1.json`，并运行
   `python scripts/validate_pc_standing_eval_suite.py`；
3. 在训练前冻结组合、预算、seed 和 Gate；
4. 失败时写 blocker、至少两个可证伪 hypothesis、LEDGER 和 STATE；
5. 用 `scripts/analyze_pc_goal153_eval_grid.py` 生成完整 `grid_summary.md`，不能只交 success 表；
6. 通过 `.venv/bin/python ../scripts/check_research_ledger.py ... --strict --require-result`。

## 6. 当前主线

- Goal153：reward + curriculum 联合 screen（P1）；Phase 3/4 起强制执行
  `research/EVALUATION_PROTOCOL.md` 的完整指标和矩阵。
- Goal154：residual authority schedule 机制研究（P1，可与 Goal153 的只读审计并行）。
- Goal152：T2 near10_2 gate training screen（sidecar，只作为权限证据，不再扩展）。
