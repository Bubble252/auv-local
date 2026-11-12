# PC 统一测评协议（standing eval contract）

**生效日期**：2026-11-11<br>
**状态**：`ACTIVE / CANONICAL / APPLIES_TO_ALL_PC_GOALS`<br>
**机器可读清单**：`configs/lowlevel/eval/pc_standing_eval_suite_v1.json`<br>
**统一分析器**：`scripts/analyze_pc_goal153_eval_grid.py`<br>
**适用**：传统 ALOS、BC、BC+残差、ALOS+残差、PPO、SAC，以及后续 REDQ/DroQ。

## 1. 目的

过去只用 success 判断策略，导致三类误读反复出现：

1. success 接近，但路线已经明显更差、动作更饱和；
2. nominal 上略有收益，换 current/noise/plant/route 后完全消失；
3. terminal 只在某个半径边界或某一类失败方向上变好，却被写成整体提升。

因此以后任何 PC 训练 Goal、机制审计和正式实验都必须同时报告
“任务结果、路线健康度、终点捕获、失败方向、控制健康度、泛化轴”。success 只是
entry point，不是主结论本身。

## 2. 硬约束

- 主成功口径固定为 `success_eta_tight_at_done`；`default=2.0 m`、`strict125=1.25 m`。
- 暂时不使用 `p_nav/eta` 位置差作为性能轴；如保留 `p_nav`，只作诊断，不参与 Gate。
- standing eval 对 C0/C1 也必须关闭 direct-command 训练脚手架；否则
  `curriculum.enabled=false` 会重新激活随机 direct command，zero-residual 不再沿路线飞。
  runner 对 SAC/PPO/BC-residual 三类 evaluator 都强制传
  `--disable-curriculum-direct-command`。
- 基线不额外注入位置偏置、导航延迟或人工时滞；压力只来自显式 current、noise、plant 和 geometry。
- dev seeds 固定 `20–59`；held-out `100–149` 关闭。
- screen 只算 screen；正式结论至少 3 个训练 seed，并跨多个压力/几何档位。
- 同一 arm 中途不得改 reward、curriculum、authority、success 或 eval split。
- `INVALID`、`FAIL`、`PASS` 必须分开；不能用放宽阈值、换 seed 或增加预算救失败。

## 3. 指标分层

### 3.1 任务结果

| 指标 | 说明 |
|---|---|
| default success | `eta_tight_at_done <= 2.0 m` |
| strict125 success | `eta_tight_at_done <= 1.25 m` |
| `GOAL` / `GOAL_PLANE_MISS` / `TIME_LIMIT` / safety | 终止类型必须分开 |
| paired success delta | 同 seed policy vs zero；不要只报聚合 success |

### 3.2 路线健康度

| 指标 | 说明 |
|---|---|
| pooled 3D/vertical/horizontal RMSE | 所有 active step 汇总，不能与 episode-mean 混用 |
| p50/p90/p95/p99/max | 暴露尾部和局部飞点 |
| path error `>1 m` 比例 | 路线偏离的持续比例 |
| `<=2 m / 2–10 m / >10 m` 分箱 RMSE | 分离 terminal、approach shell 和远段 |
| final path progress fraction | 是否真正沿路线推进，而非只靠终点附近震荡 |

### 3.3 终点捕获与终点附近分布

| 指标 | 说明 |
|---|---|
| final distance p50/p90/p95/max | 不只看是否跨过阈值 |
| final distance bins | `<=1.25 / 1.25–1.5 / 1.5–2 / 2–3 / >3 m` |
| closest approach | 是否曾接近但未在正确时刻完成 |
| 2 m entry vertical/horizontal p90 | 进入终点壳层时的几何质量 |
| near-radius occupancy | 终点 1.25/1.5/2/3 m 内停留比例 |
| signed final vertical/horizontal bias | 区分“系统性偏高/偏低/偏左/偏右” |
| success vs failure geometry decomposition | 成功/失败组的 final error、near-goal bias 分列 |

### 3.4 方向与失败分解

| 指标 | 说明 |
|---|---|
| terminal kind counts | `GOAL / GOAL_PLANE_MISS / TIME_LIMIT / safety` |
| final vertical sign | above / within / below `±0.1 m` |
| final N/E miss sign | north/south、east/west 的系统性偏移 |
| success/failure decomposition | 失败是没到、到了但错过、超时，还是安全终止 |

### 3.5 控制健康度

| 指标 | 说明 |
|---|---|
| policy-head saturation pre-scale | 策略原始输出 `abs>0.95`；需要 `policy_action_raw` |
| effective residual saturation post-scale | 乘以 global residual multiplier 后的有效动作饱和 |
| actuator/material saturation | 执行器幅度/速率或 material threshold 饱和 |
| scale/rate clip fraction | 残差被 scale 或 rate limit 裁剪的比例 |
| residual L2 ratio | residual / base；分总步与 gate-active 步 |
| residual channel RMS | rudder / stern / rpm 三通道分别报告 |
| action RMS / abs max | 排除“success 靠更激进动作” |

### 3.6 完整性

- episode 数、训练 seed 数、checkpoint SHA、environment config SHA；
- 所有 episode finite；无 NaN/Inf/flew-out；
- eval protocol、tier、scenario ID、success frame、goal radius；
- paired zero baseline 是否存在，缺失时明确写 `UNPAIRED`。

## 4. 固化矩阵

### 4.1 范式

| 范式 | base | 残差 | 算法 |
|---|---|---|---|
| `T0` | ALOS | 无 | 传统控制器 |
| `BC` | BC | 无 | 监督学习 |
| `BC-PPO` | frozen BC | actuator/command | PPO |
| `BC-SAC` | frozen BC | actuator/command | SAC |
| `ALOS-PPO` | ALOS | actuator/command | PPO |
| `ALOS-SAC` | ALOS | actuator/command | SAC |
| `future-REDQ/DroQ` | 另开预注册 | 另开预注册 | off-policy ensemble |

PPO 和 SAC 是固定对照，不在同一 Gate 中互换；REDQ/DroQ 只有在明确证明
“探索/样本效率不足”后才新开预注册。

### 4.2 课程

| 课程 | 要求 |
|---|---|
| `C-fixed@C1` | 固定 C1 分布，不晋级；用于 reward/authority，不与 `C-full` 比较课程顺序 |
| `C-full` | `C0–C6`，按 Gate 晋级/回滚到 C6；课程顺序比较对象是 `C-hard` |
| `C-hard` | 直接训练 C6 最终混合分布，隔离课程顺序价值；与 `C-full` 同预算、同最终分布 |
| `C-no-current` | 同 `C-full` 到 C6，但 actor 不看 current estimate |
| `C-oracle-current` | 仅诊断上界，不进主结果 |

课程阶段定义、深度/路径/噪声/current 阶梯沿用 Goal136 §2.5–2.6，不得在评估时悄悄换档。`C-fixed` 与 `C-full` 的最终分布不同，因此只能用于 reward/authority 对照；真正回答“课程顺序是否有用”时必须用同预算、同 C6 最终分布的 `C-full vs C-hard`。

### 4.3 residual authority

至少保留：`zero`、`near_only`、`mid_only`、`far_off_only`、`near10_2`、`shell10_3`、
`full`、`staged_0_to_1`、`full_scale_0.5`、`full_scale_0.75`。

若只测 terminal-only/full 两端，不能判断“残差在哪个距离段有用”。凡出现
“远段有害、近段有用”的结论，必须检查：

- 配置是否相同；
- residual scale 是否相同；
- checkpoint/训练预算是否相同；
- authority gate 的 realizable mask 是否一致；
- success frame 和 route seed 是否一致。

### 4.4 压力

| profile | 设置 | 用途 |
|---|---|---|
| `nominal` | base config | 复现和主表 |
| `current05/07/10/15` | Goal149 current ladder | 找 K4/策略过渡带 |
| `noise_n2` | `noise_scale_range=0.5–1.0` | 感知压力 |
| `plant20` | 五通道 `0.8–1.2×` | 动力学估计误差 |
| `combined20` | current10 + N2 + plant20 | 联合压力 |

旧 Goal134 `combined20` 若使用 current20，必须在结果里标 `legacy`，不能与本协议
的 `combined20` 静默合并。

### 4.5 路线、长度、速度

| 轴 | 档位 |
|---|---|
| route | `legacy_sine / P0 / P1 / P2 / P3 / P4` |
| length | `30 / 35 / 45 m` |
| speed | `0.9 / 1.2 / 1.4 / 1.8 / 2.2 / 2.5 m/s`；`2.5` 是 REMUS100 上界 |
| depth | `10 / 30 / 80 / 150 / 250 / 400 / 500 m`；深水只标 `SIM_ONLY_OOD` |

路线形状和长度不能只看同一个 sinusoidal bank；长度/速度与 timeout 相关时，要同时报告
timeout、final progress、path p95，而不是只报 success。

速度课程的未来版本固定在 `speed_curriculum.stages_mps`：C0 从 `1.0–1.2` 开始，
逐级抬高上界，C6 覆盖 `0.9–2.5 m/s`。该表只允许新预注册课程 Goal 启用；
Goal136/Goal143/Goal153 的既有课程速度范围不得被静默改写。强洋流下 `2.5 m/s`
地面速度可能不可达，此时必须报告不可达、饱和和 timeout，不能删除该档或改 success 口径。
注意 `speed` 是期望地速；可达性硬约束为
`v_cmd + v_current + v_variation <= 2.5 m/s`。`command_speed_range_mps`/domain
cell 已在 parse 层校验；curriculum sampler 已按最坏情况 `v_current + v_variation`
收缩电流项，确保逐 episode 联合约束成立。课程采样
不能把 `2.5` 与强流独立组合；`2.5` 端点只用于零/弱流档，或作为明确标注的 OOD
压力档报告不可达/饱和；与 `2.5` 同时出现时，速度波动也必须为零/可忽略。standing eval 的单值 `reference.speed_mps` 允许生成这种
不可达压力档，但结果必须按不可达/饱和/timeout 记录，不能算作正常可达样本。

## 5. 测评层级

| tier | 训练 seed | eval seeds | 覆盖 | 用途 |
|---|---:|---:|---|---|
| `A_screen` | 1 | 40 | nominal + current10 + noise + plant | 快速筛选 |
| `B_robustness` | 1 | 40 | 7 pressure + P2/P3 + 35/45 m + 1.2/1.4/2.5 m/s | 压力与泛化；2.5 必须保留 |
| `C_full_geometry` | 1 | 40 | 6 routes + 3 lengths + 6 speeds（0.9/1.2/1.4/1.8/2.2/2.5 m/s） | 几何审计；含 2.5 上界 |
| `D_formal` | 3 | 40 | 全部 pressure/route/length/speed | 正式结论 |

完整笛卡尔积很贵，所以按 tier 分层；但 `D_formal` 不允许只跑 nominal 或单一路线。
如果算力有限，可以先用 `A/B/C` 做 survivor selection，但不能把 selection 写成 formal。

## 6. 报告格式

每次 eval 必须生成：

- `summary.json` / `episodes.json` / `trace.npz`；
- `grid_summary.json`；
- `grid_summary.md`，至少含以下面板：

1. Scenario axes；
2. 主结果（success/plane miss/timeout/safety/episode-mean path）；
3. paired deltas vs zero；
4. Route health（pooled active-step）；
5. Terminal-near distribution；
6. Directional/failure decomposition；
7. Action/residual health；
8. Reward components（可选但推荐）。

生成场景配置：

```bash
cd settled_局部规划器_反应式
.venv/bin/python scripts/validate_pc_standing_eval_suite.py
.venv/bin/python scripts/build_pc_standing_eval_configs.py \
  --tier A_screen \
  --base-env configs/lowlevel/goal136/curriculum/C1.json \
  --output-dir artifacts/lowlevel/<goal>/standing_configs \
  --overwrite
```

arm 清单必须显式声明 evaluator、training config、checkpoint 和 action mode；例如：

```json
{
  "schema_version": "auv_pc_standing_eval_arms.v1",
  "arms": [
    {
      "tag": "zero",
      "paradigm": "BC-SAC",
      "evaluator": "sac",
      "training_config": "configs/lowlevel/goal153/train/R0_C1_sac.json",
      "action_mode": "zero-residual",
      "disable_curriculum_direct_command": true
    },
    {
      "tag": "R1",
      "paradigm": "BC-SAC",
      "evaluator": "sac",
      "training_config": "configs/lowlevel/goal153/train/R1_C1_sac.json",
      "checkpoint": "artifacts/lowlevel/<goal>/train/R1_C1/final.zip",
      "action_mode": "policy",
      "residual_scale": 1.0
    }
  ]
}
```

执行并统一分析：

```bash
.venv/bin/python scripts/run_pc_standing_eval_suite.py \
  --manifest artifacts/lowlevel/<goal>/standing_configs/manifest.json \
  --arms artifacts/lowlevel/<goal>/standing_arms.json \
  --output-root artifacts/lowlevel/<goal>/eval \
  --seed-start 20 --seed-count 40 --device cpu --max-parallel 8

.venv/bin/python scripts/analyze_pc_goal153_eval_grid.py \
  --root artifacts/lowlevel/<goal>/eval \
  --output-json artifacts/lowlevel/<goal>/eval/grid_summary.json \
  --output-md artifacts/lowlevel/<goal>/eval/grid_summary.md
```

分析器会自动发现 `tag/scenario` 和 `tag/<seed_tag>/scenario`，不局限于
`default/strict125`。旧 trace 若没有 `policy_action_raw`，策略头饱和会显示 `NA`；
这是“缺少记录”，不是 0%。

## 7. Gate 使用

- Gate 主指标按各 Goal 预注册，不能被本协议追加的指标偷换。
- 本协议追加的是**必报风险指标**：路线、terminal、方向、饱和、泛化。
- 任何 reward 提升若伴随 path RMSE、p95、饱和率或 pressure collapse 明显恶化，不能
  写成“方法有效”。
- 任何“神经网络终于赢传统方法”的结论，至少要有：
  - 跨 2 个以上压力档；或
  - 跨 2 个以上路线族/长度/速度；或
  - 3 个训练 seed 方向一致；
  - 且路线健康和 terminal 分布不退化。

## 8. 兼容与迁移

- 2026-11-11 之前的结果按旧口径保留，不追溯改数；重分析时标明 `legacy_trace`。
- Goal153 起的新分析必须使用本协议；Goal153 Phase 3/4 是本协议的第一个完整实例。
- 新 Goal 文档必须写 `eval_suite: configs/lowlevel/eval/pc_standing_eval_suite_v1.json`，
  并在完成检查中运行 validator。

