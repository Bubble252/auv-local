# Goal99：距离调度的 PPO 残差 scale 对照

**日期**：2026-11-11<br>
**状态**：`SCHEDULE_NOT_SUPPORTED / TERMINAL_DEGRADATION_PRECEDES_2M / NO_SAC`<br>
**上游**：Goal98 `SCALE_PARTIAL / TERMINAL_DIRECTION_REMAINS_BOTTLENECK / NO_SAC`<br>
**性质**：eval-only、冻结的预注册 screen；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal99 --goal-doc doc/goals/goal99_distance_scheduled_residual_scale_20261111.md --strict
```

本 Goal 记录前不得修改 schedule 集合、switch 距离、dev seed、success frame、
scenario、checkpoint、trace 字段或 Gate 阈值。失败后先补 blocker/decision，
再决定是否进入 Research Loop。

## 1. 背景与证据

> 结果：`SCHEDULE_NOT_SUPPORTED`。三组 schedule 与 constant 2.00 逐 seed 相同，
> strict 仍为 18/40。详见 `doc/goals/goal99_result_20261111.md`。

Goal98 已确认真实训练锚点是 `pc.actuator_residual_scale=1.50`，且 1.50
no-op 精确复现 Goal95 final 的 30/19。固定 scale 网格得到：

| arm | default 1.50 | default 2.00 | strict125 1.50 | strict125 2.00 |
|---|---:|---:|---:|---:|
| P_D3 | 30 | 31 | 19 | 18 |
| P_BC15 | 30 | 32 | 19 | 18 |

机制证据指向距离分区，而不是全局幅度：

- default 的 gained seed 27/49 在 `3–2 m` 窗口用 2.00 后 radial closure
  明显上升；
- strict 的 lost seed 22/29 在 `2–1.25 m` 窗口用 2.00 后 closure 从约
  `1.0` 降到 `0.44–0.69`，全部变成 `goal_plane_miss`；
- strict gained seed 44 在 2.00 下 `2–1.25 m` closure 从 `0.23–0.33`
  升到约 `2.05`；
- 所有 strict 失败都是 `goal_plane_miss`，不是 timeout 或 safety。

因此最便宜的下一步是：**远场保留 2.00 的闭合增益，近场退回 1.50 的
capture 行为**。本次只验证距离调度，不改变网络、reward 或终止协议。

## 2. 网络结构对照

本 Goal 开始前已做只读 checkpoint 对照，结论是“网络结构没有变化”：

| 项目 | Goal95 P_D3 | Goal95 P_BC15 | Goal86 冻结 BC | 结论 |
|---|---|---|---|---|
| `d_z_total` | 64 | 64 | 64 | 一致 |
| `n_z_reserved` | 4 | 4 | 4 | 一致 |
| state encoder | `[128,128]` | `[128,128]` | `[128,128]` | 一致 |
| actor / critic | `[256,128]` ELU | `[256,128]` ELU | `[256,128]` ELU | 一致 |
| path | path8 + endpoint4 | path8 + endpoint4 | path12 | 结构相同 |
| command | 3 | 3 | 3 | 一致 |
| current encoder | 64→60，reserved 补零 | 同左 | 同左 | 一致 |
| DADE | off，输入 64 维零块 | 同左 | 同左 | 一致 |
| NoisyNet | 无 | 无 | 无 | 评价走确定性 `act_inference()` |
| 冻结 BC 浮点张量 | 16 项 `max_abs=0.0` | 16 项 `max_abs=0.0` | 参照 | 逐元素一致 |

checkpoint SHA：

- D3 `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`
- BC15 `1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43`
- Goal86 BC `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`

所以本 Goal 不允许把失败归因于“网络接错、NoisyNet、容量不足”。若结果异常，
先检查 schedule 写回、距离计算和 trace，不重写主干。

## 3. 假设

### H99-A：距离调度能同时保留 default 增益和 strict capture

如果成立：

- 某个 schedule 在两臂都达到 default `>=31`、strict `>=20`；
- path 3D RMSE 相对 constant 1.50 平均恶化 `<=5%`；
- timeout/safety 不增加；
- strict 的 `2–1.25 m` closure 回到或超过 constant 1.50。

则判 `SCHEDULE_SUPPORTED`：值得预注册训练时 schedule 或 formal 复现。

如果失败：

- strict 仍低于 20；或 default 退回 30/29；或 path 恶化超过 5%。

### H99-B：收益不是简单的远/近距离分区

如果成立：

- 三个 switch 距离的 gained/lost seed 集合几乎一致；
- 或 schedule 的成功率不超过 constant 1.50/2.00 的相关结果；
- 或远场激活比例很低，实际行为仍接近 constant 1.50。

则判 `SCHEDULE_NOT_SUPPORTED`：说明 2.00 的作用不是“远场幅度”，下一步回到
terminal residual 的方向/信用审计，不扩大 scale 搜索。

### H99-C：近场仍需要 2.00，真正问题是 capture 几何

如果成立：

- schedule 在 default 有增益，但 strict 仍系统性 plane miss；
- 失败 seed 的近场 applied/effective 残差方向正确，只是 goal plane 几何
  不允许闭合；
- 把远场 2.00 与近场 1.50 切换后，strict closure 仍未改善。

则判 `GEOMETRY_LIMITED`：回到 terminal capture/guidance，不在本 Goal 改
goal radius 或终止规则。

## 4. 数据与隔离

- 训练：不训练；
- eval：dev seeds `20–59`，每 run 40 episodes；
- scenario：Goal85 unbiased default 与 strict125；
- checkpoint：Goal95 `P_D3/final.pt`、`P_BC15/final.pt`；
- success frame：固定 `both`；
- 固定 checkpoint、config、goal radius、reward、termination 和 eval split；
- held-out `100–149` 禁止打开；
- 不复用 Goal98 的 2.00 结果代替本 Goal 的新 schedule run；constant 1.50/2.00
  只作为对照表引用。

## 5. 预注册实验矩阵

### 5.1 schedule 集合

固定三组：

| ID | near scale | far scale | switch distance |
|---|---:|---:|---:|
| `S20` | 1.50 | 2.00 | 2.0 m |
| `S25` | 1.50 | 2.00 | 2.5 m |
| `S20-N175` | 1.75 | 2.00 | 2.0 m |

规则：`distance > switch_distance` 使用 far scale，否则使用 near scale。
距离用 actor 当前可观测位置到 goal 的距离；trace 记录逐环境的实际 scale。

### 5.2 矩阵

```text
3 schedules × 2 arms (P_D3, P_BC15) × 2 scenarios
= 12 eval runs
```

每个 run：

- `--seed-start 20 --seed-count 40`；
- `--success-frame both`；
- CPU/float64；
- 单线程 BLAS，8 个进程并行；
- 写 `summary.json/md` 和 `trace.npz`；
- summary 写回 `actuator_residual_scale_near/far/switch_distance_m`；
- trace 必须包含 `actuator_residual_scale`、
  `actuator_residual_schedule_distance_m`、`goal_distance_eta_m`、
  `actuator_residual_applied/effective`。调度距离是 step 前用于选择 scale 的
  距离，避免用 step 后位置误判边界。

### 5.3 命令模板

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal95_20261111/p_d3_ppo.json \
  --checkpoint artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt \
  --environment-config configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json \
  --output artifacts/lowlevel/goal99_distance_schedule_20261111/P_D3_default_S20 \
  --seed-start 20 --seed-count 40 --success-frame both --device cpu \
  --actuator-residual-scale-near 1.50 \
  --actuator-residual-scale-far 2.00 \
  --actuator-residual-scale-switch-distance-m 2.0 \
  --trace-output artifacts/lowlevel/goal99_distance_schedule_20261111/P_D3_default_S20/trace.npz
```

## 6. 指标

主指标：

- `success`（both-frame）；
- `success_pnav_tight`、`success_eta_tight_ever`、`success_eta_tight_at_done`；
- `timeout`、`safety`、`goal_plane_miss_rate`。

机制指标：

- path 3D/horizontal/vertical RMSE 相对 constant 1.50 的比值；
- `3–2 m`、`2–1.25 m`、`1.25–0 m` 窗口的 radial closure；
- `actuator_residual_scale` 在远场/近场的激活比例；
- residual applied/effective ratio、magnitude/rate clip、survival；
- 同 seed 的 gained/lost 集合。

## 7. Gate

### 7.1 `INVALID`

任一条件成立：

- schedule 参数没有完整写回 summary，或 trace 缺 `actuator_residual_scale` /
  `actuator_residual_schedule_distance_m`；
- 近场/远场边界写回与 trace 不一致；
- checkpoint SHA 不匹配；
- 出现 NaN/Inf、trace 字段缺失或 success 口径改变；
- 读取了 held-out `100–149`。

`INVALID` 不写成 `FAIL`。

### 7.2 `SCHEDULE_SUPPORTED`

存在同一个 schedule，同时满足：

- D3 与 BC15 都达到 default `>=31`、strict `>=20`；
- 相对 constant 1.50 的 path 3D RMSE 平均恶化 `<=5%`；
- timeout/safety 不增加；
- trace 有限。

### 7.3 `SCHEDULE_PARTIAL`

存在至少一个 schedule 在单 arm 或单 scenario 上超过 constant 1.50/2.00 的
相关结果，但不满足 `SCHEDULE_SUPPORTED`。只做归因，不直接 formal。

### 7.4 `SCHEDULE_NOT_SUPPORTED`

没有任何 schedule 超过 constant 1.50/2.00 的相关结果，或改善只来自 seed
翻转且 closure 没改善。不得继续盲扫 switch 距离。

### 7.5 `GEOMETRY_LIMITED`

schedule 的 residual direction/effective action 正确，但 strict 仍系统
`goal_plane_miss`。回到 terminal capture/guidance，不改 goal radius。

## 8. 执行计划

- [x] Step 0：确认 Goal98 结论、真实 scale 锚点和网络结构 parity；
- [x] Step 1：实现 actuator residual 距离调度并接评测 CLI；
- [x] Step 2：补 schedule 单测与 1 seed smoke；
- [x] Step 3：远端跑 12 个 schedule dev40 eval；
- [x] Step 4：按冻结 Gate 判断 `SCHEDULE_NOT_SUPPORTED`；
- [x] Step 5：补 result/blocker/decision、LEDGER、STATE，并运行 checker。

## 9. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不改 reward、terminal、success、goal radius；
- 不重写网络主干，不打开 NoisyNet；
- 不扩 seed，不打开 held-out；
- 不用 Goal98 的 constant 2.00 结果冒充 schedule 结果；
- 不把 `SCHEDULE_PARTIAL` 写成 `SUPPORTED`。

## 10. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal99 \
  --goal-doc doc/goals/goal99_distance_scheduled_residual_scale_20261111.md --strict
git diff --cached --check
```

不自动 commit/push，不提交 `artifacts/`、checkpoint、日志和缓存。

## 11. Goal 提示词

```text
你现在执行 Goal99：只验证 PPO actuator residual 的距离调度。

先读：
- doc/goals/goal99_distance_scheduled_residual_scale_20261111.md
- doc/goals/goal98_result_20261111.md
- doc/goals/goal87_network_history_parity_20261110.md

硬约束：
1. 不训练、不换 SAC/REDQ/DroQ；
2. 不改 reward、terminal、success、goal radius；
3. dev seeds 固定 20-59，不打开 held-out 100-149；
4. 不改网络主干，不引入 NoisyNet；
5. 先完成 CLI、trace 和单测，再跑 12 个 dev40 schedule eval。

执行顺序：
1. 验证 constant 1.50/2.00 基线可复用；
2. smoke 验证 far=2.00、near=1.50 的逐环境 scale 写回；
3. 跑 S20/S25/S20-N175 × 2 arms × 2 scenarios；
4. 按冻结 Gate 判 INVALID/SUPPORTED/PARTIAL/NOT_SUPPORTED/GEOMETRY_LIMITED；
5. 失败先写 blocker 和至少两个可证伪 hypothesis，再决定 continue/pivot/stop。
```

## 12. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或
扫 seed；每个 blocker 至少写两个可证伪 hypothesis；需要外部方法时先检索论文/
代码并写 reading note；先跑最便宜的 falsification，再决定
`continue` / `pivot` / `stop`；不自动 commit/push，不打开 held-out，不改变
任务成功定义或主算法族。
