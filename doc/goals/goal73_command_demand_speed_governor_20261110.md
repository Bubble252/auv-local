# Goal73：command-demand/曲率感知限速的只读审计与单因素闭环对照

**日期**：2026-11-10<br>
**状态**：`H73A_FALSIFIED_PIVOT_TO_CAPTURE_AUDIT`<br>
**上游**：Goal72 `LIMITER_GEOMETRY_TRADEOFF`；Goal70 `COMMAND_RATE_INFEASIBLE`；
Goal27 `H3_SR_FAIL_STOP` 中未执行的 `S1` 曲率/饱和感知限速分支

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal73 --goal-doc doc/goals/goal73_command_demand_speed_governor_20261110.md --strict
```

## 1. 背景与证据

Goal72 已经证明：启用 command rate limiter 后，terminal-local tracking 变好，
但 default 的 8 个 `newly_lost` 出现系统性终端几何恶化：

- terminal `eta` 中位恶化 `+0.668 m`，`p_nav` `+0.770 m`；
- `abs(signed offset)` `+1.736 m`，horizontal offset `+1.483 m`；
- `5` 个 plane miss、`3` 个 false-success。

Goal70 在同一批 limiter-off rich trace 上发现：lost 的 heading command rate
P90 显著高于 retained（default `1.457` vs `0.677 rad/s`），且 strict125 的
command rate 需求更高（`3.489 rad/s`）。这说明失败与“转弯/终端阶段命令需求”
有关，而不是 residual 幅度或 NoisyNet 采样。

Goal27 的 `S1` 分支原本用于检验“曲率/饱和感知限速”，但当时因为 C1/R1/G1
先失败而没有执行。当前 Goal72 把问题重新定位到“tracking 改善但横向几何恶化”，
因此重新预注册这个单因素分支，而不是换 RL 算法。

本 Goal 的问题不是“再加一点 PPO 预算”，而是：

> 在**不改变奖励、gamma、seed、checkpoint、评估协议**的前提下，只降低高
> command-demand/高曲率段的巡航速度，能否把 horizontal/signed offset 和
> false-success 压回，同时不损失 true retention？

## 2. 假设与最便宜 falsification

### H73-A：command-demand/曲率能区分 newly-lost

机制：limiter 改善末端 tracking，但在高曲率/高 heading demand 段，速度仍然过高，
舵效和 command rate 预算不足以把横向偏差压回；降低这些段的命令速度可以减小
转弯半径和横向 offset。

支持条件（测量前冻结）：

1. default 的 `newly_lost` 相对 `both_true`，pre-terminal 1 s 的 heading
   command rate P90 高 `>=0.10 rad/s`，或 pitch command rate P90 高
   `>=0.05 rad/s`，或 material saturation fraction 高 `>=0.05`；
2. strict125 至少一个对应指标同向，且幅度 `>=` 上述阈值的一半；
3. 该差异不能由 `original_lost` 子集单独驱动；newly-lost 自身必须满足条件 1。

最便宜 falsification：只读复用 Goal67/Goal71 rich trace，按 Goal72 的
`newly_lost` / `both_true` 分组计算上述指标；不新增 rollout、不训练。

证伪：newly-lost 的 command demand/饱和不高于 both-true，或差异只存在于
original-lost。此时不实现限速臂，转去审计 terminal capture/动作参数化。

### H73-B：command-demand 感知限速能压回横向几何

机制：在 command-demand 高的时段按比例降低 `desired_speed`，让 ALOS 有更多
时间完成 heading/pitch 修正，从而减小 terminal horizontal/signed offset。

支持条件（测量前冻结）：

1. default `newly_lost` 的 terminal `abs(signed offset)` 和 horizontal offset
   配对中位改善 `>=0.10 m`；
2. default true success delta `>=+3`，或 true success 不下降且 false-success
   减少 `>=2`；
3. strict125 的 geometry 方向同向，true success 不下降超过 `1` 个 seed；
4. safety/timeout delta `<=0.02`，material saturation 不恶化超过 `0.02`；
5. 关闭新 governor 时，输出与 baseline 逐元素一致（零改动等价）。

最便宜 falsification：1-seed smoke 检查控制链确实改变 actuator/eta，再做
40-seed dev；不先跑训练。

证伪：几何没有改善，或改善以 true success/timeout 为代价；此时停止该臂，
不扫速度比例、不换 seed。

### H73-C：如果限速无效，瓶颈在 terminal capture/动作参数化

机制：如果 command-demand 审计支持 H73-A，但限速臂仍不能改善 true retention，
则剩余问题在 capture 口径、terminal 参考点或动作参数化，而不是速度。

支持条件：H73-A 成立且 H73-B 失败；此时必须先做 capture label/state 审计，
不得继续加控制 authority 或换算法。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal67 limiter-off `_dwell50_m025` |
| limiter-on reference | Goal71 `_dwell50_m025_cmdlimit` |
| checkpoints | frozen Goal59 `s240/s752/s880/s1008` + `zero` |
| scenarios | `default`, `strict125` |
| dev seeds | `20–59` |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m` |
| held-out | `100–149` 关闭 |
| training | 无 |
| new rollouts | 审计阶段无；正式臂只跑闭环评估 |

禁止把 held-out 用于调参或模型选择；禁止改 reward/gamma/seed/checkpoint；
禁止用 validation 选 governor 参数后直接报 formal。

## 4. 方法

### 4.1 Step 1 只读审计

复用 Goal67/Goal71 rich trace，对每个 `scenario / checkpoint / seed` 计算：

- pre-terminal 1 s、2 s 的 command heading/pitch rate P90；
- material saturation fraction；
- `guidance_speed_scale`、mean command speed；
- 若 trace 可恢复，参考路径曲率或实际轨迹曲率代理；
- 按 Goal72 的 `newly_lost`、`both_true`、`original_lost` 分组比较。

只读审计不改变任何控制律；输出写到
`artifacts/lowlevel/goal73_command_demand_audit_20261110/`。

### 4.2 Step 2 单因素 governor

如果 H73-A 支持，新增默认关闭的 guidance 配置：

```text
command_demand_speed_limit_enabled = false
command_demand_speed_limit_heading_start_rad_s
command_demand_speed_limit_heading_full_rad_s
command_demand_speed_limit_pitch_start_rad_s
command_demand_speed_limit_pitch_full_rad_s
command_demand_speed_limit_min_scale
```

实现只改变 `desired_speed` 的缩放，不改变 reward、ALOS 增益、PD/PI 增益、
command rate limit 或终端捕获逻辑。默认关闭时必须与 baseline 逐元素一致。

### 4.3 Step 3 评估

1. 零改动等价 smoke：disabled governor vs baseline，`eta/nu/actuator/reward`
   `max_abs_diff=0`；
2. 1-seed 控制链 smoke：确认 governor 改变 actuator/eta，不是 `INVALID`；
3. 40-seed dev formal：default 主判定，strict125 一致性复核；
4. 不做参数扫描；只测试预注册的一组阈值（由 Step 1 审计的中位数/分位数决定，
   在正式跑之前写入 goal 记录）。

## 5. 执行计划

- [x] Step 0：Run Contract、服务器/本地路径、checkpoint SHA、配置 diff 审计；
- [x] Step 1：只读 command-demand/曲率审计，判 H73-A；
- [ ] Step 2：H73-A 不支持，**不实现** governor；改为 terminal capture /
  p_nav estimator bias 只读审计（见 §5.1）；
- [ ] Step 3：margin 重分类与 bias 审计；只有支持才预注册新的单因素 arm；
- [ ] Step 4：按 Gate 判读，写 blocker/ledger（已写 pivot blocker）；
- [ ] Step 5：写 result、更新 STATE/LEDGER、运行 strict checker、只 add 相关文件。

### 5.1 Step 1 结果与 pivot

只读审计使用 Goal67/Goal71 rich trace，`--jobs 10`、CPU 约 916%、
墙钟约 3.3 s、0 张 GPU。结果：

| scenario | group | n | heading rate P90 | pitch rate P90 | saturation | curvature | speed |
|---|---|---:|---:|---:|---:|---:|---:|
| default | newly_lost | 8 | 0.5236 | 0.2618 | 0.1765 | 0.0675 | 1.7270 |
| default | both_true | 59 | 0.5236 | 0.2618 | 0.2157 | 0.0460 | 1.3388 |
| strict125 | newly_lost | 3 | 0.4082 | 0.2618 | 0.0980 | 0.0107 | 1.1877 |
| strict125 | both_true | 23 | 0.5236 | 0.2618 | 0.3725 | 0.0352 | 1.6473 |

heading/pitch command rate 在两组都顶到 limiter 上下限，已经不是区分变量；
strict125 的曲率和 saturation 方向与 default 相反。因此 H73-A
`COMMAND_DEMAND_NOT_SUPPORTED`，按预注册停止 speed governor。

同一次审计发现 false-success 全部是 `sphere` 捕获：`p_nav` 距离刚好在
tight radius 内，而 true `eta` 距离在半径外；`p_nav-eta` 误差中位约
`0.45 m`（default）和 `0.61 m`（strict125），远大于 `p_nav` 距离到半径的
余量。因此 pivot 到 terminal capture / p_nav estimator bias 审计，见
`research/blockers/2026-11-10-goal73-command-demand-not-discriminative.md`。

## 6. Gate

### Step 1 审计门

| check | 阈值 |
|---|---:|
| default newly-lost vs both-true heading rate P90 gap | `>= +0.10 rad/s` |
| default newly-lost vs both-true pitch rate P90 gap | `>= +0.05 rad/s` |
| default newly-lost vs both-true material saturation gap | `>= +0.05` |
| strict125 同向 | 至少一项 `>=` 阈值一半 |

### 正式臂门

| check | 阈值 |
|---|---:|
| default newly-lost `abs(signed offset)` median improvement | `>= 0.10 m` |
| default newly-lost horizontal offset median improvement | `>= 0.10 m` |
| default true success delta | `>= +3`，或 true 不降且 false-success 减少 `>=2` |
| strict125 geometry 方向 | 同向，true success 下降 `<=1` seed |
| safety/timeout delta | `<= 0.02` |
| material saturation delta | `<= 0.02` |
| disabled governor equivalence | `max_abs_diff = 0` |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H73-A 不支持 | command demand 不是 newly-lost 的区分变量 | 停止限速臂，转 terminal capture/动作参数化审计 |
| H73-B 几何改善但 true 下降 | 用成功换几何，不可接受 | 停止，不扫比例、不换 seed |
| H73-B 几何不改善 | 速度不是瓶颈 | 停止，转 capture/动作参数化 |
| 控制链不变 | 实现/接线错误 | 修实现后重跑，不训练 |

## 7. 算力

- 服务器：8 × RTX 4090、384 CPU、约 503 GB RAM；
- Step 1 只读审计按 run 并行，`--jobs` 最多 10，每进程单线程 BLAS；
- 正式臂是 CPU/float64 闭环评估，不占 GPU；
- 不通过扩大 `num_envs`/batch 改变本 Goal 协议；训练型扩样另开预注册 arm。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal73 \
  --goal-doc doc/goals/goal73_command_demand_speed_governor_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 9. Goal 提示词

```text
你现在执行 Goal73：command-demand/曲率感知限速的只读审计与单因素闭环对照。

先读：
- doc/goals/goal73_command_demand_speed_governor_20261110.md
- doc/goals/goal72_result_20261110.md
- doc/goals/goal70_result_20261110.md
- doc/goals/goal27_result_20261109.md
- research/blockers/2026-11-10-goal72-terminal-geometry-tradeoff.md

硬约束：
1. 不改 reward/gamma/seed/checkpoint/eval split，不打开 held-out 100-149；
2. 先做 Step 1 只读审计，H73-A 不支持就停止，不实现限速臂；
3. 新 governor 默认关闭，disabled 必须与 baseline 逐元素一致；
4. 每次只改变一个因素；不扫速度比例、不换 seed、不加预算；
5. 控制链无效时修实现，不训练；几何改善但 true 下降时停止；
6. 触发 hard blocker 时写 blocker、至少 2 个可证伪 hypothesis、reading note、
   最便宜 falsification，并在 LEDGER 写 continue/pivot/stop；
7. 服务器任务放隔离目录，不覆盖 Goal67/71/72 结果；禁止 git add -A。

执行顺序：
- Step 0：Run Contract 与 SHA/配置审计；
- Step 1：只读 command-demand/曲率审计，判 H73-A；
- Step 2：支持才实现默认关闭 governor + 测试 + 零改动 smoke；
- Step 3：1-seed 控制链 smoke，通过后 40-seed dev formal；
- Step 4：按 Gate 判读，写 blocker/ledger；
- Step 5：result、STATE/LEDGER、strict checker、只 add 相关文件。
```
