# Goal77：terminal lateral/vertical geometry 与 entry-direction gate 只读审计

**日期**：2026-11-10<br>
**状态**：`GEOMETRY_GATE_NOT_SUPPORTED`<br>
**上游**：Goal76 `LABEL_CONFLATION_CONFIRMED_BUT_DEADBAND_FAIL`<br>
**结果**：`doc/goals/goal77_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal77 --goal-doc doc/goals/goal77_terminal_geometry_entry_gate_20261110.md --strict
```

本 Goal 是冻结的预注册只读 Gate。记录结果前不改候选规则、阈值、seed、
eval split 或 checkpoint 选择规则；若失败，再写 blocker / ledger。

## 1. 背景与证据

Goal76 在 tight 标签下确认了 protocol radius 的标签混淆，同时 deadband
在 strict125 的 tight false 降幅只有 `48.48%`，差 1 个样本未过 `>=50%`。
deadband 之后剩余 tight false 的几何特征是：

| scenario | n | b (m) | p_nav-eta (m) | signed offset (m) | lateral (m) | vertical (m) |
|---|---:|---:|---:|---:|---:|---:|
| default | 34 | 0.1319 | 0.3252 | -1.2187 | 1.2364 | -0.2721 |
| strict125 | 17 | 0.1456 | 0.2925 | -0.6194 | 0.7618 | 0.1279 |

沿 goal 方向的 `b` 已很小，signed offset 为负，lateral 是主要分量。并且
forced zero-residual 仍有 `5/34`（default）和 `6/17`（strict125）剩余
false，`residual_l2=0`；低噪声 s240/s752/s880 贡献了大部分失败。因此本轮
不再检查 NoisyNet、residual 幅度或 deadband，而是检查 terminal 进入几何。

## 2. 假设与最便宜 falsification

### H77-A：plane-entry gate 能去掉大部分 false-success，同时保留 true

规则：在原始 sphere `success_none` 上要求 `signed_offset >= 0`。

机制：当前 false-success 多数是 `endpoint_stop` 在球面内、但尚未到达 goal
plane 的 capture。强制 plane crossing 应能去掉这些“球内但未入场”的成功。

支持条件（测量前冻结）：default 与 strict125 都满足

1. tight false reduction `>=75%`；
2. tight true retention `>=95%`；
3. schema / mismatch `=0`，held-out 未打开。

证伪：只要任一 scenario 的 false reduction 或 true retention 未过线。

最便宜 falsification：只读复用 Goal71 episodes/trace，用最后一个 terminal
snapshot 的 `signed_offset` 重分类；不训练、不新增 rollout。

### H77-B：固定 anisotropic gate 比单独 plane gate 更稳

规则：在 plane gate 基础上，同时要求

```text
abs(lateral_offset) <= 0.75 * r_tight
abs(vertical_offset) <= 0.50 * r_tight
```

其中 `lateral/vertical` 在 goal 局部正交基中分解，不使用世界坐标 xy 代替
lateral。

机制：plane gate 可能仍允许横向/纵向擦边进入；固定比例 margin 应进一步
去掉 lateral 主导的 false，而不是继续依赖沿 goal 方向的 bias 修正。

支持条件（测量前冻结）：default 与 strict125 都满足

1. tight false reduction `>=90%`；
2. tight true retention `>=95%`；
3. 相比 plane gate，至少一个 scenario 的 false 降幅再提高 `>=10` 个百分点，
   且另一个 scenario 的 true retention 下降不超过 `2` 个百分点。

证伪：combined gate 没有比 plane gate 多去掉 false，或 true retention
跌破 `95%`。

### 诊断臂（不单独作为 Gate）

- `G2`：只加 lateral margin `abs(lateral)<=0.75*r_tight`；
- `G3`：只加 vertical margin `abs(vertical)<=0.50*r_tight`。

它们只帮助解释 false 的来源，不能在看到结果后替换 H77-A/H77-B 的主判据。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| candidate | Goal71 limiter-on `_dwell50_m025_cmdlimit` |
| checkpoints | frozen Goal59 `s240/s752/s880/s1008` + `zero` |
| scenarios | `default`, `strict125` |
| seeds | dev `20–59` |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m` |
| held-out | `100–149` 关闭 |
| training | 无 |
| new rollouts | 无 |

不覆盖 Goal71/75/76 的 trace、JSON 或 MD。服务器只读运行，结果写入
Goal77 隔离目录。

## 4. 方法

### 4.1 goal 局部坐标

对每个 terminal snapshot：

```text
delta = p_nav - goal
t = unit(goal_tangent_ned)
v = unit(e_z - dot(e_z, t) * t)
h = cross(t, v)
signed   = dot(delta, t)
vertical = dot(delta, v)
lateral  = dot(delta, h)
```

必须校验：

```text
abs(||delta||^2 - (signed^2 + vertical^2 + lateral^2)) <= 1e-8
```

若 `t` 与 `e_z` 接近平行，记为 `INVALID`，不允许换轴后继续报 Gate。

### 4.2 标签

```text
success_none = ||p_nav - goal|| <= r_tight
tight_capture = ||eta - goal|| <= r_tight
protocol_capture = ||eta - goal|| <= r_protocol
tight_true = success_none and tight_capture
tight_false = success_none and not tight_capture
```

### 4.3 候选 gate

| arm | 条件 |
|---|---|
| G0 baseline | 原始 sphere success |
| G1 plane | `signed >= 0` |
| G2 lateral | `abs(lateral) <= 0.75 * r_tight` |
| G3 vertical | `abs(vertical) <= 0.50 * r_tight` |
| G4 combined | G1 and G2 and G3 |

候选 success = `success_none and gate condition`。只做重分类，不修改
checkpoint、动作、reward 或终止器。

### 4.4 统计

- 每个 arm / scenario 的 success、protocol true、tight true、tight false；
- tight false reduction fraction；
- tight true retention；
- protocol true retention（描述性）；
- 按 checkpoint 的 false 数和 geometry 中位数；
- schema、坐标恒等式和标签 mismatch。

## 5. 执行计划

- [x] Step 0：Run Contract、路径/SHA/checkpoint 审计；
- [x] Step 1：实现 analyzer + 单元测试（坐标分解、Gate 判读、mismatch）；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器 `--jobs 10` 只读运行，拉回 JSON/MD；
- [x] Step 4：按 Gate 判 H77-A/H77-B；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| H77-A tight false reduction | 两个 scenario 都 `>=75%` |
| H77-A tight true retention | 两个 scenario 都 `>=95%` |
| H77-B tight false reduction | 两个 scenario 都 `>=90%` |
| H77-B tight true retention | 两个 scenario 都 `>=95%` |
| H77-B 相对 G1 的增益 | 至少一个 scenario false 降幅 `>=+10pp`，另一个 true retention 下降 `<=2pp` |
| schema / mismatch / 坐标恒等式 | 0 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| G1 通过 | plane-entry 是可用离线 gate | 预注册闭环 terminal plane gate arm |
| G1 失败但 G4 通过 | 需要 lateral/vertical margin | 预注册闭环 anisotropic gate arm |
| 两个主 Gate 都失败 | 现有 terminal snapshot 不足以安全 gate | 回动作参数化、reward 和 terminal capture 的时序/参考点 |
| 坐标或 mismatch 失败 | `INVALID` | 只修数据/实现后重跑，不改阈值 |

## 7. 算力

- 只读复用 10 个 run，`--jobs 10`，每进程 `OMP_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`；
- 本 Goal 不训练、不新增 rollout、不打开 held-out，0 张 GPU；
- 当前只有 10 个 run 级任务，实测墙钟约 `1.78 s`、CPU 约 `889%`、RSS 约
  `33 MB`，不为占满 384 CPU 再拆 seed；
- 若后续单次分析超过 `60 s`，再按 run×seed 分片；训练型 arm 才使用
  `num_envs 128/256/512` 与 batch 256/512 的吞吐基准；
- 8×4090 空闲不构成启动 SAC/REDQ/DroQ 的理由。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal77 \
  --goal-doc doc/goals/goal77_terminal_geometry_entry_gate_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 9. Goal 提示词

```text
你现在执行 Goal77：terminal lateral/vertical geometry 与 entry-direction gate
只读审计。

先读：
- doc/goals/goal77_terminal_geometry_entry_gate_20261110.md
- doc/goals/goal76_result_20261110.md
- research/blockers/2026-11-10-goal76-label-conflation-deadband-fail.md

硬约束：
1. 只读复用 Goal71 episodes/trace，不训练、不新增 rollout、不打开 held-out；
2. 严格在 goal 局部基分解 signed/lateral/vertical，并校验范数恒等式；
3. 只比较预注册的 G0/G1/G2/G3/G4，不扫阈值；
4. 不调 deadband、reward、gamma、seed 或 checkpoint；
5. 不换 SAC/REDQ/DroQ，不加控制 authority；
6. 禁止 git add -A；只 add 本次相关文件。
```
