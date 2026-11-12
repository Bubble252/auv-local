# Goal40：全 seed 终端 miss 分量与 p_nav/eta 不一致诊断

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal39 `INVALID_DIAGNOSTIC`，见 `doc/goals/goal39_result_20261109.md`。<br>
**目的**：不再只看 vertical-miss 子群，而是用 Goal38 四个 block 的全部 40 seeds，
分解 plane miss 的 horizontal/vertical 分量，并量化 protocol success 与 true-`eta`
terminal capture 的不一致。本 Goal 不训练、不新增 rollout、不改控制器、不打开
held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal40 --goal-doc doc/goals/goal40_terminal_component_discordance_20261109.md --strict
```

Goal39 已表明，修正 provenance 后不能继续用旧 vertical-miss 子群支撑机制解释。
本 Goal 的阈值、分类规则、seed 范围、输出和总体映射必须在读取 Goal40 聚合结果
前冻结。运行后只追加 result、ledger、state 和 decision。

## 1. 背景与证据

Goal38 修复了 command-residual wrapper 的 stale path，并在四个 fresh formal block
上验证了 `40/40` goal、goal tangent、route tangent、path progress 和 start
provenance。修正后：

| scenario | mode | terminal capture | plane miss | protocol success |
|---|---|---:|---:|---:|
| default | baseline | `0.400` | `0.275` | `0.725` |
| default | candidate | `0.400` | `0.300` | `0.700` |
| strict125 | baseline | `0.200` | `0.500` | `0.500` |
| strict125 | candidate | `0.300` | `0.525` | `0.475` |

这暴露两个必须分开的问题：

1. plane miss 的终点几何到底主要是 horizontal 还是 vertical 分量；
2. environment 的 `success` 使用 `p_nav`，而 `goal_capture_eta` 使用 true state；
   protocol success 是否掩盖了真实的终端 miss。

Goal39 的旧 vertical-miss 子群只有 `9/7/19/16` 个默认 seeds，不能支撑旧机制重裁；
但全部 `40` 个 seeds 足以做分量和协议一致性诊断，不需要降低任何旧阈值。

## 2. 数据与隔离

只使用 Goal38 fresh formal：

```text
artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

- dev seeds：`20-59`，每 block 恰好 40 seeds；
- held-out `100-149` 不读取、不打开；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- 不改变 trace、summary、config、checkpoint 或 held-out；
- `artifacts/` 不提交。

预注册 trace SHA256：

```text
default/baseline   2ffba9597bf37acc9286f994555fae4c5170e1494103572c31fc939bda4a1528
default/candidate  bb0c8ed0722604cad763d89eac33f93bc0cdac3ac313f9dcdce6672a4a0f1985
strict125/baseline fe33817f9a17c7514decc25870c67190fb649063e9e3e611f977211f3f8d8aed
strict125/candidate 78d0b88eda4d1e8a202562db7d0bff80c2212b432dd6df466f13400a43652cfa
```

实现后、读取聚合前的 analyzer/test SHA256：

```text
analyzer a361c7b9ea76199da9afb43f7ea5960afd643c25d6d2f1f5ba9084fedf18cd56
test     597b2dc60622d4c786b96bcc26e77ee77b2a41a1ec8ee2e978aa3e216b2779ac
```

## 3. 终端几何定义

对每个 seed 取 trace 的最后一行 `i_end`，使用：

```text
offset = eta[i_end, :3] - goal_ned_m[i_end]
g = normalize(goal_tangent_ned[i_end])
longitudinal = dot(offset, g)
normal = offset - longitudinal * g
horizontal = norm(normal[:2])
vertical = normal[2]
radius = scenario goal_radius_m
```

`goal_radius_m` 固定为 default `2.0 m`、strict125 `1.25 m`。

Plane-miss 子群由 summary 的 `goal_plane_miss` 定义；每个 block 必须有至少 5 个
plane-miss seeds，否则整体 `INVALID_DIAGNOSTIC`。每个 plane-miss episode 的
dominant component 定义为：

```text
horizontal if abs(horizontal) >= abs(vertical)
vertical   otherwise
```

同时报告未归一化和按 radius 归一化的中位数。`longitudinal` 只作为完整性/描述量，
因为 plane miss 发生在纵向平面附近；不把它混入 horizontal/vertical 分类。

## 4. p_nav/eta 一致性定义

对每个 block：

```text
protocol_success = summary.success
eta_capture      = summary.goal_capture_eta
protocol_success_without_eta_capture = protocol_success and not eta_capture
```

报告 `protocol_success_without_eta_capture / protocol_success`。分母为 0 时该 block
`INVALID_DIAGNOSTIC`；每个 block 的 protocol success 必须至少 10，否则 invalid。

同时报告反向 discordance `eta_capture and not protocol_success` 作为描述量，但不进入
主假设。

## 5. 可证伪 hypotheses

每个 H 在四个 block 中分别裁决。`SUPPORTED` 要求四个 block 全部满足阈值；
`PARTIAL` 为一至三个满足；`FALSIFIED` 为零个满足。`PARTIAL` 不升级。

### H-VERTICAL-CONCENTRATION：plane miss 主要沿 vertical

每个 block 的 plane-miss episodes 中，vertical dominant fraction `>= 0.60`。

- 成立：修正后剩余 plane miss 仍以 vertical 分量为主；
- 失败：horizontal 分量同样或更常见，不能继续把所有 terminal miss 归为垂直；
- 最便宜证伪：只看四个 block 的 plane-miss 分量比例，不追加 rollout。

### H-PNAV-ETA-DISCORDANCE：protocol success 掩盖 true-state 终端 miss

每个 block 中：

```text
protocol_success_without_eta_capture / protocol_success >= 0.30
```

- 成立：`p_nav` success 不能替代 true-`eta` terminal capture，后续控制器结论必须
  同时报告两者；
- 失败：协议 success 与 true-state capture 基本一致，不能把差异当作主要问题；
- 最便宜证伪：只比较 summary flags，不加载 policy。

## 6. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/seed/checkpoint/plane-miss/protocol-success 检查失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次，不解释机制 |
| 两个 H 都 `SUPPORTED` | `MULTI_FACTOR` | 先预注册 read-only `p_nav`-vs-`eta` nav error 分解；不直接改控制器 |
| 仅 `H-VERTICAL-CONCENTRATION=SUPPORTED` | `VERTICAL_COMPONENT_SUPPORTED` | 只允许预注册全 seed 的 vertical reference/state 单因素诊断 |
| 仅 `H-PNAV-ETA-DISCORDANCE=SUPPORTED` | `PROTOCOL_ETA_DISCORDANCE_SUPPORTED` | 只允许预注册 terminal estimator/protocol 审计，不训练 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_ONLY` | 只补最便宜的数据/分组验证，不训练 |
| 全部 `FALSIFIED` | `NO_SUPPORTED_COMPONENT_SIGNAL` | 停止全 seed 终端分量线，换新假设 |

## 7. 输出

```text
artifacts/lowlevel/goal40_terminal_component_discordance_20261109/
  terminal_component_diagnostic.json
  terminal_component_diagnostic.md
  terminal_component_per_seed.csv
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,goal_plane_miss,
longitudinal_m,horizontal_m,vertical_m,
abs_horizontal_over_radius,abs_vertical_over_radius,
dominant_component,protocol_success_without_eta_capture,
eta_capture_without_protocol_success
```

## 8. 执行计划

- [x] Step 0：冻结本协议、trace SHA、阈值、输出和总体映射；
- [x] Step 1：实现 `scripts/analyze_pc_goal40_terminal_components.py`；
- [x] Step 2：为几何分解、dominant component、protocol/eta discordance、block
      verdict 和总体映射写 unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、plane miss 数、protocol success 数和
      finite；
- [x] Step 5：写 `doc/goals/goal40_result_20261109.md`，按冻结映射裁决；
- [x] Step 6：更新两个 `research/STATE.md`，追加 `research/LEDGER.jsonl`，
      必要时写 decision；
- [x] Step 7：运行 focused/full pytest、strict ledger checker 和
      `git diff --cached --check`；
- [x] Step 8：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

### 8.1 运行命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python scripts/analyze_pc_goal40_terminal_components.py \
  --formal-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output artifacts/lowlevel/goal40_terminal_component_discordance_20261109
```

验证：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal40_terminal_components.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal40 \
  --goal-doc doc/goals/goal40_terminal_component_discordance_20261109.md \
  --strict

git diff --cached --check
```

## 9. Git

禁止 `git add -A`。本 Goal 只 add：

```text
doc/goals/goal40_terminal_component_discordance_20261109.md
doc/goals/goal40_result_20261109.md
scripts/analyze_pc_goal40_terminal_components.py
tests/test_pc_goal40_terminal_components.py
research/LEDGER.jsonl
research/STATE.md
../research/STATE.md
research/decisions/<若需要>
```

不提交 artifacts、trace、checkpoint、日志或大文件。只有用户明确要求才
commit/push。

## 10. 服务器边界

本 Goal 为只读 CPU 诊断，本地运行即可。若后续结果支持训练，才进入 8×RTX 4090
服务器；训练前必须重新检查 `nvidia-smi`、选择空闲卡、使用隔离目录，并把密码
只放在 shell 的 `SSHPASS` 中。

## 11. Goal 提示词

```text
你现在执行 Goal40：全 seed 终端 miss 分量与 p_nav/eta 不一致诊断。

先读：
- doc/goals/goal40_terminal_component_discordance_20261109.md
- doc/goals/goal39_result_20261109.md
- doc/goals/goal38_result_20261109.md

硬约束：
1. 只用 Goal38 四个 fresh formal block，dev seeds 20-59；
2. 不训练、不新增 rollout、不改控制器、不打开 held-out；
3. 用全部 40 seeds，不再按旧 vertical-miss 子群筛选；
4. 阈值和总体映射不得在看到结果后修改；
5. 只有 SUPPORTED 才允许预注册下一步；PARTIAL 不能升级。
```
