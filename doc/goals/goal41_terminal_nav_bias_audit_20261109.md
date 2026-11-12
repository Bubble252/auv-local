# Goal41：terminal p_nav/eta 偏差方向审计

**日期**：2026-11-09<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal40 `PROTOCOL_ETA_DISCORDANCE_SUPPORTED`，见
`doc/goals/goal40_result_20261109.md`。<br>
**目的**：只读 Goal38 fresh traces，检查 `p_nav` protocol success 掩盖 true-`eta`
terminal miss 时，terminal 估计误差是否系统性地把位置“拉向”goal。本 Goal 不训练、
不新增 rollout、不改控制器、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal41 --goal-doc doc/goals/goal41_terminal_nav_bias_audit_20261109.md --strict
```

本 Goal 只解释 Goal40 的 `p_nav`/`eta` discordance，不直接提出控制器修复。所有
阈值、block 规则、seed 和总体映射必须在读取聚合结果前冻结。

## 1. 背景与证据

Goal40 在四个 Goal38 fresh block 中发现：

| scenario | mode | protocol success | success without eta capture | fraction |
|---|---|---:|---:|---:|
| default | baseline | 29 | 13 | `0.4483` |
| default | candidate | 28 | 12 | `0.4286` |
| strict125 | baseline | 20 | 13 | `0.6500` |
| strict125 | candidate | 19 | 7 | `0.3684` |

同时 H-VERTICAL-CONCENTRATION 只有 `PARTIAL`，candidate 的 plane miss 已经转为
horizontal 主导或接近均分。因此不能继续把所有失败都归给 vertical，也不能直接用
`success` 宣称 true-state terminal capture 成功。

当前需要回答的问题是：discordant seeds 的 `p_nav` 是否系统性地比 `eta` 更接近
goal，且估计误差方向是否指向 goal。若是，这是 estimator/protocol 问题；若否，
不能把 discordance 归为简单 nav bias。

## 2. 数据与隔离

只使用 Goal38 fresh formal：

```text
artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

- dev seeds：`20-59`，每 block 恰好 40；
- held-out `100-149` 不读取、不打开；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- artifacts 不提交。

预注册 trace SHA256：

```text
default/baseline   2ffba9597bf37acc9286f994555fae4c5170e1494103572c31fc939bda4a1528
default/candidate  bb0c8ed0722604cad763d89eac33f93bc0cdac3ac313f9dcdce6672a4a0f1985
strict125/baseline fe33817f9a17c7514decc25870c67190fb649063e9e3e611f977211f3f8d8aed
strict125/candidate 78d0b88eda4d1e8a202562db7d0bff80c2212b432dd6df466f13400a43652cfa
```

实现后、读取聚合前的 analyzer/test SHA256：

```text
analyzer 35bed01c2cb903c760d8d7c1f1ea7c6c02a93ef52f05126898831443400205ef
test     bd50f0d83968c0dbcb0c124457fb0f8460a59f5898fc28c7f817f7109199d4f5
```

初次运行因 analyzer 把 6 维 `eta` 整行传给 3 维 helper 而返回
`INVALID_DIAGNOSTIC`；没有读取或解释任何 block 分数。修正只改为使用 `eta[:3]`，
并新增 6-state row 回归测试。修正后、唯一一次有效重跑前的 SHA256：

```text
analyzer 45f447d911e51090c28712c446df829dd8f7594f8918fd63f8c001898479d58c
test     1424589b8f2dfb0641725743079bd951559f65591bda593e619fc4fb4d193443
```

## 3. Discordant 子群定义

每个 block 的 discordant seeds：

```text
discordant = summary.success and not summary.goal_capture_eta
```

每个 block 至少 5 个 discordant seeds，否则整体 `INVALID_DIAGNOSTIC`。这是
Goal40 已观测到的子群，不用于重新筛 vertical miss，也不降低旧 vertical 阈值。

对每个 seed 取 trace 最后一行，定义：

```text
nav_error_vec = p_nav[:3] - eta[:3]
nav_error_m = norm(nav_error_vec)
pnav_goal_distance_m = goal_distance_pnav_m[-1]
eta_goal_distance_m = goal_distance_eta_m[-1]
distance_gap_m = eta_goal_distance_m - pnav_goal_distance_m
to_goal_vec = goal_ned_m[:3] - eta[:3]
direction_cos = dot(nav_error_vec, to_goal_vec) /
                (norm(nav_error_vec) * norm(to_goal_vec))
```

`p_nav_closer = distance_gap_m > 1e-6`。`toward_goal = direction_cos > 0`。

必须验证 `pnav_goal_distance_m` 与 `norm(p_nav[:3] - goal)`、`eta_goal_distance_m`
与 `norm(eta[:3] - goal)` 在 `1e-6 m` 内一致；否则 invalid。

## 4. 可证伪 hypotheses

每个 H 在四个 block 中分别裁决。四个 block 全部满足为 `SUPPORTED`，一至三个为
`PARTIAL`，零个为 `FALSIFIED`。

### H-NAV-DISTANCE-BIAS：p_nav 系统性比 eta 更接近 goal

每个 block 的 discordant seeds 中：

```text
fraction(p_nav_closer) >= 0.80
```

- 成立：discordance 至少部分来自 terminal nav distance 偏乐观；
- 失败：p_nav 并不系统性更接近 goal，discordance 不能归为简单距离偏置；
- 最便宜证伪：只比较 terminal distance，不加载 policy。

### H-NAV-DIRECTION-BIAS：估计误差方向系统性指向 goal

每个 block 的 discordant seeds 中：

```text
fraction(toward_goal) >= 0.80
```

- 成立：terminal estimator 有 toward-goal 偏差，需继续审计 estimator/protocol；
- 失败：误差方向混合，不能把 discordance 写成统一 nav bias；
- 最便宜证伪：只算 terminal `nav_error_vec` 与 goal 方向余弦。

## 5. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/seed/checkpoint/discordant/距离一致性失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次，不解释机制 |
| 两个 H 都 `SUPPORTED` | `NAV_BIAS_MECHANISM_SUPPORTED` | 预注册 estimator calibration/nav-age 审计；不训练控制器 |
| 仅 `H-NAV-DISTANCE-BIAS=SUPPORTED` | `DISTANCE_METRIC_BIAS_SUPPORTED` | 预注册 protocol metric/termination 审计；不改控制器 |
| 仅 `H-NAV-DIRECTION-BIAS=SUPPORTED` | `DIRECTION_BIAS_SUPPORTED` | 预注册 nav error 方向/外推诊断；不改控制器 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_NAV_BIAS` | 只补最便宜的分组验证，不训练 |
| 全部 `FALSIFIED` | `NO_SUPPORTED_NAV_BIAS` | 停止该 nav-bias 解释，回到分量/控制器问题 |

## 6. 输出

```text
artifacts/lowlevel/goal41_terminal_nav_bias_audit_20261109/
  terminal_nav_bias_diagnostic.json
  terminal_nav_bias_diagnostic.md
  terminal_nav_bias_per_seed.csv
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,
pnav_goal_distance_m,eta_goal_distance_m,distance_gap_m,p_nav_closer,
nav_error_m,direction_cos,toward_goal,current_estimate_age_s
```

## 7. 执行计划

- [x] Step 0：冻结本协议、trace SHA、discordant 阈值和总体映射；
- [x] Step 1：实现 `scripts/audit_pc_goal41_terminal_nav_bias.py`；
- [x] Step 2：为距离一致性、p_nav closer、方向余弦、block verdict 和总体映射写
      unit tests；
- [x] Step 3：运行脚本，生成 JSON/MD/CSV；
- [x] Step 4：核对 40/40 seeds、checkpoint、discordant seed 数和 finite；
- [x] Step 5：写 `doc/goals/goal41_result_20261109.md`，按冻结映射裁决；
- [x] Step 6：更新两个 `research/STATE.md`，追加 `research/LEDGER.jsonl`，必要时
      写 decision；
- [x] Step 7：运行 focused/full pytest、strict ledger checker 和
      `git diff --cached --check`；
- [x] Step 8：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

### 7.1 运行命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python scripts/audit_pc_goal41_terminal_nav_bias.py \
  --formal-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output artifacts/lowlevel/goal41_terminal_nav_bias_audit_20261109
```

验证：

```bash
.venv/bin/python -m pytest -q tests/test_pc_goal41_terminal_nav_bias.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal41 \
  --goal-doc doc/goals/goal41_terminal_nav_bias_audit_20261109.md \
  --strict

git diff --cached --check
```

## 8. Git

禁止 `git add -A`。本 Goal 只 add：

```text
doc/goals/goal41_terminal_nav_bias_audit_20261109.md
doc/goals/goal41_result_20261109.md
scripts/audit_pc_goal41_terminal_nav_bias.py
tests/test_pc_goal41_terminal_nav_bias.py
research/LEDGER.jsonl
research/STATE.md
../research/STATE.md
research/decisions/<若需要>
```

不提交 artifacts、trace、checkpoint、日志或大文件。只有用户明确要求才 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal41：terminal p_nav/eta 偏差方向审计。

先读：
- doc/goals/goal41_terminal_nav_bias_audit_20261109.md
- doc/goals/goal40_result_20261109.md
- doc/goals/goal38_result_20261109.md

硬约束：
1. 只用 Goal38 fresh traces，dev seeds 20-59；
2. 不训练、不新增 rollout、不改控制器、不打开 held-out；
3. 只审计 Goal40 的 protocol-success-without-eta-capture 子群；
4. 阈值和总体映射不得在看到结果后修改；
5. 只有 SUPPORTED 才允许进入下一步审计。
```
