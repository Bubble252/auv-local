# Goal38：命令残差 wrapper 路径 provenance 修复与 Goal31 有效重评

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / H1_FALSIFIED`<br>
**上游**：Goal37 `INVALID_DIAGNOSTIC`，见
`doc/goals/goal37_result_20261109.md` 和
`research/blockers/2026-11-09-goal37-trace-path-provenance.md`。<br>
**结果**：`doc/goals/goal38_result_20261109.md`。<br>
**目的**：修复 `BatchedPCCommandResidualEnv` 缓存旧 `path` 的根因，并用 fresh
rollouts 重跑 Goal31 的四组 dev formal，得到 path-provenance 一致、可判读的 H1
结果。本 Goal 不训练、不改控制器、不扫参数、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal38 --goal-doc doc/goals/goal38_command_residual_path_provenance_repair_20261109.md --strict
```

本 Goal 的修复根因已经由静态代码检查和本地最小复现确认，不是参数搜索。若
provenance、控制不变性或 H1 Gate 失败，先按 blocker 记录，不通过换 seed、放宽
阈值或增加训练预算来救。

## 1. 背景与根因

Goal37 发现 Goal31 的四个 formal block 中，起状态跟随
`path_bank[seed % 32]`，但 trace 的 `goal_ned_m`、`goal_tangent_ned`、
`route_tangent_ned` 和 path progress 跟随 `path_bank[0]`。Goal37 因此裁决
`INVALID_DIAGNOSTIC`，没有裁决控制器机制。

后续只读代码审计和本地最小复现定位到根因：

```text
src/auv_lowlevel/batch/pc_command_residual.py:85
    self.path = env.path
```

`BatchedPCCommandResidualEnv.__init__` 在构造时缓存了当时的 `env.path`，而
`BatchedPCLowLevelEnv.reset()` 会在 reset 后把 `env.path` 重新绑定到
`path_bank[seed % 32]`。wrapper 自己已有 `path` 属性，因此
`__getattr__` 不再动态委托；这使 wrapper 的 `path` 永远停在初始化时的 path 0。

本地复现结果（修复前）：

```text
base._path_index[:8]      = [20, 21, 22, 23, 24, 25, 26, 27]
base.path.goal(seed=20)   = path_bank[20].goal_ned_m
wrapper.path.goal(seed=20)= path_bank[0].goal_ned_m
wrapper.env.path.goal     = path_bank[20].goal_ned_m
```

因此起点和底层的 ALOS/控制链使用 seed 对应路径；evaluator 的 goal、goal
tangent、route tangent、path RMSE、pre-goal mask 和 terminal geometry 读取
wrapper 的 stale path 0。Goal30/31 的 command-residual 评估和 Goal27 的
C1/R1 command-residual 路径指标都可能受影响；H3-SR 使用
`command_residual.enabled=false` 的 action-residual 路径，不受此 wrapper bug
影响。

## 2. 假设与证伪条件

### H-EVAL-ALIAS：stale path 只污染 evaluator 的路径派生量

机制：真实控制由 `BatchedPCLowLevelEnv.step()` 使用动态 `env.path` 执行；
wrapper 的 stale `path` 只被 evaluator 的 `inner.path` 访问。因此修复后
`actual_actuator`、`eta`、`nu`、`requested_action` 和 `terminal_kind` 应与
Goal31 原始 trace 逐 seed、逐 step 一致。

若成立：所有控制字段最大差接近零，只有 goal/route/path 指标变化。<br>
若失败：控制字段出现可分辨差异，说明 stale path 还进入了控制链，必须先定位
第二处路径来源，不能解释 H1。

最便宜证伪：修复 wrapper，跑一个 seed 的 baseline/candidate，与原始 Goal31
同 seed trace 对齐比较；通过后再跑 40 seeds。

### H-H1-RECHECK：原始 Goal31 H1 失败可能是 provenance 伪差

机制：Goal31 的 terminal capture、horizontal/vertical offset 和 path RMSE
都依赖被污染的 `inner.path`；修复后这些量可能改变方向或幅度。因此必须用相同
checkpoint、相同 config、相同 seeds 和相同 Gate 重新裁决 H1。

若成立：修复后的 H1 Gate 通过，或至少出现与原始 FAIL 不同的可分辨机制信号。<br>
若失败：修复后仍无 capture/offset 改善且护栏通过，则 H1 FALSIFIED，停止该
terminal path-preserving 路线。

最便宜证伪：seed `20` 与 seed `41` 的四组 smoke 只检查 provenance 和链路，
不据此判 H1；H1 只由 40-seed dev formal 裁决。

## 3. 数据与隔离

- 只使用 dev seeds `20-59`；
- held-out `100-149` 不读取、不打开、不评估；
- 不训练、不新增 checkpoint、不改变 policy 参数或训练配置；
- checkpoint 固定为 `/tmp/goal30_evidence/c1_final.pt`，SHA256 必须为
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- environment config 固定为 Goal31 实际使用的：

```text
default:
  configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
strict125:
  configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
```

- candidate 只额外使用
  `configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json` 作为
  `--guidance-config`；
- 原始 Goal31 四组 `summary.json`/`trace.npz` 只用于 control-invariance 对照，
  不作为修复后的 H1 证据；
- path bank 固定为 `path_bank_size=32`、`path_bank_seed=41`；
- 修复后的 fresh 输出写入新的隔离目录：

```text
artifacts/lowlevel/goal38_path_provenance_repair_20261109/
  smoke/seed20/...
  formal/{default,strict125}/{baseline,candidate}/{summary.json,trace.npz}
  provenance_audit.json
  goal31_gate_recheck.json
```

`artifacts/` 不提交。

## 4. 方法

### 4.1 最小代码修复

删除 `BatchedPCCommandResidualEnv` 对 `path` 的构造期缓存，使 wrapper 的
`path` 通过 `__getattr__` 动态委托给底层 `BatchedPCLowLevelEnv.path`。不改
`step()`、控制率、action scale、reward、guidance 或训练配置。

在 `tests/test_lowlevel_pc_command_residual.py` 增加回归测试：

1. 用 `path_bank_size > 1` 的 config reset 多个 seed；
2. 断言 `wrapper.path` 与 `wrapper.env.path` 的 goal/tangent 一致；
3. 断言不同 seed 的 path index 对应 `seed % path_bank_size`；
4. 用 `reset_indices` 替换部分行 seed，断言 wrapper 的 path 跟随更新；
5. 断言 wrapper 的 path 修复没有改变零残差 stepwise B0 等价性。

### 4.2 Provenance 审计

新增只读脚本 `scripts/audit_pc_goal38_path_provenance.py`，读取 fresh formal
traces，逐行检查：

- `episode_seed`、`path_bank_size=32`、seed 在 `20-59`；
- `goal_ned_m` 与 `path_bank[seed % 32].goal_ned_m` 的最大距离 `<=1e-8 m`；
- `goal_tangent_ned` 与对应 path endpoint tangent 的最大角度/向量差
  `<=1e-8`；
- 每个 trace 行 `eta` 投影到对应 path 的 `route_tangent_ned` 与记录值一致，
  容差 `1e-6`；`path_progress_m` 与重投影 progress 一致，容差 `1e-6 m`；
- 第一行 `eta` 到对应 path start 的距离不超过 start jitter 上界
  `sqrt(2 * 0.5^2 + 0.5^2) + 1e-6 m`；
- 所有几何量和指标 finite。

同时把 fresh trace 与原始 Goal31 同 scenario/mode/seed/step 的 trace 对齐，
比较 `actual_actuator`、`eta`、`nu`、`requested_action` 和 `terminal_kind`。

控制不变性比较要求两边使用**相同 batch 组成**。`SyntheticSensorSuite` 的
noise seed 依赖 reset 时传入的完整 seed list（`sum(seed * (index + 1))`），
因此单 seed smoke 与原始 40-seed formal 的 `p_nav` 不可直接比较，不能把
batch-size 差异误判成第二处路径 bug。smoke 只检查 provenance；控制不变性只在
修复后的 40-seed formal 与原始 40-seed Goal31 formal 之间检查。

## 5. 执行计划

- [x] Step 0：冻结本协议、checkpoint SHA、seed、config、Gate 和输出目录；
- [x] Step 1：修复 wrapper stale path，增加回归测试；只改
      `src/auv_lowlevel/batch/pc_command_residual.py` 和相关测试；
- [x] Step 2：运行 focused tests：`pytest tests/test_lowlevel_pc_command_residual.py`；
- [x] Step 3：运行 seed `20` 的四组 smoke，生成 fresh trace；
- [x] Step 4：运行 provenance + control-invariance 审计；若任一失败，写 blocker
      并停止，不跑正式；
- [x] Step 5：运行 40-seed dev formal（default/strict125 × baseline/candidate）；
- [x] Step 6：重跑 provenance + control-invariance 审计，确认四个 block 全部
      `40/40`；
- [x] Step 7：用未修改的 `scripts/analyze_pc_goal31_gate.py` 对 fresh formal
      重算 H1 Gate；
- [x] Step 8：写 `doc/goals/goal38_result_20261109.md`，更新 ledger/STATE，
      写 blocker/decision（若失败）；
- [x] Step 9：运行 focused/full pytest、strict ledger checker、
      `git diff --cached --check`；
- [x] Step 10：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

### 5.1 运行命令

修复后先跑 focused tests：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python -m pytest tests/test_lowlevel_pc_command_residual.py -q
```

seed `20` smoke：

```bash
OUT=artifacts/lowlevel/goal38_path_provenance_repair_20261109
for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"
    fi
    .venv/bin/python scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint /tmp/goal30_evidence/c1_final.pt \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 20 --seed-count 1 \
      --device cpu --dtype float64 \
      --output "$OUT/smoke/seed20/$scenario/$mode" \
      --trace-output "$OUT/smoke/seed20/$scenario/$mode/trace.npz" \
      $EXTRA
  done
done
```

smoke provenance 审计：

```bash
.venv/bin/python scripts/audit_pc_goal38_path_provenance.py \
  --trace-root "$OUT/smoke" \
  --reference-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --expected-seed-start 20 --expected-seed-count 1 \
  --skip-control-invariance \
  --output "$OUT/smoke/provenance_audit.json"
```

40-seed dev formal：

```bash
for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"
    fi
    .venv/bin/python scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint /tmp/goal30_evidence/c1_final.pt \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 20 --seed-count 40 \
      --device cpu --dtype float64 \
      --output "$OUT/formal/$scenario/$mode" \
      --trace-output "$OUT/formal/$scenario/$mode/trace.npz" \
      $EXTRA
  done
done

.venv/bin/python scripts/audit_pc_goal38_path_provenance.py \
  --trace-root "$OUT/formal" \
  --reference-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output "$OUT/provenance_audit.json"

.venv/bin/python scripts/analyze_pc_goal31_gate.py \
  --formal-root "$OUT/formal" \
  --output "$OUT"
```

## 6. Gate（测量前冻结）

### 6.1 修复与 provenance Gate

以下全部通过才允许解释 H1：

| check | 阈值 |
|---|---:|
| wrapper 动态 path | focused tests 全通过 |
| 四 block seeds | 每个 block `40/40`，seeds 恰为 `20-59` |
| goal provenance | 每 block `40/40` seeds 匹配 `path_bank[seed % 32]`，误差 `<=1e-8` |
| goal tangent provenance | 每 block `40/40`，误差 `<=1e-8` |
| route tangent provenance | 所有 trace 行匹配对应 path 重投影，误差 `<=1e-6` |
| path progress provenance | 所有 trace 行误差 `<=1e-6 m` |
| start provenance | `40/40` seeds 在 start jitter 上界内 |
| finite | 0 NaN/Inf、0 缺失行、0 seed 重复 |

任一失败：`INVALID_PROVENANCE_REPAIR`，不判 H1。

### 6.2 控制不变性 Gate

对修复前后的相同 `(scenario, mode, seed, step)`：

| 字段 | 最大绝对差 |
|---|---:|
| `actual_actuator` | `<=1e-8` |
| `eta` | `<=1e-8` |
| `nu` | `<=1e-8` |
| `requested_action` | `<=1e-8` |
| `terminal_kind` | 完全相等 |

若任一超过阈值：`UNEXPECTED_CONTROL_CHANGE`，先定位第二处 stale path，不判 H1。

### 6.3 H1 Gate（复制 Goal31 §6，不改阈值）

#### 控制链有效性

- 至少 20/40 个 seed 的 `actual_actuator` 最大差 `>1e-6`；
- 至少 20/40 个 seed 的 `eta` 最大差 `>1e-6`；
- 若两者逐位一致，裁决 `INVALID_CONTROL_CHAIN`，不是 H1 FAIL。

#### 主效果（三项同时满足）

| check | 阈值 |
|---|---:|
| `terminal_eta_capture` | 至少一个 scenario `delta >= +0.05`，另一个 `delta >= 0.00` |
| terminal horizontal-normal offset | 两 scenario paired median `<= -0.05 m`；至少一个 bootstrap 95% CI 上界 `<0.0` |
| terminal vertical offset | 两 scenario paired median `<= +0.05 m` |

#### 护栏（任一失败即 Gate FAIL）

| check | 阈值 |
|---|---:|
| `success_protocol` | 两 scenario paired delta 都 `>= -0.025` |
| `plane_miss` | 两 scenario paired delta 都 `<= +0.025` |
| `timeout` / `safety` | 都 `<= 0.0` |
| terminal speed ratio | paired median delta `<= +0.02` |
| terminal material saturation | paired median delta `<= +0.03` |
| path 3D RMSE | paired median delta `<= +0.10 m` |
| vertical path RMSE | paired median delta `<= +0.10 m` |

#### 裁决映射

| 观察 | 判读 | 下一步 |
|---|---|---|
| provenance/控制不变性失败 | `INVALID_*` | 修数据/代码后只重跑一次，不解释 H1 |
| capture/offset 无改善但护栏通过 | `H1_FALSIFIED` | 停止 terminal path-preserving 路线，不扫 lookahead/window |
| 主效果与护栏全部通过 | `H1_SUPPORTED` | 仍不直接打开 held-out；先预注册 held-out 确认 Goal |
| capture 改善但护栏恶化 | `GATE_FAIL_GUARDRAIL` | 不以终端收益换安全/路径完整性 |
| seed 噪声无法区分 | `AMBIGUOUS` | 按 Research Contract 写 blocker，不换 seed 救 |

## 7. 影响范围与后续记录

- 修复后 fresh 结果只替代 Goal31 formal 的 H1 证据，不追溯修改 Goal31 原
  result；
- 在 Goal27 result 追加说明：使用 command-residual wrapper 的 path 指标需以
  Goal38 fresh 结果为准；H3-SR（`command_residual.enabled=false`）不受影响；
- Goal30 的 command-residual 路径指标也受 wrapper 影响，但其 protocol
  success/plane-miss 由底层环境 termination 决定；除非另有新 Goal，不重开
  Goal30 的窗口扫描；
- 不重开 reward、gamma/lambda、seed、terminal window、pitch gain、lookahead
  或 authority 搜索。

## 8. Git

禁止 `git add -A`。本 Goal 只 add：

```text
doc/goals/goal38_command_residual_path_provenance_repair_20261109.md
doc/goals/goal38_result_20261109.md
scripts/audit_pc_goal38_path_provenance.py
src/auv_lowlevel/batch/pc_command_residual.py
tests/test_lowlevel_pc_command_residual.py
tests/test_pc_goal38_path_provenance.py
research/LEDGER.jsonl
research/STATE.md
../research/STATE.md
research/blockers/<若失败>
research/decisions/<若失败>
```

代码提交建议：

```bash
git add src/auv_lowlevel/batch/pc_command_residual.py \
  tests/test_lowlevel_pc_command_residual.py
git diff --cached --check
git commit -m "fix(pc): delegate command-residual path after reset"
```

实验与文档提交建议：

```bash
git add doc/goals/goal38_command_residual_path_provenance_repair_20261109.md \
  doc/goals/goal38_result_20261109.md \
  scripts/audit_pc_goal38_path_provenance.py \
  tests/test_pc_goal38_path_provenance.py \
  research/LEDGER.jsonl research/STATE.md ../research/STATE.md
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal38 \
  --goal-doc doc/goals/goal38_command_residual_path_provenance_repair_20261109.md \
  --strict
git commit -m "research: repair Goal31 path provenance and re-evaluate H1"
```

推送前确认远端和分支：

```bash
git remote -v
git branch --show-current
git push origin "$(git branch --show-current)"
```

本回合不自动 commit/push；只有用户明确要求才执行提交或推送。

## 9. Goal 提示词

```text
你现在执行 Goal38：修复命令残差 wrapper 的 stale path，并用 fresh rollouts 重评
Goal31 H1。

先读：
- doc/goals/goal38_command_residual_path_provenance_repair_20261109.md
- doc/goals/goal37_result_20261109.md
- research/blockers/2026-11-09-goal37-trace-path-provenance.md

硬约束：
1. 只修 wrapper 的 path 委托，不改控制率、reward、guidance、checkpoint 或训练配置；
2. dev seeds 20-59，held-out 100-149 不打开；
3. 先 smoke + provenance/control-invariance，再跑 40-seed formal；
4. H1 只由 fresh formal 按 Goal31 原阈值裁决；
5. provenance 或控制不变性失败时不解释 H1。
```
