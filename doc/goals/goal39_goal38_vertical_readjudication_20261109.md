# Goal39：基于 Goal38 一致 provenance 的垂直 miss 机制重裁

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / INVALID_DIAGNOSTIC`<br>
**上游**：Goal38 `H1_FALSIFIED`，见 `doc/goals/goal38_result_20261109.md`。<br>
**结果**：`doc/goals/goal39_result_20261109.md`。<br>
**目的**：用 Goal38 fresh、path-provenance 一致的四个 formal trace，重跑已经冻结的
Goal34/35/36 三个只读分析器，重新裁决全程垂直延续、早期 pitch/交接和初始几何
机制。本 Goal 不训练、不新增 rollout、不改控制器、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal39 --goal-doc doc/goals/goal39_goal38_vertical_readjudication_20261109.md --strict
```

本 Goal 是数据 provenance 修复后的重裁，不是参数搜索。Goal34/35/36 的阈值、
分组规则和总体映射全部保持不变；不得在看到 Goal39 聚合结果后修改。若任一
provenance 或完整性检查失败，整体输出 `INVALID_DIAGNOSTIC`，不解释机制。

## 1. 背景与证据

Goal37 已证明原 Goal31 formal trace 的 start state 跟随
`path_bank[seed % 32]`，但 evaluator 的 goal、goal tangent、route tangent 和
path progress 跟随 `path_bank[0]`。因此：

- Goal34 的 40/30/20 m 垂直延续、相位和路径斜率统计建立在 mixed provenance 上；
- Goal35 的 initial offset、pitch tracking 和 20 m handoff 统计同样受影响；
- Goal36 的 world/reference/start-dynamics 初始几何分解也受影响。

Goal38 修复了 `BatchedPCCommandResidualEnv` 的 stale path alias，并验证：

- 四个 block 的 goal、goal tangent、route tangent、path progress 和 start
  provenance 均为 `40/40`；
- fresh trace 与原始 Goal31 同 seed/step trace 的 `actual_actuator`、`eta`、
  `nu`、`requested_action` 最大差均为 `0.0`；
- `terminal_kind` 完全一致；
- held-out `100-149` 未打开。

因此现在第一次具备重裁 Goal34/35/36 的数据条件。不能把 Goal37 的
`INVALID_DIAGNOSTIC` 直接改写成机制结论，也不能沿用旧的 Goal35/36 数值。

## 2. 数据与隔离

只使用以下四个 fresh formal block：

```text
artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal/
  default/{baseline,candidate}/{summary.json,trace.npz}
  strict125/{baseline,candidate}/{summary.json,trace.npz}
```

- dev seeds：`20-59`，每 block 恰好 40 seeds；
- held-out：`100-149` 不读取、不打开；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- path bank：`path_bank_size=32`、`path_bank_seed=41`；
- environment config 与 Goal38 完全一致；
- 不加载 policy、不 step environment、不训练、不新增 rollout；
- 不修改 seed、config、trace、checkpoint 或 analyzer 阈值。

预注册 trace SHA256：

```text
default/baseline   2ffba9597bf37acc9286f994555fae4c5170e1494103572c31fc939bda4a1528
default/candidate  bb0c8ed0722604cad763d89eac33f93bc0cdac3ac313f9dcdce6672a4a0f1985
strict125/baseline fe33817f9a17c7514decc25870c67190fb649063e9e3e611f977211f3f8d8aed
strict125/candidate 78d0b88eda4d1e8a202562db7d0bff80c2212b432dd6df466f13400a43652cfa
```

若任一 SHA256 改变，本 Goal 输出 `INVALID_DIAGNOSTIC`，不读取聚合裁决。

## 3. 冻结分析器与假设集

三个分析器保持字节不变，预注册 SHA256：

```text
Goal34 analyzer a118d81adb65f02f5eb00b83b1e8084b335b2d9aa0037ef8beff868d0080ae4e
Goal35 analyzer fd6a95c4e7f979c29f2e6f221eef091f40b9614a1ddf307a04502f0a4588b27c
Goal36 analyzer 423bc666093a4c5f03c0835d5e849412b0fe327c77c3e3611db9a15e87e682f4
```

### 3.1 Goal34：全程垂直延续、相位和斜率

继续使用原冻结判据：

- `H-GLOBAL`：40/30/20 m 垂直误差是否延续并排序 terminal vertical error；
- `H-PHASE`：terminal window 内 sign-matched closure 与换向延迟是否解释 miss；
- `H-SLOPE`：path vertical slope 是否与持续垂直偏置同向。

输出裁决只能是原脚本的 `SUPPORTED` / `PARTIAL` / `FALSIFIED` /
`INVALID_DIAGNOSTIC`。

### 3.2 Goal35：早期 initial offset、pitch tracking 和 20 m 交接

继续使用原冻结判据：

- `H-INITIAL`：第一行垂直偏置是否主导 terminal miss；
- `H-TRACK`：reference pitch 方向是否正确、实际 pitch 是否跟踪不足；
- `H-HANDOFF`：20 m 交接是否引入不连续并加剧垂直 miss。

### 3.3 Goal36：初始几何的状态/参考/起始动力学分解

继续使用原冻结判据：

- `H-STATE`：world vertical separation 是否解释/排序初始和 terminal error；
- `H-REFERENCE`：goal/path tangent 投影是否制造并排序垂直偏置；
- `H-START-DYNAMICS`：初始 pitch、pitch rate、vertical velocity 是否有独立排序能力。

## 4. Goal39 可证伪 hypotheses

Goal39 本身只检验“旧裁决是否对 provenance 敏感”，不新增控制器机制。

### H-PROVENANCE-SENSITIVE：至少一个整体裁决改变

若成立应看到：Goal34/35/36 中至少一个 overall verdict 或关键 H verdict 与旧
Goal31 数据上的结果不同，且差异可由 goal/tangent/progress 修复解释。

若失败应看到：三个分析器的 overall verdict 和关键 H 方向全部与旧结果一致。

最便宜证伪：只跑三个冻结分析器，比较 overall 和 per-H verdict。

### H-MECHANISM-SURVIVES：至少一个机制在 fresh provenance 上仍达到 SUPPORTED

若成立应看到：至少一个冻结 H 在 fresh 四个 block 上仍为 `SUPPORTED`，并且其
方向、统计量和分组差满足原阈值。

若失败应看到：所有 H 都是 `FALSIFIED` 或仅 `PARTIAL`，没有可预注册的单因素
控制器候选。

最便宜证伪：先看 Goal34/35/36 的 H verdict，不追加 rollout。

## 5. 总体裁决映射（读取结果前冻结）

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入 SHA、seed、checkpoint 或 analyzer 输出无效 | `INVALID_DIAGNOSTIC` | 修 provenance/输入后只重跑一次，不解释机制 |
| 至少一个 H 为 `SUPPORTED` | `MECHANISM_SIGNAL_FOUND` | 只选最便宜的单因素机制，预注册新的 Goal40 干预 |
| 无 SUPPORTED、至少一个 `PARTIAL` | `PARTIAL_ONLY` | 只允许补最便宜的数据/分段验证，不训练 |
| 全部 `FALSIFIED` | `NO_SUPPORTED_MECHANISM` | 停止该垂直机制线；只能在新用户假设下换问题 |
| 结果与旧分析完全一致 | `PROVENANCE_STABLE` | 保留可信的旧机制结论，但仍按原阈值裁决 |

`PARTIAL` 不得升级为 `SUPPORTED`，`INVALID` 不得写成 `FAIL`。

## 6. 执行计划

- [x] Step 0：冻结本协议、输入 trace SHA、analyzer SHA、阈值和输出目录；
- [x] Step 1：验证四个 trace 的 SHA、40 seeds、checkpoint 和 Goal38 provenance；
- [x] Step 2：运行 Goal34 冻结分析器；
- [x] Step 3：运行 Goal35 冻结分析器；
- [x] Step 4：运行 Goal36 冻结分析器；
- [x] Step 5：核对三个输出的 validation、overall verdict、H verdict 和
      vertical-miss seed 数；
- [x] Step 6：写 `doc/goals/goal39_result_20261109.md`，按冻结映射给出
      `MECHANISM_SIGNAL_FOUND` / `PARTIAL_ONLY` / `NO_SUPPORTED_MECHANISM`；
- [x] Step 7：未出现 SUPPORTED；改为预注册 Goal40 全 seed terminal-component
      诊断，不降低旧垂直 miss 样本阈值；
- [ ] Step 8：更新两个 `research/STATE.md`，追加 `research/LEDGER.jsonl`，
      必要时写 decision；
- [ ] Step 9：运行 focused/full pytest、strict ledger checker 和
      `git diff --cached --check`；
- [ ] Step 10：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

### 6.1 运行命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

OUT=artifacts/lowlevel/goal39_goal38_vertical_readjudication_20261109

.venv/bin/python scripts/analyze_pc_goal34_global_vertical.py \
  --formal-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output "$OUT/goal34"

.venv/bin/python scripts/analyze_pc_goal35_early_vertical.py \
  --formal-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output "$OUT/goal35"

.venv/bin/python scripts/analyze_pc_goal36_initial_geometry.py \
  --formal-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output "$OUT/goal36"
```

验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_pc_goal34_global_vertical.py \
  tests/test_pc_goal35_early_vertical.py \
  tests/test_pc_goal36_initial_geometry.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal39 \
  --goal-doc doc/goals/goal39_goal38_vertical_readjudication_20261109.md \
  --strict

git diff --cached --check
```

## 7. Git

禁止 `git add -A`。本 Goal 只 add：

```text
doc/goals/goal39_goal38_vertical_readjudication_20261109.md
doc/goals/goal39_result_20261109.md
research/LEDGER.jsonl
research/STATE.md
../research/STATE.md
research/decisions/<若需要>
```

不提交任何 `artifacts/`、checkpoint、trace、日志或大文件。

建议提交信息：

```bash
git add doc/goals/goal39_goal38_vertical_readjudication_20261109.md \
  doc/goals/goal39_result_20261109.md \
  research/LEDGER.jsonl research/STATE.md ../research/STATE.md
git diff --cached --check
git commit -m "research: re-adjudicate vertical miss on clean provenance"
git push origin "$(git branch --show-current)"
```

本回合不自动 commit/push；只有用户明确要求才执行。

## 8. 服务器边界

本 Goal 是只读 CPU 分析，允许在本地运行。若后续 Goal40 需要训练或 GPU 评估，
必须使用 8×RTX 4090 服务器的 `oceansim` 环境，先检查 `nvidia-smi` 选择空闲卡，
使用隔离输出目录，不能覆盖旧 checkpoint 或他人任务。服务器密码只通过 shell
环境变量 `SSHPASS` 传递，不写入仓库、日志或命令文件。

## 9. Goal 提示词

```text
你现在执行 Goal39：基于 Goal38 fresh provenance 重裁垂直 miss 机制。

先读：
- doc/goals/goal39_goal38_vertical_readjudication_20261109.md
- doc/goals/goal38_result_20261109.md
- doc/goals/goal34_result_20261109.md
- doc/goals/goal35_result_20261109.md
- doc/goals/goal36_result_20261109.md

硬约束：
1. 只跑 Goal34/35/36 冻结分析器，不改阈值、不训练、不新增 rollout；
2. 输入只用 Goal38 四个 fresh formal block，dev seeds 20-59；
3. held-out 100-149 不打开；
4. 任一 provenance/seed/checkpoint 检查失败即 INVALID_DIAGNOSTIC；
5. 只有 SUPPORTED 才能预注册下一步单因素干预，PARTIAL 不能升级。
```
