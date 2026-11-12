# Goal42：terminal estimator calibration 与 nav-age 审计

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / RESIDUAL_CALIBRATION_BIAS_SUPPORTED`<br>
**上游**：Goal41 `NAV_BIAS_MECHANISM_SUPPORTED`，见
`doc/goals/goal41_result_20261109.md`。<br>
**目的**：审计 Goal41 的 toward-goal nav bias 是稳定 residual calibration offset，
还是由 terminal navigation age/delay 解释。本 Goal 不训练、不改 policy/controller、
不扫参数、不打开 held-out；只允许一次确定性的 instrumentation replay，用于补齐现有
trace 缺失的 nav-age 字段。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal42 --goal-doc doc/goals/goal42_estimator_calibration_nav_age_audit_20261109.md --strict
```

Goal41 已证明 discordant subgroup 的 `p_nav` 系统性更接近 goal，且 nav-error 方向
在 45/45 个 seeds 上指向 goal。Goal41 没有证明这是 calibration offset、nav age 或
短暂 estimator event。Goal42 只回答这个归因问题。

## 1. 背景与证据

Goal41 的四个 block 中，discordant seeds 的 `p_nav` closer fraction 和 toward-goal
fraction 都是 `1.0000`；median direction cosine 为 `0.5741-0.6673`，median terminal
`current_estimate_age_s` 为 `0.28-0.39 s`。

但 `current_estimate_age_s` 描述 current estimate，不是 `p_nav` 的 navigation age。
Goal38 trace 没有记录 `nav_age_s`，因此不能用它直接审计 navigation delay。Goal42
必须先补齐这个测量字段，并在完全相同的 policy、checkpoint、configs、seeds 和
termination 下验证控制轨迹不变。

## 2. 数据、隔离与 instrumentation replay

固定输入：

- Goal38 fresh formal，dev seeds `20-59`，四个 block 各 40 seeds；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- baseline/candidate configs、guidance overrides 和 endpoint-stop 设置与 Goal38 相同；
- held-out `100-149` 不读取、不打开；
- 不训练、不新增 checkpoint、不改 policy/controller/guidance/reward/termination；
- artifacts 不提交。

允许的唯一新 rollout 是 **instrumentation replay**：在同一 checkpoint、configs、
dev seeds 和 deterministic CPU/float64 下重跑四个 block，只为 trace 增加以下字段：

```text
nav_age_s
usbl_age_s
dvl_bottom_age_s
usbl_valid
usbl_outlier
```

已有字段 `p_nav`、`eta`、`goal_ned_m`、`current_estimate_age_s`、
`current_estimate_valid` 和 `dvl_bottom_valid` 继续保留。不得用 replay 选择
checkpoint、改 seed、改 terminal 规则或替换 Goal38 trace。

### 2.1 Replay 有效性 Gate

按 `(scenario, mode, seed, step)` 对齐 fresh replay 与 Goal38 trace。以下字段必须在
所有 row 上 bitwise identical：

```text
eta
nu
actual_actuator
requested_action
terminal_kind
```

同时要求：

- 每个 block 40/40 seeds，row 数和 step 序列一致；
- 新增字段全部 finite，`nav_age_s >= 0`；
- summary 的 checkpoint、seeds、success、goal_capture_eta 与 Goal38 一致。

任一失败，整体 `INVALID_REPLAY`；修复 instrumentation 后只重跑一次，不解释机制。

实际 replay Gate：`PASS`，`4/4` blocks，无 invalid reason。四个 block 的全部
45 个 Goal38 trace 字段 bitwise identical；`eta`、`nu`、`actual_actuator`、
`requested_action`、`terminal_kind` 的 `max_abs_diff` 均为 `0.0`；新增
`nav_age_s`、`usbl_age_s`、`dvl_bottom_age_s`、`usbl_valid`、`usbl_outlier`
全部 finite，age 非负，USBL flags 在 `[0, 1]`。

## 3. Terminal 定义

对每个 seed 取 trace 最后一行 `i_end`。discordant subgroup 仍为：

```text
discordant = summary.success and not summary.goal_capture_eta
```

定义 terminal goal direction：

```text
u_t = normalize(goal_ned_m[i_end] - eta[i_end, :3])
```

nav-age 修正使用 terminal `age = nav_age_s[i_end]`，在 true trajectory 上对
`t_end - age` 的 `eta[:3]` 做 linear interpolation，得到 `eta_lag`。要求
`0 <= age <= time_s[i_end]`，否则该 seed invalid。

逐 seed 计算：

```text
raw_error_vec          = p_nav[i_end, :3] - eta[i_end, :3]
delay_vec              = eta_lag - eta[i_end, :3]
residual_error_vec     = p_nav[i_end, :3] - eta_lag
raw_projection_m       = dot(raw_error_vec, u_t)
delay_projection_m     = dot(delay_vec, u_t)
residual_projection_m  = dot(residual_error_vec, u_t)
```

`raw_projection_m > 1e-6` 表示原始 terminal error 指向 goal；
`residual_projection_m > 1e-6` 表示即使按记录的 nav age 回看 true state，误差仍指向
goal。`delay_projection_m` 仅作为描述量，不要求固定符号。

## 4. 可证伪 hypotheses

每个 H 在四个 block 中分别裁决。四个 block 全部满足为 `SUPPORTED`，一至三个为
`PARTIAL`，零个为 `FALSIFIED`。

### H-NAV-AGE-ASSOCIATION：discordant seeds 的 terminal nav age 更高

每个 block 中分别取 discordant 与 non-discordant seeds 的 terminal `nav_age_s`：

```text
median(nav_age_s | discordant) - median(nav_age_s | non-discordant) >= 0.10 s
```

并用固定 `bootstrap_seed=20261109` 的 5000 次 group bootstrap 计算该 median
difference 的 95% CI；
要求 CI 下界 `> 0`。每个 block 至少 5 个 discordant 和 10 个 non-discordant seeds，
否则 invalid。

- 成立：terminal nav age 与 discordance 同向，但不能单独证明因果；
- 失败：不能把 Goal41 的 bias 归为更高 nav age；
- 最便宜证伪：只用 terminal age 分组比较，不加载 policy。

### H-RESIDUAL-CALIBRATION-BIAS：age 修正后 toward-goal bias 仍存在

每个 block 的 discordant seeds 中：

```text
fraction(residual_projection_m > 1e-6) >= 0.80
```

- 成立：记录到的 nav age 不能解释 Goal41 的 toward-goal bias，支持 residual
  estimator/calibration offset；
- 失败：age 修正后残留方向混合，不能把 bias 归为稳定 calibration；
- 最便宜证伪：只算 terminal `p_nav - eta(t-age)` 的 goal projection。

## 5. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| replay Gate 或任一输入/样本/age 检查失败 | `INVALID_REPLAY` | 修 instrumentation 后只重跑一次，不解释机制 |
| 两个 H 都 `SUPPORTED` | `AGE_ASSOCIATED_WITH_RESIDUAL_BIAS` | 预注册只读 calibration/estimator 单因素诊断；不训练 |
| 仅 `H-RESIDUAL-CALIBRATION-BIAS=SUPPORTED` | `RESIDUAL_CALIBRATION_BIAS_SUPPORTED` | 预注册只读 estimator bias 审计；不改控制器 |
| 仅 `H-NAV-AGE-ASSOCIATION=SUPPORTED` | `NAV_AGE_ASSOCIATION_ONLY` | 只补最便宜的 age 分组验证，不训练 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_AGE_BIAS` | 只补数据/分组验证，不训练 |
| 全部 `FALSIFIED` | `NO_SINGLE_SOURCE_SUPPORTED` | 停止 nav-age/calibration 单因素解释，回到 estimator 结构问题 |

`SUPPORTED` 只允许进入下一项 read-only 审计，不授权训练、controller 修复、参数扫描
或 held-out。

## 6. 输出

```text
artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/
  replay_gate.json
  terminal_age_bias_diagnostic.json
  terminal_age_bias_diagnostic.md
  terminal_age_bias_per_seed.csv
  formal/{default,strict125}/{baseline,candidate}/trace.npz
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,
nav_age_s,usbl_age_s,dvl_bottom_age_s,
raw_projection_m,delay_projection_m,residual_projection_m,
raw_toward_goal,residual_toward_goal,
current_estimate_age_s
```

## 7. 执行计划

- [x] Step 0：冻结本协议、replay Gate、hypotheses、样本门槛和总体映射；
- [x] Step 1：为 evaluator 增加可选 trace instrumentation，并写 replay equivalence
      tests；验证缺省输出与 Goal38 兼容；
- [x] Step 2：按 §2 重跑四个 block，生成 `replay_gate.json`；Gate 失败立即停止；
- [x] Step 3：实现 `scripts/audit_pc_goal42_estimator_calibration_nav_age.py` 和
      unit tests，覆盖 age interpolation、projection、block verdict 和总体映射；
- [x] Step 4：读取聚合前记录 analyzer/test SHA256；
- [x] Step 5：运行 analyzer，生成 JSON/MD/CSV，核对 40/40 seeds、样本门槛和 finite；
      `terminal_age_bias_diagnostic.{json,md}` 与 160-row per-seed CSV 已生成，45 个
      discordant seeds 全部 finite；
- [x] Step 6：写 `doc/goals/goal42_result_20261109.md`，按冻结映射裁决；
      `H-NAV-AGE-ASSOCIATION=FALSIFIED`、`H-RESIDUAL-CALIBRATION-BIAS=SUPPORTED`，
      总体为 `RESIDUAL_CALIBRATION_BIAS_SUPPORTED`；
- [x] Step 7：更新两个 `research/STATE.md`，追加 `research/LEDGER.jsonl`，必要时写
      decision；
- [x] Step 8：运行 focused/full pytest、Ruff、strict ledger checker 和
      `git diff --cached --check`；
      最终 full suite `1035 passed`；Ruff、strict ledger 和 whitespace check 均通过；
- [x] Step 9：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

每一步的 Git 规则：

```bash
git add doc/goals/goal42_estimator_calibration_nav_age_audit_20261109.md
git diff --cached --check
# 仅当用户明确要求时：
git commit -m "docs(goal42): pre-register estimator nav-age audit"
git push
```

Step 1/3 的代码提交只包含 evaluator instrumentation、analyzer、tests 和文档；不得
加入 replay trace、artifact、checkpoint 或日志。Step 5 的 commit 只包含 result、
ledger、state、decision 和代码/文档，不包含 `artifacts/`。

### 7.1 运行命令

Step 1 冻结的 instrumentation/replay-gate SHA256：

```text
evaluate_pc_ppo.py               380a6792383f6f1b4fab1062ef207350f3aba04c8853f6901ac1b52eff36fca5
audit_pc_goal42_replay.py        cccc825cc7df7fc660342d8487c80bb02dd7c9c8832082c07010abc2a6bf1053
replay instrumentation test      981e67126583b4562cd418d3ab4b06142a2cd59c9e95acd6fd4c023f024cf51f
replay gate test                 8210759ec1236fdc6012fbce748faf74e0748b2237483575195eacc58fca4dd1
```

Step 2 replay 已按上述 evaluator SHA 完成。首次 gate 调用因 auditor 读取
`args.replay_root` 而 argparse 只提供 `--formal-root`，在读取任何 trace 比较结果前
退出；没有重新 rollout，也没有改 evaluator。修正只改为使用 `args.formal_root` 并
增加 CLI 回归测试。gate 重跑前修正 SHA256：

```text
audit_pc_goal42_replay.py        65b15f1d1386f53c27be9186d8ddf2ce6d863ebc43aa9a1e4262c724eca0af7c
replay gate test                 c7392866d72e45b6372463a1de6a2e2884d878f005e61828c551ccc790cd9a5b
```

Step 3/4 冻结的 analyzer/test SHA256：

```text
audit_pc_goal42_estimator_calibration_nav_age.py
  99f9f44e9ec97ec80a202e1e3201e036328bb4387e5a5541433e78dfc1294bc1
test_pc_goal42_estimator_calibration_nav_age.py
  44e70df448b6c673876fcecc34b6ab0a31465d8d2ebcab306c6254d6fac534a0
```

Step 1 先运行 focused tests：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal42_replay_instrumentation.py
.venv/bin/ruff check scripts/evaluate_pc_ppo.py tests/test_pc_goal42_replay_instrumentation.py
```

Step 2 的正式 replay 命令必须复用 Goal38 已有的 evaluation runner/config；在执行前
把实际命令、输入/输出目录和 checkpoint 写入本协议补充段，禁止临时改 seed 或 config。

Step 2 唯一一次 replay（在读取任何新 trace 前冻结）：

```bash
OUT=artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109
REFERENCE=artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal

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
      --trace-sensor-ages \
      $EXTRA
  done
done

.venv/bin/python scripts/audit_pc_goal42_replay.py \
  --formal-root "$OUT/formal" \
  --reference-root "$REFERENCE" \
  --output "$OUT/replay_gate.json"
```

Step 5：

```bash
.venv/bin/python scripts/audit_pc_goal42_estimator_calibration_nav_age.py \
  --formal-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --reference-root artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal \
  --output artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109

.venv/bin/python -m pytest -q tests/test_pc_goal42_estimator_calibration_nav_age.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal42 \
  --goal-doc doc/goals/goal42_estimator_calibration_nav_age_audit_20261109.md \
  --strict

git diff --cached --check
```

## 8. 停止条件

出现以下任一情况立即停止，不得转成 controller FAIL：

- replay control invariant 失败；
- 新增 nav-age 字段不是 finite，或 `nav_age_s` 为负；
- 任一 block discordant `< 5` 或 non-discordant `< 10`；
- age interpolation 越界；
- 需要改 policy/controller/config/seed 才能继续。

## 9. 服务器边界

Step 1/3/5 是 CPU/read-only 分析，本地执行。仅在 Step 2 需要正式 CPU/float64 replay
且本地吞吐不足时，才使用 8×RTX 4090 服务器；使用前重新检查 `nvidia-smi` 和隔离目录，
密码只通过 `SSHPASS` 传递，不写入仓库、日志或命令历史文件。

## 10. Goal 提示词

```text
你现在执行 Goal42：terminal estimator calibration 与 nav-age 审计。

先读：
- doc/goals/goal42_estimator_calibration_nav_age_audit_20261109.md
- doc/goals/goal41_result_20261109.md
- doc/goals/goal40_result_20261109.md

硬约束：
1. 只用 Goal38 dev seeds 20-59，同一 checkpoint；
2. 不训练、不改 policy/controller/config/termination，不扫参数；
3. 只允许一次 instrumentation replay，且必须先通过 bitwise control invariance Gate；
4. 阈值、样本门槛和总体映射不得在看到结果后修改；
5. 不打开 held-out 100-149；只有 SUPPORTED 才允许下一项 read-only 审计。
```
