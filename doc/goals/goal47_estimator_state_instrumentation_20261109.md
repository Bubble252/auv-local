# Goal47：realized estimator-state instrumentation 与 OU/event 分解

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / REALIZED_OU_DOMINANT`<br>
**上游**：Goal46 `DYNAMIC_PERSISTENT_WITH_OU_ASSOCIATION`，见
`doc/goals/goal46_result_20261109.md`。<br>
**目的**：在不改变 policy、controller、config、seed、termination 或评估协议的前提下，
用一次确定性的 CPU/float64 instrumentation replay 记录逐 step 的 realized
`nav_OU`、`nav_event`、USBL innovation 和 DVL hold drift，并检验 realized OU
是否解释 Goal46 的 `nav_drift_sigma` association。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal47 --goal-doc doc/goals/goal47_estimator_state_instrumentation_20261109.md --strict
```

这是探索型只读 instrumentation goal。若 replay 或 component decomposition 连续两次失败，
必须按“数据 → 标签 → 评测 → baseline → 架构 → 文献”进入 autoresearch loop；不得把
instrumentation/Gate 失败解释成 controller 需要调参。

## 1. 背景、问题与边界

Goal46 已确认：

- dynamic residual 在四个 block 的五个 `0.9 s` subwindow 中持续 toward goal；
- sampled `nav_drift_sigma` 与 late dynamic projection 在 `4/4` blocks 同向，
  median Spearman `rho=0.8214`，stratified one-sided `p=0.00005`；
- accepted USBL outlier leak 为 0，DVL outages 全是 early-only，因此这两个 exposure
  的 rank association 是 `INSUFFICIENT`，不是 `FALSIFIED`；
- `nav_drift_sigma` 只是 episode sampler 参数，不是 realized `nav_OU(t)`。

本 Goal 只回答：

```text
realized nav_OU(t) 和 nav_event(t) 的 late projection，能否精确重建 Goal45 的
dynamic residual，并解释 Goal46 的 sigma association？
```

不回答“OU 是否通过某个控制器反馈路径造成失败”，也不做反事实 rollout、重训或修复。

## 2. 固定输入、隔离与 provenance

### 2.1 固定实验协议

- dev seeds：`20-59`；四个 block：`default|strict125 × baseline|candidate`；
- held-out `100-149`：不读取、不打开、不用于 checkpoint/seed/threshold 选择；
- sensor profile：`realistic_calibrated`；sensor seed：`81300`；
  `nav_bias_calibration_fraction=0.80`；
- checkpoint SHA256：
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- 只允许一次正式 CPU/float64 instrumentation replay；`device=cpu`，`dtype=float64`；
- replay 的旧字段必须与 Goal42 trace bitwise 相同，success/goal-capture/plane-miss 必须相同；
- default trace 原有字段和未开启新 flag 的行为不得改变。

### 2.2 冻结输入 SHA256

```text
configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
  6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f
configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1
configs/lowlevel/pc_goal27_c1_ppo_default_27001.json
  aa8b46b33b2cec682ccc21ecb2e64755c53ab14af8bf5091bbf65b51926c577f
configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json
  f8222ddad652e361c3f001fd53bea295699a8b53580c6315e6d47c25c9468b12
src/auv_lowlevel/config.py
  6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182
```

```text
formal/default/baseline/trace.npz
  fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8
formal/default/candidate/trace.npz
  dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548
formal/strict125/baseline/trace.npz
  aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387
formal/strict125/candidate/trace.npz
  b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02
```

```text
goal45 fixed_calibration_residual_per_seed.csv
  5edd8e3b13dad9854428114085b9bd71c4f15988b9c56fe4cd3f72c96ed0fabc
goal46 dynamic_residual_association_per_seed.csv
  8ccfb2ca081322987ce3f7527a9a773d103fffce009cfc1a34fa26149d293cb8
goal46 dynamic_residual_association_diagnostic.json
  e847ddab3e5ae68dcad0862d2988a86f74cbddeb40cc769fc06ce2dd040cc71b
```

### 2.3 实现前基线

```text
scripts/evaluate_pc_ppo.py
  380a6792383f6f1b4fab1062ef207350f3aba04c8853f6901ac1b52eff36fca5
src/auv_lowlevel/batch/sensors.py
  c1e6f985ae8bc18191e9baee58db59380bee6d8894e636670659338d2af32932
scripts/audit_pc_goal42_replay.py
  65b15f1d1386f53c27be9186d8ddf2ce6d863ebc43aa9a1e4262c724eca0af7c
```

### 2.4 服务器边界

已验证的 8×RTX 4090 主机（8 张 `NVIDIA GeForce RTX 4090`，384 CPU，`/data`
可用空间在开始前不少于 100 GB）用于 outcome-level corroboration。实际运行放在隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal47_instrumentation_20261109
```

Python：`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`。只通过 SSH key 或
临时 `SSHPASS` 环境变量认证，不把密码写入仓库、日志、命令或 artifact。GPU 不参与
确定性 CPU replay。服务器与 Goal42 reference 的 Intel CPU 不同；Step 3 preflight
发现 26 个旧字段存在 `4.55e-13` 以内的跨架构舍入差，因此 server trace 不能作为
bitwise formal trace，只能作为 outcome-level corroboration。正式 bitwise replay
必须在与 Goal42 reference 相同的本地 CPU 架构上执行；详见
`research/blockers/2026-11-09-goal47-cross-architecture-bitwise.md`。

## 3. 要冻结的 instrumentation 字段

新增字段只在显式 `--trace-estimator-state` 时写入 trace。字段语义是“该 control step
更新后、进入 nav delay queue 前”的 realized estimator state；不得使用 truth position、
current truth 或 future state。

| 字段 | shape | 含义 |
|---|---:|---|
| `nav_ou` | `(N,3)` | realized OU state |
| `nav_event` | `(N,3)` | realized total event offset，含 initial calibration reset |
| `nav_event_usbl` | `(N,3)` | 自 episode 起始累计的 accepted-USBL innovation |
| `nav_event_dvl` | `(N,3)` | 自 episode 起始累计的 DVL hold drift |
| `nav_bias` | `(N,3)` | episode sampled calibration bias |
| `usbl_due` | `(N,)` | 本步 USBL due |
| `usbl_accepted` | `(N,)` | due 且未被 gate 拒绝 |
| `usbl_rejected` | `(N,)` | outlier 且被 gate 拒绝 |
| `usbl_innovation` | `(N,3)` | `raw_usbl - predicted_nav`；非 due 行为 0 |
| `usbl_innovation_applied` | `(N,3)` | accepted 时 `gain * innovation`，否则 0 |
| `usbl_raw` | `(N,3)` | 本步 raw USBL；非 due 保持 last |
| `usbl_predicted` | `(N,3)` | `eta_truth + bias + OU + event`（只写 trace，不进入 actor） |
| `dvl_bottom_outage` | `(N,)` | 本步 bottom hold 是否 active |
| `nav_dvl_outage_delta` | `(N,3)` | 本步加入 event 的 DVL drift delta |

必须满足：

```text
nav_event = -nav_bias_calibration_fraction * nav_bias
            + nav_event_usbl + nav_event_dvl
nav_dynamic_delta = nav_OU + nav_event
                    + nav_bias_calibration_fraction * nav_bias
```

第二条等于 Goal45 的 `late_error - fixed_residual`，其中
`fixed_residual=(1-fraction)*nav_bias`。`usbl_predicted` 只用于 provenance；若它能
改变 observation、action、RNG 或 controller，则整个 replay 判为 `INVALID`。

## 4. 可证伪 hypotheses 与统计定义

先按 Goal44/45 的定义重算每个 seed 的 exact late error：

```text
queue_age_s = min(nav_age_s over episode)
exact_error(t) = p_nav(t) - eta(t - queue_age_s)
late window = [t_end - 5.0 s, t_end - 0.5 s]
u_t = normalize(goal_ned_m - eta_terminal_xyz)
dynamic_projection_m = dot(median(late_error) - 0.2*nav_bias, u_t)
```

realized component 必须在 source row `t - round(queue_age_s/dt)` 读取，与该 trace
row 的 delayed `p_nav` 对齐。`dynamic_projection_m` 与 Goal45 CSV 的差不得超过
`1e-4 m`；late window 至少 100 rows，四个 block 各 40 seeds。

所有 component share 使用 signed projection 除以正的 `dynamic_projection_m`。报告每个
discordant seed 的 OU、USBL、DVL、event share；block 统计量是这些 seed 的中位数。

### H-OU-REALIZED：realized OU 主导 late dynamic projection

对每个 block：

```text
reconstruction_max_abs_error_m <= 1e-6（整个 block 全部 discordant records）
median(OU projection share over discordant seeds) >= 0.50
median(dynamic_projection_m) >= 0.10 m
```

- 成立：realized OU sample path 直接解释 majority of dynamic projection，而不只是 sigma
  参数同向；
- 失败：OU path 不主导，或 instrumentation identity 不能闭合；
- 最便宜证伪：只用 late-window source-aligned `nav_ou + event_delta` 重建 projection。

### H-NAV-EVENT：accepted USBL innovation 主导 late dynamic projection

对每个 block：

```text
accepted USBL updates in episode >= 1
median(USBL cumulative projection share over discordant seeds) >= 0.50
```

- 成立：accepted USBL innovation 是 majority component；
- 失败：USBL cumulative component 不是 majority；
- 最便宜证伪：累计 `usbl_innovation_applied` 后取 late projection share。

### H-DVL-HOLD：DVL bottom outage hold drift 主导 late dynamic projection

对每个 block：

```text
median(DVL cumulative projection share over discordant seeds) >= 0.30
```

- 成立：DVL hold drift 是 material component；
- 失败：DVL component 接近零或只是 minor；
- 最便宜证伪：累计 `nav_dvl_outage_delta` 后取 late projection share。

没有 late outage 时不能把“本次 trace 中零贡献”外推成未来一定不会发生；结果文字必须写成
“observed traces”限制。

### H-SIGMA-PARTIAL：realized OU 解释 Goal46 的 sigma association

在每个 block 的 discordant seeds 中，对 `X=ou_sigma_m`、`Y=dynamic_projection_m`、
`Z=realized OU projection` 做 rank transform，并用 OLS 去掉 `Z` 的线性影响，计算
partial Spearman `rho_sigma_given_ou`。Block pass 条件：

```text
unadjusted rho_sigma_dynamic >= 0.50
partial rho_sigma_given_ou <= 0.30
rho_drop = unadjusted - partial >= 0.20
```

- 成立：realized OU 基本吸收 sampled sigma 的 association；
- 失败：控制 realized OU 后 sigma association 仍强；
- 最便宜证伪：只做 rank residual correlation，不引入新 replay。

### 4.1 裁决映射

```text
SUPPORTED  = 4/4 blocks pass
PARTIAL    = 1-3 blocks pass
FALSIFIED  = 0/4 blocks pass
INSUFFICIENT = H-NAV-EVENT 在所有 block 都没有 accepted USBL update
```

| 条件 | 总体裁决 | 允许下一步 |
|---|---|---|
| 任一 hash/seed/parameter/window/replay identity 失败 | `INVALID_INSTRUMENTATION_REPLAY` | 只修 instrumentation；不解释机制、不训练 |
| H-OU-REALIZED 与 H-SIGMA-PARTIAL 均 `SUPPORTED`，其余不 `SUPPORTED` | `REALIZED_OU_EXPLAINS_SIGMA_ASSOCIATION` | 预注册只读 OU-conditioned 评估或停止当前解释链；不训练 |
| H-OU-REALIZED 与 H-NAV-EVENT 均 `SUPPORTED` | `MIXED_REALIZED_OU_USBL` | 先做可区分两者时间方向的只读审计；不训练 |
| H-OU-REALIZED 与 H-DVL-HOLD 均 `SUPPORTED` | `MIXED_REALIZED_OU_DVL` | 先核实 hold exposure 的 late timing；不训练 |
| 仅 H-OU-REALIZED `SUPPORTED` | `REALIZED_OU_DOMINANT` | 预注册 realization-conditioned 只读 audit；不训练 |
| 仅 H-NAV-EVENT `SUPPORTED` | `REALIZED_USBL_EVENT_DOMINANT` | 预注册 USBL stream 只读审计；不训练 |
| 仅 H-DVL-HOLD `SUPPORTED` | `REALIZED_DVL_HOLD_DOMINANT` | 先做 DVL outage timing 只读审计；不训练 |
| 至少一个 `PARTIAL` 且无 `SUPPORTED` | `PARTIAL_ESTIMATOR_COMPONENT_EVIDENCE` | 只补数据/采样验证；不训练 |
| 全部 `FALSIFIED` 或 `INSUFFICIENT` | `REALIZED_COMPONENTS_NOT_SUPPORTED` | 回到 estimator 记录 schema/时间对齐；不训练 |

任何裁决都不授权训练、controller 修复、参数扫描、新 seed、新 replay 或 held-out。
association、component share 和 reconstruction 都不是 causal identification。

## 5. Replay Gate（看到新 trace 前冻结）

| check | 阈值 |
|---|---|
| block/seed 完整性 | 4 blocks × 40 seeds，0 missing |
| 旧 trace 字段 | Goal42 每个 reference field bitwise equal |
| 控制字段 | `eta/nu/actual_actuator/requested_action/terminal_kind` bitwise equal |
| episode flags | `success/goal_capture_eta/goal_plane_miss` 全部一致 |
| 新增字段 | 全部存在、finite；shape 合法；flags 在 `{0,1}` |
| event identity | 每个 row 的 `nav_event - init - usbl - dvl` 最大误差 `<=1e-9 m` |
| late reconstruction | 全部 discordant records 最大误差 `<=1e-6 m` |
| 样本 | 每 block discordant `>=5`、non-discordant `>=10`、late rows `>=100` |

任一 Gate 失败即 `INVALID_INSTRUMENTATION_REPLAY`；修复只允许改 instrumentation 的
记录/合并逻辑或分析代码，不允许改 policy、controller、config、seed 或 eval split。

## 6. 输出

```text
artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/
  replay_gate.json
  estimator_state_diagnostic.json
  estimator_state_diagnostic.md
  estimator_state_per_seed.csv
  estimator_state_per_block.csv
  formal/{default,strict125}/{baseline,candidate}/
    summary.json
    summary.md
    trace.npz
```

`estimator_state_per_seed.csv` 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,queue_age_s,window_rows,
late_projection_m,dynamic_projection_m,ou_projection_m,event_projection_m,
usbl_projection_m,dvl_projection_m,ou_share,event_share,usbl_share,dvl_share,
reconstruction_error_m,accepted_usbl_count,late_dvl_hold_fraction,ou_sigma_m
```

`artifacts/`、trace、checkpoint、日志不提交。

## 7. 执行计划

- [x] Step 0：冻结本协议、Research Contract、输入/checkpoint SHA256、hypotheses、
      replay Gate 和总映射；在实现前检查 frozen checkpoint 与 server。
      - 运行：本文件审阅 + `sha256sum`；
      - 验证：上述 hash 与固定输入一致；
      - Git：只 `git add doc/goals/goal47_estimator_state_instrumentation_20261109.md`，
        `git diff --cached --check`，用户明确要求时才 commit。
- [x] Step 1：只在 `src/auv_lowlevel/batch/sensors.py` 增加 pure diagnostic state，
      在 `scripts/evaluate_pc_ppo.py` 增加 opt-in `--trace-estimator-state`；不改 observation、
      action、RNG、controller、termination 或默认 trace 字段。
      - 运行：`.venv/bin/python -m pytest -q tests/test_pc_goal47_estimator_state_instrumentation.py`；
      - 验证：字段存在、finite、event identity、disabled 默认路径；
      - Git：`git add src/auv_lowlevel/batch/sensors.py scripts/evaluate_pc_ppo.py tests/test_pc_goal47_estimator_state_instrumentation.py`；
- [x] Step 2：实现 `scripts/audit_pc_goal47_estimator_state.py` 和 focused tests；
      在读取任何新 trace 前把 evaluator/sensors/analyzer/gate tests 的 SHA256 追加到本文件。
      - 运行：focused analyzer tests + Ruff；
      - 验证：identity、window、share、partial-rho 和 verdict 逻辑；
      - Git：只 add 本 Goal 的 analyzer/test/文档。
- [x] Step 3：先在 8×4090 server 的隔离目录同步已冻结代码、checkpoint 和 Goal42 reference，
      做 block × 40 seeds 的机器兼容 preflight（不能只跑 1 seed，因为 sensor sampler RNG
      依赖 40-seed batch 组成）。server 与 reference 架构不同，preflight 已记录
      `NOT_BITWISE_CROSS_ARCH` 及 max abs diff；随后在与 Goal42 reference 相同的本地 CPU
      上执行唯一一次正式四 block replay。
      - 运行：本地 evaluator 与 `scripts/audit_pc_goal42_replay.py`；server 保留 corroboration；
      - 验证：`replay_gate.json= PASS`，四 block 40/40，control/old fields bitwise；
      - 失败：立即停止，不换 config/seed、不重跑 formal。
- [x] Step 4：把 replay 输出拉回本地，运行 analyzer；核对 160 rows、discordant 样本、
      source alignment、reconstruction error、finite 和 component shares。
      - 运行：`scripts/audit_pc_goal47_estimator_state.py`；
      - 验证：`INVALID_INSTRUMENTATION_REPLAY` 不出现；
      - Git：artifacts 不 add。
- [x] Step 5：写 `doc/goals/goal47_result_20261109.md`，按冻结映射裁决；更新两个
      `research/STATE.md`、追加 `research/LEDGER.jsonl`，必要时写 decision。
      - 运行：strict ledger checker；
      - Git：只 add result/state/ledger/decision。
- [x] Step 6：运行 focused/full pytest、Ruff、strict checker、`git diff --cached --check`；
      不自动 commit/push。

## 8. 命令与 Git

实现和本地 focused checks：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal47_estimator_state_instrumentation.py
.venv/bin/python -m pytest -q tests/test_pc_goal47_estimator_state.py
.venv/bin/ruff check src/auv_lowlevel/batch/sensors.py scripts/evaluate_pc_ppo.py \
  scripts/audit_pc_goal47_estimator_state.py \
  tests/test_pc_goal47_estimator_state_instrumentation.py \
  tests/test_pc_goal47_estimator_state.py
```

正式 replay 的唯一冻结命令（在与 Goal42 同架构的本地 CPU 执行；先记录 Step 2 SHA）：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
OUT=artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109
REFERENCE=artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal

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
      --trace-estimator-state \
      $EXTRA
  done
done

.venv/bin/python scripts/audit_pc_goal42_replay.py \
  --formal-root "$OUT/formal" \
  --reference-root "$REFERENCE" \
  --output "$OUT/replay_gate.json"
```

server corroboration 使用相同四个 config/mode，只检查 row count、summary flags 和
数值近邻；其结果标记为 `NOT_BITWISE_CROSS_ARCH`，不进入 analyzer 的 formal root。

analyzer：

```bash
.venv/bin/python scripts/audit_pc_goal47_estimator_state.py \
  --formal-root artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109/formal \
  --goal42-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --goal45-csv artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/fixed_calibration_residual_per_seed.csv \
  --goal46-csv artifacts/lowlevel/goal46_dynamic_residual_association_audit_20261109/dynamic_residual_association_per_seed.csv \
  --output artifacts/lowlevel/goal47_estimator_state_instrumentation_20261109
```

完成检查：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal47 \
  --goal-doc doc/goals/goal47_estimator_state_instrumentation_20261109.md \
  --strict
git diff --cached --check
```

### 7.1 Step 2 冻结 SHA256 与本地 preflight

在读取 Goal47 formal trace 前冻结：

```text
src/auv_lowlevel/batch/sensors.py
  4459e1d33c239dd7122544bf4ab169ac1c365899f4d412917f670c97868f766d
scripts/evaluate_pc_ppo.py
  25ea6d4d093b439110019e64275a80e24757d1aa98a4156ae9c0eee2d1b21975
scripts/audit_pc_goal47_estimator_state.py
  10f5eb34d14101f7c9d34736a82932e4a800b0c7739c6102d21400fad165e19a
tests/test_pc_goal47_estimator_state_instrumentation.py
  acd52cbdf5d2cb0d74a82e046b8129761c4c9d285d30ca3ea5e537c809359e98
tests/test_pc_goal47_estimator_state.py
  609fabfd19310ba7aa3f79afe0fecf3546e5e1da78f96cf728ea80f260c66b5f
scripts/audit_pc_goal42_replay.py
  65b15f1d1386f53c27be9186d8ddf2ce6d863ebc43aa9a1e4262c724eca0af7c
```

Step 2 focused tests：`32 passed`；Ruff：`All checks passed!`。

本地 40-seed preflight（`default/baseline`）与 Goal42 reference 的 row count、全部旧
trace 字段、41 个旧字段的 shape/dtype/bytes、summary flags 均完全一致；新字段全部
finite，event identity 最大误差 `6.66e-16 m`。曾尝试 1-seed preflight，因
`SyntheticSensorSuite` 的 sampler seed offset 依赖 40-seed batch 组成而产生不同
sensor draw；该 1-seed run 不作为兼容证据，也未进入 formal/analysis。

上述 run 只用于实现验证；Goal47 formal 是在同架构本地 CPU 上唯一一次四 block replay，
server 只保留 outcome-level corroboration。

### 7.2 Environment amendment：bitwise formal 回到同架构 CPU

Step 3 server preflight 后，按
`research/blockers/2026-11-09-goal47-cross-architecture-bitwise.md` 的 evidence：

- server 32 threads：26 个旧字段非 bitwise，max abs diff `4.55e-13`；
- server 1 thread：相同 26 个字段仍非 bitwise；
- local 40 seeds / 32 threads：全部旧字段 bitwise PASS；
- local 40 seeds / 1 thread：全部旧字段 bitwise PASS；
- server 与 reference 的 summary flags、row count 完全一致。

因此本 amendment 只改变 formal 执行机器：local matching-architecture CPU 负责 bitwise
Gate 和 analyzer；server 负责 outcome-level corroboration。checkpoint、config、seeds、
late window、hypotheses、thresholds 和 verdict mapping 均未改变。上一行的“server 上唯一
formal replay”由本节明确 supersede，避免把跨架构 trace 冒充 bitwise PASS。

Git 只 add 本 Goal 相关文件，禁止 `git add -A`。若用户明确要求提交，建议分两次：

```bash
git commit -m "feat: instrument estimator state for goal47"
git commit -m "docs: record goal47 estimator-state audit"
git push origin main
```

第二步只在 result/state/ledger 完成且检查通过后执行；没有明确要求时不得 commit/push。

### 7.3 Result-time analysis-code correction

正式 replay 和第一次 full analyzer 输出后，发现分析代码有三个实现错误，均在最终
verdict 前修正，且不改变 checkpoint、config、seed、eval split、held-out、late
window、hypothesis 或任何阈值：

1. **eta shape validator**：初版 analyzer 对 `eta` width 的期望写错；实际
   trace schema 是 `(rows, 6)`。修正只改分析代码的 validator。
2. **componentwise median non-additivity**：初版用
   `median(OU) + median(event)` 对比 `median(dynamic)`。median 不满足可加性，
   部分 seed 会产生约 `1e-3 m` 量级的伪 reconstruction error。修正为在
   source-aligned late window 上使用 component mean projection：

   ```text
   mean(dynamic) = mean(OU) + mean(event)
   mean(event)   = mean(USBL) + mean(DVL)
   ```

   `dynamic_projection_median_m` 仍保留，用于 Goal45 CSV 对齐和冻结的 median
   dynamic 门槛；share 使用 mean projection。
3. **discordant sign validator**：初版错误要求 discordant 的 dynamic
   projection 为正。Goal45 定义是
   `discordant = success and not goal_capture_eta`，且只要求 `late_projection`
   为正；seed 22 baseline 的 late projection 为正、dynamic projection 为负，
   属于 Goal46 已知的负 dynamic records。修正只改 validator 的检查对象。

另外在 per-seed/per-block CSV 中显式输出 `event_identity_error_m` 和
`event_identity_error_max_m`，便于直接复核 `<=1e-9 m` Gate；该字段不参与裁决。

修正后分析代码 SHA256：

```text
scripts/audit_pc_goal47_estimator_state.py
  4c61e4939ac773a9e656af2e16ed9e4a598bc7b3431a58030c88ece22f888048
tests/test_pc_goal47_estimator_state.py
  70544ea9b38409edfa136d62677d526675cafe8bd42fbf92025414631a320bde
```

修正后 full analyzer 只重跑一次；overall verdict、四个 hypothesis verdict 和所有
冻结阈值均未改变。详见 `doc/goals/goal47_result_20261109.md` §7。

## 9. 停止条件

- checkpoint/config/reference/input SHA256 不匹配；
- 新 instrumentation 改变默认 trace 字段、observation、action、RNG、controller、
  termination 或 seed；
- replay 不是四个 block × 40 seeds，或任一旧字段/control/summary 不 bitwise equal；
- 新字段非 finite、flag 非法、event identity 或 late reconstruction 超界；
- 需要改 policy、config、seed、eval split 或 held-out 才能继续；
- server 机器兼容 preflight 在未读取 formal 结果前失败，且无法在预注册的 CPU/float64
  路径内修复。不得静默改成本地结果或换 config 重跑。

## 10. Goal 提示词

```text
你现在执行 Goal47：realized estimator-state instrumentation 与 OU/event 分解。

先读：
- doc/goals/goal47_estimator_state_instrumentation_20261109.md
- doc/goals/goal46_result_20261109.md
- doc/goals/goal45_fixed_calibration_residual_audit_20261109.md

硬约束：
1. 只用 dev seeds 20-59、同一 checkpoint 和四 block 协议；
2. 不训练、不加载新 policy、不改 controller/config/seed/termination；
3. 只增加 opt-in trace instrumentation，旧字段必须 bitwise invariant；
4. 只允许一次正式 CPU/float64 replay；held-out 100-149 保持关闭；
5. threshold、component share 定义和 verdict mapping 在看到新 trace 后不得修改。

执行顺序：
- Step 0 冻结输入与 server provenance；
- Step 1 实现 pure diagnostic fields + focused tests；
- Step 2 实现 analyzer，冻结 SHA；
- Step 3 在 8×4090 server 隔离目录做 1-seed smoke 和唯一正式 replay；
- Step 4 核验 replay Gate、identity、样本和 finite；
- Step 5 写 result、state、ledger，跑 full pytest/Ruff/strict checker；
- 不自动 commit/push，除非用户明确要求。
```
