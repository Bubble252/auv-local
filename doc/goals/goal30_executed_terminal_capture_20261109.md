# Goal30：可执行控制链上的终端横向捕获

**日期**：2026-11-09<br>
**状态**：`SCREEN_PASS_CHAIN_VALID_METRIC_CORRECTED`<br>
**候选**：只把 ALOS-R 终端对准窗口从 `20 m` 前移到 `30 m`，不改速度、控制器增益、reward、checkpoint 或网络。

> 2026-11-09 screen 追加：seed `41` 的 default/strict125 paired screen 显示控制链
> 有效：candidate 的 `actual_actuator` 最大变化约 `5.77 rpm`，`eta` 最大变化约
> `1.29 m`，不再出现 Goal29 的逐位一致。screen 同时暴露 `goal_capture_eta`
> 是“曾经进入半径”而不是“终止时仍在半径内”：default candidate 的最小 eta
> 距离为 `1.81 m`，但 termination 距离为 `2.06 m` 且仍是 plane miss。
>
> 因此 formal 主指标改为 `terminal_eta_capture`（termination 时
> `eta_distance <= goal_radius`）。`success_eta`（ever-capture）继续报告，但不再
> 单独作为通过依据。该修正在任何 formal 数据产生之前记录；详见
> `doc/goals/goal30_screen_result_20261109.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal30 --goal-doc doc/goals/goal30_executed_terminal_capture_20261109.md --strict
```

`required: false` 表示这是测量前冻结的预注册 Gate。记录 Gate 前不得修改候选、
阈值、seed、eval split、held-out 或 checkpoint 选择规则。若 Gate 失败并触发
autoresearch，则补 blocker、至少 2 个可证伪 hypothesis、reading note 和 ledger
decision。

## 1. 背景与证据

Goal27 已停止 command residual / supervised residual 主线；Goal28 证明现有
`endpoint_stop` 的 protocol success 用 `p_nav` 判定，明显高估 true-state `eta`
capture，并且 plane miss 主要来自终端 horizontal-normal offset，而不是纵向过冲、
速度过高或 material saturation。

Goal29 在 D3/G26-P2 的 C3 直执行器策略上测试同一 `20 m -> 30 m` 窗口，得到
`INVALID_CONTROL_CHAIN`：candidate 的 ALOS command 最多变化约 `0.86 rad`，但
`eta`、`nu`、`p_nav` 和 `actual_actuator` 与 baseline 逐位一致。原因是 C3 actor
直接输出执行器动作，ALOS guidance 只进入 `observation.command` 和 reward。

因此 Goal30 不再改 C3 observation 层，而是在真正执行 ALOS command 的链路上测试：

```text
ALOS command -> BatchedPCCommandResidualEnv（零残差）
             -> BatchedRemus100Teacher PD/PI
             -> [rudder, stern, rpm]
```

这是一次**控制链切换**，不是 D3/G26-P2 的续跑。Goal30 的 baseline 是 B0：
`ALOS -> PD/PI -> actuator`、零命令残差、可部署控制器；不能把 B0 结果冒充
D3/G26-P2，也不能用 Goal27 的 C1 residual 结果作为本 Goal 的收益。

## 2. 研究问题与可证伪假设

### H1：在真实执行链上，更早进入终端对准窗口能降低 true-state 横向 miss

机制：B0 的 ALOS 在 `goal_distance <= alignment_distance` 后直接指向目标。当前
窗口只有最后 `20 m`；在存在较大横向偏差时，剩余时间不足以让 PD/PI 闭环把真实
位置收敛到目标半径。前移到 `30 m` 应增加横向修正时间，同时保持相同的目标点、
速度指令和控制律。

若成立应看到：

- candidate 的 `actual_actuator` 与 baseline 不同，且 `eta` 轨迹随之改变；
- `success_eta` 提升，或 paired terminal horizontal-normal `eta` offset 下降；
- terminal yaw error 同方向改善；
- `success_protocol`、plane miss、timeout、safety 不恶化；
- terminal speed ratio 和 material saturation 不升高。

若失败应看到：

- actuator 改变但横向 offset 不变或变差；
- 只有 `success_protocol` 变化而 `success_eta` 不变；
- 提前对准导致更大横切、plane miss、timeout 或 safety；
- terminal speed/saturation 上升，说明收益来自更大权限或速度，而不是横向几何。

最便宜的证伪：固定 seed `41` 的 paired screen，default 与 strict125 各一次；不训练。

### H2：终端横向 miss 不能靠固定 PD/PI 的更早对准修复

机制：如果 B0 的 PD/PI 在最后几十米已经没有足够的横向权限或相位裕度，单纯提前
对准点只会改变路径形状，不能降低 terminal horizontal-normal offset。

若成立应看到：正式 40-seed dev 上 `success_eta` 与横向 offset 都没有达到 Gate，
或协议成功上升但 true-state capture 不变。

若失败应看到：横向 offset 与 `success_eta` 在至少一个 scenario 上稳定改善，且
不是由 speed/saturation 或 protocol 口径变化造成。

最便宜的证伪：同一 paired screen；若 screen 已显示 actuator 改变但横向 offset
明显变差，则不再跑 formal。

### H3：控制链审计必须先于效果解释

机制：Goal29 的失败说明“command 变了”不等于“执行器变了”。Goal30 必须在解释
任何收益前证明 candidate 的 `actual_actuator` 和 `eta` 确实被 guidance 改变。

若成立应看到：candidate vs baseline 的 `max_abs_actual_actuator_diff > 0` 且
`max_abs_eta_diff > 0`。

若失败应看到：两者逐位一致；此时裁决为 `INVALID_CONTROL_CHAIN`，不是 H1 FAIL。

最便宜的证伪：seed `41` 的 trace 逐位审计，不跑 formal。

## 3. 数据与隔离

- 训练数据：本 Goal 不训练；只加载现有 C1 checkpoint 以满足评估脚本的模型契约。
- 主 checkpoint：Goal27 C1 `final.pt`，SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。
- 训练配置：`configs/lowlevel/pc_goal27_c1_ppo_default_27001.json`。
- 环境配置：`configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json`
  与 `..._strict125_20261109.json`；两者必须保持 `command_residual.enabled=true`。
- 动作模式：`--action-mode zero-command-residual`。
- screen seeds：`41`，default 与 strict125 各一组 paired baseline/candidate。
- formal seeds：固定 dev `20–59`，default 与 strict125 各一组 paired baseline/candidate。
- held-out：`100–149` 完全冻结，本 Goal 不读取、不评估。
- 评估协议：`endpoint_stop`、`realistic_calibrated`、CPU/float64、`path_dim=8`。
- oracle fallback：禁止。
- 禁止：训练新策略、改 checkpoint、改 eval split、改 goal radius、改 success 阈值、
  扫 alignment window、扫 seed、用 validation 调参后直接报 formal。

## 4. 方法

- baseline B0：C1 training/env + C1 `final.pt` + `zero-command-residual`；
  ALOS guidance 直接进入 PD/PI，零命令残差。
- candidate：同一 baseline，仅额外传
  `--guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json`，
  把 `terminal_alignment_min_m/max_m` 从 `20.0` 改为 `30.0`。
- 除 alignment window 外，candidate 与 baseline 必须使用同一 checkpoint、同一
  environment config、同一 seed、同一传感器配置和同一终止规则。
- `--guidance-config` 在环境 override 校验之后替换 guidance，因此不会改变
  checkpoint 的训练契约；但 candidate 必须保留完整审计信息。
- 外部论文/代码：本 Goal 不引入新方法，不需要新的 reading note；若 Gate 失败并
  进入 autoresearch，再按 AGENTS 补文献调研。

## 5. 执行计划

- [ ] Step 0：Run Contract；确认 C1 checkpoint SHA256、环境配置
  `command_residual.enabled=true`、评估脚本接受 `zero-command-residual`。
- [ ] Step 1：实现只读 trace 分析器 `scripts/analyze_pc_goal30_traces.py` 与对应测试；
  输出 protocol/eta success、plane miss、terminal horizontal-normal offset、yaw
  error、speed ratio、material saturation 和 paired delta。
- [ ] Step 2：seed `41` screen，default 与 strict125 各跑 baseline/candidate；
  做 `eta`、`nu`、`actual_actuator`、`command` 逐位审计。
- [ ] Step 3：screen 通过后跑 40-seed dev formal：default/strict125 ×
  baseline/candidate，共 4 组；每组输出 summary 与 trace。
- [ ] Step 4：按第 6 节 Gate 计算 paired 指标、bootstrap CI 和护栏。
- [ ] Step 5：写 `doc/goals/goal30_result_20261109.md`；若 FAIL/INVALID 或进入
  autoresearch，补 `research/blockers/`、reading note、`research/LEDGER.jsonl`
  的 `decision` 与 `next`；更新 `research/STATE.md`。
- [ ] Step 6：只 add 本次相关文件；运行测试和 ledger checker；`git diff --cached
  --check`；按第 7 节 commit/push 命令执行。未得到明确指示前不自动 push。

### Step 0 运行契约

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

# 若本地没有 C1 final.pt，从服务器复制到 /tmp/goal30_evidence/
mkdir -p /tmp/goal30_evidence
scp -P 43127 \
  yhx@152.136.183.51:/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/artifacts/ppo_c1_27001/final.pt \
  /tmp/goal30_evidence/c1_final.pt

sha256sum /tmp/goal30_evidence/c1_final.pt
# 期望 0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9

.venv/bin/python - <<'PY'
import json
from pathlib import Path
for name in (
    "pc_goal27_c1_command_residual_default_20261109.json",
    "pc_goal27_c1_command_residual_strict125_20261109.json",
):
    config = json.loads(Path("configs/lowlevel", name).read_text())
    assert config["command_residual"]["enabled"] is True, name
    print(name, config["guidance"]["terminal_alignment_min_m"],
          config["guidance"]["terminal_alignment_max_m"])
PY
```

### Step 2 screen 命令

```bash
CKPT=/tmp/goal30_evidence/c1_final.pt
OUT=artifacts/lowlevel/goal30_executed_terminal_capture_20261109

for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json"
    fi
    .venv/bin/python scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint "$CKPT" \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 41 --seed-count 1 \
      --device cpu --dtype float64 \
      --output "$OUT/screen/seed41/$scenario/$mode" \
      --trace-output "$OUT/screen/seed41/$scenario/$mode/trace.npz" \
      $EXTRA
  done
done
```

### Step 3 formal 命令

```bash
for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json"
    fi
    .venv/bin/python scripts/evaluate_pc_ppo.py \
      --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
      --checkpoint "$CKPT" \
      --environment-config "$ENV" \
      --action-mode zero-command-residual \
      --seed-start 20 --seed-count 40 \
      --device cpu --dtype float64 \
      --output "$OUT/formal/$scenario/$mode" \
      --trace-output "$OUT/formal/$scenario/$mode/trace.npz" \
      $EXTRA
  done
done
```

### Step 4 分析命令

```bash
.venv/bin/python scripts/analyze_pc_goal30_traces.py \
  --trace "default:baseline=$OUT/formal/default/baseline" \
  --trace "default:candidate=$OUT/formal/default/candidate" \
  --trace "strict125:baseline=$OUT/formal/strict125/baseline" \
  --trace "strict125:candidate=$OUT/formal/strict125/candidate" \
  --output "$OUT"
```

分析器必须报告：

- `success_protocol`、`success_eta`（ever-capture）与 `terminal_eta_capture`
  （termination 时真实状态在半径内）；三者不能互相替代；
- plane miss、timeout、safety、miss distance；
- terminal horizontal-normal `eta` offset（最后 `2.0 s` 与 termination 状态）；
- terminal yaw error、terminal speed ratio、material saturation；
- 每个 seed 的 paired candidate-baseline delta，以及 10,000 次 seed bootstrap
  的 95% CI；
- path RMSE 只作诊断，不能作为 terminal capture 的通过依据。

## 6. Gate

### 6.1 Screen Gate（seed 41，default + strict125）

| check | 阈值 |
|---|---|
| 控制链有效 | candidate 的 `max_abs_actual_actuator_diff > 1e-9` 且 `max_abs_eta_diff > 1e-9` |
| 数值稳定 | 0 NaN / 0 Inf / 0 flew-out |
| 护栏 | safety、timeout、plane miss 均不高于 baseline |
| 方向 | candidate 不能在两个 scenario 同时把 horizontal-normal offset 恶化 `>0.50 m`，或把 yaw error 恶化 `>0.25 rad` |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| actuator 与 eta 逐位一致 | `INVALID_CONTROL_CHAIN` | 停止 formal；写 result 与 pivot decision |
| chain 有效但两个 scenario 都明显恶化 | H1 screen-falsified | 停止 formal；不扫窗口、不扫 seed |
| chain 有效且无护栏恶化 | screen pass | 跑 40-seed formal |

screen 只用于验证可执行性和排除灾难性方向，不作为统计收益结论。

### 6.2 Formal Gate（40 dev seeds，default + strict125）

| check | 阈值 |
|---|---|
| 主效应 | 两个 scenario 方向一致；至少一个 scenario 满足 `terminal_eta_capture` 提升 `>=0.05`，或 paired median horizontal-normal `eta` offset 下降 `>=0.25 m`（且 `>=10%`），bootstrap 95% CI 不跨零 |
| 协议成功护栏 | `success_protocol` 在任一 scenario 的下降不超过 `0.025` |
| plane miss 护栏 | plane miss 在任一 scenario 的上升不超过 `0.025` |
| 安全护栏 | timeout、safety 在任一 scenario 的上升不超过 `0.025`，且无新增 safety termination |
| 终端速度/饱和 | terminal speed ratio 与 material saturation 的 paired 均值不上升超过 `0.05` |
| 数值稳定 | 0 error / 0 NaN / 0 flew-out |

通过条件：主效应成立，且所有护栏在 default/strict125 都通过。path RMSE、protocol
success 或单次 smoke 不能替代主效应。

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| chain 无效 | `INVALID_CONTROL_CHAIN` | 停止；修控制链需新预注册 |
| `success_eta` 上升但 `terminal_eta_capture` 不变 | `INVALID_METRIC` / 曾经捕获不等于终端捕获 | 不判 PASS；用 terminal 指标重新裁决 |
| 主效应不成立但护栏通过 | `FAIL` / H1 falsified | 停止 alignment window 主线，不调 gain/seed |
| 主效应成立但护栏失败 | `FAIL` | 停止；不得用 protocol 或 safety 换指标 |
| 主效应 CI 跨零且 effect 接近 seed 噪声 | `INCONCLUSIVE` | 进入 autoresearch，不直接加 seed 救结果 |

若同一 Gate 连续失败两次，或修复后指标仍不移动，按 AGENTS 进入 autoresearch：
写 blocker、至少 2 个可证伪 hypothesis、paper reading note、最便宜 falsification，
并在 `research/LEDGER.jsonl` 写 `decision`（`continue` / `pivot` / `stop`）。

## 7. Git

禁止 `git add -A`。只 add 本次相关文件；不提交 `artifacts/`、checkpoint、trace、
日志或大 PDF。提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q \
  tests/test_pc_goal30_traces.py \
  tests/test_pc_goal29_guidance_window.py \
  tests/test_batch_alos_r.py

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal30 \
  --goal-doc doc/goals/goal30_executed_terminal_capture_20261109.md \
  --strict

git diff --cached --check
git status --short
```

建议 commit 信息：

```text
docs(goal30): preregister executed-chain terminal capture screen
```

只 add 文档、分析脚本、测试和 research 记录，例如：

```bash
git add doc/goals/goal30_executed_terminal_capture_20261109.md \
  scripts/analyze_pc_goal30_traces.py \
  tests/test_pc_goal30_traces.py \
  research/STATE.md
git commit -m "docs(goal30): preregister executed-chain terminal capture screen"
# 未得到明确指示前不 push
```

## 8. Goal 提示词

```text
你现在执行 Goal30：在可执行控制链上验证终端横向捕获。

先读：
- doc/goals/goal30_executed_terminal_capture_20261109.md
- doc/goals/goal29_screen_result_20261109.md
- doc/goals/goal28_result_20261109.md
- research/STATE.md

硬约束：
1. baseline 必须是 B0 / zero-command-residual / ALOS -> PD/PI -> actuator；
   不能冒充 D3/G26-P2。
2. 只测 terminal alignment window 20 m -> 30 m 单因子；不改 reward、gamma、
   gain、checkpoint、seed 或网络。
3. 先 seed 41 screen；只有控制链有效且无护栏恶化才跑 40-seed formal。
4. dev 固定 20-59；held-out 100-149 完全冻结。
5. 同时报告 success_protocol、success_eta（ever-capture）与 terminal_eta_capture；
   path RMSE 不能替代 terminal capture。
6. 不提交 artifacts、checkpoint、trace 或日志；不自动 push。

执行顺序：
- 先跑 Step 0 契约检查和 checkpoint SHA256；
- 跑 seed 41 screen 并做 actuator/eta 逐位审计；
- screen 通过后跑 40-seed formal；
- 按 Gate 写 result、ledger、STATE；
- 运行测试、ledger checker 和 git diff --cached --check。
```
