# Goal31：保留路径几何的终端收敛控制

**日期**：2026-11-09<br>
**状态**：`FORMAL_FAIL_H1_FALSIFIED`（结果见 `goal31_result_20261109.md`）<br>
**候选**：在同一个 20 m 终端窗口内，把“直接指向 goal”替换为“沿参考路径前视点做纯追踪”；不改窗口、速度、控制器增益、reward、checkpoint 或种子。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal31 --goal-doc doc/goals/goal31_terminal_path_preserving_control_20261109.md --strict
```

`required: true` 表示这是探索型 goal。预注册 Gate 在记录前不得修改；若 Gate 连续
失败、指标不移动或需要外部方法，按 AGENTS 补 blocker、至少两个可证伪 hypothesis、
reading note 和 `research/LEDGER.jsonl` decision。

## 1. 背景与证据

### 1.1 已证伪的路线

- Goal27 已停止 command residual PPO、reward 权重、gamma/lambda、seed 搜索和
  H3-SR supervised residual；不得重开。
- Goal29 是 `INVALID_CONTROL_CHAIN`：在 C3 直执行器策略中，guidance 只进入
  `observation.command` 和 reward，没有改变实际执行器。
- Goal30 在真实执行链上测试终端窗口 `20 m -> 30 m`，40-seed dev 的 default 和
  strict125 都失败：protocol success 下降 `0.125/0.100`，plane miss 上升
  `0.125/0.100`，terminal offset 和 3D/vertical path RMSE 同时变差。

Goal30 的机制审计给出一个重要约束：更大的终端窗口覆盖了约 35 m 路径的 82%，
把大部分路径跟踪替换成直接 goal seeking，导致路径切割和垂直方向误差变差。
因此 Goal31 不再扫 `terminal_alignment_min_m/max_m`，也不通过提前切换窗口来
增加终端修正时间。

### 1.2 当前 baseline 的精确行为

当前 B0 链路为：

```text
ALOS guidance -> zero-command-residual -> REMUS100 PD/PI -> actuator
```

在 `robust_terminal_enabled=true` 且 `goal_distance <= 20 m` 时，batch guidance
会覆盖 path-following 的 heading/pitch：

- `terminal_ground = normalize(goal - position)`；
- `desired_heading` 和 `desired_pitch` 直接指向 goal；
- `reference_position` 被替换成 goal；
- 终端速度比例仍为 1.0，PD/PI 和 actuator 权限不变。

这个分支消除了路径几何，只保留“到 goal 的直线视线”。Goal30 已证明更早进入这个
分支会恶化捕获；但还没有测试“在同一个分支开始时间下，保留路径几何”的候选。

### 1.3 关键指标口径

- `success_protocol`：沿用环境的 goal termination，使用 `p_nav`；
- `success_eta`：曾经进入 true-state 半径的 ever-capture，只作诊断；
- `terminal_eta_capture`：termination 时 `eta_distance <= goal_radius`，是本
  Goal 的主捕获指标；
- `terminal horizontal-normal / vertical eta offset`：termination 时 true-state
  相对 goal 的平面法向/垂直分量；
- `plane_miss`、`timeout`、`safety`、terminal speed ratio、material saturation
  和 path RMSE 是护栏，不允许用它们换主指标。

不得把 `success_eta` 的 ever-capture 当成 terminal capture，也不得用 path RMSE
单独替代终端捕获。

## 2. 研究问题与可证伪假设

### H1：保留路径几何能降低终端横向/垂直偏差

机制：baseline 在最后 20 m 用直线视线指向 goal，忽略了终点前路径的切向和曲率；
在欠驱动、有限转艏/俯仰速率下，直线视线容易在终端产生横切和过冲。候选在同一个
20 m 窗口内改为追踪参考路径上的前视点：

```text
s_target = min(projection_s + lookahead_distance, path_length)
p_target = path.point_at_s(s_target)
terminal_direction = normalize(p_target - position)
```

当 `p_target` 到达路径终点时，它自然退化为 goal；因此候选不需要新的切换距离，
也不通过参数扫描改变终端窗口。路径前视点同时保留 horizontal 和 vertical 几何，
预期降低 terminal horizontal-normal offset 和 vertical offset。

若 H1 成立应看到：

- candidate 的 `actual_actuator`、`eta` 和 terminal offset 与 baseline 不同；
- `terminal_eta_capture` 至少在一个 scenario 提升 `+0.05`，另一个 scenario
  不下降；
- paired terminal horizontal-normal offset 中位数在两个 scenario 都下降至少
  `0.05 m`，且至少一个 scenario 的 bootstrap 95% CI 上界小于 0；
- vertical offset 不恶化超过 `0.05 m`；
- protocol success、plane miss、timeout、safety、speed 和 saturation 不越过护栏。

若 H1 失败应看到：

- 终端前视点改变，但 terminal capture 不变，offset 没有稳定下降；
- 或 path RMSE 改善而 terminal capture/offset 不改善；
- 或 protocol success/plane miss/safety 恶化，说明路径保持换来了不可接受的代价。

最便宜的证伪：seed `41` 的 default/strict125 paired screen；不训练，不打开 held-out。

### H2：失败来自终端执行权限或状态估计，而不是参考点几何

机制：如果 PD/PI 在终点前已经无法提供足够的横向/垂直权限，或者 `p_nav` 与 true
`eta` 的偏差主导了 plane miss，那么只替换终端参考点不会改善 terminal capture。

若 H2 成立应看到：

- actuator 和 `eta` 确实改变，但 terminal offset 与 capture 不变；
- terminal material saturation/rate clipping 已接近上限，或 `p_nav` 与 `eta`
  的 terminal offset 差异大于候选收益。

若 H2 失败应看到：在 saturation 不增加、terminal speed ratio 基本不变的条件下，
terminal capture 或横向/垂直 offset 稳定改善。

最便宜的证伪：同一 seed `41` screen，读取已有 `actual_actuator`、
`material_saturated`、`eta/p_nav` 和 terminal offset；不新增训练。

### H3：默认关闭必须与 Goal30 B0 逐位一致

机制：候选是新的终端控制模式，不能让默认路径产生数值漂移。

若 H3 成立应看到：flag 为 false 时，同一 seed、同一 checkpoint、同一环境的
`command`、`actual_actuator`、`eta`、`nu` 和 `p_nav` 与 Goal30 baseline 逐位一致
（容差 `0`，除浮点序列化外）。

若 H3 失败应看到：默认为 false 仍改变任一执行量；此时不得继续实验，必须先修复
回归。

最便宜的证伪：离线 guidance unit test + seed `41` baseline replay，不跑 formal。

## 3. 数据与隔离

- 训练数据：本 Goal 不训练；只加载现有 C1 checkpoint 满足评估脚本契约。
- checkpoint：`/tmp/goal30_evidence/c1_final.pt`，SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。
- 环境：`configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json` 与
  strict125 对应文件；`command_residual.enabled=true`，评估时使用
  `--action-mode zero-command-residual`。
- default：`goal_radius_m=2.0`；strict125：`goal_radius_m=1.25`。
- dev seeds：`20–59`，40 个 paired episodes；screen 只使用 seed `41`。
- held-out：`100–149` 冻结，本 Goal 不打开。
- 评估：`endpoint_stop`、`realistic_calibrated`、CPU/float64、`path_dim=8`。
- oracle fallback：禁止。
- 禁止：训练、改 checkpoint、改 eval split、改 goal radius、改 success 阈值、
  改 reward、改 gamma/lambda、扫 seed、扫 terminal window、用 validation 调参。

## 4. 方法

### 4.1 单因子候选

新增 `GuidanceConfig.terminal_path_preserving_enabled: bool = False`。默认 false
时完全走 Goal30 baseline 的直接 goal 分支；candidate 只把该字段设为 true。

候选在 terminal 分支内部只改变目标方向的计算：

```text
baseline: terminal_ground = normalize(goal - position)
candidate: terminal_ground = normalize(p_target - position)
```

其中 `p_target` 使用 ALOS 已经计算好的路径前视点，即 `s_target` 由现有
`projection.progress_m`、`max(horizontal_lookahead, vertical_lookahead)` 和路径
总长度决定。候选不新增 gain、不新增距离阈值，也不改变 `terminal_alignment_*`。

为保持数值安全和可解释性：

- `p_target` 与当前位置重合时，回退到 `normalize(goal - position)`；
- 仍使用同一 `terminal_speed_ratio`、`minimum_speed_ratio`、current/crab 补偿；
- 仍返回 `terminal_phase=true`，便于 trace 审计；
- 不改变非 terminal 分支的 heading/pitch/lookahead。

### 4.2 与现有分支的关系

- baseline：`robust_terminal_enabled=true` 且新 flag false，保持直接 goal seeking；
- candidate：`robust_terminal_enabled=true` 且新 flag true，终端使用路径前视点；
- 非 terminal 阶段：两者使用完全相同的 path-following 分支；
- guidance window：两者都使用 `terminal_alignment_min_m=max_m=20.0`。

### 4.3 外部方法依据

使用已有 Goal30 reading note：
`research/paper_reading/2026-11-09-goal30-terminal-capture-control.md`。本 Goal
不引入新论文实现；若 Gate 失败并进入 autoresearch，再按 AGENTS 补 reading note。

## 5. 执行计划

每一步完成后按本节 Git 命令提交对应文件；未得到明确指示前不自动 push。

### Step 0：Run Contract 与静态审计

- [x] 确认 checkpoint SHA256；
- [x] 确认 default/strict125 的 goal radius、`command_residual.enabled=true` 和
  terminal window 都是 20 m；
- [x] 确认新 flag 默认 false；
- [x] 审计现有 terminal branch 确实覆盖 heading/pitch；
- [x] 记录 baseline 的 `success_protocol`、`success_eta`、`terminal_eta_capture`
  口径。

运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
sha256sum /tmp/goal30_evidence/c1_final.pt

.venv/bin/python - <<'PY'
import json
from pathlib import Path
for name in (
    "pc_goal27_c1_command_residual_default_20261109.json",
    "pc_goal27_c1_command_residual_strict125_20261109.json",
):
    data = json.loads(Path("configs/lowlevel", name).read_text())
    assert data["command_residual"]["enabled"] is True
    assert data["guidance"]["terminal_alignment_min_m"] == 20.0
    assert data["guidance"]["terminal_alignment_max_m"] == 20.0
    print(name, data["plant"]["goal_radius_m"])
PY
```

提交文档：

```bash
git add doc/goals/goal31_terminal_path_preserving_control_20261109.md research/STATE.md
git diff --cached --check
git commit -m "goal31: preregister terminal path-preserving control"
git push origin main
```

### Step 1：实现 guidance 与配置

- [x] 新增 `GuidanceConfig.terminal_path_preserving_enabled`，默认 false；
- [x] config parser 接受并校验该字段；
- [x] `BatchedThreeDAlosGuidance` 只在 terminal 分支使用路径前视点；
- [x] 新增 candidate guidance profile；
- [x] 不改变任何已有默认配置。

验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_pc_goal31_terminal_path_preserving.py \
  tests/test_batch_alos_r.py \
  tests/test_lowlevel_pc_command_residual.py
```

提交实现：

```bash
git add src/auv_lowlevel/config.py \
  src/auv_lowlevel/batch/guidance.py \
  configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json \
  tests/test_pc_goal31_terminal_path_preserving.py
git diff --cached --check
git commit -m "goal31: add opt-in terminal path-preserving guidance"
git push origin main
```

### Step 2：离线/回放审计

- [x] 用纯 guidance 测试确认 20 m 窗口和 20.1 m 外行为；
- [x] 确认 default false 与 Goal30 baseline 的命令逐位一致；
- [x] 确认 true 时 terminal target 在路径上，且 `s_target` 不会超过 path end；
- [x] 检查不产生 NaN/Inf，p_target 与 position 重合时有安全回退；
- [x] screen 前不跑 formal。

运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_pc_goal31_terminal_path_preserving.py \
  tests/test_pc_goal30_traces.py \
  tests/test_pc_goal30_failure.py
```

### Step 3：seed 41 screen

- [x] default/strict125 各跑 baseline/candidate；
- [x] 输出 trace 和 summary；
- [x] 检查控制链是否真正改变；
- [x] 检查 terminal capture、offset、path RMSE、speed、saturation、timeout/safety；
- [x] 若命令/执行器逐位一致，裁决 `INVALID_CONTROL_CHAIN`，不跑 formal；
- [x] screen 未出现逐位一致或灾难性安全回归，按预注册跑 formal。

screen 命令：

```bash
CKPT=/tmp/goal30_evidence/c1_final.pt
OUT=artifacts/lowlevel/goal31_terminal_path_preserving_20261109

for scenario in default strict125; do
  ENV=configs/lowlevel/pc_goal27_c1_command_residual_${scenario}_20261109.json
  for mode in baseline candidate; do
    EXTRA=""
    if [ "$mode" = candidate ]; then
      EXTRA="--guidance-config configs/lowlevel/pc_goal31_terminal_path_preserving_20261109.json"
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

.venv/bin/python scripts/analyze_pc_goal30_traces.py \
  --trace "default:baseline=$OUT/screen/seed41/default/baseline" \
  --trace "default:candidate=$OUT/screen/seed41/default/candidate" \
  --trace "strict125:baseline=$OUT/screen/seed41/strict125/baseline" \
  --trace "strict125:candidate=$OUT/screen/seed41/strict125/candidate" \
  --output "$OUT/screen/seed41"
```

### Step 4：40-seed dev formal

- [x] 只有 Step 3 通过后才执行；
- [x] default/strict125 × baseline/candidate，共 4 组、每组 40 seeds；
- [x] 保存 summary 和 trace；
- [x] 按第 6 节 Gate 计算 paired rate、bootstrap CI 和护栏；
- [x] 不打开 held-out。

formal 命令：

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

### Step 5：结果、research 记录与完成检查

- [x] 写 `doc/goals/goal31_result_20261109.md`；
- [x] 若 FAIL/INVALID，写 blocker、至少两个可证伪 hypothesis、reading note、
  `research/LEDGER.jsonl` decision 和 `research/STATE.md`；
- [x] 结果为 FAIL，已按 required autoresearch 写记录；
- [ ] 运行 checker 和 staged diff 检查；
- [ ] 提交本次相关文件；不提交 artifacts、checkpoint、trace、日志、大 PDF。

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal31 \
  --goal-doc doc/goals/goal31_terminal_path_preserving_control_20261109.md \
  --strict
git diff --cached --check
```

## 6. Gate

### 6.1 控制链有效性

- 至少 20/40 个 seed 的 `actual_actuator` 最大差 `> 1e-6`；
- 至少 20/40 个 seed 的 `eta` 最大差 `> 1e-6`；
- 若两者逐位一致，裁决 `INVALID_CONTROL_CHAIN`，不是 H1 FAIL。

### 6.2 主效果

以下三项必须同时满足：

| check | 阈值 |
|---|---:|
| terminal_eta_capture | 至少一个 scenario `delta >= +0.05`，另一个 `delta >= 0.00` |
| terminal horizontal-normal offset | 两个 scenario 的 paired median `<= -0.05 m`；至少一个 scenario 的 bootstrap 95% CI 上界 `< 0.0` |
| terminal vertical offset | 两个 scenario 的 paired median `<= +0.05 m` |

其中 `terminal_eta_capture` 是 termination 时 true-state 捕获，不是 ever-capture。

### 6.3 护栏

以下任一失败即 Gate FAIL：

| check | 阈值 |
|---|---:|
| success_protocol | 两个 scenario 的 paired delta 都 `>= -0.025` |
| plane_miss | 两个 scenario 的 paired delta 都 `<= +0.025` |
| timeout / safety | 不得新增；delta 都 `<= 0.0` |
| terminal speed ratio | paired median delta `<= +0.02` |
| terminal material saturation | paired median delta `<= +0.03` |
| path 3D RMSE | paired median delta `<= +0.10 m` |
| vertical path RMSE | paired median delta `<= +0.10 m` |

允许报告 path RMSE 改善，但不能用 path RMSE 代替 terminal capture。

### 6.4 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 执行器逐位一致 | INVALID_CONTROL_CHAIN | 修控制链接线，不宣称 H1 FAIL |
| capture/offset 无改善但护栏通过 | H1 FALSIFIED | 停止参考点路线，转 H2 的权限/估计诊断 |
| capture 改善但 path/plane/safety 恶化 | 不通过 | 不以终端收益换护栏 |
| 只在 ever-capture 改善 | 不通过 | 保留为诊断，不能用 `success_eta` 判 PASS |
| seed 噪声无法区分 | 不通过 | 进入 autoresearch，按 contract 记录 |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件；提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal31 --goal-doc doc/goals/goal31_terminal_path_preserving_control_20261109.md --strict
git diff --cached --check
```

不要提交：

- `artifacts/`、checkpoint、trace、日志；
- 数据集和大 PDF；
- `settled_SUPERMIMIC_Copy` 的任何改动。

未得到明确指示前不自动 commit / push。

## 8. Goal 提示词

```text
你现在执行 Goal31：在保持 20 m 终端窗口不变的前提下，测试“沿参考路径前视点纯追踪”的终端控制，目标是降低 true-state terminal horizontal/vertical offset 并提高 terminal_eta_capture。

先读：
- doc/goals/goal31_terminal_path_preserving_control_20261109.md
- doc/goals/goal30_result_20261109.md
- research/paper_reading/2026-11-09-goal30-terminal-capture-control.md

硬约束：
1. 不扫 terminal_alignment_min_m/max_m，不改 reward/gamma/lambda/seed/checkpoint。
2. 不改 eval split，不打开 held-out 100-149。
3. 默认 false 必须与 Goal30 B0 逐位一致。
4. 先静态/离线审计，再 seed 41 screen，再 40-seed formal。
5. 主指标使用 terminal_eta_capture；success_eta 只是 ever-capture 诊断。
6. 不要把 path RMSE 当作 terminal capture 的替代。

执行顺序：
- Step 0：Run Contract 和静态审计；
- Step 1：实现 opt-in guidance + 配置 + 单测；
- Step 2：默认关闭等价性与 NaN/越界审计；
- Step 3：seed 41 default/strict125 paired screen；
- Step 4：仅 screen 通过后跑 40-seed dev formal；
- Step 5：Gate、result、research record、checker、按第 7 节提交。
```
