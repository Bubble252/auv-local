# Goal29：终端横向捕获窗口与真实状态捕获

**日期**：2026-11-09<br>
**状态**：`SCREEN_INVALID_CONTROL_CHAIN`<br>
**候选**：只把 ALOS-R 终端对准窗口从 `20 m` 前移到 `30 m`，不改速度、控制器增益、reward、checkpoint 或网络。

> 2026-11-09 screen 追加：1-seed screen 判定为
> `INVALID_CONTROL_CHAIN`。在 C3 D3/G26-P2 中 ALOS guidance 只进入
> `observation.command` 和 reward，不进入 actor 的执行器动作；candidate 的
> `eta/nu/actuator` 与 baseline 逐位一致。详见
> `doc/goals/goal29_screen_result_20261109.md`。本文件保留原预注册内容，不把
> 该 screen 写成 H1 FAIL，也不运行 40-seed formal。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal29 --goal-doc doc/goals/goal29_terminal_horizontal_capture_20261109.md --strict
```

`required: false` 表示这是测量前冻结的预注册 Gate。记录 Gate 前不得修改候选、
阈值、seed、eval split、held-out 或 checkpoint 选择规则。若 Gate 失败并触发
autoresearch，则补 blocker、至少 2 个可证伪 hypothesis、reading note 和 ledger
decision。

## 1. 背景与证据

Goal27 已停止 command residual / supervised residual 主线。Goal28 的只读诊断给出
了更具体的失败机制：

- 现有 `endpoint_stop` 的 protocol success 用 `p_nav` 判定，明显高估 true-state
  `eta` capture；D3 的 `success_protocol/success_eta` 为
  `0.725/0.425`（default）和 `0.500/0.250`（strict125）。
- plane miss 的 median horizontal-normal offset 为 `2.18–2.58 m`，占 miss
  distance 的 `81%–87%`；tangent offset 只有 `0.011–0.013 m`。
- plane miss 的 terminal yaw error 约 `0.75 rad`，而 true success 约
  `0.19–0.31 rad`。
- Goal28 已证伪“终端速度过高/纵向过冲”和“执行器权限不足”：miss 组的 terminal
  speed ratio 和 material saturation 都低于 true success。

现有 ALOS-R 在 `robust_terminal_enabled=true` 时，于 `goal_distance <= alignment_distance`
后直接指向目标；`alignment_distance` 由
`clamp(terminal_alignment_horizon_s * desired_ground_speed, terminal_alignment_min_m,
terminal_alignment_max_m)` 得到。当前 default/strict125 的
`terminal_alignment_min_m == terminal_alignment_max_m == 20.0`，因此只有最后
`20 m` 使用直接对准。

Goal28 trace 的只读分解还显示：对 plane-miss episode，距目标 `30 m` 和 `20 m` 时
的 median horizontal-normal offset 都约 `7.6 m`；到 `10 m` 约 `3.9 m`，到 `5 m`
仍约 `3.7 m`。这段证据不是因果结论，但说明当前 20 m 对准窗口不足以在过平面前
消除横向误差，且最后一个 `5 m` 内横向误差几乎不再收敛。

因此本 Goal 只测试一个机制上最小、可解释的单因子干预：**更早进入直接对准窗口**。

## 2. 研究问题与可证伪假设

### H1：更早进入终端对准窗口能降低真实状态横向 miss

机制：当前 20 m 窗口在存在较大初始横向偏差时，剩余时间不足以在 AUV 航向/速度
响应约束下把真实位置收敛到目标半径；把窗口前移到 30 m 可增加约 8 s 的横向修正
时间，同时保持相同的目标点、速度和控制律。

若成立应看到：

- `success_eta` 提升，plane miss 下降；
- plane-crossing 的 median horizontal-normal `eta` offset 至少下降 `20%`；
- terminal yaw error 至少下降 `15%`；
- `success_protocol`、timeout、safety 不恶化，terminal speed/saturation 不升高。

若失败应看到：

- 横向 offset 和 yaw error 基本不动，或仅 `success_protocol` 上升而
  `success_eta` 不变；
- 提前对准导致路径横切、timeout、safety 或 plane miss 变差；
- terminal speed/saturation 上升，说明收益来自更大权限/速度而非横向几何。

最便宜的证伪：固定 hard-case seed 上的 paired 1-seed screen；不训练。

### H2：收益必须来自终端几何，而不是 success 口径或速度/饱和

机制：如果 30 m 对准窗口真的修复横向捕获，收益应同时出现在 true-state 横向误差
和航向对齐上，而不是只让 `p_nav` 更早进入半径。

若成立应看到：

- `success_eta` 与 terminal horizontal-normal `eta` offset 同方向改善；
- terminal yaw error 改善；
- terminal speed ratio 与 material saturation 不高于 baseline；
- `success_protocol` 不出现“上升但 `success_eta` 不同步”的指标伪改善。

若失败应看到：只有 `success_protocol` 改善，或横向 offset/yaw error 无改善；此时
判为指标伪改善或机制不成立，不能继续调增益/seed。

最便宜的证伪：同一 paired screen 同时报告 protocol/eta success、terminal 横向
offset、yaw error、speed ratio 和 material saturation。

### H3：不是“再加速度/执行器权限”问题

Goal28 已给出负证据：plane miss 的 terminal speed ratio 和 material saturation
都低于 true success。本 Goal 不重新测试速度/braking/权限扫描；若 30 m 对准窗口
失败，也不允许用“提高速度、放宽动作限制、继续 residual PPO”作为补救。

可证伪预测：若 30 m 对准窗口的改善只有伴随更高 speed/saturation 才出现，则 H1
不成立，不能把收益归因于横向捕获。

## 3. 数据、基线与隔离

### 3.1 数据与协议

| 项目 | 值 |
|---|---|
| dev seeds | `20–59`，固定 |
| held-out | `100–149`，本 Goal 完全不打开 |
| 场景 | `default`（goal radius `2.0 m`）与 `strict125`（goal radius `1.25 m`） |
| 协议 | `endpoint_stop`、`realistic_calibrated`、CPU/float64、`path_dim=8` |
| 训练 | 无；不更新任何 checkpoint |
| screen | 固定 stress seed `41`，只用于方向性 debug，不作正式结论 |
| formal | 40 seeds，paired baseline/candidate，完成 screen 后才运行 |

### 3.2 Baseline

| arm | checkpoint | action mode | SHA256 |
|---|---|---|---|
| D3 frozen base | H3-SR final checkpoint（内含冻结 D3 actor） | `frozen-base` | `7549e99edd376c27eef35a5c11c29e1e22888903944f0d2a8cb61bd76212d152` |
| G26-P2 | Goal26 P2 final checkpoint | `policy` | `afcda74ff3171a003fff032a7b7238731802ffb439c034f981419ea6443dd9b8` |

注意：本地 `checkpoints/goal26/d3/pc_bc_actor.pt` 只有 actor，直接喂给
`evaluate_pc_ppo.py` 会报 `checkpoint has no rsl_rl infos mapping`。D3 必须继续用
H3-SR final checkpoint + `--action-mode frozen-base` 复现，不得把 H3-SR residual
当作候选收益。

### 3.3 环境配置

| arm | default | strict125 |
|---|---|---|
| D3 | `configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json` | `configs/lowlevel/pc_goal27_a1_reward_v2_strict125_20261109.json` |
| G26-P2 | `configs/lowlevel/pc_goal26_env_default_p2.json` | `configs/lowlevel/pc_goal26_env_strict125_p2.json` |

上述四个配置的 guidance block 在测量前完全相同。候选只通过 `--guidance-config`
替换 guidance，不修改这些 base 配置。

### 3.4 禁止项

- 不训练、不 warm-start、不改 checkpoint、不改 reward；
- 不打开 held-out `100–149`；
- 不扫 seed、不换 checkpoint、不放宽 goal radius 或 Gate；
- 不以 path RMSE 改善替代 terminal capture；
- 不用 speed/braking/action-limit/residual 作为失败后的补丁。

## 4. 候选与方法

### 4.1 唯一候选：terminal alignment window `20 m -> 30 m`

新增 guidance-only profile：

```text
configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json
```

相对 baseline 只改变两个 guidance 字段：

```yaml
terminal_alignment_min_m: 30.0
terminal_alignment_max_m: 30.0
```

保持不变：

```yaml
terminal_alignment_horizon_s: 6.0
terminal_capture_distance_m: 20.0  # robust terminal 下不用于 capture 判定
terminal_braking_enabled: false
terminal_speed_ratio: 1.0
minimum_speed_ratio: 0.75
current_crab_enabled: true
route_speed_floor_enabled: true
saturation_governor_enabled: false
```

选择 `30 m` 的理由：它比当前窗口多约 `8 s`（按 `1.2–1.5 m/s` 估计），但仍明显
短于全部路径，不会把整个 episode 变成“从起点直线追目标”。不测试 35/40 m，
避免把一个机制问题变成窗口超参数扫描。

### 4.2 与已有方法的关系

这是对 Goal28 已定位的 terminal horizontal capture 机制的最小干预；不改变 ALOS
路径跟踪律、PD/PI 控制器、观测、actor 或 residual 接口。它测试的是“更早切换为
目标对准是否足够”，不是新策略学习。

### 4.3 实现约束

- 只新增 guidance profile 和一个隔离单测，不修改 `guidance.py` 算法；
- 单测必须证明候选与 baseline 的差异只有两个 alignment 字段；
- 单测必须证明 `25 m` 时 baseline 不进入 terminal phase、候选进入，且两者速度
  指令相同；
- 评估必须用相同 checkpoint、相同 seed、相同 base environment，只替换
  `--guidance-config`。

## 5. 执行计划

- [x] Step 0：Run Contract 与只读复核；
      `git status --short`、checkpoint SHA256、base guidance equality、Goal28 trace
      指标；写本文件。验证：`check_research_ledger.py` 对 Research Contract 通过。
- [x] Step 1：创建 `pc_goal29_terminal30_guidance_20261109.json` 与
      `tests/test_pc_goal29_guidance_window.py`；
      运行 `.venv/bin/python -m pytest -q tests/test_pc_goal29_guidance_window.py tests/test_batch_alos_r.py`。
      只 add 本 Goal 的配置、测试和文档。
- [x] Step 2：1-seed paired screen（seed `41`，`default`，D3 与 G26-P2）；
      每个 arm 跑 baseline/candidate，输出到
      `artifacts/lowlevel/goal29_screen_20261109/seed41/{arm}/{baseline,candidate}/`；
      同时写 trace 以提取 terminal 横向 offset/yaw error。
      验证：candidate 配置 log 中只有两个 alignment 字段不同；trace finite；
      无 held-out。
- [x] Step 3：screen 裁决为 `INVALID_CONTROL_CHAIN`：候选 command 改变但
      `eta/nu/actuator` 与 baseline 逐位一致。已写
      `goal29_screen_result_20261109.md`，停止本 Goal，不跑 formal、不改参数重跑。
- [ ] Step 4：未运行。C3 actor 不执行 ALOS guidance，40-seed formal 没有可测候选；
      只有新预注册的 executed-control-chain Goal 才允许重新测试 alignment window。
      输出到 `artifacts/lowlevel/goal29_formal_20261109/`。
      每个 run 记录 summary 与 trace；验证 seeds 均为 `20–59`、40/40 完整。
- [x] Step 5：完成 screen 的 control-chain audit，汇总 `success_protocol`、
      `success_eta`、plane miss、terminal 横向 offset、yaw error、
      speed ratio 和 material saturation；
      输出 `artifacts/lowlevel/goal29_screen_20261109/control_chain_audit/`。
- [x] Step 6：按 §6 Gate 写 `goal29_screen_result_20261109.md`，更新
      `research/STATE.md` 与 `research/LEDGER.jsonl`；运行 checker。
- [ ] Step 7：只 add 本 Goal 相关文件，运行 `git diff --cached --check`；
      不自动 commit / push，除非用户明确要求。

### Step 2 screen 命令模板

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

# D3 frozen-base baseline
.venv/bin/python scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
  --checkpoint /tmp/goal28_evidence/h3sr_final.pt \
  --environment-config configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json \
  --action-mode frozen-base --seed-start 41 --seed-count 1 \
  --device cpu --dtype float64 \
  --output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/baseline \
  --trace-output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/baseline/trace.npz

# D3 candidate
.venv/bin/python scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/pc_goal27_a1_ppo_default_27006.json \
  --checkpoint /tmp/goal28_evidence/h3sr_final.pt \
  --environment-config configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json \
  --guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json \
  --action-mode frozen-base --seed-start 41 --seed-count 1 \
  --device cpu --dtype float64 \
  --output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/candidate \
  --trace-output artifacts/lowlevel/goal29_screen_20261109/seed41/d3/candidate/trace.npz
```

G26-P2 使用同一命令，替换：

```text
--checkpoint /tmp/goal28_evidence/p2_final.pt
--environment-config configs/lowlevel/pc_goal26_env_default_p2.json
--action-mode policy
--training-config configs/lowlevel/pc_goal26_ppo_p2_default.json
```

如果 `/tmp/goal28_evidence/*.pt` 不存在，先从已确认的服务器 Goal27 artifacts 复制到
本机临时目录；不要重新训练，也不要把 checkpoint 提交到 Git。

### Step 4 formal 命令约定

对以下 8 个组合各运行一次 `--seed-start 20 --seed-count 40`：

```text
arm × scenario × mode
d3       default   baseline
d3       default   candidate
d3       strict125 baseline
d3       strict125 candidate
g26_p2   default   baseline
g26_p2   default   candidate
g26_p2   strict125 baseline
g26_p2   strict125 candidate
```

baseline 不加 `--guidance-config`；candidate 加
`--guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json`。
其余参数必须完全一致，且每组都写 summary 与 trace。

## 6. Gate

### 6.1 有效性与隔离

| check | 通过条件 |
|---|---|
| seed | screen 仅 seed `41`；formal 全部 `20–59`，每 run 40 episodes |
| held-out | `100–149` 读取次数为 0 |
| checkpoint | SHA256 与 §3.2 一致；D3 使用 `frozen-base` |
| 配置隔离 | candidate 相对 base 只改变 `terminal_alignment_min_m`/`max_m` 为 `30.0` |
| 协议 | 全部 `endpoint_stop`、`realistic_calibrated`、CPU/float64、path_dim 8 |
| 数值 | finite；无 NaN；outcome 互斥且每个 episode 恰有一个 terminal kind |
| 训练 | 0 次训练、0 次 checkpoint 更新、0 次 warm-start |

### 6.2 Screen Gate（seed 41，仅方向性）

screen 不产生正式性能结论，只决定是否值得跑 40 seeds。

| check | 通过条件 |
|---|---|
| 横向几何 | D3 和 G26-P2 的 candidate terminal horizontal-normal `eta` offset 都不高于 baseline |
| 真实捕获 | candidate 的 `success_eta` 不低于 baseline；优先要求至少一个 arm 从 miss 变 success |
| 口径 | 不出现“只有 `success_protocol` 改善、`success_eta` 不变” |
| 护栏 | timeout/safety 不增加；terminal speed/saturation 不高于 baseline |

若 screen 不通过：写 screen result，停止；不得换 seed、改窗口到 35/40 m 或加
speed/braking/residual 重试。若 screen 因配置/日志/数据链路失败，判 `INVALID`，
只修链路后重跑同一 seed，不改候选。

### 6.3 Formal Gate（40 seeds，测量前冻结）

主指标是 true-state capture，不是 path RMSE。所有判定均以 paired episode 为准。

| check | 通过条件 |
|---|---|
| 主指标 A | 至少 3/4 个 arm×scenario cell 的 `success_eta` 提升 `>= +0.05`，且没有 cell 下降超过 `0.025` |
| 主指标 B | 至少 3/4 cell 的 plane-crossing median horizontal-normal `eta` offset 下降 `>= 20%`，且没有 cell 变差超过 `10%` |
| 航向机制 | 至少 3/4 cell 的 terminal yaw error median 下降 `>= 15%` |
| protocol 护栏 | 每个 cell 的 `success_protocol` 下降不超过 `0.025`；不得用 protocol success 替代 eta |
| 终止护栏 | plane miss、timeout、safety 每个 cell 均不得增加超过 `0.025` |
| 控制护栏 | terminal speed ratio、material saturation 不得相对 baseline 增加 `>= 0.05` |
| 稳定性 | 0 NaN、0 infinite、0 未解释的 `INVALID` episode |
| 复现 | baseline/candidate 除 guidance profile 外，命令、checkpoint、seeds、协议完全一致 |

补充报告但不作为 Gate：pre-goal 3D RMSE、path RMSE、episode time、miss distance、
outcome transition 数。path RMSE 改善不能抵消 terminal Gate 失败。

### 6.4 裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 任一配置/协议/seed 隔离失败 | `INVALID` | 只修数据链路；不改候选、阈值或 seed |
| Screen 不通过 | `SCREEN_FAIL_STOP` | 写结果并停止；不进入 40-seed，不扫窗口 |
| Formal 达到全部主指标与护栏 | `PASS` | 可作为 terminal guidance 的候选配置；仍需新 Goal 做泛化/压力测试 |
| Formal 有部分改善但未达 Gate | `FAIL_PARTIAL` | 保留负结果；进入 autoresearch blocker，先判断机制是否可救，不得换 seed 重跑 |
| Formal 无改善或横向/yaw 不变 | `FAIL_MECHANISM` | 停止 20→30 m 窗口路线；进入 autoresearch，另提至少 2 个可证伪机制假设 |
| plane miss/timeout/safety 显著恶化 | `REJECT_SAFETY` | 立即停止；不得用放宽护栏或增加预算救结果 |
| 只有 protocol success 改善 | `FAIL_METRIC` | 判为口径伪改善；不得写成 true-state capture 收益 |

## 7. Git 与复现

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
python ../scripts/check_research_ledger.py \
  --goal Goal29 \
  --goal-doc doc/goals/goal29_terminal_horizontal_capture_20261109.md --strict
git diff --cached --check
```

预期 add 文件：

```text
doc/goals/goal29_terminal_horizontal_capture_20261109.md
configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json
tests/test_pc_goal29_guidance_window.py
```

完成后才 add result、analysis 脚本（如有）、`research/STATE.md` 和
`research/LEDGER.jsonl`。不要提交 `artifacts/`、`.npz`、checkpoint、日志或大型
数据文件。没有用户明确要求时，不自动 commit / push。

## 8. Goal 提示词

```text
你现在执行 Goal29：终端横向捕获窗口与真实状态捕获。

先读：
- doc/goals/goal29_terminal_horizontal_capture_20261109.md
- doc/goals/goal28_result_20261109.md
- doc/goals/goal28_terminal_capture_plane_miss_20261109.md
- research/STATE.md

硬约束：
1. 只测试 terminal alignment window 20 m -> 30 m 这一个因子；
2. 不训练、不 warm-start、不改 checkpoint、不打开 held-out 100–149；
3. 固定 dev 20–59；screen 只用 seed 41；formal 必须 paired；
4. 同时报告 success_protocol 和 success_eta，不能用 protocol success 代替真实捕获；
5. 不扫 speed/braking/action limit/residual/seed；失败后按 Gate 停止或进入 autoresearch。

执行顺序：
- 先验证配置文件隔离和单测；
- 再跑 seed 41 的 D3/G26-P2 screen；
- screen 通过后才跑 40-seed formal；
- 按 §6 Gate 写 result、ledger、STATE，并运行 checker。
```
