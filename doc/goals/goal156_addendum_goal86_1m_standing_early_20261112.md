# Goal156 Addendum：Goal86 1M checkpoint 的多环境早期诊断

**日期**：2026-11-12<br>
**状态**：`DIAGNOSTIC_COMPLETE / EARLY_PRESSURE_SUPPORTED / NOT_FORMAL / HELDOUT_CLOSED`<br>
**上游**：Goal154 authority、Goal156 5M formal、standing A_screen<br>
**性质**：复用已完成训练的 Goal86 1M T2-SAC checkpoint，提前检查 residual 收益是否能在
current/noise/plant 压力下保留。不训练、不换算法、不打开 held-out，也不替代 Goal156
5M 正式 Gate。

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
  max_outer_loops: 1
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal156 --goal-doc doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md --strict --require-result
```

## 1. 问题

Goal86 的 1M checkpoint 在 default/strict125 上只有边缘正向信号，且 full residual
有 path 代价；Goal154 表明 `near10_2` 部署 gate 能在不增加 path 的情况下保留成功收益。
但现有证据主要来自 nominal 环境，尚不知道收益是否会在 current、noise 和 plant
mismatch 下消失。

## 2. 先验假设

### H-EARLY-A：压力下仍有边缘正收益

若 `near10_2` 相对 zero 在至少 3/4 压力档的 default success 不低于 zero，且总
default 改善方向为正，则 1M 信号不是纯 nominal 偶然。该结果只提高 5M 诊断优先级，
不能写成正式 Gate PASS。

### H-EARLY-B：压力下收益消失

若 current10/noise_n2/plant20 中至少 2 档出现明显下降，或 path/vertical 代价扩大，
则说明 1M residual 仍主要拟合 nominal；5M 正式 Gate 后应优先做压力归因，而不是继续
放大 residual scale。

## 3. 固定设计

| 项目 | 值 |
|---|---|
| checkpoint | Goal86 T2-SAC 1M，seeds `86201–86203` |
| deployment variants | `full / near10_2 / mid_only / far_off_only` |
| baseline | zero-residual，同 A_screen 环境 |
| pressure | `nominal / current10 / noise_n2 / plant20` |
| route/length/speed | `legacy_sine / 35 m / 1.2 m/s` |
| success | default `2.0 m`、strict125 `1.25 m` |
| dev seeds | `20–59`，40 个 |
| precision | CPU / float64 |
| held-out | `100–149` 关闭 |

## 4. 执行

```bash
bash scripts/run_goal86_1m_standing_early.sh
```

输出：`artifacts/lowlevel/goal86_1m_standing_early_20261112/standing_eval/`。

## 5. 结果（2026-11-12，Goal86 1M checkpoint）

104/104 个 eval 成功，CPU/float64，`3 seeds × 40 episodes = 120` per pressure/radius。

### 5.1 pooled success

| variant | default success | Δ vs zero | strict125 success | Δ vs zero | path Δ default | path Δ strict |
|---|---:|---:|---:|---:|---:|---:|
| zero | `294/480` | 0 | `159/480` | 0 | 0 | 0 |
| `near10_2` | `313/480` | **`+19`** | `172/480` | **`+13`** | `-0.0014 m` | `-0.0020 m` |
| `full` | `310/480` | `+16` | `166/480` | `+7` | `+0.2603 m` | `+0.2594 m` |
| `mid_only` | `319/480` | `+25` | `174/480` | `+15` | `-0.0025 m` | `-0.0033 m` |
| `far_off_only` | `320/480` | `+26` | `184/480` | `+25` | `-0.0043 m` | `-0.0054 m` |

### 5.2 压力分层（default）

| pressure | zero | `near10_2` Δ | `full` Δ | `mid_only` Δ | `far_off_only` Δ | `near10_2` path Δ | `near10_2` vertical Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | `81` | `+6` | `+4` | `+7` | `+6` | `-0.0009` | `-0.0056` |
| current10 | `54` | `+2` | `+9` | `+4` | `+8` | `-0.0017` | `-0.0048` |
| noise_n2 | `81` | `+7` | `+4` | `+8` | `+7` | `-0.0010` | `-0.0056` |
| plant20 | `78` | `+4` | `-1` | `+6` | `+5` | `-0.0019` | `-0.0053` |

strict125 下 `near10_2` 为 nominal `+4`、current10 `+0`、noise `+6`、plant `+3`。

### 5.3 结构与边界

- `near10_2` 在 4/4 压力档 default 不低于 zero，strict 为 3/4 正向、current10 持平；
  因此判 `EARLY_PRESSURE_SUPPORTED`，但只是 1M 诊断。
- per-seed default：nominal `+1/+1/+4`、current10 `+1/-1/+2`、noise `+1/+1/+5`、
  plant `+1/+1/+2`。收益仍集中在 seed 86203，current10 也最弱。
- `full` 的 path 代价为 `+0.260 m`；`near10_2` 的 path Δ 为 `-0.0014 m`，说明
  “训练时 full、部署时 gate”能收回 path 代价。
- `near10_2` 的 2–10 m residual RMS 约 `0.890`，>10 m 约 `0.0001`，gate 行为正确。
- **纵向没有修好**：episode-mean vertical 仍约 `1.06 m`，进入 2 m 球时 vertical p95
  default `1.79 m`、strict `1.68 m`；near-goal vertical mean 只改善约 `0.029 m`。
  success 增量来自边缘样本越过球壳，不是纵向捕获整体修复。
- 无 timeout、无 safety；失败仍以 plane miss 为主。

### 5.4 对 5M 的含义

5M 结果出来后优先检查：

1. `current10` 的 default/strict 是否仍明显弱于 nominal；
2. `near10_2` 的 success 增量是否伴随 vertical entry p95 下降；
3. 若 success 过但 vertical 不动，直接进入 terminal vertical credit / 动作参数化，
   不再扫 residual scale；
4. `mid_only` 在 eval-only 1M 上更高，但 Goal155 的 mid-only 训练臂更差，不能用这次
   诊断改写 Goal156 的 `near10_2` 主 arm。

### 5.5 residual 方向审计

对每个 episode 在 active、goal distance `2–10 m` 区间计算
`corr(stern residual ch1, path_vertical_error)`：

| train seed | nominal | current10 | noise_n2 | plant20 | 判读 |
|---|---:|---:|---:|---:|---|
| `86201` | `-0.56 / -0.28` | `-0.46 / -0.25` | `-0.60 / -0.24` | `-0.27 / -0.59` | success/failure 均为负 |
| `86202` | `+0.71 / +0.88` | `+0.22 / +0.86` | `+0.68 / +0.89` | `+0.51 / +0.84` | 强正相关 |
| `86203` | `+0.22 / +0.62` | `+0.31 / +0.43` | `+0.16 / +0.62` | `+0.05 / +0.46` | 弱到中正相关 |

（表中为 default 的 success/failure 中位数；完整 strict125 结果见
`artifacts/lowlevel/goal86_1m_standing_early_20261112/standing_eval/residual_vertical_summary.md`。）

关键观察：

- residual 方向符号跨训练 seed 不一致，不能把“放大 near residual”当成通用修复；
- 失败 episode 的相关性通常比成功更强，而 success/failure 的 `|residual|` 都约
  `0.03–0.04`，说明不是“有幅度但全被压小”，而是动作方向/credit 不稳定；
- 这与此前 Goal155 的跨 seed 符号不一致一致，支持下一步做 terminal vertical
  credit / 动作参数化证伪，而不是继续扫 scale。

## 6. 判读边界

- 这是 1M checkpoint 的早期诊断，不是 Goal156 5M 正式结果；
- 不据此改 Goal156 阈值、seed、checkpoint 或算法；
- 若诊断与 5M 正式结果冲突，以预注册 5M Gate 为准；
- 只用于决定 5M 结果出来后优先分析 nominal、pressure 还是 terminal capture。
