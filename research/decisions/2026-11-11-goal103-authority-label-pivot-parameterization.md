# Goal103 decision：authority 标签不等于 causal authority，转向通道级参数化审计

**日期**：2026-11-11<br>
**状态**：`CONTROL_AUTHORITY_LIMITED / PARAMETERIZATION_NOT_CAUSALLY_ISOLATED / PIVOT_ACTION_PARAMETERIZATION / NO_SAC`

## 1. 触发

Goal102 已判定 PPO 剂量从最小步数起伤害 BC 先验。Goal103 对 B20 strict 的 20 个
失败 seed 做 terminal-capture 审计，检查：

- goal-radius 邻域和 plane-miss 几何；
- terminal gate 与 residual clip；
- p_nav/eta 分叉；
- 网络接线与冻结 BC parity。

## 2. 结果

### 2.1 预注册分类触发

```text
shell failures     = 6/20
authority failures = 14/20
nav/eta failures   = 3/20
classification     = CONTROL_AUTHORITY_LIMITED
```

反事实 success：

```text
radius 1.25 -> 20/40
radius 1.50 -> 26/40
radius 1.75 -> 30/40
radius 2.00 -> 33/40
```

### 2.2 但 authority 不是因果判别器

| outcome | magnitude clip median | rate clip median |
|---|---:|---:|
| failure | 0.746 | 0.500 |
| success | 0.872 | 0.613 |

成功集的 clip 更高。失败/成功都由大幅值 BC residual 驱动，区别主要是方向与终端
几何，而不是“有没有 authority”。

`material_saturation_rate=0` 是诊断盲区：wrapper 在送入环境前已经 clamp，环境侧
`material_saturated` 看不到这次 clamp。不能据此说“没有物理饱和”。

### 2.3 终端几何

20 个失败中：

- 13/20 vertical miss 大于 horizontal；
- vertical median `1.485 m`，horizontal median `1.043 m`；
- radial excess min/median/p90/max = `0.015/0.534/2.918/4.261 m`；
- 只有 6 个在壳层 `+0.25 m` 内。

因此既不是单纯“差一点半径”，也不是单一水平误差。

### 2.4 网络结构

- `pc_policy.py` SHA 与 Goal89 parity 一致；
- actor `131→256→128→3`、critic `131→256→128→1`；
- state encoder `20→128→128→60`、path `12→64→64`、
  endpoint `4→32→64`、current `3→64→60`；
- `d_z_total=64`、`n_z_reserved=4`；
- DADE off、无 NoisyNet；
- Goal95 两臂各 22 个 frozen-BC 张量与 Goal86 `max_abs_diff=0.0`。

```text
network_wiring = ok
checkpoint_contract = ok
noisenet = not involved
```

### 2.5 动作包线

BC label 的 residual scale 是 `1.0`，Goal95 训练是 `1.5`，B20 评估是 `2.0`。
在最后 3 m，失败集归一化 stern 输出：

```text
median = 0.391
p95    = 1.516
abs>0.5 = 38.4%
abs>1.0 = 11.5%
```

全 strict run 的物理 residual delta：

```text
rudder mean abs = 0.354 rad  (limit 0.262 rad)
stern  mean abs = 0.365 rad  (limit 0.349 rad)
rpm    mean abs = 249 rpm
```

所以 B20 已把 BC 残差放大到标签包线之外，执行动作大量贴边。

## 3. 判读

1. 不是网络接线/容量问题：22/22 冻结张量完全一致，结构无错位。
2. 不是 NoisyNet：deterministic mean，无 NoisyNet。
3. 不是单纯 shell 问题：只有 6/20 在 `+0.25 m` 内。
4. 不是单纯 p_nav/eta：3/20 分叉，不是主因。
5. `CONTROL_AUTHORITY_LIMITED` 是预注册标签，但不能直接解释为“限幅导致失败”；
   成功集同样甚至更贴边。
6. 当前最可检验的机制是 **大幅值、逐通道独立、非保方向的 residual 参数化**：
   scale 2.0 让 BC 输出大面积饱和，success/failure 由方向而非幅值区分。

## 4. 决策

`decision = pivot`

不训练新算法，不换 SAC/REDQ/DroQ，不扩大网络，不改 reward/terminal/success/
goal radius，不打开 held-out。

下一 Goal 只做 **eval-only 通道级动作参数化审计**：

- 固定 B20 checkpoint、Goal85 unbiased 环境、dev seeds `20–59`；
- 分别调整 rudder/stern/rpm 通道的 effective scale；
- 或比较逐通道 clamp 与 joint norm/方向保持投影；
- 每个候选必须同时报告 default、strict125、path、timeout、safety；
- 任何候选只有在 default 不低于 33、strict 至少 +1、path 不恶化超过 5% 时才有
  资格进入后续训练；
- 若通道参数化无效应，再进入 reward/terminal capture audit。

## 5. 证据

- `doc/goals/goal103_b20_terminal_capture_audit_20261111.md`
- `doc/goals/goal103_result_20261111.md`
- `research/blockers/2026-11-11-goal103-authority-label-not-causal.md`
- `artifacts/lowlevel/goal103_terminal_capture_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal103_terminal_capture_20261111/phase0_summary.md`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/P_BC15/final.pt`
