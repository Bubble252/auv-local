# Goal104 decision：通道参数化有弱信号，继续 eval-only，不开 SAC

**日期**：2026-11-11<br>
**状态**：`CHANNEL_SIGNAL_WEAK / CONTINUE_ACTION_PARAMETERIZATION / NO_SAC / HELDOUT_CLOSED`

## 1. 触发

Goal103 发现 B20 strict 失败与大幅值逐通道 residual 和终端几何相关，但 authority clip
不是 outcome 判别器。Goal104 因此在固定 P_D3、scale 2.0、Goal85 unbiased、dev
`20–59` 下，只调整 `(rudder, stern, rpm)` 通道增益。

## 2. 结果

- C0 精确复现 `33/40` default、`20/40` strict；
- 26/26 run 的 checkpoint SHA、scale、channel gains、seed 和 trace 校验通过；
- `R075`、`R125` 通过冻结 Gate：default `33`、strict `21`、path 不恶化、timeout/safety 0；
- 但两者都只新增 1 个 strict seed，方向相反，exact McNemar 均为 `p=1.0`；
- `U050` 是最强 post-hoc 方向：strict `23/40`（`+4/-1`，`p=0.375`），default `32/40`；
- `U150` strict `19/40`，与 `U050` 形成 rpm 方向对照；
- 网络 parity：22/22 frozen-BC 张量完全一致，无 NoisyNet，无非有限参数。

## 3. 判读

`R075/R125` 的 strict 增益是 goal-plane 边界翻转：

- seed 33：C0 final `1.365 m` -> `1.236 m`；
- seed 56：C0 final `1.265 m` -> `1.249 m`；
- 两者都只是进入 1.25 m 半径，不是全轨迹改善。

所以 Gate 可以写 `CHANNEL_SIGNAL`，但不能写“稳定提升”或“rudder authority 方向已找到”。
当前没有“方向正确但探索/样本效率不足”的证据，不能开 SAC。网络容量和 NoisyNet 同样
不是本次主因。

## 4. 决策

```text
decision = continue
```

继续 eval-only 参数化线，不改变主算法族：

1. 不训练 PPO/SAC/REDQ/DroQ；
2. 不打开 held-out；
3. 下一 Goal 预注册 `C0/U075/U050/U050+R075/U050+R125`；
4. 新 Gate 必须同时要求 default `>=33`、strict `>=23`、path `<=1.05*C0`、
   timeout/safety 0；
5. 只有 dev Gate 通过后，才允许做一次 held-out 确认。

## 5. 证据

- `doc/goals/goal104_result_20261111.md`
- `doc/goals/goal104_channel_parameterization_audit_20261111.md`
- `artifacts/lowlevel/goal104_channel_parameterization_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal104_channel_parameterization_20261111/phase0_summary.md`
- `artifacts/lowlevel/goal104_channel_parameterization_20261111/posthoc_pairing.json`
- `artifacts/lowlevel/goal104_channel_parameterization_20261111/network_parity.json`
