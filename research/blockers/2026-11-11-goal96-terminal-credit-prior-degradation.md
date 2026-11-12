# Blocker：terminal residual 方向不成立，PPO 放大先验但没有闭环增益

日期：2026-11-11<br>
Goal：Goal96<br>
状态：`AUDIT_COMPLETE / DIRECTION_NOT_SUPPORTED / PPO_DEGRADES_PRIOR / NO_SAC`

## 1. 现象

Goal95 的监督残差已经安装，residual/base 约 `0.55`，但 PPO final 的
default success 从 initial `31` 降到 `30`，strict 保持 `19`；frozen BC 是
default `31`、strict `20`。必须回答：终端 residual 的方向/权限/reward
credit 是否有问题，还是 PPO 更新在破坏先验。

## 2. 证据

### 2.1 方向不成立

12 个 run 的 `action_effect_score` 在 `3–2 m` 和 `2–1.25 m` 窗口的
success−failure bootstrap 95% CI 全部跨 0。按预注册判据记为
`DIRECTION_NOT_SUPPORTED`。

`cos_prior` 不能单独证明“朝目标”；Phase A 中 failure 甚至常常比 success
更接近 BC 先验。因此不能把失败写成“方向对，只是幅度小”或“NoisyNet 把
方向打乱”。

### 2.2 权限足够

terminal window 的 effective/applied survival 最低为 `0.9192`，高于预注册
`0.90`。动作包线不是主瓶颈，记为 `AUTHORITY_SUFFICIENT`。

### 2.3 Reward formal misaligned，但折扣后 terminal 份额很小

pre-terminal radial closure 在所有 12 个 run 都能区分 success/failure，CI
下界均大于 0。名义 terminal positive share 约 `1.00`，按 `<0.80` 判据记为
`REWARD_CREDIT_MISALIGNED_FORMAL`。

但 PPO `gamma=0.995` 折扣后，terminal 项只占总 return 的 `1.2%–1.4%`。
单个 `300` goal bonus 落在总 reward 略低于 `300` 的 terminal 行，造成名义
占比接近 1；不能据此断言 goal bonus 是唯一主导项。

### 2.4 逐 seed 配对：PPO 放大先验，没学出新方向

Phase C 新增同 seed 配对：

| group | initial | final | frozen | initial→final |
|---|---:|---:|---:|---|
| D3 default | 31 | 30 | 31 | lost seed 26 |
| D3 strict | 19 | 19 | 20 | 无翻转，frozen gained 33 |
| BC15 default | 31 | 30 | 31 | lost seed 26 |
| BC15 strict | 19 | 19 | 20 | 无翻转，frozen gained 33 |

终端窗口 initial→final 的 mean cosine 为 `0.907–0.969`，但 mean residual L2
ratio 为 `1.10–1.52`，closure delta 为 `-0.035–0.000`。唯一丢失的 seed 26：

| group | window | cos | L2 ratio | closure |
|---|---|---:|---:|---:|
| D3 default | 3–2 | 0.9465 | 2.2862 | 1.2066→0.8846 |
| BC15 default | 3–2 | 0.9986 | 1.3175 | 1.0823→0.8556 |

即 PPO 的确定性 residual 与 initial 方向高度一致，主要变化是幅度放大；
放大没有提高径向 closure，反而在丢失 seed 上下降 `0.23–0.32 m/s`。

## 3. 网络结构排除项

- 当前 `PCActorCritic` 与 Goal86/87/89 parity：`d_z_total=64`、
  `n_z_reserved=4`、state encoder `[128,128]`、actor/critic `[256,128]` ELU、
  path12、command3、current encoder 存在、DADE off；
- frozen BC 22 个 entry 与 Goal86 P-A checkpoint `max abs diff=0.0`；
- 当前没有 NoisyNet；低层 PPO 是 `Normal(mean, std=0.1)`，评价走确定性
  `act_inference()`；
- `use_noisy` 只存在于 Goal58/59 DDPG 谱系，不属于当前 PPO/frozen-BC 谱系。

```text
network_wiring_bug = false
network_capacity_primary = false
noisenet_dominant = false
```

## 4. 假设裁决

- `H96-A`：terminal residual 方向能分离 success/failure。FALSIFIED。
- `H96-B`：失败由 terminal 动作权限不足导致。FALSIFIED。
- `H96-C`：terminal reward credit 完全错位且主导。PARTIAL/FORMAL：
  名义占比 > 0.80，但折扣后只有 1.2%–1.4%，pre-terminal closure 仍有信号。
- `H96-D`：PPO 保持或改善 initial/frozen 先验。FALSIFIED：default 31→30，
  frozen 31/20；final 主要放大 residual 幅度，没有新方向。
- `H96-E`：NoisyNet 或网络接线主导。FALSIFIED。
- `H96-F`：当前证据足以开 SAC 同协议对照。FALSIFIED：没有“方向正确但
  样本/探索效率不足”的证据。

## 5. 结论与 next

```text
decision = pivot
target   = PPO prior-preserving terminal residual update
SAC      = not yet
```

下一步不是换算法，而是预注册一个 PPO 先验保持实验：在 terminal window 对
residual 幅度或对 initial/frozen prior 的偏离加约束，验证能否在保持
default `31/40`、strict `20/40` 的同时提高 closure 和 success。只有该实验
明确出现“方向正确、effective 增量存在，但闭环仍不改善”的直接证据，才重新
评估 SAC/REDQ/DroQ。

## 6. 证据

- `doc/goals/goal96_result_20261111.md`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseA.json`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseB.json`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.json`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.md`
- `scripts/analyze_pc_goal96_paired_terminal.py`
