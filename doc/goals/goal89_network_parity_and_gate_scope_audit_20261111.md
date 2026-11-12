# Goal89 网络结构对照与 gate scope 审计

**日期**：2026-11-11<br>
**状态**：`AUDIT_COMPLETE / NETWORK_PARITY_OK / HISTORICAL_LINEAGE_OK / GATE_SCOPE_BUG_CONFIRMED`<br>
**上游**：Goal86 P-A BC、Goal87 network parity、Goal88 S1.25、Goal89 G20_5/G10_3<br>
**性质**：只做审计与诊断，不改 reward、terminal、success、goal radius、seed 或 held-out。

## 1. 结论

这次问题**不是网络结构接错**。Goal89 PPO checkpoint 中的冻结 BC 分支与
Goal86 P-A BC checkpoint 逐张量完全一致；`d_z_total=64`、path12、command3、
current3、`c_hat=0`、reserved slots 也都符合原契约。

真正的问题是 **Goal89 训练时 gate 作用在“BC 残差 + PPO 残差”的总和上**，
而评估器的 `gated-residual` 只 gate PPO 增量。两种语义不同：

```text
Goal89 旧训练：
ALOS + scale * gate * (r_BC + r_PPO)

Goal89 旧评估 gated-residual：
ALOS + scale * (r_BC + gate * r_PPO)
```

Goal86 P-A checkpoint 的标签是：

```text
r_BC = normalize(D3_actuator - ALOS_actuator)
```

所以旧训练在远离 goal 时把 `r_BC` 也关掉了，动作退化成纯 ALOS；
success 从 BC 的 `31/40, 20/40` 掉到 `23/40, 14/40` 正是这个原因。

旧 blocker 中“frozen BC 是绝对动作、actuator 接口错误”的判断**撤回**。
Goal87/88/89 使用 `pc.action_interface=actuator` 在语义上是正确的。
`goal89_v2` 的 absolute 实验全部标为
`INVALID_WRONG_BC_CHECKPOINT_CONTRACT`，不能用于 Gate。

## 2. 结构逐项对照

Goal86 checkpoint：
`artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`

| 项目 | Goal86 BC | Goal89 PPO 冻结分支 | 结果 |
|---|---|---|---|
| `d_z_total` | 64 | 64 | 一致 |
| `n_z_reserved` | 4 | 4 | 一致 |
| `actor` 输入 | 64+3+64=131 | 64+3+64=131 | 一致 |
| `actor_hidden_dims` | 256,128 | 256,128 | 一致 |
| `state_encoder_hidden_dims` | 128,128 | 128,128 | 一致 |
| `path_dim` | 12 | 12 | 一致 |
| `path_encoder` | 12→64→64 | 12→64→64 | 一致 |
| `current_encoder` | 3→64→64 | 3→64→64 | 一致 |
| `activation` | ELU | ELU | 一致 |
| `c_hat` | 全零 | 全零 | 一致 |
| command 列 | 3 列训练后为 0 | 冻结加载 | 一致 |
| current 权重 | 全零 | 冻结加载 | 一致 |
| path endpoint 列 | 最后 4 列全零 | 冻结加载 | 一致 |

权重级比对：`frozen_bc_actor*` 的 22 个 state entry 与 BC checkpoint
**max abs diff = 0.0**。因此不存在“隐藏层宽度错位”“path/current 输出维度错位”
或“能加载但语义错位”。

Goal89 PPO 自身的可训练分支：

- `path_encoder` 输入 path8，输出 64；`path_endpoint_encoder` 输入 endpoint4，
  输出 64，末层零初始化；
- `actor_current_encoder` 输出 60 个 active 维，再补 4 个 reserved 零；
- 这些与 Goal86/Goal87 的既有设计一致，不是本次回归。

### 2.1 历史代码谱系对照

为排除“新版自测通过、但和历史实现语义不同”的盲点，额外只读对照了三代代码：

| 代码版本 | 关键文件 SHA256 | 用途 |
|---|---|---|
| Goal26 success-first | `pc_policy.py` `3e9d10ad...e1f4f` | 最早一批 frozen-BC + residual 代码 |
| Goal80 terminal reference | `pc_policy.py` `6e868325...6f4a4` | `z_t64` 主干基线 |
| 当前 Goal89/90/91 | `pc_policy.py` `2e1ebe0c...bb1a6` | path12/endpoint4、frozen residual、gate |

代码级结论：

1. `PCActorCritic` 的 state encoder 仍是 `[128,128]`、`d_z_total=64`、
   `n_z_reserved=4`，actor/critic 仍是 `[256,128]`、ELU；主干没有重写。
2. `base_action()` 的 frozen-BC 流水线从 Goal26 到当前没有被替换：state encoder
   → path encoder → 可选 current encoder → 拼 command3/零 DADE → actor。[^lineage-base]
3. 当前新增的是接口扩展：path12 拆成 BC 已见的 path8 和 endpoint4；frozen
   path encoder 输出已是 64 维时不再错误补 4 个 reserved slot；gate 只包住
   `r_PPO` 的环境侧组合。
4. 因此“网络结构接线错误”仍被排除。真正的谱系变化是 **frozen base 的语义**：
   Goal26 加载的是绝对 D3 BC，Goal86 P-A 之后加载的是
   `r_BC = normalize(D3_actuator - ALOS_actuator)`。两者都能用同一个
   `base_action()`，但 checkpoint 含义不同，不能拿 Goal26 的动作幅度直接类比。
5. 兼容 checkpoint 的 path12/cmd3 扩展是严格前缀复制 + 新列零填充：actor `0/2/4`
   权重、encoder 三层、path 前 8 列均逐元素一致；新增 command3、endpoint4、
   current encoder 初始输出为零。Goal89 PPO 的 22 个 `frozen_bc_*` entry 与
   Goal86 checkpoint `max abs diff = 0.0`。

一句话：

```text
network_wiring = same
interface_extension = path12 endpoint + gate scope
base_semantics = absolute D3 BC -> D3-minus-ALOS residual BC
```

[^lineage-base]: 这里比较的是结构；Goal86 P-A checkpoint 的语义由
`scripts/pretrain_pc_goal86_pa_residual.py` 的 `residual_semantics.target` 明确写成
`normalize(D3_actuator - ALOS_actuator)`，不能因为加载器复用 BC schema 就误判成绝对 BC。

## 3. Step 7：deterministic policy vs zero learned residual

在**相同 checkpoint、相同 gate、相同 dev40、default+strict125**下比较：

- `policy`：`r_BC + r_PPO`；
- `frozen-base`：只保留 `r_BC`，强制 `r_PPO=0`。

### 3.1 旧 Goal89 gate 训练臂

12 组配对（2 gate profiles × 3 seeds × 2 scenarios）全部为：

```text
policy success == frozen-base success
```

| arm | default policy/BC | strict125 policy/BC |
|---|---:|---:|
| G20_5 86201/02/03 | 23/23, 23/23, 23/23 | 14/14, 14/14, 14/14 |
| G10_3 86201/02/03 | 22/22, 22/22, 22/22 | 13/13, 13/13, 13/13 |

PPO 确定性增量只让物理残差平均变化约 `1–2` 个单位（总残差约 30），
path 和 plane-miss 也几乎逐 episode 相同。**PPO 增量在 success 层面为零，
success 退化完全来自 gate 把 BC 一起关掉。**

### 3.2 Goal88 S1.25 无 gate 训练臂

| seed | default policy/BC | strict125 policy/BC | path policy/BC |
|---|---:|---:|---:|
| 86201 | 31/31 | 20/19 | 2.320/2.303 |
| 86202 | 31/31 | 20/19 | 2.339/2.303 |
| 86203 | 31/31 | 19/19 | 2.303/2.303 |

关键结论：

- **BC 残差放大到 1.25 本身**就把 strict125 从 `17/17/17` 提到 `19/19/19`；
- PPO 只再加 `+0/+1/+0`（合计 +2/120），不稳健；
- PPO 同时把 path 拉差约 `0.017–0.036 m`，这正是 Goal88 formal 的
  `+0.0231 m` 退化来源。

因此 Goal88 的主收益应归因于 **BC 残差幅度的动作参数化**，不是 PPO
学到了有效的终端策略。

## 4. 探索噪声诊断

Goal89 PPO 训练记录：

- `mean_noise_std` ≈ `0.10`，40 iterations 后仍为 `0.0986`；
- 评估时 deterministic `residual_action_abs_mean` ≈ `0.0188`；
- 噪声/确定性信号 ≈ `5.3×`；
- 物理推力尺度约：确定性 `36 rpm`，噪声 `191 rpm`。

也就是说，训练时 PPO 的探索噪声远大于它最终学到的残差均值。策略倾向于把
均值压小以躲避噪声，这与 actor 末层范数持续缩小、5M 不再提升 success 的
观察一致。这个证据支持“**探索/优化是 PPO 分支的主要瓶颈**”，但还不能
单独证明样本效率不足；需要先做只 gate PPO 增量的 corrected screen。

## 5. 下一步

1. **不要**继续用 Goal89 旧 G20_5/G10_3 判 Gate；它们标
   `INVALID_GATE_SCOPE`。
2. 新开 Goal90：`output_mode=residual`，gate 只作用于 `r_PPO`：
   `ALOS + scale*(r_BC + gate*r_PPO)`。
3. 同协议 1M/dev40/default+strict125，先确认 corrected gate 是否能恢复
   BC 的 `31/40, 19–20/40` 且 path 不再退化。
4. 若 corrected gate 仍只得到 BC 水平，开一个预注册的 PPO exploration
   screen（例如 noise std 0.03 vs 0.10）；只有该 screen 证明探索是主因，
   才按用户规则开 SAC 同协议对照。
5. 不打开 held-out `100–149`，不改 reward/terminal/success/goal radius。

## 6. 证据

- `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`
- `artifacts/lowlevel/goal89_gate_20261110/G20_5_seed*/final.pt`
- `artifacts/lowlevel/goal89_gate_20261110/G10_3_seed*/final.pt`
- `artifacts/lowlevel/goal89_gate_20261110/*_eval_*_dev40*/summary.json`
- `artifacts/lowlevel/goal89_gate_20261110/*_eval_*_dev40_frozenbase/summary.json`
- `artifacts/lowlevel/goal88_scale_20261110/scale1.25_seed*_eval_*_step7_*/summary.json`
- `doc/goals/goal86_bc_prior_traditional_residual_20261110.md`
- `doc/goals/goal87_network_history_parity_20261110.md`
- `doc/goals/goal88_result_20261110.md`

## 7. 文档与 Git

- 旧 blocker `2026-11-11-goal89-action-interface-composition.md` 保留但标
  `RETRACTED`；
- 新 blocker `2026-11-11-goal89-gate-scope.md`；
- 新 decision `2026-11-11-goal89-gate-scope-pivot.md`；
- 不自动 commit/push；不提交 `artifacts/`、checkpoint 和日志。
