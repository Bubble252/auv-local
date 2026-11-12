# [RETRACTED] Blocker：Goal89 action composition 结论错误

> **撤回说明（2026-11-11）**：本文件把 Goal86 P-A checkpoint 误判为
> “绝对动作 BC”，据此把 Goal87/88/89 的 `actuator` 接口判为错误。该判断
> 不成立。Goal86 checkpoint 的 `residual_semantics.target` 明确是
> `normalize(D3_actuator - ALOS_actuator)`，`actuator` 接口语义正确。
> 本文件保留作为失败审计历史，但不再作为有效结论。
>
> 正确分析见 `research/blockers/2026-11-11-goal89-gate-scope.md` 和
> `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`。

日期：2026-11-11<br>
Goal：Goal89 / Goal87 / Goal88 frozen-BC residual<br>
状态：`ACTIVE / INVALID_ACTION_COMPOSITION / NETWORK_PARITY_OK / CORRECTED_PROTOCOL_REQUIRED`

## 1. 现象

Goal87/88/89 的配置同时使用了：

```text
base_policy.mode = frozen_pc_bc_residual
pc.action_interface = actuator
```

但 `frozen_pc_bc_residual` 的 `policy.act_inference()` 返回的是
`frozen_BC + residual`，而 `BatchedPCActuatorResidualEnv.step()` 把输入解释为
“相对 ALOS 的归一化执行器残差”，再计算：

```text
ALOS_actuator + pc.actuator_residual_scale * full_scale * input
```

因此实际闭环动作是：

```text
ALOS + scale * (frozen_BC_action + learned_residual)
```

不是预注册的：

```text
frozen_BC_action + learned_residual
```

这不是网络主干或 checkpoint 接线错误，而是动作接口的语义错误。历史 Goal26
的 frozen D3 BC residual 使用 `pc.action_interface=absolute`；Goal87/88/89
改成了 `actuator`，导致 frozen BC 输出被错误地当作残差。

同一错误也影响了 `gated-residual` eval：它虽然只乘了
`base_action + residual` 中的 residual 部分，但随后仍把合并动作传给 actuator
wrapper，整体仍是 `ALOS + scale * (BC + gated_residual)`，不能作为干净的
residual-only gate 证据。

## 2. 直接证据

- `src/auv_lowlevel/batch/pc_policy.py`：frozen residual 模式下
  `_actor_mean()` 返回 `base + residual`，`base_action()` 返回冻结 BC 的
  normalized PC action。
- `src/auv_lowlevel/batch/pc_actuator_residual.py`：wrapper 再次计算 ALOS base，
  并把输入加上 `residual_scale * full_scale`。
- `configs/lowlevel/goal88/*.json`、`configs/lowlevel/goal89/*.json`：
  `pc.action_interface=actuator`。
- `configs/lowlevel/pc_goal26_ppo_p2_default.json`：历史同族 arm 使用默认
  `environment -> absolute`，没有 actuator wrapper。
- Goal88 BC/S1.25 的 `actuator_residual_abs_mean` 约 50–65，远大于训练残差；
  这个数值主要来自 frozen BC 与 ALOS 的差，而不是 learned residual。

## 3. 受影响结果

以下结果标记为 `INVALID_ACTION_COMPOSITION`，不能作为干净证据：

- Goal87 H87-O2/H87-O4 中所有 `pa_bc_frozen_residual*` 的 actuator arm；
- Goal88 `scale1.0/1.25/1.5/1.75/2.0` 1M 与 S1.25 5M formal；
- Goal89 `G20_5/G10_3` 训练 arm；
- Goal89 `evalonly_G20_5/G10_3` 四组 eval；它们仍是错误合并动作上的诊断。

网络结构对照本身仍然有效：`PCActorCritic` 主干、`z_t64`、path8+endpoint4、
frozen BC checkpoint 和 reserved-slot 补零没有新接线错误。

## 4. 可证伪 hypothesis

### H89C-A：absolute 接口能恢复 frozen-BC + residual 的真实语义

- 支持：zero-residual 时 absolute 闭环动作与同观测 frozen BC 逐元素一致；
  非零 residual 只改变 `action - base`，不再出现 ALOS 二次叠加。
- 否定：absolute 动作仍与 BC 不一致，或环境配置的 actuator envelope 不允许
  BC action 直接通过。
- 最便宜证伪：单观测 action trace，比较 `frozen_bc`、`absolute policy` 和
  `actuator-wrapper policy` 三者的 physical actuator。

### H89C-B：只 gate residual 不会改变远处 BC 行为

- 支持：距离大于 `enter_m` 时动作逐元素等于 frozen BC；进入窗口后才偏离，
  且 PPO storage 中保存的动作与 log-prob 仍是 policy 实际采样的 residual。
- 否定：gate 后远处动作仍偏离 BC，或训练/评估动作与 storage 不一致。
- 最便宜证伪：一个 `actor_decision_stride=1` 的 smoke，记录 `base`、`gate`、
  `residual`、`applied` 四组张量并做逐行等式检查。

### H89C-C：此前 scale/success 现象主要是 ALOS+BC 错位，而不是 residual scale

- 支持：修正为 absolute 后，旧 S1.25 checkpoint 的 success/path 与 actuator
  wrapper 结果显著不同。
- 否定：两种组合的逐 episode 结果几乎相同，说明当前动作包络下二次叠加恰好
  近似等价。
- 最便宜证伪：对同一个 checkpoint 做 eval-only absolute vs actuator 配对。

## 5. 已排除项

- 不是网络宽度、特征维度、checkpoint 前缀复制或 path8/endpoint4 接线错误；
  Goal87 的逐元素 parity 仍成立。
- 不是 NoisyNet；这些 arm 没有 NoisyNet。
- 不是 reward/terminal/goal radius；错误发生在 `env.step()` 前的动作组合层。

## 6. 下一步

1. 在 Goal89 中加入显式 `base_policy.output_mode=residual`，要求 absolute
   接口；训练时由 vector env 组合 `frozen_BC + gate * sampled_residual`。
2. 评估 `policy` 模式同样组合，`frozen-base` 仍只输出 BC，`zero-residual`
   仍等于 BC。
3. 先跑 1-step/1-iteration action parity smoke，再重跑同协议 1M：
   corrected `A1.0`、`A1.25`、`AG20_5`、`AG10_3`。
4. 旧 Goal89 结果保留但标 `INVALID_ACTION_COMPOSITION`；不据此判 Gate 失败，
   也不换 SAC。

## 7. 证据路径

- `src/auv_lowlevel/batch/pc_policy.py`
- `src/auv_lowlevel/batch/pc_actuator_residual.py`
- `src/auv_lowlevel/batch/pc_vec_env.py`
- `scripts/train_batch_lowlevel_ppo.py`
- `scripts/evaluate_pc_ppo.py`
- `configs/lowlevel/goal26/`、`configs/lowlevel/goal88/`、`configs/lowlevel/goal89/`
- `artifacts/lowlevel/goal89_gate_20261110/evalonly_*_dev40/summary.json`
- `doc/goals/goal89_terminal_gated_actuator_residual_20261110.md`

