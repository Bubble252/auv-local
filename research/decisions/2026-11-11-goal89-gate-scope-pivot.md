# Decision：Goal89 旧 gate scope 作废，转 Goal90 corrected gate

日期：2026-11-11<br>
Goal：Goal89<br>
Decision：`pivot`<br>
Next：Goal90

## 决策

1. 撤回 `2026-11-11-goal89-action-interface-composition.md` 中
   “frozen BC 应使用 absolute 接口”的结论。
2. Goal87/88/89 的 `pc.action_interface=actuator` 保持有效。
3. Goal89 `G20_5`/`G10_3` 训练臂标 `INVALID_GATE_SCOPE`，不用于 Gate。
4. `goal89_v2_20261111` 全部标 `INVALID_WRONG_BC_CHECKPOINT_CONTRACT`。
5. 新开 Goal90，预注册：

```text
ALOS + scale * (r_BC + gate * r_PPO)
```

其中 `r_BC` 始终保留，gate 只作用于 PPO 增量。

## 依据

- Goal86 checkpoint 明确是 `D3−ALOS` 执行器残差；
- 冻结分支与 BC checkpoint 权重逐位一致；
- Goal89 旧训练 gate 作用在总和，远处退化成纯 ALOS；
- `policy` vs `frozen-base` 的 12 组配对 success 完全相同；
- Goal88 无 gate 下，BC 放大 1.25 本身已有 strict125 19/19/19，
  PPO 只再加 +0/+1/+0，并把 path 拉差约 0.02 m。

## 不改变

- 不换 PPO/SAC 主算法族；
- 不改 reward、terminal、success、goal radius；
- 不打开 held-out 100–149；
- 不改旧 Gate 阈值；Goal90 是新预注册 arm。
