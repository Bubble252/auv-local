# Goal171/172 阻塞：route reward 已接线，但残差不产生净路线收益

**日期**：2026-11-12<br>
**状态**：`RESIDUAL_SCALE_HARMFUL / REWARD_WIRING_CONFIRMED / SAMPLE_EFFICIENCY_UNRESOLVED / HELDOUT_CLOSED`<br>
**上游**：Goal171 `SCREEN_NO_CANDIDATE / FORMAL_GATE_NOT_RUN`<br>
**性质**：eval-only 反事实诊断，不训练、不打开 held-out、不写正式 Gate。

## 1. 现象

Goal171 的 8 arms × 200k 单 seed screen 没有路线收益；全矩阵最好的 3D route
delta 只有 `-0.00245 m`，而 Gate171-A 要求 `>=0.05 m` 且相对 `>=5%`。

这不是“差一点”。为区分方向问题和接线问题，Goal172 在同一 checkpoint 上做
current10 的 40-seed eval-only 放大：

| arm | radius | scale | Δ3D RMSE | relative | improved segments | 安全/速度 |
|---|---|---:|---:|---:|---:|---|
| `R1_BC_PPO` | default | `2×` | `+0.0588 m` | `+3.37%` | 1/5 | 0 overspeed |
| `R1_BC_PPO` | default | `5×` | `+0.2152 m` | `+12.32%` | 2/5 | 1 timeout |
| `R1_BC_PPO` | default | `10×` | `+0.5797 m` | `+33.19%` | 1/5 | 253 overspeed, 10 timeout |
| `R1_ALOS_PPO` | default | `2×` | `+0.0490 m` | `+2.81%` | 0/5 | 0 |
| `R1_ALOS_PPO` | default | `10×` | `+0.0541 m` | `+3.11%` | 0/5 | 0 |
| `R1_BC_SAC` | default | `2×` | `+0.0484 m` | `+2.78%` | 0/5 | 0 |
| `R1_BC_SAC` | default | `10×` | `+0.0491 m` | `+2.82%` | 0/5 | 0 |
| `R1_ALOS_SAC` | default | `2×` | `+0.0480 m` | `+2.76%` | 0/5 | 0 |
| `R1_ALOS_SAC` | default | `10×` | `+0.0462 m` | `+2.65%` | 0/5 | 0 |

strict125 同向。没有出现“放大后路线单调改善”的单元。PPO-BC 在较高 scale 还
出现明显速度/超时退化。

## 2. Reward 归因

同一 frozen base、同一 seed、零 residual 下比较 R0 old-route 和 R1 route-health：

- 两个 radius 的 40 个 seed 轨迹完全一致，`max_abs_route_delta = 0.0 m`；
- R1 trace 中出现了 `reward_path_cross`、`reward_path_vertical`、
  `reward_path_progress`、`reward_route_cross_huber`、`reward_route_vertical_huber`、
  `reward_route_middle_gate`、`reward_route_reward` 等项；
- `reward_route_middle_gate` 平均差约 `+0.50`，`reward_route_reward` 约 `+0.90`，
  `reward_total` 约 `+0.85`。

因此 **reward 不是没接上**。零残差轨迹相同但 reward breakdown 不同，符合
“同一 base 上 reward 配置改变”的预期。当前缺少的是：训练是否把这段 reward
差异转成更好的闭环路线。

## 3. 可证伪假设

### H172-A：方向有用，只是幅度/探索不足

判定：`FALSIFIED`。

2×/5×/10× 放大没有出现 3D route 改善；PPO-BC 在 5× 已经退化，10× 明显超出
速度包线。不能把“残差太小”作为训练不足的解释。

### H172-B：reward 没有进入优化信号

判定：`NOT_SUPPORTED_FOR_WIRING / REMAINS_OPEN_FOR_OPTIMIZATION`。

运行环境和 trace 的 reward 项确实改变；不能再说“reward 没接上”。但如果训练
200k 后 R0/R1 的路线仍几乎相同，可能是优化没有利用 reward，而不是接线错误。

### H172-C：某个动作通道拖累整体

判定：`RUNNING`。

当前在 `2×` 上补 rudder-only / stern-only / rpm-only 归因。若某一通道单独放大
能改善路线，说明当前三通道残差的组合/信用分配有问题；若三通道都退化，说明
方向本身不可用。

### H172-D：SAC 的样本效率更高，所以应换 SAC/DroQ

判定：`NOT_SUPPORTED`。

同一 200k 预算下 SAC 也没有路线收益；PPO-BC 的大残差反而更容易伤速度和轨迹。
没有证据支持直接换算法或上 REDQ/DroQ。

## 4. 最便宜裁决

```text
decision = continue
```

不换算法、不扩大 seed 救当前 200k 结果、不把 `NO_CANDIDATE` 写成 formal fail。

下一步：

1. 等 channel 归因完成；如果没有通道净收益，关闭“仅靠 residual 幅度”的支线；
2. 预注册 Goal173：保持 Goal85 position/no-delay、Goal86 z_t/BC、Goal171
   route-health reward 和 C3 课程，改为 `3 seeds × 1M` 的同协议训练；
3. 训练前先做 base-policy value anchor / critic normalization 的小规模
   falsification，避免 residual RL 的 cold-start 把 1M 预算浪费在零解附近；
4. 仍保持 PPO/SAC 两种算法；REDQ/DroQ 只在明确证明样本效率不足后才开；
5. 任何正式评价继续跑完整 `4 levels × 2 radii + zero`，不缩成
   `default/strict125`。

## 5. 产物

```text
artifacts/lowlevel/goal171_residual_attribution_20261112/
  screen/
  screen_channel/          # 运行中
  reward_attribution/
  analysis/goal172_residual_attribution.json
  analysis/goal172_residual_attribution.md
```

所有结果标记 `DIAGNOSTIC_NOT_GATE`，held-out 未打开。
