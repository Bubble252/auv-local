# Decision：停止 goal-tangent 前移，转向 termination/capture 审计

日期：2026-11-10<br>
关联：Goal80<br>
决定：`pivot`

## 判据

Goal80 两个 scenario 都得到 `TERMINAL_REFERENCE_HURTS`：

- default true `97 -> 80`，newly lost `18`，plane-miss rate `+0.120`；
- strict125 true `44 -> 38`，newly lost `10`，plane-miss rate `+0.060`；
- default/strict 新增丢失中分别有 `17/18`、`8/10` 是 plane miss；
- zero-residual 仍出现 plane-miss `+0.075/+0.075`；
- extension `0.0` 与 Goal71 baseline 逐位等价，extension `1.0` 实际改变
  terminal command 和轨迹；
- timeout/safety 没有增加，heading tracking 反而改善。

因此失败不是“extension 没生效”、不是 NoisyNet 采样、也不是优化/样本效率；
而是前移目标让载具在 tight capture 前更早穿过 goal plane。

## 决定

- 停止 goal-tangent 前移线；不扫 extension `0.5/2/4 m`；
- 不扫 terminal window、reward、gamma、seed、residual scale；
- 不启动 SAC/REDQ/DroQ；
- 新开预注册 Goal81 只读 termination/capture 审计，使用已有 trace：
  plane-miss guard、tight capture、terminal `p_nav/eta` 最小距离和 crossing offsets；
- 只有离线审计证明存在可恢复的 termination 规则，才设计单因素闭环 arm；
- 保持 dev `20–59`、held-out `100–149` 关闭、CPU/float64 和既有 checkpoint 协议。

## 下一实验的预注册要素

1. H81-A：plane-miss guard 在 tight capture 前触发是新增丢失的主因；
2. H81-B：仅修改 termination/capture 规则即可在记录轨迹上恢复 true，且不增加
   false success、timeout 或 safety；
3. H81-C：若 A/B 均不支持，再查动作参数化或 controller bandwidth；
4. 最便宜实验：对 Goal80 trace 做离线重放，不训练、不新增 rollout；
5. 训练型 arm 的前置条件：先做 `num_envs 128/256/512 × batch 256/512` 单卡吞吐
   基准，再按 seed 分 8 卡并行。
