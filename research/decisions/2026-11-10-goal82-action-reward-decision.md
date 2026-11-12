# Decision：tracking 有增益但 outcome 不动，转向 mapping/capture frame 审计

日期：2026-11-10<br>
关联：Goal82<br>
决定：`pivot`

## 判据

Goal82 只读复用 20 个已记录 run、800 个 episode：

- H82-A 支持，但支持来自 `tracking_error_heading` / `tracking_error_pitch`，
  不是 residual 三通道幅度；heading/pitch/speed 的 `abs_mean` 在 lost/retained
  之间几乎相同；
- H82-B 不支持：scale clip 和 rate clip 在四个 group 都是 `0`，material
  saturation 与 direction cosine 没有稳定方向；
- H82-C 支持：`reward_terminal` 和 `reward_goal_plane_miss` 在 `100%`
  episode 中只在 done 行激活；preterminal 仍有 terminal_progress，且 lost
  的 terminal_progress 更低；
- H82-D 支持但主要是 strict125-specific：policy 的 heading tracking 优于
  zero，但 true success 反而更低（baseline `-0.069`，candidate `-0.106`）；
- H82-E 不支持：当前不是“action 和 reward 都没有区分度”的完全无信号状态。

修正后的 trace 语义是：

```text
corrected_command[t] = base_command[t] + residual[t]
```

`command_physical[t]` 是执行后的下一步命令，不是当前 residual 的基座。修正后
H82-A/D 的结论没有翻转；原始结果保留用于审计。

## 决定

- 停止把 reward 作为当前主因；
- 不启动 SAC/REDQ/DroQ；
- 不重开 Goal27 失败臂；
- 不扫 residual scale、NoisyNet、dwell、capture radius 或 reference extension；
- 下一 Goal 先做只读的 mapping / capture frame / terminal capture 审计：
  1. residual 的 base、时序、符号、尺度、rate limit 与 command→actuator 响应；
  2. guidance/reward/termination/success 是否使用一致的 `p_nav`/`eta` 和
     goal-tangent/path-local frame；
  3. actuator→state→`p_nav`/`eta`→capture 的哪一环丢掉了 tracking 改善；
  4. 只有链路一致后，才设计单因素 reward 或 terminal objective arm。

## 训练型 arm 的前置条件

1. 先做 `num_envs 128/256/512 × batch_size 256/512` 单卡吞吐基准；
2. 记录 `env_transitions/s`、gradient steps/s、GPU 利用率、显存、CPU、wall；
3. 按 seed 分 8 张 GPU，一卡一个 seed；
4. 扩大采样/批量必须独立预注册，不能混进正式 Gate；
5. SAC 只在出现“residual 方向正确、但探索/样本效率不足”的直接证据后开启；
   REDQ/DroQ 只在样本效率证据明确时考虑。

## 证据

- `doc/goals/goal82_action_reward_attribution_20261110.md`
- `doc/goals/goal82_result_20261110.md`
- `artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.json`
- `artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.md`
- `research/paper_reading/2026-11-10-goal80-terminal-reference-plane-miss.md`
