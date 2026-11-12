# Decision：guard 不是抢跑，转向动作参数化与 reward 审计

日期：2026-11-10<br>
关联：Goal81<br>
决定：`pivot`

## 判据

Goal81 只读重放 Goal71 baseline 与 Goal80 candidate 的 20 个 run、800 个
episode：

- 所有 lost 都是 `GOAL_PLANE_MISS`，`100%` 从未进入 `r_tight`；
- lost 在终止前 `0.88–1.36 s` 已经过最近点，随后外离 `0.60–1.25 m`，
  最后一秒斜率仍为 `+0.58–+1.13 m/s`；
- `dwell_observed` 中位数为 `51` 步，guard 已经等满 50 步，不是 crossing 当步
  抢跑；
- `R_cross_tight` 在四个 group 都恢复 `0` 个 lost；`R_cross_guard` 在
  candidate 只恢复 `28.6%/22.2%`，baseline strict125 恢复 `5/7` 但新增 `13`
  个 false，达不到可推广条件；
- default 的 true `eta` 曾进入 tight、但 `p_nav` 未进入的比例较高
  （policy `1/1` baseline、`8/15` candidate），是真实次级因素；strict125
  是混合结果（baseline `2/6` 刚过阈值、candidate `1/7` 不过），不能外推；
- lost/retained residual L2 比值 `1.10–1.32`，baseline default 反而为 `0.705`;
  zero-residual 仍有 `14` 个 lost 且 geometry 100% 同向。NoisyNet/residual
  幅度不是主因。

## 决定

- 停止 plane-miss guard、dwell、capture radius 和 extension 的继续扫描；
- 不启动 SAC/REDQ/DroQ；
- 不重跑 Goal27 失败臂；
- 下一 Goal 只读审计动作参数化与 reward：
  - terminal 前后 heading/pitch/speed command-to-actual 通道；
  - residual 三通道幅度、方向、rate clip、material saturation；
  - reward 各分量在 terminal 前窗口的占比与 terminal success/plane-miss 激活；
  - 只有在审计显示明确可修正通道后，才设计单因素训练 arm。

## 训练型 arm 的前置条件

1. 先做 `num_envs 128/256/512 × batch_size 256/512` 单卡吞吐基准；
2. 记录 `env_transitions/s`、gradient steps/s、GPU 利用率、显存、CPU、wall；
3. 按 seed 分 8 张 GPU，一卡一个 seed；
4. 扩大采样/批量必须独立预注册，不能混进正式 Gate；
5. SAC 只在出现“残差方向正确、但探索/样本效率不足”的直接证据后开启；
   REDQ/DroQ 只在样本效率证据明确时考虑。

## 证据

- `doc/goals/goal81_terminal_termination_event_audit_20261110.md`
- `doc/goals/goal81_result_20261110.md`
- `artifacts/lowlevel/goal81_terminal_events_20261110/goal81.json`
- `artifacts/lowlevel/goal81_terminal_events_20261110/goal81.md`
