# Blocker：Goal139 fixed mixed-horizon 没有形成跨 seed 收益

**日期**：2026-11-11<br>
**Goal**：Goal139<br>
**状态**：`MIXED_SINGLE_SEED_ONLY / MIXED_SAFETY_STOP / PPO_NO_CREDIT_HORIZON_EFFECT`<br>
**证据**：`doc/goals/goal139_result_20261111.md`、`artifacts/lowlevel/goal139_mixed_horizon_credit_20261111/analysis/goal139_sac.json`、`.../goal139_ppo.json`

## 现象

- SAC M50：1/3 seeds 在至少两个 level 有信号；nominal strict 比 N0 低 15；
- SAC M75：0/3 seeds；combined20 出现 6 个 timeout，触发 safety stop；
- M50 的 residual L2、path ratio、authority ratio 与 N0 同量级，没有异常残差；
- PPO G980/G950 与 G995 几乎完全相同，residual 仍约 1%。

## 已排除

| 排除项 | 证据 |
|---|---|
| 评估使用了训练 mixed 起点 | `evaluate_pc_sac.py` 使用 Goal134 环境配置，起点为 0 |
| `start_progress_full_fraction` 没生效 | 256 reset census：M50 `50.39%` full，M75 `27.34%` full，起点 0/20 m 正确 |
| residual 过大/过小 | M50 `0.41–0.48`，N0 `0.39–0.48`；path/authority 未越护栏 |
| held-out 泄漏 | `100–149` 未打开 |
| PPO gamma 未实际改变 | G995/G980/G950 三档 manifest 与结果均已记录，success 差异最多 1 个 strict |

## 可证伪假设

1. **H139-B1：mixed-start 实现错误，fraction 实际没有按配置生效。**
   - 预测：真实 reset census 的 full-path 比例应显著偏离 50%/25%，或起点不落在
     0/20 m。
   - 最便宜证伪：用真实 M50/M75 配置重建 256 个 reset，直接统计 mask 与
     `episode_start_progress_m`。
   - 结果：**FALSIFIED**。M50 `50.39%`、M75 `27.34%`，起点正确。

2. **H139-B2：固定比例混合不够，静态 near-terminal 密度不能替代阶段式课程。**
   - 预测：提高 T15 比例（M50→M75）应带来方向一致改善；若反而更差，则说明
     “更多 near-terminal”不是有效课程轴。
   - 最便宜证伪：比较 M50/M75 的训练 episode 密度、评估 strict/default、timeout。
   - 结果：**SUPPORTED**。M75 训练 episode 更多，但 nominal/stress 更差，
     combined20 6 个 timeout。

3. **H139-B3：失败来自 SAC seed 方差，86203 的正信号可通过增加 seeds 救回。**
   - 预测：同一 seed 在多个 level 的正向应稳定，且 M75 也应出现相同方向。
   - 最便宜证伪：读现有 3-seed 配对；不扩 seed、不挑 seed。
   - 结果：**NOT SUPPORTED AS GATE**。只有 M50/86203 一个 seed 出现跨 level
     正信号；M75/86203 没有复制。不能把它当作部署收益。

4. **H139-B4：动作参数化、reward 或 terminal capture 是当前主因。**
   - 预测：若是主因，应看到 residual/path/authority 异常，或 terminal capture
     几何在成功/失败间系统性分离。
   - 最便宜证伪：复用本轮 residual/path/authority 统计与 Goal141 的 terminal
     contract 审计。
   - 结果：**PARTIAL**。本轮 residual 与 N0 同量级，动作幅度未异常；Goal141
     已独立发现 vertical envelope 偏松，但它不能解释 mixed-horizon 的 nominal
     guard 失败。该问题进入 Goal140/terminal-contract 队列，不与本 Gate 合并。

## 不做什么

- 不换 REDQ/DroQ；
- 不扩 8 seeds 或 5M formal；
- 不继续扫 full-path/T15 固定比例；
- 不挑 86203 报收益；
- 不打开 held-out，不改 reward/terminal/success/radius。
