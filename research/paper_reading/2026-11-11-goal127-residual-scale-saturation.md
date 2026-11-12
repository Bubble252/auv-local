# Reading note：residual RL 的尺度、critic 与饱和

**日期**：2026-11-11<br>
**关联**：Goal126 / Goal127<br>
**问题**：SAC P-A 残差在 1M 后跨 seed 退化，是动作 scale、值学习，还是 reward/terminal credit？

## 检索路径

- arXiv API：`residual reinforcement learning`、`soft actor-critic saturation`、
  `AUV reward shaping`；
- 关键词筛选：bounded residual、base-policy prior、critic normalization、tanh 饱和；
- GitHub 搜索未找到这些新论文的稳定公开训练仓库，因此本次只使用论文结论，不复制
  未验证代码。

## 论文 1：What Makes Value Learning Efficient in Residual Reinforcement Learning?

- arXiv:2602.10539
- 要点：residual RL 的两个瓶颈是 base-policy transitions 缺失导致的 “cold start”，
  以及 residual 相对 base action 的 “structural scale mismatch”；DAWN 用 base-policy
  warmup 和 critic normalization 解决。
- 对本项目：P-A 已用 frozen BC residual 作为 base prior，但 1M SAC 的 critic 仍可能
  没有足够 near-base 价值锚点。Goal127 的低倍率诊断说明“只改增益”不够，下一步
  应检查 reward/critic 是否能区分 near-success 与 timeout。

## 论文 2：CANDO: Conservative Action-Noise Directional Optimization

- arXiv:2611.04543
- 要点：action-space residual 往往比固定 latent noise-space 更样本高效；但需要
  conservative/gated adaptation 保护预训练策略结构。
- 对本项目：当前 P-A 是 action-space residual，架构方向合理；但无约束 SAC 在
  7/8 seed 上破坏 base prior，支持后续用 conservative residual authority，而不是
  直接扩大动作或更换算法。

## 论文 3：Unthrottling the Tanh Jacobian in SAC: A Negative Result

- arXiv:2609.09478
- 要点：SAC 的 tanh Jacobian 饱和是真实现象，但强行补 Jacobian 会导致策略几乎
  全程饱和并显著降低 return；饱和不等于任务最优解在边界。
- 对本项目：Goal126 在 1.0 倍率有 26.9% 动作饱和；但 Goal127 在 0.125–0.5 无饱和
  仍退化。因此不能把“解饱和”当作解决方案，也不能把高倍率饱和解释成全部原因。

## 论文 4：Residual Reinforcement Learning from Demonstrations

- arXiv:2106.08050
- 要点：在 conventional controller / demonstration policy 上学习 residual，可用
  sparse task reward 提升高维任务；关键是 base 与 residual 的接口清晰。
- 对本项目：P-A 的 `frozen BC + bounded residual` 接口本身有文献支持；当前问题
  不是接口不成立，而是 1M 学到的 residual 没有跨 seed 捕获 terminal success。

## 综合判断

```text
bounded action residual = 合理
base-policy prior = 合理
critic scale mismatch = 仍需检查
action saturation = 不是低倍率失败主因
reward/terminal credit = 下一优先项
```

## 对 Goal128 的约束

- 不引入新 RL 算法；
- 不改 success/goal radius；
- 先用 eval-only trace 检查 reward breakdown、terminal 几何和 pre-terminal 轨迹；
- 只有找到稳定、可证伪的 reward/terminal 缺陷，才允许重新训练。
