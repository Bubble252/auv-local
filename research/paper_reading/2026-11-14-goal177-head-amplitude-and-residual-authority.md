# Reading note：residual policy 的“小幅修正”与执行器权限

**日期**：2026-11-14<br>
**对应 blocker**：`research/blockers/2026-11-14-goal177-head-authority-residual-too-small.md`<br>
**检索问题**：residual policy 为什么需要足够可见的动作幅度，以及如何区分
“表征存在”与“动作权限没有形成有效 credit”。

## 1. 论文与来源

- Tom Silver, Kelsey Allen, Josh Tenenbaum, Leslie Kaelbling,
  *Residual Policy Learning*, arXiv:1812.06298 (2018)。
  https://arxiv.org/abs/1812.06298
- Charles Schaff, Matthew R. Walter,
  *Residual Policy Learning for Shared Autonomy*, RSS 2020,
  DOI `10.15607/rss.2020.xvi.072`；arXiv:2004.05097。
  https://arxiv.org/abs/2004.05097

检索同时尝试了 `paper-search` 的 Semantic Scholar / arXiv / Crossref 源；
Semantic Scholar 返回 HTTP 429，arXiv 和 Crossref 可用。上述两篇的标题、作者、
摘要和 DOI 已通过 arXiv API / Crossref 核对。

## 2. 方法/结论

### Silver et al., Residual Policy Learning

论文把 residual policy 定义为：在已有的、可能不可微的控制器之上学习一个修正，
而不是从零学习控制。它在 MuJoCo 的部分可观测、传感器噪声、模型不匹配和控制器
标定误差任务中显示，residual 能利用好但不够完美的初始控制器；论文还强调
residual 适合长时程、稀疏奖励任务，但并没有声称“任意小的 residual 都足以改变
闭环”。

对本项目的含义：

- 在 ALOS/frozen BC 上叠加 residual 的方向是合理的；
- 若 residual 经过 scale、channel gain、rate limit 后只剩很小的有效动作，闭环
  行为不移动并不奇怪；
- 必须先测 residual 的**有效动作幅度**，不能只看 head 参数是否更新。

### Schaff & Walter, Residual Policy Learning for Shared Autonomy

这篇工作学习一个模型无关的 residual，以最小幅度调整人的动作，使一组与目标无关
的约束得到满足；实验包括连续控制的 Lunar Lander 和 6-DOF quadrotor reaching。
它的核心不是“residual 越小越好”，而是 residual 仍需足够影响被控对象并满足约束。

对本项目的含义：

- 可以把冻结 BC 看成基础策略，把 PPO 看成受约束的修正器；
- 但若 PPO 分支的方向被 BC 分支覆盖，或其有效幅度低于动作噪声，credit 仍无法
  形成；
- 因此需要将 `bc_delta` 和 `ppo_delta` 分开记录，而不是只看 composed action。

## 3. 与当前 blocker 的对应关系

Goal176 已证明 `z_t` 对纵向变量的 held-out R² 很高，排除了“latent 完全没有纵向
信息”。Goal177 Stage A v3 进一步显示：

- head-only 方向存在，sign consistency 约 `1.0`；
- head-only authority 随 wrapper scale 上升，但 `head residual/action_std` 最大
  只有 `0.0737`；
- composed authority 的 `0.4–0.65` 主要来自冻结 BC residual，不能作为 PPO head
  的证据。

两篇 residual policy 论文支持“保留基础控制器并学习修正”的总体思路，但不能直接
证明本项目的 `frozen_bc_residual_scale=0.25`、`action_std=0.1` 或当前通道增益是
合适的。它们给出的可迁移原则是：residual 必须被单独测量，且其修正幅度必须足以
影响闭环。

## 4. 可复现实验与迁移边界

最便宜的后续实验是 eval-only head-only gain/scale sweep，并同时记录：

- `head residual/std`；
- `bc_delta` 与 `ppo_delta` 的符号/相关系数；
- magnitude/rate clipping；
- cross/vertical/forward route 与 success/timeout/safety。

不能直接从这两篇论文迁移出具体 AUV 的 scale 数值，也不能据此宣称“冻结 encoder
一定优于微调最后一层”。它们的边界是连续控制和 residual 学习范式；AUV 的
ALOS/BC 组合、通道单位、rate limit 与任务奖励仍需本项目自己的 Gate 证据。
