# Goal37 reading note：局部路径框架、goal-frame 与初始偏置伪差

日期：2026-11-09<br>
触发：Goal36 `NO_SUPPORTED_MECHANISM`，参考投影大但负相关；需要确认
goal-tangent 初始垂直误差是否可能是终点切向投影伪差。<br>
检索：`paper-search`，arXiv 2018–2026；摘要通过 arXiv API 逐条核验。

## 1. 检索问题

- 3D path following 的误差应在什么坐标系定义？
- 局部路径切向/法向框架与终点切向投影是否是同一个量？
- 如果 goal tangent 与起点 route tangent 夹角很大，用 goal tangent 分解起点
  误差会不会制造一个近常数的虚拟垂直偏置？

检索 query：

```text
AUV 3D path following local tangent cross-track error frame
Serret-Frenet path following autonomous underwater vehicle
underwater vehicle line-of-sight path following guidance 3D
```

Semantic Scholar 在本次检索中返回 HTTP 429，重试 4 次仍失败；arXiv 结果可用。

## 2. 已核验论文

### Yao & Cao (2019/2020)

**Path Following Control in 3D Using a Vector Field**，arXiv `1911.02304`。

- 目标是把机器人控制到一条 3D desired path 的局部邻域；
- 论文推导 local path-following error 在路径足够小邻域内指数收敛，并证明局部
  ISS；
- 误差定义依赖 desired path 的局部几何，而不是某个终点平面的切向。

对本项目的含义：Goal35/36 用 `goal_tangent` 定义整条 episode 的垂直法向，会把
起点到终点的纵向距离投影到终点法向。该量是 **goal-frame** 量，不是
path-following 的 local error；两者在弯曲路径起点不等价。

### Yang, Chavez-Galaviz & Mahmoudian (2026)

**An Adaptive Fixed-Time Line-of-Sight Guidance Scheme for 3D Path Following of
Underwater Vehicles: Theory and Experiment**，arXiv `2609.14756`。

- 3D LOS 的指标是 cross-track error 和 vertical-track error；
- 论文报告 time-varying lookahead 在 curved path 上降低 cross-track/vertical-track
  RMSE；
- 这些误差是路径跟踪坐标中的局部误差，不是 goal-plane terminal capture。

对本项目的含义：局部 path frame 的 vertical-track error 才是与控制器反馈一致
的量；goal-plane 的 terminal vertical offset 仍可作为终端指标，但不能拿来解释
episode 起点的控制器初始条件。

## 3. 综合判断

1. **局部路径误差与终点切向投影必须分开。** 路径跟踪文献的 cross-track /
   vertical-track 是在当前路径点/局部切向附近定义；Goal35/36 的
   `initial_vertical_m` 是相对最终 goal tangent 的投影。
2. **几何上可以出现“大 goal-frame 初始偏差、小 local-frame 初始误差”。**
   当起点 route tangent 与 goal tangent 夹角较大时，终点切向投影会包含
   `-longitudinal * goal_tangent_z` 的近常数项。
3. **这不是说 Goal35/36 无效。** 它们正确描述了 goal-plane terminal metric；
   但把同一 goal-frame 量当成“控制器初始垂直偏置”会混淆标签。
4. **Goal37 应先做只读重投影。** 只有 local-frame 初始误差仍然大、或 local
   40/20 m 误差仍排序 terminal，才能把问题归因于真实初始状态或局部垂直控制。

## 4. 对 Goal37 的约束

- 按 `seed % 32` 重建 Goal31 path bank，不能直接用 trace 的 `goal_tangent`
  代替局部路径切向；
- 同时报告 goal-frame 与 local-frame，禁止混用；
- 先检查 path/goal/tangent/start jitter 一致性，再解释误差；
- 不训练、不新增 rollout、不改控制器、不打开 held-out；
- local-frame 40/20 m carryover 成立后，才允许预注册局部垂直跟踪候选。

## 5. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| Yao & Cao 2019/2020, arXiv 1911.02304 | arXiv API 摘要 | verified |
| Yang et al. 2026, arXiv 2609.14756 | arXiv API 摘要 | verified |
| Goal31 reading note, `2026-11-09-goal31-terminal-path-vertical-gap.md` | 本地 note + 其已核验摘要 | verified context |

本次没有把 Semantic Scholar 的 429 结果当作文献证据；只使用 arXiv API 已返回的
两篇摘要。
