# Reading note：路径跟踪评价为什么不能只看终点

**日期**：2026-11-12<br>
**用途**：Goal170 路线健康指标设计依据

## 1. 文献线索

`paper-search` 本次查询覆盖 arXiv、OpenAlex、Crossref；Semantic Scholar 和
DBLP 有 429/解析失败，阅读结论不依赖单一来源。

与当前问题最相关的方向：

- 3D AUV path following / ALOS：关注 cross-track error、along-track error、
  vertical error、heading/pitch tracking 和 control effort，而不是终点是否恰好
  落入一个小球。
- prescribed-performance tracking：把误差的瞬态/稳态包络作为主要评价对象，
  强调整段轨迹的约束满足情况。
- trajectory evaluation：单一 endpoint error 会掩盖路径中段的大偏差，需要
  时间对齐后的整体误差、分段误差和最大/分位数误差。
- segment-based trajectory similarity：分段统计能定位误差发生在哪个路径区间，
  比 pooled RMSE 更能解释控制策略的差异。

代表性检索结果：

| 主题 | 代表工作 | 对本 goal 的启示 |
|---|---|---|
| 3D AUV path following | Zieglmeier et al., *Data-Enabled Predictive Control with Predictive Adaptive Line-of-Sight Guidance for 3-D Path Following of AUVs*, arXiv:2510.25309 | 用 3D 路径误差和制导误差评价，不把 terminal success 当唯一目标 |
| prescribed performance | Wang et al., *Path Following With Prescribed Performance for Under-Actuated AUVs*, IEEE T-ITS 2023 | 关注误差包络、瞬态和整段约束 |
| trajectory tracking 评价 | Palacín et al., *Evaluation of the Path-Tracking Accuracy of a Three-Wheeled Omnidirectional Mobile Robot*, Sensors 2021 | 同时报告 RMSE、最大误差和分段/重复实验统计 |
| trajectory metric | Daum et al., *Comparing Trajectories from Positions Alone: Curvature-Based Time Alignment and Drift Error Metric*, arXiv:2609.14936 | 进度/时间对齐后比较轨迹，避免只用终点 |
| segment similarity | Mao et al., *A Segment-Based Trajectory Similarity Measure in the Urban Transportation Systems*, Sensors 2017 | 分段误差可定位局部失败区间 |

## 2. 本项目的可计算量

当前 `evaluate_pc_sac.py` / `evaluate_pc_ppo.py` 已经记录：

- `path_progress`；
- `path_error_3d`；
- `path_horizontal_error`（signed cross-track）；
- `path_vertical_error`（signed vertical）；
- `ground_velocity_body_mps`、`eta_ned`、`time_s`；
- residual/action saturation 和 terminal metadata。

因此不需要改环境，也不需要新增 reward，就可以构造 route-health replay。

## 3. 采用的评价原则

1. **时间加权 + 弧长加权并存**：前者惩罚停滞时的持续偏差，后者避免停滞导致
   某些路径段被过度采样。
2. **绝对误差 + signed bias 并存**：只看 RMSE 会掩盖持续单侧偏航；只看均值
   会掩盖大幅振荡。
3. **整段 + 五段 + 10% 关键点并存**：整段回答总体是否更好，分段回答在哪里
   更好，关键点回答优势是否只出现在终点附近。
4. **完成度/效率为约束**：不能通过减速、悬停或缩短实际推进来降低误差。
5. **success 降级为二级诊断**：终点进圈仍然重要，但不能代替整条路线健康度。

## 4. 不采用的做法

- 不用单一 `success` 作为主 Gate；
- 不用一个没有预注册权重的“综合分数”掩盖横向/垂向/分段退化；
- 不用只在终点附近计算的 near-goal 指标替代整条路线；
- 不因为某个 seed 的 success 翻转就扩 seed 或改 radius。

## 5. 后续可扩展

若 Goal170 显示分段差异稳定，下一轮可把关键点升级为“路径走廊/关键航点”
评价：在每个 10% 弧长位置设置局部通过窗口，并记录错过窗口后的恢复时间。
这仍然不改变控制契约，只是让评价更接近真实任务中的路线健康度。
