# Reading note：terminal homing / cross-track authority

**日期**：2026-11-12<br>
**关联**：Goal167 `GATE167_B_FAIL_TERMINAL_HORIZONTAL_CAPTURE`<br>
**用途**：为 Goal168 的只读 terminal capture authority 审计提供最小文献背景；
不改变当前 Gate，不引入新算法。

## 1. 本轮检索

检索式：

```text
AUV terminal homing pure pursuit cross-track goal capture guidance
```

来源：arXiv、OpenAlex、Crossref、Semantic Scholar（Semantic Scholar 本轮
返回 429，未采用其结果）。这是 metadata-level screen，未下载全文；只用于
确定下一步应该审计“横向 terminal authority”，而不是直接改 reward 或算法。

## 2. 相关条目

### 2.1 AUV cross-track error 与 LOS guidance

- Ghosh & Mandal, 2024, *AUV way-point tracking at constant velocity:
  cross-track error (CTE) and line-of-sight (LOS) guidance*,
  Engineering Research Express, DOI `10.1088/2631-8695/ad36b0`.
- 与本项目的关系：终点失败的主项是横向 CTE，不是纵向；LOS/ALOS 的
  lookahead 与 cross-track gain 直接决定终端收敛速度。
- 可迁移审计：把失败 episode 的 required turn-rate 与 LOS 可提供的
  turn-rate 分开，避免把“制导未收敛”误判为“RL 信用分配”。

### 2.2 AUV homing/docking 的任务优先级

- Zuo et al., 2021, *A Unified Approach for Underwater Homing and Docking
  of over-Actuated AUV*, JMSE, DOI `10.3390/jmse9080884`.
- Thomas et al., 2021, *A Unifying Task Priority Approach for Autonomous
  Underwater Vehicles Integrating Homing and Docking Maneuvers*, JMSE,
  DOI `10.3390/jmse9020162`.
- 与本项目的关系：terminal homing 通常需要从 path-following 任务切换到
  goal-capture 任务；如果切换点太晚，横向误差没有足够时间收敛。
- 可迁移审计：检查 `goal_distance<=20 m` 的 direct-homing 是否真的持续
  生效，以及 endpoint 附近是否仍被 path projection 主导。

### 2.3 pure-pursuit / homing 的终端几何

- Rusnak, 2024, *Optimal pure-pursuit missile guidance*, Aerotecnica
  Missili & Spazio, DOI `10.1007/s42496-023-00185-2`.
- Li et al., 2023, *Three-Dimensional Optimal Homing Guidance Without
  Terminal Maneuverability Advantage*, JGCD, DOI `10.2514/1.G007483`.
- 与本项目的关系：终端捕获不仅取决于“朝向 goal”，还取决于剩余距离内
  可用的横向机动能力；没有 maneuverability advantage 时，晚切换会直接
  导致 plane miss。

## 3. 对本项目的含义

Goal167 已经证明：

- actuator residual 能产生大幅动作，但净终端位移只有约 `1 cm`；
- command residual 在 full authority 下也没有 success 翻转；
- 失败全部是横向 `GOAL_PLANE_MISS`。

因此下一步不应再增加 residual 训练预算，而应先做：

```text
required turn-rate / lateral acceleration
vs
available rudder / turn-rate / crab / current authority
```

如果 required 在包线内，再预注册 deterministic terminal homing 对照；
如果 required 超包线，则记录当前 success 契约在 P2 终点不可达，不能用
放大 radius 或换 seed 来救。

## 4. 不直接迁移

- 不因为文献支持 pure-pursuit 就直接替换 ALOS；
- 不在 Goal167 结果上事后改 reward/terminal/success/radius；
- 不把本次 metadata screen 写成全文复现；
- 不引入 PPO/SAC/REDQ/DroQ 之外的算法族。

