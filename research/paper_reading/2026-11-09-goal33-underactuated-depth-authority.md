# Reading note：underactuated AUV depth/pitch authority

日期：2026-11-09<br>
触发：Goal33 `NO_SUPPORTED_MECHANISM`；stern 限幅频繁但与 shortfall 反向，需要
外部文献判断下一步应查全程垂直跟踪还是继续查终端 authority。

## 1. 检索

使用 `paper-search` 查询：

```text
underwater vehicle depth control pitch fin saturation
AUV terminal docking vertical guidance control
```

通过 `7897` 代理检索 arXiv、OpenAlex、Crossref，共得到 34 条去重结果。直接检索
Semantic Scholar 遇到 429，arXiv 直连超时，已改用代理；本 note 只使用已核验的
OpenAlex/Crossref 元数据和摘要。

## 2. 关键文献

### Du, Zhou & Wang (2023)

**Reference Model-Based Deterministic Policy for Pitch and Depth Control of
Autonomous Underwater Vehicle**，DOI `10.3390/jmse11030588`。

摘要要点：DRL 控制 pitch/depth 时容易出现 command saturation 和 response
overshoot；作者用 reference-model / twin-delayed deterministic policy 平滑目标，
改善响应并保留鲁棒性。

对本项目的含义：pitch/depth 是一个受饱和和响应超调约束的独立控制问题。Goal33
看到 stern limit binding 频繁但 shortfall 与 binding 反向，说明不能只用“饱和更多”
解释失败；需要检查参考/响应相位和全程误差。

### Liu, Xiang & Duan (2024)

**ADRC-SMC-based disturbance rejection depth-tracking control of
underactuated AUV**，DOI `10.1002/rob.22312`。

摘要要点：在 underactuated AUV 上采用 ADRC-SMC pitch autopilot 与 adaptive LOS
双环，ALOS 估计 angle of attack 并补偿 inner-loop pitch tracking；有 field
experiments。

对本项目的含义：深度跟踪依赖 outer guidance 与 inner pitch loop 的耦合；
terminal angle/phase 和 pitch tracking 需要单独看。Goal33 的“更多 stern 限幅”
不是充分解释，和该文的双环视角一致。

### Zhong, Yu, Wang et al. (2022)

**Adaptive depth tracking of underwater vehicles considering actuator
saturation: Theory, simulation and experiment**，DOI
`10.1016/j.oceaneng.2022.112517`。

元数据已核验；摘要接口未返回。

对本项目的含义：执行器饱和是 depth tracking 的正式控制问题，但通常需要自适应/
bounded controller，而不是通过 terminal reference 参数扫参解决；下一步应先确认
饱和是共享工作区还是 shortfall 的排序变量。

### Yu, Zhong, Lian et al. (2021)

**An experimental study of adaptive bounded depth control for underwater
vehicles subject to thruster's dead-zone and saturation**，DOI
`10.1016/j.apor.2021.102947`。

元数据已核验；摘要接口未返回。

对本项目的含义：dead-zone 与 saturation 会改变深度响应；如果 Goal34 的
sign-matched closure 长期不足，应优先检查执行器死区/换向，而不是继续增加
pitch gain。

### Zuo, Wang, Xiao et al. (2021)

**A Unified Approach for Underwater Homing and Docking of over-Actuated
AUV**，DOI `10.3390/jmse9080884`。

元数据已核验；摘要接口未返回。

对本项目的含义：docking/homing 通常需要明确的 relative pose/guidance 结构；
本项目的 terminal plane capture 仍使用路径状态，不能把路径 RMSE 当作终点机制。

### Yang, Liu, Liang et al. (2024)

**Anti-disturbance control strategy in capture stage for AUV dynamic base
docking with optical guided constraints**，DOI `10.1016/j.oceaneng.2024.118946`。

元数据已核验；摘要接口未返回。

对本项目的含义：capture stage 被单独建模，terminal 误差与外部扰动/光学导引约束
共同作用；后续诊断应把 terminal 误差分段，而不是把整个 episode 混成一个标签。

## 3. 综合判断

1. **pitch/depth control 是饱和、死区、超调和双环耦合问题。** 现有文献支持把
   stern saturation 当作工作区描述，而不是直接当作 shortfall 的因果排序变量。
2. **Goal33 的反向相关不荒谬。** 更差 episodes 可能有更大的入口误差和更低的
   closure，但未必有更多角度限幅；限幅可能是共享基线现象。
3. **下一步应看全程和相位。** 优先检验 H-GLOBAL（40/30/20 m 垂直误差延续）和
   H-PHASE（sign-matched closure、换向滞后），而不是继续扫 authority 阈值。
4. **不建议现在改控制器。** 没有单一机制通过前，pitch gain、reward 或 terminal
   window 的参数修改都会缺少可证伪依据。

## 4. 对 Goal34 的约束

- 只读现有 Goal31 traces；
- 不训练、不新增 rollout、不打开 held-out；
- 先预注册 H-GLOBAL/H-PHASE/H-SLOPE 的判定阈值；
- 主指标是全程/分段垂直误差、sign-matched closure、换向时间；不是 path RMSE；
- 任何控制器候选必须等单一机制通过后再预注册。

## 5. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| Du et al. 2023, DOI 10.3390/jmse11030588 | OpenAlex 摘要 + Crossref 结果 | verified |
| Liu et al. 2024, DOI 10.1002/rob.22312 | OpenAlex 摘要 + Crossref/OpenAlex 结果 | verified |
| Zhong et al. 2022, DOI 10.1016/j.oceaneng.2022.112517 | OpenAlex/Crossref 元数据 | title verified |
| Yu et al. 2021, DOI 10.1016/j.apor.2021.102947 | OpenAlex/Crossref 元数据 | title verified |
| Zuo et al. 2021, DOI 10.3390/jmse9080884 | OpenAlex 元数据 | title verified |
| Yang et al. 2024, DOI 10.1016/j.oceaneng.2024.118946 | OpenAlex 元数据 | title verified |

未把只有标题的文献写成具体实验数值结论。
