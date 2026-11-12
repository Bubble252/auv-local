# 阅读笔记：ground/water velocity、DVL 与速度反馈契约

**日期**：2026-11-12<br>
**服务 Goal**：Goal163 speed-envelope audit / Goal164 speed-contract repair<br>
**检索范围**：AUV 运动学、DVL bottom/water track、海流补偿、速度环 anti-windup

## 1. 结论

水下机器人的速度量至少有三种，不能混用：

1. **ground velocity**：相对海底/NED 的速度，`eta` 的时间导数；
2. **water-relative velocity**：相对水体的速度，DVL water track 测量；
3. **current velocity**：水体相对海底的速度。

在 body frame 下应满足：

```text
v_ground_body = v_water_body + current_body
```

本项目动力学把 `nu` 用作 `v_ground_body`（`position_dot = R @ nu`，
水动力用 `nu - current`），但 `ground_velocity_body()` 和 `sensors.py` 仍按
`nu = v_water_body` 处理，导致 DVL bottom、actor 观测、reward 和 evaluation
同时多加了 current。

## 2. 参考框架

### 2.1 Fossen 的 6-DOF 运动学

**Reference**：Thor I. Fossen, *Handbook of Marine Craft Hydrodynamics and
Motion Control*, Wiley, 2011. DOI `10.1002/9781119994138`。

要点：`eta_dot = J(eta) nu` 中的 `nu` 是 body-fixed velocity；若模型显式包含
海流，必须区分 ground-relative 与 water-relative 两种 `nu`，并保证
`eta_dot`、水动力、DVL 和控制器使用同一约定。

对本项目的含义：`eta` 差分已经证明当前实现中的 `nu` 是 ground velocity。
任何返回 `nu + current` 的“ground” helper 都是重复计流。

### 2.2 DVL bottom-track 与 water-track

**Reference**：Kinsey J. C., Eustice R. M., Whitcomb L. L. *A survey of
underwater vehicle navigation: recent advances and new challenges.* IFAC
Conference on Manoeuvring and Control of Marine Craft, 2006.

要点：DVL bottom-track 测的是相对海底的速度（ground velocity）；water-track
测的是相对水体的速度。两者之差可用于估计海流，但不能把 bottom-track 再叠加
一次 current。

对本项目的含义：`sensors.py` 应生成
`dvl_bottom = nu`、`dvl_water = nu - current_body`；当前实现正好相反。

### 2.3 速度环 anti-windup

**Reference**：Bui P. D. H., You S.-S., Kim H.-S., Lee S.-D. *Dynamics
modeling and motion control for high-speed underwater vehicles using
H-infinity synthesis with anti-windup compensator.* Journal of Ocean
Engineering and Science, 2022. DOI `10.1016/j.joes.2021.07.002`。

要点：执行器饱和时，积分器必须条件积分、back-calculation 或显式限幅；仅有
`max_integral` 常数不等于 anti-windup 在减速阶段有效。

对本项目的含义：Goal163 已看到 terminal 超速步的 integral 接近 400 rpm cap。
修完速度语义后，如果真实超速仍存在，应优先做条件积分/回收，而不是加 reward
penalty 或换 RL 算法。

## 3. 与本项目证据的连接

- `nu-vs-eta` MAE = 0.000：`nu` 是地速；
- `trace ground-vs-eta` MAE = 0.286/0.360 m/s：旧 ground 字段多计 current；
- 旧超限 7.2%/14.7%，真实超限 0.31%/0.51%：评价 bug 是主因；
- 真实超速全在 terminal phase：残余问题是终点减速/积分回收，不是全程强流；
- ALOS 命令是水相对速度，控制器用 `nu-current` 跟踪；actor/reward 却用
  `nu+current`，学习信号与控制目标不一致。

## 4. 最小可迁移组件

1. **单一速度契约**：在配置中显式声明 `nu` 是 ground velocity；
2. **两个 helper**：`ground_velocity_body()` 返回 `nu`，
   `water_velocity_body()` 返回 `nu - current`；
3. **传感器修正**：DVL bottom/water 分别返回 ground/water；
4. **消费者审计**：actor、critic、reward、hold、evaluation 明确标注使用哪一个；
5. **版本化 replay**：旧配置显式保留 legacy，避免覆盖历史结果；
6. **corrected zero replay**：先证明真实地速，再谈 terminal anti-windup。

## 5. 不迁移的部分

- 不把“评价字段错了”当成“真实超速不存在”；真实峰值仍略高于 2.5；
- 不直接放宽 2.5 m/s；
- 不因为速度语义 bug 就换 PPO/SAC/REDQ/DroQ；
- 不用 reward penalty 替代传感器/控制器的契约修复。

