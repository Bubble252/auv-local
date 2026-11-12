# 阅读笔记：速度包线、anti-windup 与增益调度

**日期**：2026-11-12<br>
**服务 Goal**：Goal162 speed-contract blocker / Goal163 speed-envelope audit<br>
**检索范围**：AUV 速度环、执行器饱和、anti-windup、RL/自适应 PID 和控制器参数残差

## 1. 结论

Goal162 的问题不是“再给 residual 一个更大的动作”，而是闭环速度安全前提失效。
速度环应由传统控制保证低速/高速可达性、积分限幅和反向制动；RL 只在明确的
姿态/路径通道上做有界修正。课程采样可以保证输入包线，但不能替代闭环安全性。

## 2. 论文

### 2.1 Ma et al.：速度环由 PI 保持，RL 控制舵/升降舵

**Paper**：Ma Dongfang, Chen Xi, Ma Weihao, Zheng Huarong, Qu Fengzhong.
*Neural Network Model-Based Reinforcement Learning Control for AUV 3-D Path
Following.* IEEE Transactions on Intelligent Vehicles, 2024; online DOI
`10.1109/TIV.2023.3282681`。

要点：ModelPPO 负责 rudder/elevator 等姿态控制，传统 PI 维持所需航速。其分解
避免让 RL 同时重学速度环和姿态环。

对本项目的含义：Goal162 已把动作限制为 heading/pitch gain residual，这是正确
方向；但速度环仍必须有独立的安全包线和 anti-windup，不能只依赖 manifest 的
`reference + current + variation` 上界。

### 2.2 Wang et al.：SAC 调自适应 PID，但基座仍是稳定 PID

**Paper**：Wang Yuxuan, Hou Yaochun, Lai Zhounian, Cao Linlin, Hong Weirong,
Wu Dazhuan. *An adaptive PID controller for path following of autonomous
underwater vehicle based on Soft Actor–Critic.* Ocean Engineering, 2024.
DOI `10.1016/j.oceaneng.2024.118171`。

要点：SAC 用于调节 PID，而不是直接替代底层闭环；速度和姿态的稳定性仍由控制
结构保证。

对本项目的含义：若 Goal163 证实速度环饱和，应优先修 PID/限幅，而不是把
`Δspeed` 加回 action。后续 SAC 对照也只比较姿态通道，不把速度安全交给策略。

### 2.3 Bui et al.：H-infinity 与 anti-windup

**Paper**：Bui Phuc Duc Hong, You Sam-Sang, Kim Hwan-Seong, Lee Sang-Do.
*Dynamics modeling and motion control for high-speed underwater vehicles using
H-infinity synthesis with anti-windup compensator.* Journal of Ocean Engineering
and Science, 2022. DOI `10.1016/j.joes.2021.07.002`。

要点：高速水下航行器的执行器饱和会使名义控制器失稳；anti-windup 被显式纳入
综合，而不是事后调参。

对本项目的含义：speed loop 的积分状态需要 hard cap、conditional integration 或
back-calculation；`max_speed_integral_rpm` 存在不等于 anti-windup 生效。Goal163
应记录 integral 与 RPM 的关系，并检查减速阶段是否能快速回收。

### 2.4 Guerrero et al.：饱和非线性 PID 的稳定性

**Paper**：Guerrero J., Torres J., Creuze V., Chemori A., Campos E.
*Saturation based nonlinear PID control for underwater vehicles: Design,
stability analysis and experiments.* Mechatronics, 2019.
DOI `10.1016/j.mechatronics.2019.06.006`。

要点：执行器饱和下的非线性 PID 需要明确的稳定性和抗扰设计，尤其在海流和参数
变化下。

对本项目的含义：强流下的 overspeed 不是单靠 reward penalty 能解决的问题；
应先用闭环轨迹把速度、转向和 current 的压力分开，再决定是否需要 ground-speed
safety cap 或更保守的 speed command envelope。

## 3. 与本项目证据的连接

- Goal149 已看到 current 越大，rudder/rpm saturation 与 ground-speed `>=2.5`
  比例同步上升；
- Goal162 的 H3 zero 与 gain policy 超限几乎相同，说明 residual 不是必要条件；
- Goal162 的典型超速点在 `18.7 m`，远在 `10→2 m` gain gate 之外；
- 课程 v2 已严格限制 `speed + current + variation <= 2.5`，所以后续不能靠
  再次改采样范围来“修”闭环 overspeed。

## 4. 最小可迁移组件

1. **独立速度安全包线**：以实际 ground speed 为准，不以 command speed 或配置上界
   代替；先做只读统计，再决定 soft/hard cap 的落点。
2. **anti-windup 诊断**：记录 `speed_integral_rpm`、commanded/actual RPM、speed
   error、减速事件和超速持续时间；验证积分在减速阶段是否回退。
3. **ground-speed cap**：优先在速度环输出层限制，而不是在 reward 中软惩罚；
   任何 cap 都必须有可证明的减速/制动路径，不能只是截断 RPM。
4. **转向瞬态分层**：按 turn rate、current、goal distance 分箱；区分直线超速和
   转向诱导超速。
5. **姿态 residual 保留**：Goal162 的 heading/pitch gain residual 可以继续研究，
   但必须等速度契约恢复有效后再比较 success，不把 `Δspeed` 放回策略动作。

## 5. 不迁移的部分

- 不因为速度超限就换 SAC/PPO/REDQ/DroQ；目前没有探索或样本效率证据。
- 不把速度超限解释成 NoisyNet 或 deterministic evaluation 噪声。
- 不用 reward penalty 代替执行器安全约束。
- 不放宽 `2.5 m/s`、success radius 或 held-out 规则。

## 6. 下一步

Goal163 先做 deterministic replay 和只读 trace 审计；只有 H162-S1 或 H162-S2
得到明确支持后，才写最小控制器修正，并重新跑同一 zero/policy 配对协议。
