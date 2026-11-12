# Goal114 reading note：pitch onset handoff 与 rate/authority 的时间顺序

**日期**：2026-11-11<br>
**关联**：Goal114 `MIXED_PITCH_HANDOFF`<br>
**性质**：内部证据综合 + 控制/学习文献概念对照；不引入新训练方法。

## 1. 为什么看时间顺序

Goal113 的 3 m 静态窗口把 rate、authority、coupling 混在一个统计窗里，无法区分
“谁先坏”。Goal114 用 success envelope 和连续 3 帧持续越界，把问题改成：

```text
pitch command rate / rate error
  vs stern near-limit
  vs speed error / current
  vs pitch error
```

这不是因果识别，而是时间先后筛选。时间先后只能排除“某机制在 2 m 前完全没动”，
不能单独证明因果。

## 2. 结果映射

11 个 failure 中：

- 8 个是 mixed handoff；
- 2 个 rate first；
- 1 个 speed/current first；
- 0 个 authority first；
- 0 个 tracking first。

中位 rate/coupling onset 比 authority/tracking 早约 0.18 s，但这个差值小于
预注册的 0.20 s mixed 窗口。也就是说，rate 和 coupling 有“略早”的方向性信号，
但不足以压成单机制。

## 3. 控制视角的解释

一个串级姿态/航向控制系统里，command rate 是 reference 层信号，pitch rate 是
plant 响应，stern 是执行器权限，speed/current 是外部工作点。它们本来就可能在同一
个接近阶段同时变化：

- reference 加速 -> command rate 先升；
- plant 响应滞后 -> rate error 和 pitch error 随后形成；
- 低速/洋流改变 -> stern 权限和 plant 响应同时变差；
- stern 接近限幅 -> 不是“先坏”，而是与 rate/coupling 同步进入饱和。

因此 `MIXED_PITCH_HANDOFF` 更像“接近阶段的联合工作点变化”，而不是单一模块故障。
下一步应把 command/reference 的产生与 plant 响应拆开，而不是继续在 3 m/2 m 窗口
里投票。

## 4. 对 SAC 的含义

Goal108 已显示：

- PPO 增量很小且没有闭环收益；
- SAC 在未对齐的 path12 初始化下把确定性 residual 推到饱和；
- NoisyNet absent，frozen BC 22/22 tensor parity。

Goal114 没有新增“样本效率不足”的证据。mixed handoff 反而说明：在 command、
plant、authority 同时变化时，直接换 off-policy 算法会把多个耦合因素一起交给
replay 优化，难以归因。因此当前不应开 SAC/REDQ/DroQ。

## 5. 下一步 reading 方向

如果 Goal115 仍指向 joint handoff，优先查：

1. cascaded attitude/heading control 的 rate command 生成与限幅；
2. fin/stern authority 随 speed/current 的变化；
3. reference shaping / command filter 的相位滞后；
4. residual action 与所需 pitch 修正的符号一致性。

不把“更大 residual”当作默认修复；Goal104/107 已显示失败种子的更大 residual
反而有害。
