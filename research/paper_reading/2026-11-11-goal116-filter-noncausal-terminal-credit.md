# Goal116 reading note：filter 不是可调主因，下一层是 terminal closure/credit

**日期**：2026-11-11<br>
**关联**：Goal116 `FILTER_NONCAUSAL`<br>
**性质**：内部证据综合 + 控制系统概念对照；不引入新算法。

## 1. 结果映射

Goal115 看到的是：

- raw pitch target 高频；
- failure 的 limiter 输出更长时间贴 15 deg/s；
- failure 的 tracking deficit 更大。

Goal116 直接改变 limiter：

- 7.5、15、30 deg/s 与关闭四臂 success 为 `20/20/20/21`；
- 原 11 failure 全部仍是 failure；
- timeout/safety 全为 0；
- closest approach 中位数只移动约 1–8 cm。

因此 filter 是失效路径上的“过路点”，不是可以通过调大/调小直接修复的主因。

## 2. 控制视角

串级控制中，rate limiter 只改变 reference 的相位和斜率，不改变：

- reference 的终端几何目标；
- plant 的 vertical/pitch authority；
- dense reward 对 terminal credit 的分配；
- residual 在最后阶段的符号和幅度。

如果终点几何已经在 2 m 前偏离，或者末段 closure 掉到接近 0，调整 rate limiter
只能在有限范围内移动轨迹。Goal107/109/111 已经分别指出：

- terminal reward 可分，但 dense path/command 项份额更大；
- failure 的 terminal vertical/pitch/closure 明显更差；
- last-2s closure 掉到约 0.12 m/s；
- closest approach 失败集中在外侧。

因此下一步应看 terminal closure 和 action credit，而不是继续扫 filter。

## 3. 对算法选择的意义

当前没有证据支持：

```text
PPO/SAC 样本效率不足
```

Goal108 中 PPO 没有增量收益，SAC 在 path 参数化不对齐时大残差饱和；这不是
“同协议 SAC 更高效”的证据。Goal116 也没有出现“方向正确但探索不足”的模式。

只有 Goal117 直接证明：

1. residual/action 方向与所需 terminal 修正同号；
2. 在相同动作方向下，闭环仍不动；
3. 增大探索/样本预算可能改变结果；

才值得开预注册 SAC 对照。否则继续在动作参数化、reward credit、terminal
capture 内做最小干预。
