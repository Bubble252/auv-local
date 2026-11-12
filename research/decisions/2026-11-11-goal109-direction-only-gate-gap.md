# Goal109 decision：方向单独过线，转事件级 terminal 审计

**日期**：2026-11-11<br>
**状态**：`DECISION_RECORDED / PIVOT_TO_GOAL110_EVAL_ONLY`

## 触发

Goal109 冻结 Gate 输出：

```text
NO_DIRECTION_SIGNAL
reason = direction_separates_but_magnitude_does_not_no_preregistered_branch
```

但 `direction_strong=true`，且严格来说出现的是“方向单独过线”。这不是 NoisyNet、
网络接线、residual saturation 或 C0 不可复现；是预注册分支没有覆盖的组合。

## 决策

```text
decision = pivot
```

不把结果事后改判，不打开训练，不扫 residual scale/channel gain。

## 下一步

开 Goal110 **eval-only**：以首次进入 2.0 m、closure 首次跌破 0.2 m/s 为锚点，
检查 rpm 方向翻转与 vertical/pitch/closure 的时间先后。

- 若 rpm 先反转、closure 后下降：才允许预注册方向保持的 rpm 参数化实验；
- 若 vertical/pitch 先恶化：转 terminal capture 几何或 terminal reward 审计；
- 若两者同时发生：只做联合预注册，不接受单通道 scale 扫描。

不改 reward、terminal、success、goal radius；不打开 held-out；不训练 PPO/SAC/
REDQ/DroQ/DADE。
