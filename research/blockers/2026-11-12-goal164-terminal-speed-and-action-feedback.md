# Goal164 blocker：速度契约正确，但控制器拿到错误的速度反馈

## 现象

- corrected `nu_ground` zero replay 的真实 `eta` 峰值仍为 `2.558/2.623 m/s`；
- 超速仅占 `0.23%/0.41%`，且 100% 在 terminal phase；
- 非 terminal 超速严格为 0；
- legacy trace 与旧实现逐元素一致，说明不是速度契约改坏了历史行为。

## 已排除

- `nu`/`eta` 语义错误：MAE `4.7e-14`；
- `ground_velocity_body` 重复加流：corrected trace 中 `ground=nu` 完全一致；
- noisenet 主导：zero policy 同样超速；
- 强流转向主导：超速 step 的 turn rate 并不高于正常 step；
- 仅仅放宽课程：课程采样已经保证 `speed+current+variation<=2.5`。

## 新发现

PC residual/controller wrapper 的 actor feedback 分支使用：

```text
controller_nu = actor_state[:, 6:12]
```

但实际布局是：

```text
[p_nav(0:3), attitude(3:6), angular_rate(6:9), linear_velocity(9:12)]
```

所以控制器把 `p,q,r` 当成 `u,v,w`，速度环的 water surge 反馈失真。这会让速度
积分和 RPM 指令持续偏置，和 trace 中 overspeed requested RPM 远高于 normal
的现象一致。

## 可证伪 hypothesis

1. H165-A：修正排列并使用 DVL water-relative surge 后，terminal 峰值显著下降，
   无需算法替换。证伪：修正后峰值和超速比例不变。
2. H165-B：修正反馈后仍有 terminal 超速，则剩余问题才是 anti-windup 或终端
   减速策略。证伪：修正后所有 episode 都满足 `eta<=2.5`。

## 最便宜实验

先加合成测试验证 `actor_state -> controller_nu` 排列，再跑同一 corrected
zero-only replay；不训练，不扩 seed。

