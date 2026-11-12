# Reading note：residual authority 与动作参数化

**日期**：2026-11-11<br>
**触发**：Goal134 压力测评显示 7/8 SAC checkpoint 退化；唯一获益的 86201 学习
残差 RMS 最小。需要判断这是 authority/饱和问题，还是 learned direction 本身错误。

## 检索

使用 `paper-search`（arXiv / Semantic Scholar / OpenReview）检索：

```text
residual reinforcement learning action authority saturation robot control
```

返回 16 条，直接相关的有：

1. Rana et al., *Residual Skill Policies: Learning an Adaptable Skill-based
   Action Space for Reinforcement Learning for Robotics*, CoRL 2022,
   arXiv:2211.02231.
2. Eßer et al., *Action Space Design in Reinforcement Learning for Robot Motor
   Skills*, CoRL 2024, OpenReview `GGuNkjQSrk`.
3. Liu et al., *Visual Reinforcement Learning with Residual Action*, AAAI 2025,
   OpenReview `ZxdDnjogOb`.
4. Qu et al., *ResSafe: Learning Safety Filtering with Residual Reinforcement
   Learning for Humanoids*, CoRL 2026 Spotlight, OpenReview `rEmMzoUnpn`.

Semantic Scholar 本次返回 HTTP 429，未使用其结果；以上结论只依赖可见标题/
摘要与 OpenReview 元数据，不把未读全文的细节当成证据。

## 与本题相关的要点

### 1. 残差是“在基座策略上做小幅适配”的接口

Rana et al. 的摘要明确把 low-level residual policy 描述为在 skill space 上做
fine-grained adaptation。这个表述与我们的结构一致：frozen BC/ALOS 是基座，
SAC 只应提供小幅修正。若学习残差幅值接近或超过基座，接口语义已经从
“residual”变成“replacement”，闭环稳定性会下降。

### 2. 动作空间/authority 是一阶设计变量

Eßer et al. 的标题直接指向 action space design；Liu et al. 与 ResSafe 也都把
residual action 作为核心接口。这些工作支持一个保守但可检验的判断：

```text
在压力下，先调 residual authority / norm / cap，
再考虑换 RL 算法。
```

这不是说这些论文证明了“SAC 在 AUV 上 authority 过大”，而是说动作参数化本身
是必须单独控制的变量，不能用算法均值掩盖。

## 与 Goal134 证据的对应

Goal134 的 current15 dose-response：

| arm | scale 1.0 | 0.5 | 0.25 | zero |
|---|---:|---:|---:|---:|
| 86201 strict/default | 16/29 | 17/25 | 16/26 | 14/24 |
| 86202 strict/default | 8/20 | 9/23 | 13/24 | 14/24 |
| 86204 strict/default | 7/18 | 11/20 | 12/23 | 14/24 |

86202/86204 随 authority 降低单调回到 zero，支持 `AUTHORITY_DOMINANT`；
86201 在 0.25 仍有 +2/+2，说明其方向也有贡献。这个分解比“SAC 不行”更具体：

```text
多数 seed：authority 过大，残差把基座推坏；
86201：低 authority + 小幅正确方向。
```

## 对下一个 Goal 的含义

不应换 PPO/SAC，也不应开 REDQ/DroQ。更合理的下一步是保持 SAC，预注册一个
authority-regularized / norm-capped residual 训练项，让更多 seed 学到
86201 式的“小残差、正确方向”：

- 训练时约束 learner residual 的 RMS 或相对 frozen BC 的比例；
- 评价时仍用 deterministic mean，不改 success/radius/terminal；
- 先 screen 8 seeds，再决定是否扩 formal；
- 如果 norm cap 让所有 seed 都退化成 zero，则说明方向信号不足，应回到
  reward/terminal/state，而不是继续加 cap。

## 局限

- 检索结果不是 AUV 专用；没有直接证据表明这些方法在 REMUS100/ALOS 上有效。
- 只读了摘要/标题级信息；正式引用前应读全文。
- 当前 dose-response 只测 current15；不能外推到所有压力档。
