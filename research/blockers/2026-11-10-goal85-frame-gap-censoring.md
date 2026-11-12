# Goal85 blocker：done-frame success gap 在 endpoint-stop 下被 censoring 污染

日期：2026-11-10<br>
Goal：Goal85<br>
状态：`INVALID_METRIC_CENSORING`（不是算法 FAIL）

## 现象

Goal85 预注册 Gate 要求：

```text
eta_done vs p_nav_done paired gap 95% CI 包含 0
```

20-run / 800-episode 结果为：

```text
p_nav_done - eta_done = -24.88 pp
95% CI = [-31.00, -18.63] pp
```

按字面判读会得到“frame residual 仍然存在”。但进一步审计显示，done 帧的
`p_nav - eta` 距离没有系统偏差。

## 证据

### 1. 距离偏差接近 0

同一 800 episodes 的 done 帧：

```text
mean(p_nav_done_distance - eta_done_distance) = +0.0157 m
95% CI = [+0.0009, +0.0299] m
std = 0.133-0.148 m
```

`0.0157 m` 远小于预注册的 `0.05 m` 阈值；没有 Goal84 那种 `-20 pp` 量级的系统性
frame gap。

### 2. 成功样本被截断在半径边缘

endpoint-stop 一旦看到 `eta` 进入 tight radius 就终止，因此成功 episode 的
`eta_done` 距离紧贴边界：

| scenario | tight radius | eta_done mean（成功样本） |
|---|---:|---:|
| default | `2.00 m` | `1.985 m` |
| strict125 | `1.25 m` | `1.236 m` |

零均值 `p_nav` 噪声的标准差约 `0.135 m`，足以让边界附近的样本约一半落到球外：

| scenario / root | eta_done=true 中 p_nav 球外比例 |
|---|---:|
| default / baseline | `51.3%` |
| default / candidate | `47.3%` |
| strict125 / baseline | `52.9%` |
| strict125 / candidate | `47.2%` |

反过来，`eta_done=false` 的样本中只有 `0–8.7%` 的 `p_nav` 落在球内。

### 3. 这不是新的系统偏差

- `nav_bias=0`、`nav_drift=0`、`nav_usbl_gain=0`、所有 delay `0`；
- 10 万步 audit：`mean(p_nav-eta)` 最大 `1.44e-4 m`，drift 最大 `2.16e-4 m`；
- done 帧距离 bias 只有 `+0.0157 m`。

## 可证伪 hypothesis

### HCG-A：负 success gap 由 endpoint-stop censoring 产生

如果成立，成功样本的 `eta_done` 应集中在半径边缘，且 `p_nav` 球外比例接近 50%；
done 帧距离 bias 应接近 0。

**结果：SUPPORTED。** default 成功距离 `1.985/2.0`，strict125 `1.236/1.25`；
球外比例 `47.2–52.9%`；距离 bias `+0.0157 m`。

### HCG-B：如果存在真实 frame residual，done 帧距离 bias 应同向且大于 0.05 m

如果成立，`mean(p_nav_done - eta_done)` 应稳定偏离 0，并随 scenario/root 同向。

**结果：FALSIFIED。** 最大绝对 bias `0.0299 m`（CI 上界），均值 `0.0157 m`。

### HCG-C：在非截断帧上，success gap 应消失

如果成立，取首次进入 tight radius 前的固定窗口或只用失败样本，gap 不应保持
`-25 pp`。

**结果：PARTIAL。** 失败样本的 `p_nav` 球内比例只有 `0–8.7%`，没有“p_nav 假成功”
主导；成功样本的球外比例由边界噪声解释。

## 最便宜 falsification

不新增 rollout、不训练、不改 success。直接对已有 800 episodes 的
`terminal_p_nav_distance_m` / `terminal_eta_distance_m` 做配对统计。该分析已经
完成，成本为 0 个新 episode。

## 判定

```text
原 success-gap Gate 行：INVALID_METRIC_CENSORING
censoring-safe distance bias：PASS
算法主结果：仍然 valid，因为 official success 使用 eta_tight_at_done
```

不能把本条写成 `FRAME_RESIDUAL`，也不能据此改 `goal_radius`、success 或算法。
后续报告一律同时给出：

```text
done-frame distance bias
done-frame success gap (diagnostic only, censored)
near-miss shell / eta_min distribution
```

## 决策

`continue`：Goal85 的位置契约通过；学习残差在 nominal 下仍为负效果。按
`research/AUTORESEARCH_LOOP.md` 进入强制的 Goal85b plant mismatch，再进入
Goal86 的 BC 先验 / PPO-SAC / 传统控制器 + RL 残差对照。
