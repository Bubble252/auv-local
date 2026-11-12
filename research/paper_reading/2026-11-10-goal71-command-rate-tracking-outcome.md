# Reading note：command rate limiter 的 tracking 改善为何不等于 terminal capture

日期：2026-11-10<br>
关联 Goal：Goal71<br>
触发：H71-B 支持、H71-A 失败；limiter 降低 tracking error 但没有改善 true retention

## 1. 本地证据

- default original-lost 的 heading command-rate P90 从 `1.457` 降到
  `0.524 rad/s`，heading error 从 `0.930` 降到 `0.315 rad`，但 true delta 为 `-2`；
- strict125 的 heading command-rate P90 从 `3.489` 降到 `0.524 rad/s`，pitch
  error 从 `0.596` 降到 `0.410 rad`，但 original-lost 恢复 `0/7`，true delta 仅 `+1`；
- zero-residual 也没有收益：default `19 -> 17`，strict125 `11 -> 11`；
- 执行器未饱和，terminal phase 在两组都是 `1.0`。

这说明“command 更平滑、tracking 误差更小”与“进入 true capture 球”是两个不同
问题，后者仍受终端几何、速度和实际 authority 影响。

## 2. 检索

使用 `paper-search`：

```text
AUV command rate limiter terminal capture tracking
AUV command rate limiting tracking error terminal capture underactuated vehicle
command filter backstepping AUV trajectory tracking input saturation
```

时间范围 2021–2026。Semantic Scholar 连续 HTTP 429；全源并行检索等待过久后中止。
受限到 arXiv + Crossref 的非并行检索成功，返回 10 + 10 个 unique records。
以下只采用能核验标题/DOI/摘要的直接相关文献，不把泛化的 command-filter 控制律
当成 terminal capture 证据。

## 3. 直接相关文献

### Tombul 2021, “Simultaneous feedforward online command rate limiter filters for existing controllers”

- DOI：`10.3906/elk-2007-147`；Turkish Journal of Electrical Engineering &
  Computer Sciences。
- 摘要核验：OpenAlex 返回完整 abstract；Crossref 核验标题与 DOI。
- 核心：为已有 controller 增加 online command rate limiter，显式考虑 velocity、
  acceleration 和 jerk constraints；论文明确指出 demand signal 的 rate-of-change
  limitation 会改变 control effort 和 system response，并用 fin actuation 例子
  比较不同 rate-limiting scheme。
- 对 Goal71 的含义：limiter 不是“只去掉坏 command”的无害过滤器；它会改变闭环
  response。Goal71 中 tracking error 下降、但 speed error/终端轨迹和 true outcome
  改变，和这一机制一致。tracking 改善不能自动等同于任务成功。

### 终端捕获与估计器边界（本地已核验阅读链）

- Goal30 reading note：terminal capture/LOS/CTE 与普通 path tracking 不是同一指标；
- Goal49 reading note：actuator saturation、control allocation、tracking feedback
  是相互独立的通道；
- Goal65 reading note：plane crossing、dwell、`p_nav` 与 true `eta` capture shell
  必须分开；只改终止时间不会自动让载具进入收紧球。

这些本地证据共同约束 Goal71 的解释：limiter 改善 command tracking，并不能证明
terminal capture 的几何/估计器/label 问题已经解决。

## 4. 综合判断

1. command rate limiter 的收益首先是 command-side smoothness；它会牺牲一部分
   高频修正能力，并可能改变 speed/pitch 耦合与接近轨迹。
2. Goal71 的 H71-B 是真实的 tracking 改善，但 H71-A 失败说明 tracking 不是
   true retention 的充分统计量。
3. 下一步应优先做 terminal capture 的逐 seed 几何/时序审计，而不是把 limiter
   再调快、再扫 reward/gamma，或直接换 SAC/REDQ/DroQ。
4. 只有在 Goal72 明确显示优化/探索失败或样本效率失败时，才允许新开算法对照；
   当前没有这两类证据。

## 5. 对 Goal72 的约束

- 只读复用 Goal67/Goal71 rich trace，保持 dev `20–59`、held-out `100–149` 关闭；
- 配对检查 crossing 时刻、`eta`/`p_nav` 距离、terminal signed offset、
  speed/height 耦合、plane-miss timing 与 false-success；
- 不改变 reward、gamma、seed、checkpoint 或 controller；
- 若几何审计仍不能区分，再预注册 controller bandwidth/terminal-aware action
  参数化 arm；
- 不因为算力空闲而静默扩大 `num_envs`/batch 或启动算法替换。
