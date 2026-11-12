# Reading note：terminal dwell、捕获球与 AUV 接近几何

日期：2026-11-10<br>
关联 Goal：Goal65<br>
用途：解释 50 步 dwell 失败，并约束 Goal66 的 estimator/calibration 审计

## 1. 本地证据链

- Goal59：deterministic residual 已饱和；
- Goal60/61：zero-residual 也复现 `p_nav/eta` terminal 偏差；
- Goal63：immediate plane-miss suppression 在 default 通过、strict125 失败；
- Goal64：strict125 的 6/6 lost true 为 plane miss，minimum `p_nav` 中位数
  `1.172 m > 1.00 m`，perp offset 中位数 `1.065 m`；
- Goal65：50-step dwell 不改变结论，strict125 仍 0/6 进入收紧半径。

## 2. 方法学结论

terminal capture 不应该只用“是否穿过一个平面”来判断。至少要区分：

1. 控制目标使用的 `p_nav` 与真实 `eta`；
2. GOAL 球半径；
3. goal-plane guard；
4. crossing 后的短暂 dwell；
5. 成功标签使用的 true capture 半径。

Goal65 证明，单独增加 dwell 只会延迟 plane-miss，不会自动让载具进入更紧球。
如果下一步仍只改终止时间，而不检查 `p_nav/eta` 偏差、terminal command 和 reward，
就会继续在同一几何失败上扫参数。

## 3. 为什么这不是 NoisyNet 问题

- dwell 不改变 policy 的均值动作或噪声采样；
- forced zero-residual 在 strict125 的 retention 为 `0.857`，但仍未通过 default
  zero Gate；
- Goal64 的 lost/retained residual L2 中位数几乎相同，scale/rate clip 均为 0；
- 关闭或缩小 NoisyNet 不会改变“未进入收紧球”的终端几何。

因此，当前证据不支持把 residual 放大、关闭 NoisyNet 或改用 SAC/REDQ/DroQ 作为
这个 Gate 的修复。

## 4. 外部检索状态

本次尝试用 `paper-search` 检索 AUV docking / terminal homing / capture-region
相关工作，但 Semantic Scholar 返回 429，OpenAlex/arXiv 组合查询在 90 秒内未完成。
因此本 note 不把任何外部论文结论当作本次 Gate 的证据；下一步若需要引用外部
方法，必须补充可核验的 reading note 后才能写入设计。

## 5. 对 Goal66 的约束

- 只读比较 `p_nav` 与 true `eta` 的 terminal 偏差；
- 检查 crossing 后的 heading/pitch/speed command 是否继续推离目标；
- 检查 reward/terminal capture 标签是否与真实目标一致；
- 不训练、不打开 held-out、不改正式 controller；
- 只有明确 optimization/exploration 证据才预注册 SAC 对照；
- 只有明确 sample efficiency 证据才考虑 REDQ/DroQ；
- 不把一次 dwell 失败写成“算法失败”，下一条主线必须先有预注册 Gate。
