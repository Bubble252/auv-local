# Goal26 residual PPO path-reward blocker

日期：2026-11-08<br>
Goal：Goal26<br>
状态：`OPEN`；预注册 P1/P2 已停止

## 1. 现象

在 frozen D3 BC 上学习 action residual，并给 ALOS 环境追加 3D path reward，连续两个
预注册权重都没有产生有效的 path-tracking 改善：

- P1：`w_path_3d=0.15`；
- P2：`w_path_3d=0.30`；
- 两臂都使用 `4096 env × 128 steps × 200 iterations`、seed `26001`、GPU 1；
- 两臂都通过 success/safety/plane-miss/timeout 护栏，但都没有通过效果门禁。

## 2. 证据

40-seed dev、`endpoint_stop`、`realistic_calibrated`、CPU/float64：

| scenario | method | success | pre-goal RMSE | full RMSE | plane miss | residual/base L2 |
|---|---|---:|---:|---:|---:|---:|
| default | D3 BC | 0.7250 | 2.849170 | 2.763411 | 0.2750 | 0.0 |
| default | P1 | 0.7000 | 2.835630 | 2.754396 | 0.3000 | 0.02724 |
| default | P2 | 0.7250 | 2.832894 | 2.750314 | 0.2750 | 0.01806 |
| strict125 | D3 BC | 0.5000 | 2.829135 | 2.752011 | 0.5000 | 0.0 |
| strict125 | P1 | 0.4750 | 2.814198 | 2.742351 | 0.5250 | 0.02705 |
| strict125 | P2 | 0.4750 | 2.811860 | 2.738334 | 0.5250 | 0.01795 |

关键比较：

- P1 pre-goal 改善 `0.013540 m` / `0.014937 m`；
- P2 把路径奖励翻倍后，改善只增加到 `0.016275 m` / `0.017276 m`；
- P2 的 residual/base ratio 反而从约 `2.7%` 降到约 `1.8%`；
- 两臂 safety 和 timeout 都是 `0.0`；
- default plane miss 在 P2 回到 D3 的 `0.275`，strict125 仍增加 `0.025`；
- zero-residual 机制已验证：初始 residual 为 `0.0`，frozen BC 参数漂移为 `0.0`。

路径奖励并没有“弱到完全没进 reward”：

- `w_path_3d=0.15/0.30` 已显式进入 `_path_reward`；
- P2 的平均 episode reward 比 P1 更负（约 `-6.10` 对 `-5.36`），说明 reward 项确实
  改变了；
- 但行为没有相应改善，而是 residual 幅度收缩。

## 3. 已排除项

- 不是 zero-residual 初始化错误：初始 residual 为 0，frozen BC audit 通过；
- 不是 checkpoint 选错：固定使用第 200 轮 `final.pt`，SHA256 已记录；
- 不是传感器 profile 或 dev seed 不一致：两臂共用同一 40-seed dev 协议；
- 不是 success 护栏被牺牲后换来的假改善：success 没有提升，plane miss 没有改善；
- 不是单纯 reward 权重不够：预注册预算内翻倍到 `0.30` 仍失败，且 residual 更小；
- 不是 path-zero 诊断新增的完全路径依赖：D3 本身 path-zero success 也是 `0.0`，该诊断
  在这里不可单独作为归因。

## 4. 可证伪 hypothesis

### H1：路径奖励与终止/指令跟踪奖励冲突

机制：success-first 环境下，终止和指令跟踪项的梯度远强于 path 项；策略选择把 residual
缩到接近零来避免破坏 success，而不是用 residual 修路径。

可证伪预测：对 D3/P1/P2 的固定轨迹做离线 reward decomposition，若 path 项只占很小的
return/advantage 份额，H1 成立；若 path 项已占显著份额但 residual 仍收缩，则 H1 不足，
应转向 residual 表达或信用分配。

### H2：直接 action residual 的探索不够结构化

机制：`[rudder, stern, rpm]` 上的小 residual 噪声大、时间上不连贯，难以形成持续的横向/
垂向路径修正。ZPRL 类工作指出 action residual 容易出现 noisy/poorly structured
exploration；本项目 residual/base L2 ratio 只有 `1.8%–2.7%`，与该机制一致。

可证伪预测：在冻结 actor 的 64D `z_t` 上学习零初始化 residual，若 1-seed screen 的
pre-goal 改善明显超过 action residual，H2 成立；若仍无改善，则不是接口问题。

### H3：128-step 的信用分配窗口太短

机制：50 Hz 下 128 steps 只有 `2.56 s`，而路径修正需要数十秒；PPO 很难把路径误差归因
到早期动作，策略因此只学到局部动作平滑。

可证伪预测：保持其余条件不变，增加 decision stride/rollout horizon 或使用 action
chunking；若 pre-goal 改善显著超过 P1/P2，H3 成立；若仍无改善，则不是 horizon 问题。

## 5. 最便宜的 falsification experiment

已执行的 P2 是 H1 的最便宜预注册检验：只把 `w_path_3d` 从 0.15 翻倍到 0.30，其他条件
固定。结果是失败，因此不再扫第三档。

下一步最便宜的新实验不是再训练大模型，而是：

1. 在固定 D3 轨迹上做离线 reward decomposition，量化 path/terminal/command 项占比；
2. 若 path 份额不低，直接做一个 1-seed 的 `z_t residual` smoke；
3. 只有 smoke 的动作平滑性和 zero-residual 都通过，才扩到 40-seed dev。

## 6. 当前结论与 next

- Goal26 的 action-residual + path-reward 路线在本预算内 **停止**；
- 不扩训练 seed，不扫第三档 `w_path_3d`，不修改成功口径；
- 下一步优先级：`z_t residual` > horizon/action chunking > 继续 action residual 权重搜索；
- 新路线必须先通过 zero-residual、success-first 护栏和 1-seed screen，再谈正式训练。

## 7. 外部文献交叉检查（2026-11-08）

已补读并记录：

- `research/paper_reading/2026-11-08-differentiable-residual-navigation.md`

三篇文献与当前失败模式的对应关系：

- DiffPhysDrone 用可微物理 + 150-step BPTT（约 10 s）和稠密物理损失，不是 PPO；说明
  “长窗口一阶梯度 + 稠密目标”可能是关键差异；
- FoPG RPL 明确报告 PPO / PPO RPL 在 perceptive local navigation 上训练 `1e8`
  samples 仍不能收敛，而 SHAC RPL 在约 `1e5` steps 内达到 `>99%`；
- E2DRL 同时使用 SAPO、软穿透奖励、终止/进度项和 DAgger teacher-student，说明
  success/termination 不能只放在评估里；
- ZPRL 指出 action residual 可能产生 noisy、poorly structured exploration，支持测试
  `z_t` latent residual。

因此当前 blocker 的主判断更新为：

> 奖励设计有结构性缺口，但 Goal26 不改善的主因不能只归因于 reward 权重；PPO 的信用
> 分配窗口、直接 action residual 的表达方式，以及 BC/ALOS 在高曲率路径上的系统性
> horizontal error 同样是主因候选。

后续仍然先做离线 return/advantage decomposition，再做 reward-v2 与 `z_t residual` 的
单变量 screen；不继续扫 `w_path_3d`。
