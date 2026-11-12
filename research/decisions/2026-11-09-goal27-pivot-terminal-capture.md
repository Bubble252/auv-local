# Goal27 decision: PIVOT

日期：2026-11-09<br>
决策：`pivot`，停止命令残差 PPO 主线

## 触发

同一接口效果门禁连续失败（C1、R1）。autoresearch 的 H1（命令跟踪惩罚）和
H2（GAE horizon）分别由 H1/G1 臂证伪；A1 的 action residual + reward-v2 只有
约 `0.02 m` 的弱改善，且 success 只在 seed 间交换。

## 放弃的路线

- 继续扫 `w_path_3d`、命令跟踪权重、gamma/lambda、rollout 长度；
- 通过换训练 seed、加预算或放宽门禁“救”命令残差；
- 把 C1p 的 privileged 结果当作部署收益；
- 把 A1 的 D3 基座优势误写成 A1 残差收益。

## 新路线

先围绕任务级 terminal capture/plane miss 建立预注册 Gate，再决定是否测试
supervised residual（BC minus PD/PI）或 off-policy residual。新路线必须：

1. 固定 D3、G26-P2、B0 基线；
2. 继续使用 dev `20–59`，不打开 held-out `100–149`；
3. 先跑 1-seed screen，不以长训练代替机制验证；
4. 在启动前写 Research Contract、阈值和失败后的停止规则。

## 证据

- `doc/goals/goal27_result_20261109.md`
- `research/blockers/2026-11-09-goal27-command-residual-collapse.md`
- `research/paper_reading/2026-11-09-goal27-residual-rl-credit-assignment.md`
