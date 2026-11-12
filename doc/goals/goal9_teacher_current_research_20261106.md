# Goal9 教师抗流方法调研与 V-0D 教师筛选

日期：2026-11-06

状态：只读调研 + 纯教师 20-seed 评估；未重新训练 BC，未修改旧配置。

## 问题

V-0D 的 BC-0D 在 default 下 3D RMSE 约 `12.24 m`，strict125 下约 `12.69 m`。
此前判断主因不是 BC 拟合失败，而是它模仿的教师本身受时变海流影响。现在要回答：

1. 仓库里是否已经有可观测海流下的抗流方法？
2. 这些方法能否直接作为新的 BC 教师？
3. 收益到底来自海流补偿、速度下限，还是终端逻辑？

## 已有的相关组件

### 真实的 current field

`guidance.compensate_current` 和 `base_controller.use_current_field` 可以读取真实海流，
但旧 PC 实验默认关闭。原因是环境契约要求 actor/部署侧只能看到传感器估计，
不能直接使用真实海流；相关测试也会把打开真实 current 的配置判为泄漏。

因此不能简单把 `compensate_current` 打开后作为可部署教师。它只能作为训练特权
诊断，不是本次首选。

### 可观测海流估计

环境已经有 `current_estimate_body`，带延迟、低通、噪声、bias、随机游走和失效窗口。
`_current_estimate_for_guidance()` 会把它转换到 NED，并检查：

- 估计年龄不超过 `current_estimate_max_age_s`；
- 速度不超过 `current_estimate_max_speed_mps`；
- 数值有限。

这是一条真实可部署的观测链路，不是真实海流作弊。

### ALOS-R 当前蟹行

`BatchedThreeDAlosGuidance` 已有 `current_crab_enabled`：

- 用估计海流计算“地速方向 -> 水速方向”的 crab angle；
- 有最大 crab 角限制；
- 估计失效时回退到普通 ground-direction 指令；
- 不会把真实 current 直接塞进 actor。

同时还有：

- `robust_terminal_enabled`：进入终点捕获后直接朝目标修正；
- `route_speed_floor_enabled`：用 DVL 地速反馈维持最低前进速度；
- `saturation_governor_enabled`：按饱和程度降低命令速度。

### DADE

DADE 是历史编码/隐式海流推断分支，不是显式海流补偿器。此前在 residual policy
上没有稳定收益，且本轮问题是教师标签质量；因此这次不把 DADE 当作首选的教师修复。

## 纯教师 screening

评估器：

```text
scripts/evaluate_pc_teacher.py
```

固定：seed `0–19`、CPU、float64；环境为 V-0D 的 reference/current/sensor 设置。

| 组别 | 配置 | 3D RMSE (m) | success | timeout | saturation |
|---|---|---:|---:|---:|---:|
| R0 | 当前 V-0D 教师 | `11.99` | `0.15` | `0.85` | `0.0091` |
| R2B | ALOS-R：crab + route floor + robust terminal | `4.35` | `0.85` | `0.15` | `0.1838` |
| R3 | ALOS-R：crab + governor | `7.90` | `0.35` | `0.65` | `0.0935` |

R2B 配置：

`configs/lowlevel/odarrl_control_alos_r2b_crab_floor_init100.json`

SHA256：

`6a961c7238a97c43a4bcf7a03f09359593dc0883c5f5ba7be1b3a85524625a36`

产物：

- `artifacts/lowlevel/pc_v0d_teacher_current_screen_20261106/`
- smoke：`artifacts/lowlevel/pc_v0d_teacher_current_smoke_20261106/`

## 组件消融

在 R2B 基础上只关闭一个组件，仍为 seed `0–19`：

| 组别 | 关闭内容 | 3D RMSE (m) | success | timeout |
|---|---|---:|---:|---:|
| R2B | 无 | `4.35` | `0.85` | `0.15` |
| NO_CRAB | 关闭海流蟹行 | `4.74` | `0.80` | `0.20` |
| NO_FLOOR | 关闭 route speed floor | `4.87` | `0.70` | `0.30` |
| NO_BOTH | 同时关闭 crab 和 floor | `4.87` | `0.70` | `0.30` |

解释：

- 最大收益来自 `robust_terminal`：R0 -> NO_BOTH 已经解释了大部分 RMSE 和 success 改善；
- 海流蟹行仍有小幅增量：相对 NO_CRAB，3D RMSE 改善约 `0.38 m`，success 增加 `0.05`；
- route speed floor 也有小幅增量，但单独看不如 crab 稳健；
- R3 的 saturation governor 组合反而明显弱于 R2B，不采用。

所以不能把 R2B 的收益全部归因于“抗流”；它是“更强的终端捕获 + 可观测海流蟹行 +
速度下限”的组合。若论文要宣称海流补偿，需要单独报告 NO_CRAB 对照。

## strict125

使用同一 V-0D reference，仅把目标半径改为 `1.25 m`：

| 组别 | 3D RMSE (m) | success | timeout |
|---|---:|---:|---:|
| R0 教师（V-0D strict125） | `12.71` | `0.05` | `0.95` |
| R2B 教师（strict125） | `9.39` | `0.15` | `0.85` |

strict125 下 R2B 仍改善约 `26%` RMSE，但 success 只有 `0.15`，不能直接说“已经解决”。
strict125 的终点半径更紧，R2B 的终端捕获参数仍可能偏宽松；此外 strict125 下
material saturation 为 `0.2116`，高于 default 的 `0.1838`，需要作为风险记录。

## 结论与建议

1. 仓库里确实已有比当前 V-0D 教师更好的方法：ALOS-R R2B。
2. 但 R2B 不是纯海流补偿的胜利；主要收益来自 robust terminal，海流蟹行是增量。
3. R2B 适合作为下一版 BC 教师的候选，不应直接替换 actor，也不应把 DADE 当默认修复。
4. 若重采 BC，必须保持 V-0D actor 契约不变：state + 8D path，command 仍为空；
   只替换教师动作标签和示范分布。
5. 新 BC 仍需 default 和 strict125 双环境 20-seed 门禁；strict125 不能只看 default
   的改善。
6. 不应直接打开真实 current field；教师只能使用带延迟/噪声/失效的 current estimate。

## 下一步（尚未执行）

建议先新增一个 V-0D 教师配置的候选副本，把 R2B 的 guidance 字段合并进 V-0D 环境，
保留 `pc_control_mode=path_learned_speed` 和 actor 0D 契约；然后：

1. 用新教师采集 20-seed pure-teacher 对照，确认 default/strict125；
2. 再用同一采集脚本生成 0D BC 示范，训练新的 `R-BC-0D-current`；
3. 以 default 和 strict125 的 20-seed 闭环结果决定是否进入残差 PPO；
4. 不覆盖旧 `R-BC-0D` checkpoint、旧配置或旧 artifacts。

## 产物

```text
artifacts/lowlevel/pc_v0d_teacher_current_screen_20261106/
artifacts/lowlevel/pc_v0d_teacher_current_ablation_20261106/
artifacts/lowlevel/pc_v0d_teacher_current_strict125_20261106/
logs/pc_v0d_teacher_current_screen_20261106/
logs/pc_v0d_teacher_current_ablation_20261106/
logs/pc_v0d_teacher_current_strict125_20261106/
```
