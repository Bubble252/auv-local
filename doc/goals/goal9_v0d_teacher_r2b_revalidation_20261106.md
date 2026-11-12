# Goal9 V-0D + R2B 教师复验执行记录

日期：2026-11-06

状态：v1 阶段 0–4 已完成，教师门禁 PASS、BC default 门禁 FAIL；
2026-11-07 的 v2.1 饱和动作补救也已完成并 FAIL；
同日 Goal16 的 crab-preserving 真限速教师 screen 完成，但两个变体均未过
rudder/stern 饱和门禁。三轮均按预注册停止，未进入残差 PPO。

## 重要技术修正

最初计划把 R2B 的 guidance 字段合并进 V-0D 的
`pc_control_mode=path_learned_speed` 配置。实测发现这不会启用 ALOS：

- `BatchedPCLowLevelEnv._path_command()` 只按路径切线生成 heading/pitch，
  完全不读取 `current_crab_enabled`、`route_speed_floor_enabled`、
  `robust_terminal_enabled` 等字段。
- 该错误组合的 2-seed 结果为 `RMSE=13.23 m, success=0.0`，与旧 R0 一致。
- 原始 R2B 配置的 2-seed 结果为 `RMSE=7.75 m, success=0.5`。

因此教师配置改为从 R2B 复制，显式使用 `pc_control_mode=alos` 和
`reward.mode=pc_command_track`。V-0D 的 actor 契约仍在 BC/PPO 层面保持：
state + 8D path，command_dim=0；不使用教师配置的 command reward 训练 PPO。

## 阶段 0：契约审计与配置

- [x] 记录 V-0D 基线 SHA256：
  `383b3c61e1bf8d227470788727ca056bc62740fb61ce43e7c3ab24dc03fcefcb`
- [x] 记录 R2B 候选 SHA256：
  `6a961c7238a97c43a4bcf7a03f09359593dc0883c5f5ba7be1b3a85524625a36`
- [x] 逐字段 diff：V-0D 与 R2B 的 plant、reference、current、sensors、
  termination、observation、actuator、normalization、evaluation 完全相同；
  差异只在 guidance、reward、experiment。
- [x] 新增可复现生成器：
  `scripts/make_pc_v0d_teacher_r2b_variants.py`
- [x] 生成教师 default/strict125/消融配置，以及 R0 strict125 基线。
- [x] 确认教师配置 `guidance.compensate_current=false`、`pc_control_mode=alos`、
  `reward.mode=pc_command_track`。
- [x] 确认 R0 strict125 是 V-0D 的纯副本，只改 `goal_radius_m=1.25` 和元数据。
- [x] `load_config` 加载全部新配置成功。
- [x] 契约/单元测试：
  `41 passed in 1.59s`
  （`tests/test_batch_alos_r.py`、`tests/test_lowlevel_pc_contract.py`、
  `tests/test_lowlevel_pc_v0d.py`、`tests/test_lowlevel_pc_teacher.py`）

## 配置与 SHA256

| 配置 | SHA256 | 用途 |
|---|---|---|
| `odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json` | `383b3c61...fcefcb` | R0 default |
| `odarrl_control_alos_r2b_crab_floor_init100.json` | `6a961c72...625a36` | R2B 原始候选 |
| `odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json` | `8e4ac934...464b23` | R2B default 教师 |
| `odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json` | `398dd219...d688ae` | R2B strict125 |
| `odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json` | `f1d6416e...4c3cef` | R0 strict125 |
| `..._nocrab.json` | `798b82c3...9a4f09` | 关闭 crab |
| `..._nofloor.json` | `7fa60215...9247c` | 关闭 speed floor |
| `..._noboth.json` | `0e3162eb...f8d8c3` | 关闭 crab + floor |
| `..._noterminal.json` | `ac3ba888...eb38c` | 关闭 robust terminal |

完整 SHA 以 `sha256sum configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_*.json`
和运行 manifest 为准。

## 阶段 0 smoke 证据

固定 seed 0–1、CPU、float64、V-0D 环境：

| 配置 | 3D RMSE | success | 说明 |
|---|---:|---:|---|
| 错误合并（path_learned_speed + R2B guidance） | `13.2261 m` | `0.00` | 证明 guidance 未生效 |
| R2B 原始配置 | `7.7460 m` | `0.50` | ALOS 教师参考 |
| 新教师配置（R2B 复制 + alos） | `7.7460 m` | `0.50` | 与原始 R2B 完全一致 |
| R0 strict125 | `13.2261 m` | `0.00` | 2-seed；正式 20-seed 见阶段 1 |

产物：

- `artifacts/lowlevel/pc_v0d_teacher_r2b_revalidation_20261106/smoke_r2b_default/`
- `.../smoke_r2b_original/`
- `.../smoke_r2b_merged/`
- `.../smoke_r0_strict125/`

## 阶段 1：正式 20-seed 纯教师复验

固定 seed 0–19、CPU、float64；服务器
`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`。

| 组别 | 3D RMSE (m) | success | timeout | safety | material sat |
|---|---:|---:|---:|---:|---:|
| R2B default | **4.3531** | **0.850** | 0.150 | 0.000 | 0.1838 |
| NO_TERMINAL default | 4.3531 | 0.850 | 0.150 | 0.000 | 0.1838 |
| NO_CRAB default | 4.7379 | 0.800 | 0.200 | 0.000 | 0.1851 |
| NO_FLOOR default | 4.8713 | 0.700 | 0.300 | 0.000 | 0.0887 |
| NO_BOTH default | 4.8663 | 0.700 | 0.300 | 0.000 | 0.0895 |
| ALOS_PLAIN default | 4.8663 | 0.700 | 0.300 | 0.000 | 0.0895 |
| R0 default | 11.9876 | 0.150 | 0.850 | 0.000 | 0.0091 |
| R2B strict125 | **9.3916** | **0.150** | 0.850 | 0.000 | 0.2116 |
| R0 strict125 | 12.7146 | 0.050 | 0.950 | 0.000 | 0.0087 |

- [x] R0 default
- [x] R2B default
- [x] R0 strict125
- [x] R2B strict125
- [x] NO_CRAB default
- [x] NO_FLOOR default
- [x] NO_BOTH default
- [x] NO_TERMINAL default
- [x] ALOS_PLAIN default（补充对照）

预注册门禁见
`doc/goals/goal9_v0d_teacher_r2b_revalidation_prompt_20261106.md` 阶段 2。

## 阶段 2：门禁与决策

- [x] 汇总 20-seed 表格
- [x] 主门禁判定：
  - R0 default sanity：11.9876 m / 0.15，落在 10.5–13.5 m、0.05–0.25，通过。
  - R2B default：4.3531 m <= 5.0，success 0.85 >= 0.75，
    R2B/R0 = 0.363 <= 0.50，通过。
  - R0 strict125：12.7146 m >= 11.5，通过。
  - R2B strict125：9.3916 m <= 10.0，success 0.15 >= 0.10，
    R2B/R0 = 0.738 <= 0.80，通过。
  - safety：全部 0.000，不高于 R0 + 0.02，通过。
- [x] **决策：PASS；进入 BC 重采。**

### 归因修正

- `NO_TERMINAL` 与 R2B 完全一致（4.3531 m / 0.85）。
  原因：当前 `terminal_speed_mps=0` 且 `terminal_speed_ratio=1.0`，
  guidance 的 terminal 分支退化为与普通 capture 相同的行为。
  因此“主要收益来自 robust_terminal”的旧结论不成立，必须撤回。
- `ALOS_PLAIN`（关闭 crab/floor/terminal，但保留 ALOS 制导）为
  4.8663 m / 0.70，说明 R0 -> R2B 的主要收益来自 **从 path_command
  切换到 ALOS 制导**，不是 terminal 开关。
- `NO_CRAB` 相对 R2B 恶化 8.8%，`NO_FLOOR` 恶化 11.9%，
  两者同时关闭恶化 11.8%；crab 和 speed floor 有增量作用，
  但存在交互，不能简单相加。
- 论文表述应以 `ALOS 制导` 为主，`crab` 与 `route speed floor` 为增量；
  不再宣称 `robust_terminal` 是主要来源。

产物：

- 服务器 `artifacts/lowlevel/pc_v0d_teacher_r2b_revalidation_20261106/`
- 服务器 `logs/pc_v0d_teacher_r2b_revalidation_20261106/`

## 阶段 3–4：BC 重采与评估

### 数据与训练

- [x] 初始示范：`64 env × 4096 step = 262,144` transitions，
  220 episodes，0 safety events。
- [x] 第一轮 BC：网络与旧 `R-BC-0D` 完全一致
  （`d_z_total=64`、`actor=[256,128]`、`state_encoder=[128,128]`、
  `path_encoder=[64]`、`actor_command_dim=0`、`path_dim=8`）。
- [x] 第一轮 BC 结果：best epoch 59，validation loss 0.0433，
  验证集物理 RMSE：rudder 0.0828 rad、stern 0.0814 rad、RPM 105.92。
- [x] 按预注册做一轮 DAgger 补救：
  `32 env × 4096 step = 131,072` transitions、90 episodes、34 goals、
  0 safety events；学生-教师动作 L2 均值 0.692、p95 1.961。
- [x] 合并示范：`262,144 + 131,072 = 393,216` transitions。
- [x] DAgger 重训：best epoch 45，validation loss 0.1276，
  验证集物理 RMSE：rudder 0.1596 rad、stern 0.1218 rad、RPM 174.00。
- [x] 旧 BC 对照：best epoch 126，validation loss 0.00683，
  RPM RMSE 36.55。

### 闭环结果（seed 0–19，同 V-0D 环境）

| 策略 | default 3D RMSE | default success | strict125 3D RMSE | strict125 success | path-zero 3D RMSE | path-zero success |
|---|---:|---:|---:|---:|---:|---:|
| 旧 R-BC-0D | 12.2396 m | 0.15 | 12.6950 m | 0.05 | — | — |
| 新 R-BC-0D（初始示范） | 9.0944 m | 0.50 | 9.9476 m | 0.40 | 15.5926 m | 0.00 |
| 新 R-BC-0D + DAgger | **8.9129 m** | **0.55** | 10.1828 m | 0.30 | 12.8576 m | 0.10 |

### 门禁判定

- [x] default 失败：
  - 初始 BC `9.0944 > 6.0`，DAgger BC `8.9129 > 6.0`；
  - success `0.50/0.55 < 0.70`；
  - 相对同 harness 旧 BC：`9.0944/12.2396 = 0.743`，
  `8.9129/12.2396 = 0.728`，均 `> 0.70`。
- [x] strict125 通过：
  - 初始 BC `9.9476 <= 10.5`，success `0.40 >= 0.10`，
    相对旧 BC `0.784 <= 0.85`，相对教师 `1.059 <= 1.20`；
  - DAgger BC `10.1828 <= 10.5`，success `0.30 >= 0.10`，
    相对旧 BC `0.802 <= 0.85`，相对教师 `1.084 <= 1.20`。
- [x] path-zero 恶化：新 BC `9.0944 -> 15.5926`，
  DAgger BC `8.9129 -> 12.8576`，说明路径输入被使用。
- [x] **总体判定：BC 失败（default 主门禁未过）；不进入残差 PPO。**
- [x] 一轮 DAgger 补救已用尽；按预注册停止，不再加数据、换损失或调门禁。

### 失败原因诊断

- 离线动作预测并不差：初始 BC 与教师动作相关系数约
  `[0.968, 0.960, 0.924]`，DAgger BC 约 `[0.959, 0.951, 0.908]`。
- 但教师动作高度饱和：rudder `55.27%`、stern `19.85%`；
  初始 BC 只有 `21.54%`、`8.26%`，DAgger BC 只有 `27.48%`、`9.14%`。
  这是 MSE 行为克隆把 bang-bang 动作“平均化”的典型现象：
  预测均值接近，但控制权限的尾部被抹平。
- actor 观测不含 `current_estimate_body`；R2B 教师的 crab 使用了该估计，
  因此存在可观测性缺口。不过离线动作相关仍很高，当前最直接的闭环损失
  更像是“饱和/切换动作被平滑”，而不是单纯的 current 不可见。
- 结论：继续用同一 MSE BC 加更多数据或更多 DAgger 轮次，收益预期有限；
  下一轮应优先改动作表示/损失，或降低教师的 bang-bang 程度。

## 阶段 5–6：残差 PPO

未启动；阶段 4 的 default 主门禁失败，按协议不得进入 PPO。
正式 `3×50` 也未启动。

## 后续建议（需另开 goal 预注册）

1. 饱和保持的 BC：把动作建模为 sign/magnitude 分类、mixture density，
   或对饱和样本加权；先验证能否把 rudder/stern 饱和率从 20%/8% 提到接近教师。
2. 平滑教师：降低 ALOS 的 heading/pitch 增益或启用 saturation governor，
   先做纯教师 screen，再重采 BC；这会牺牲部分教师上限，但更容易模仿。
3. 可观测性：若部署允许，把 `current_estimate_body` 或短历史加入 actor；
   否则教师不应把主要收益建立在使用 actor 看不到的 current 上。
4. 只有新 BC 在 default 达到 `<= 6.0 m`、success `>= 0.70`、
   `<= 0.70×旧 BC` 后，才恢复“冻结 BC + 残差 PPO”阶段。

## 2026-11-07：v2.1 饱和动作补救结果

完整报告见 `doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`。

### 平滑教师

4 个 config-only 平滑教师均未通过阶段 1 门禁：

| variant | default RMSE | success | rudder sat | stern sat | strict RMSE | strict success |
|---|---:|---:|---:|---:|---:|---:|
| h15 | 5.2095 | 0.7000 | 0.4909 | 0.2265 | 9.1435 | 0.2000 |
| h20 | 4.5904 | 0.8000 | 0.4972 | 0.1720 | 9.7623 | 0.2000 |
| rate | 4.3531 | 0.8500 | 0.5426 | 0.2130 | 9.3916 | 0.1500 |
| gov | 4.4421 | 0.8500 | 0.5931 | 0.2454 | 8.7052 | 0.2500 |

`stage1_decision.json` 的 `selected_variant=null`，因此没有用平滑教师重采示范。

### 加权 MSE BC

- 数据集仍为 v1 合并示范，SHA256
  `eb21837960f89c5295af4946bd3fb914471602f6cbd15a33f40aa9ecdb2d1387`；
- 唯一训练配方：`saturation_weight=2.0`、`threshold=0.9`、seed 0；
- best epoch 11，checkpoint SHA256
  `31c568c30cb5cb6627ba97bf60991eb3300be615bc5ec4aff9bcab714f47023f`；
- default `3D RMSE=7.4591 m`、success `0.65`；
- strict125 `9.7894 m`、success `0.45`；
- path-zero `11.6752 m`、success `0.05`，相对正常路径恶化 56.5%；
- 0.99 饱和口径下，predicted rudder/stern saturation 为
  `16.42%/7.65%`，低于 v1 DAgger BC 的 `29.59%/9.10%`。

default RMSE `7.4591 > 7.0`，因此新 BC 为 **FAIL**，不能进入 PPO。
未运行 PPO screen，未启动正式 `3 training seeds × 50 eval seeds`，
未扫描第二组权重或门禁。

## 2026-11-07：Goal16 crab-preserving 真限速教师结果

完整报告见 `doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md`。

上一轮 `smooth_rate` 是 no-op，因为 `saturation_governor_enabled=false` 时
`_limit_guidance_command()` 直接返回。Goal16 先把
`command_rate_limit_enabled` 与 governor 解耦，再保留 crab、关闭 governor、
显式开启命令 rate limit，只测试两组速率。

| variant | default RMSE | success | rudder sat | stern sat | strict RMSE | strict success |
|---|---:|---:|---:|---:|---:|---:|
| `rate035_018` | 4.7499 | 0.80 | 0.6184 | 0.2548 | 8.6849 | 0.25 |
| `rate020_012` | 4.9943 | 0.80 | 0.5897 | 0.2109 | 9.0071 | 0.30 |

- 两组都通过了 RMSE、success、strict125、timeout 和 safety 门禁；
- 两组都未通过 default rudder `<=40%` 与 stern `<=20%` 门禁；
- `selected_variant=null`，决策为 `stop_goal16_teacher_gate_failed`；
- 未重采示范、未训练新 BC、未进入 PPO、未启动正式 `3×50`；
- 结论：限制命令变化率不等于降低 actuator 幅值饱和；继续扫第三组 rate
  不属于本 Goal，应另开 Goal 研究动作表示或直接约束 actuator 幅值。

## Git 建议

不自动 commit/push。若用户要求提交，只添加本 goal 新增文件：

```bash
git add doc/goals/goal9_v0d_teacher_r2b_revalidation_prompt_20261106.md \
  doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md \
  scripts/make_pc_v0d_teacher_r2b_variants.py \
  scripts/collect_pc_bc_dagger_demos.py \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current*.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json
git commit -m "Add V-0D R2B teacher revalidation setup"
git push origin <当前分支>
```

不要使用 `git add -A`。
