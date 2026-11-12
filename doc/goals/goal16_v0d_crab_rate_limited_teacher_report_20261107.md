# Goal16 V-0D 保留 Crab 的真限速教师结果报告

日期：2026-11-07

状态：**阶段 0–3 完成；两个纯限速教师均未通过教师门禁；按预注册停止。**
未重采示范，未训练 BC，未进入 PPO，未启动正式 `3×50`。

## 结论

1. `command_rate_limit_enabled` 已与 `saturation_governor_enabled` 解耦，旧配置不写
   新字段时仍跟随 governor，默认行为不变。
2. 两组纯限速教师都保留了 crab、`compensate_current=false` 和既有 actor 契约。
3. 两组教师的 RMSE/success/strict125/timeout/safety 都达到了预注册门禁，但
   **default rudder/stern 饱和率没有达到 40%/20% 上限**。
4. `rate035_018` default 为 `4.7499 m / 0.80`，饱和率 `61.84% / 25.48%`；
   `rate020_012` default 为 `4.9943 m / 0.80`，饱和率 `58.97% / 21.09%`。
5. 两者都没有优于 R2B 的 `4.3531 m / 0.85`，也没有把饱和率降到门禁内；因此
   `selected_variant=null`，决策为 `stop_goal16_teacher_gate_failed`。
6. 本轮没有降低门禁、没有删除 crab、没有加入 current 特权观测、没有改 actor 契约，
   也没有用旧 `smooth_rate` no-op 结果替代正式实验。

## 代码修正

### 限速开关解耦

- `src/auv_lowlevel/config.py` 新增可选字段：
  `guidance.command_rate_limit_enabled: bool | None = None`；
- 配置解析支持 `true/false/null`；旧配置缺省值为 `None`；
- `src/auv_lowlevel/batch/pc_env.py::_limit_guidance_command()` 使用兼容语义：

```python
enabled = guidance.command_rate_limit_enabled
if enabled is None:
    enabled = guidance.saturation_governor_enabled
if not enabled:
    return target
```

因此旧配置行为不变，新配置可以单独开启或关闭命令限速。

### 新增文件

- `scripts/make_pc_v0d_r2b_rate_teacher_variants.py`；
- `scripts/run_pc_v0d_r2b_rate_teacher_screen_20261107.sh`；
- `scripts/summarize_pc_v0d_r2b_rate_teacher_20261107.py`；
- `configs/lowlevel/pc_v0d_r2b_rate_teacher_variants_20261107.manifest.json`。

### 验证

本地与服务器均运行：

```bash
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
  python -m pytest -q \
  tests/test_lowlevel_pc_env.py tests/test_lowlevel_config.py tests/test_batch_alos_r.py
```

- 本地：`69 passed`；
- 服务器：`65 passed`；
- `ruff check`：通过；
- 开关测试覆盖：无字段 + governor off/on、显式 true/false、旧行为兼容。

## 教师配置

全部配置从 R2B default 复制，只改实验元数据、目标半径和以下 guidance 字段：

- `command_rate_limit_enabled=true`；
- `saturation_governor_enabled=false`；
- `heading_rate_limit_rad_s` / `pitch_rate_limit_rad_s` 使用下表值；
- `current_crab_enabled=true`；
- `compensate_current=false`。

| variant | heading rate | pitch rate | default config SHA256 | strict125 config SHA256 |
|---|---:|---:|---|---|
| `rate035_018` | 0.35 rad/s | 0.18 rad/s | `3b2d906b42a041abecab84e18725486737c0c73596ef7b7a099e592cdc07f182` | `abbf70848d42376f1a79d772ca957c2ceff1122f722cb6ed71e506eb592b8a95` |
| `rate020_012` | 0.20 rad/s | 0.12 rad/s | `8f4d764b0f5f685f97669d4c8ee581289fee8509b5f3d8192e36bacc0dfd09cd` | `267aafebb1d4f24f61b365271184c9dfc1b28e783a898eb7d795d2f3c37d5fc6` |

## 20-seed 纯教师结果

评估协议：seed `0–19`、CPU、`float64`、default + strict125、每格 20 episodes。

| variant | profile | 3D RMSE | success | timeout | safety | rudder sat | stern sat |
|---|---|---:|---:|---:|---:|---:|---:|
| R2B baseline | default | 4.3531 | 0.85 | 0.15 | 0.00 | 55.27% | 19.85% |
| R2B baseline | strict125 | 9.3916 | 0.15 | 0.85 | 0.00 | — | — |
| `rate035_018` | default | 4.7499 | 0.80 | 0.20 | 0.00 | 61.84% | 25.48% |
| `rate035_018` | strict125 | 8.6849 | 0.25 | 0.75 | 0.00 | 73.49% | 29.50% |
| `rate020_012` | default | 4.9943 | 0.80 | 0.20 | 0.00 | 58.97% | 21.09% |
| `rate020_012` | strict125 | 9.0071 | 0.30 | 0.70 | 0.00 | 70.56% | 29.22% |

strict125 的 RMSE/success 比 R2B 更好，但 default 饱和门禁仍然失败；该收益不能替代
预注册的 default 动作饱和条件。

## 门禁判定

| check | rate035_018 | rate020_012 |
|---|---|---|
| default RMSE <= 6.0 m | PASS | PASS |
| default success >= 0.70 | PASS | PASS |
| default rudder sat <= 40% | **FAIL** | **FAIL** |
| default stern sat <= 20% | **FAIL** | **FAIL** |
| strict125 RMSE <= 10.5 m | PASS | PASS |
| strict125 success >= 0.10 | PASS | PASS |
| timeout <= baseline + 0.10 | PASS | PASS |
| safety <= baseline + 0.02 | PASS | PASS |
| overall | **FAIL** | **FAIL** |

决策：

- `passing_variants=[]`；
- `selected_variant=null`；
- `decision=stop_goal16_teacher_gate_failed`。

## 阶段 4–6

未执行。按 Goal16 停止条件，两个教师都未通过门禁时：

- 不重采 262,144 transitions；
- 不训练新的 MSE BC；
- 不进入残差 PPO；
- 不启动正式 `3 training seeds × 50 evaluation seeds`；
- 不继续扫描第三组 rate、学习率、seed 或降低门禁。

## 产物与证据

服务器根目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_v0d_r2b_rate_teacher_20261107/
```

| 文件 | SHA256 |
|---|---|
| `stage0_audit.json` | `83dc0606a67cd253020f51a9d3f46f0edc276680a1eb8c0b509bbf37b6ca653c` |
| `teacher_screen/stage1_decision.json` | `8d374728b29159f6676dc82e46154a4b27df6b82850f46595a2a2b670a0127ab` |
| `teacher_screen/run_manifest.json` | `7736e6d1d06733c23477f43920a174a597b8eaf8b90a40a5f403a53ffdbbd419` |
| `teacher_screen/rate035_018_default/teacher_summary.json` | `16acc4ba558632c2eb84151309dec02344d875d7bdfb8cc4da8e5f58f892c716` |
| `teacher_screen/rate035_018_strict125/teacher_summary.json` | `1c0ba589f068fd84494819efb67b8f7b83b46d52a6405ba6f9c4622526ecf1aa` |
| `teacher_screen/rate020_012_default/teacher_summary.json` | `a9881e7a87065e7a906fd1690e27ce2ba28ec2b3da909a7872bda8ecf79dea26` |
| `teacher_screen/rate020_012_strict125/teacher_summary.json` | `b032b12d52b0d16f4ce33d499e3829b1af8dae950c547457d8a7f654bb2195bd` |

运行审计：

- 服务器仓库无 `.git`；本地 HEAD 记录为 `06c05b6`；
- 每个评估 run 的命令、配置 SHA、summary SHA、seed、device、dtype 记录在
  `teacher_screen/run_manifest.json`；
- 日志：
  `logs/pc_v0d_r2b_rate_teacher_20261107/`；
- 结束后无教师/BC/PPO/eval runner 残留。

## 解释与下一步

本轮说明“限制 ALOS 命令的变化率”并不等价于“降低最终 actuator 幅值饱和”。在
crab 补偿持续参与时，低频命令仍可能长期把 rudder/stern 推在幅值边界上；0.99
饱和率是动作幅值指标，不是命令 rate 指标。

按预注册规则，Goal16 到此停止。若继续，必须另开 Goal 并预注册新的机制性假设，例如：

- sign/magnitude 或 mixture-density 动作头；
- 直接约束 actuator 幅值/切换频率的教师；
- 重新审计 crab 的幅值需求，而不是继续扫第三组 rate。

## 最终输出表

| 问题 | 回答 |
|---|---|
| 限速开关是否已与 governor 解耦？旧配置行为是否保持不变？ | 是；新旧四种开关组合均有测试，旧配置缺省 `None` 时继续跟随 governor。 |
| 两个纯限速教师是否通过？哪个最好？rudder/stern 饱和率是多少？ | 均未通过；性能上 `rate035_018` 的 default RMSE 更低，但 rudder/stern sat 为 `61.84%/25.48%`；`rate020_012` 为 `58.97%/21.09%`。 |
| 是否重采 BC？数据集 SHA256 和样本数是多少？ | 否；教师门禁失败，未生成演示数据。 |
| 新 BC 的 default/strict125/path-zero 指标是多少？ | 未训练新 BC，无指标。 |
| BC 是 PASS、CONDITIONAL 还是 FAIL？ | 未执行 BC。 |
| 若进入 PPO，screen 结果是什么？正式 3×50 是否启动？ | 未进入 PPO；未启动正式 `3×50`。 |
| 是否修改过预注册门禁、crab、观测契约或评估协议？ | 否。 |

## Git 建议

不自动 commit/push。若用户要求提交，只添加本 Goal 相关文件，不使用
`git add -A`：

```bash
git add src/auv_lowlevel/config.py \
  src/auv_lowlevel/batch/pc_env.py \
  tests/test_lowlevel_config.py \
  tests/test_batch_alos_r.py \
  scripts/make_pc_v0d_r2b_rate_teacher_variants.py \
  scripts/run_pc_v0d_r2b_rate_teacher_screen_20261107.sh \
  scripts/summarize_pc_v0d_r2b_rate_teacher_20261107.py \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate035_018_20261107.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate035_018_20261107_strict125.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate020_012_20261107.json \
  configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_rate020_012_20261107_strict125.json \
  configs/lowlevel/pc_v0d_r2b_rate_teacher_variants_20261107.manifest.json \
  doc/goals/goal16_v0d_crab_rate_limited_teacher_prompt_20261107.md \
  doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md \
  doc/goals/goal9_execution_state.md \
  doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md
git commit -m "test(pc): stop at crab-preserving rate teacher gate"
```

推送前先检查 `git status`；不要提交 `artifacts/` 或 `logs/`。
