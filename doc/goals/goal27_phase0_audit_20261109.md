# Goal27 Phase 0 审计

日期：2026-11-09<br>
状态：`PHASE0_LOCAL_PASS_SERVER_UNAVAILABLE`

## 1. 本地资源

本地机器：

- 根盘 `/`：253G，已用 240G，可用约 116M，使用率 100%；
- `/dev/shm`：16G，已用 12G，可用约 4G；
- GPU：RTX 4060 Laptop，8188 MiB，已用约 1587 MiB，利用率约 61%；
- 本地只适合只读分析、CPU 评估和小规模 smoke，不适合 4096 环境训练。

## 2. 服务器状态

尝试结果：

- `capri_target`：跳板机 `202.112.113.15:2228` 连接被拒；
- `capri_target_old`：跳板机 `202.112.113.78:2258` 连接被拒；
- `vla101`：可以连接，主机为 `di-20260613120619-84fwm`，有 2×A100-80GB，
  但没有 `/data/yhx/underwater-robotics`，不是 Goal26 使用的
  `zhulab3-AS-4125GS-TNRT`。

因此本阶段不能把 Goal27 的大规模训练伪装成服务器结果。服务器恢复后，训练必须使用
隔离目录，不覆盖 Goal24/25/26 产物。

## 3. D3 只读基线

checkpoint：

```text
checkpoints/goal26/d3/pc_bc_actor.pt
SHA256 7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73
```

与 `goal26_phase0_audit_20261108.md` 记录的 SHA256 一致。

metadata：

```text
d_z_total = 64
n_in_reserved = 8
n_z_reserved = 4
actor_hidden_dims = [256, 128]
actor_state_names = [x, y, z, phi, theta, psi, p, q, r, u, v, w]
actor_command_dim = 0
path_dim = 8
path_encoder_alpha = 1.0
```

## 4. Goal26 只读证据

Goal26 已记录：

- P1/P2 action residual 的 pre-goal RMSE 改善约 `0.014–0.017 m`；
- success 没有提升；
- residual/base L2 比例 `1.8%–2.7%`；
- zero-residual 机制通过，frozen BC 参数漂移为 0；
- P1/P2 checkpoint 和 40-seed 评估在服务器隔离目录，本地只有结果文档。

本 Goal 不重复跑 Goal26 P1/P2 作为主实验；G26-P2 只作历史对照。

## 5. 执行器事实

REMUS100 动作合同只有：

```text
[rudder, stern_plane, propeller_rpm]
```

`src/auv_lowlevel/batch/dynamics.py:531-543`：

- rudder 产生 sway force 和 yaw moment；
- stern plane 产生 heave force 和 pitch moment；
- propeller 产生 surge force。

没有独立 sway/heave 推力器。横向误差必须通过艏向/速度间接修正，因此 H3 的
执行器约束是必须测量的，而不是可以用更大 PPO 预算默认绕过的。

## 6. 本地验证

Python 环境：

- 系统 `python3`：3.8.10，torch 2.4.1+cu121；
- 项目 `.venv`：3.10.18，torch 2.7.0+cpu，`rsl_rl` 可用；
- 本地 4096 环境训练不可行，CPU 评估和单测可用。

已运行：

```text
.venv/bin/python -m pytest -q \
  tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_pc_policy.py \
  tests/test_lowlevel_pc_teacher.py

39 passed in 1.39s
```

## 7. 裁决与下一步

- Phase 0 本地审计：PASS；
- 服务器训练：UNAVAILABLE，暂不启动；
- 下一步：在本机做 Step 1 离线 reward/advantage 分解，输出只读诊断；
- Step 2 的 `command_residual` 实现先做零残差单测和 CPU smoke；
- 服务器恢复后，才允许跑 4096 环境 PPO screen；
- 未使用 held-out `100–149`，未覆盖旧产物，未 commit/push。
