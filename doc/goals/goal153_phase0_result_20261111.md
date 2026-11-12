# Goal153/154 Phase 0：residual 分箱与成败只读审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_READ_ONLY / NO_TRAINING`<br>
**数据**：Goal151 的 full / near10_2 / shell10_3，3 个 T2-SAC checkpoint，dev40，default/strict125<br>
**脚本**：`scripts/audit_pc_goal153_phase0.py`<br>
**JSON**：`artifacts/lowlevel/goal153_reward_curriculum_20261111/phase0_goal151.json`

## 1. 结论

本轮只读审计给出三条直接结论：

1. **full residual 的成败不能由远段 residual 幅度解释。**
   default 下成功 episode 的 `>10 m` residual RMS 为 `1.471`，失败为 `1.413`；
   两者几乎一样，失败甚至略低。
2. **真正区分成败的是 path error，尤其是横向/纵向 terminal error。**
   full/default：成功 `horizontal 2.02 / vertical 0.77 / path3d 2.22`，
   失败 `horizontal 3.87 / vertical 1.21 / path3d 4.08`。
3. **权限干预比 residual 幅度本身更关键。**
   `near10_2` 把 `>10 m` residual 压到 `~2e-5`，path3d 从 full 的 `2.62` 降到
   `2.34`，success 从 `85/120` 提到 `88/120`。这说明“在哪里开 residual”比
   “residual 整体多大”更能解释 path 代价。

因此 Goal153 的 reward 不应只惩罚 residual 幅度；应把 `vertical_error` /
`heave_rate` / terminal path error 作为 potential，Goal154 再用权限 schedule/soft
constraint 控制 residual 的作用位置。

## 2. 分箱表

### full / default

| outcome | n | <=2 m residual | 2–10 m residual | >10 m residual | vertical RMSE | horizontal RMSE | path3d RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| success | 29 | 1.423 | 1.210 | 1.471 | 0.769 | 2.021 | 2.220 |
| failure | 11 | NA | 1.306 | 1.413 | 1.213 | 3.867 | 4.083 |

### near10_2 / default

| outcome | n | <=2 m residual | 2–10 m residual | >10 m residual | vertical RMSE | horizontal RMSE | path3d RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| success | 32 | 1.407 | 0.809 | `2.5e-5` | 0.898 | 1.668 | 1.974 |
| failure | 8 | NA | 0.841 | `2.1e-5` | 1.432 | 3.446 | 3.782 |

### shell10_3 / default

| outcome | n | <=2 m residual | 2–10 m residual | >10 m residual | vertical RMSE | horizontal RMSE | path3d RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| success | 30 | 0.000 | 0.701 | `1.3e-4` | 0.854 | 1.604 | 1.900 |
| failure | 10 | NA | 0.766 | `1.0e-4` | 1.431 | 3.293 | 3.637 |

## 3. 判读

- 失败 episode 的 horizontal RMSE 约是成功 episode 的 `1.9–2.0×`；vertical RMSE
  约 `1.4–1.6×`。失败主因是 terminal/path geometry，而不是“residual 不够大”。
- full 下远段 residual 与 path3d 的 episode 相关系数为 `-0.45`；near/shell 下为
  `+0.30`。这说明 full 的远段 residual 与 path error 的关系被其他因素掩盖，不能
  用简单相关判断因果；Goal151 的 gate 干预才是因果证据。
- shell 的 `<=2 m` residual 为 0，却仍有 success 增益，说明 shell 收益来自
  approach shell 的修正，不是 capture-zone 修正；这支持 Goal154 研究“在哪个距离
  开 residual”，而不是只研究“开多大”。

## 4. 对 Goal153/154 的直接影响

- Goal153：R1/R2 的 potential 应针对 `vertical_error` 和 `heave_rate`，而不是
  额外加一个“residual 大幅惩罚”；否则会把必要修正一起压掉。
- Goal154：重点比较 hard near gate、soft gate floor、距离加权 L2、barrier 和
  staged schedule；不要只比较“full vs off”。
- 后续 result 必须同时报告 success、path3d、horizontal RMSE、vertical RMSE 和
  residual 三个距离分箱，不能只报 success。
