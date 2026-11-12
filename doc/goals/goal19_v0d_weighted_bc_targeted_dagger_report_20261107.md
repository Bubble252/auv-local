# Goal19 V-0D：Weighted BC 定向 DAgger 结果报告

日期：2026-11-07

状态：**已执行并收口。判定 `FAIL_RMSE`，不进入残差 PPO。**

协议：`doc/goals/goal19_v0d_weighted_bc_targeted_dagger_prompt_20261107.md`

## 1. 结论

针对 dev 失败轨迹补一轮 DAgger，确实把 held-out default 闭环从
`7.4591 m / 0.65` 推进到 `6.6236 m / 0.70`，首次通过 default 主门禁。

但 strict125 从 `9.7894 m / 0.45` 回退到 `10.5982 m / 0.25`：

- RMSE 比预注册上限 `10.5 m` 高 `0.0982 m`；
- success 下降 `0.20`；
- 5 个原本 success 的 strict125 seed 变为 timeout。

因此按预注册协议判定 **`FAIL_RMSE`**，不能冻结新 BC，不能进入残差 PPO
screen，也不能事后把 strict125 门禁从 `10.5` 放宽。

本 Goal 的主要科学结论是：

> 在 default dev 失败分布上补充 on-policy DAgger 数据，可以明显改善同一工作点，
> 但当前这一轮补数据没有迁移到 strict125；它更像 default 条件分布的定向修复，
> 还不是跨工作点的普遍改进。

## 2. 固定条件

- student 起点：
  `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt`；
- 原始数据 SHA256：
  `eb21837960f89c5295af4946bd3fb914471602f6cbd15a33f40aa9ecdb2d1387`；
- 新 DAgger：7 个 dev seed，每个 `4096` 步，`num_envs=1`；
- 合并数据：`421,888` 个样本、`330` 个 episode；
- 新增样本占比：`28,672 / 421,888 = 6.80%`；
- actor：12D state + 8D path，`command_dim=0`；
- 动作：连续 `[rudder, stern, rpm]`；
- 网络：state `[128,128]`、actor `[256,128]`、path `[64]`、`d_z_total=64`；
- 训练：`w=2.0`、`threshold=0.9`、`batch=4096`、`lr=1e-3`、`seed=0`；
- 评估：CPU float64，held-out test seeds `0–19`。

## 3. 数据与 seed 隔离

test seeds `0–19` 只用于最终评估，没有参与 DAgger 采集。

dev seeds `20–39` 的 baseline default 为 `6.8103 m / 0.70`，其中 6 个 timeout。
按预注册规则冻结的 target seeds 为：

| seed | baseline default RMSE | outcome |
|---:|---:|---|
| 37 | 13.2806 | timeout |
| 34 | 12.1107 | timeout |
| 24 | 11.4603 | timeout |
| 20 | 11.3818 | timeout |
| 39 | 11.2411 | timeout |
| 31 | 10.9790 | timeout |
| 35 | 6.2238 | success，按最高 RMSE success 补足 |

每个 seed 采集 4096 个 transition；每个数据集均满足 `pc_v1`、8D path、
`command_dim=0`，且 student checkpoint SHA256 一致。

## 4. 训练结果

| 项目 | 数值 |
|---|---:|
| best epoch | 20 |
| epochs completed | 51 |
| best weighted validation loss | 0.125403 |
| best unweighted validation loss | 0.115678 |
| checkpoint SHA256 | `acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3` |
| merged dataset SHA256 | `080f4ac3a99ce5f41da119038f0de5a626ea8af1c6b3d9cb7de472d337169d78` |

验证集物理 RMSE：

- rudder `0.13892 rad`；
- stern `0.12720 rad`；
- RPM `180.21`。

离线动作诊断：

| 指标 | 新 BC |
|---|---:|
| sign accuracy：rudder/stern | `0.9747 / 0.9708` |
| predicted/teacher 饱和比 | `0.4340 / 0.3891` |
| 饱和召回 | `0.4284 / 0.3658` |
| 大幅反转率（任一通道） | `0.0 / s` |
| predicted zero fraction | `0.0291 / 0.0688` |

按 Goal18 后的新证据，这些量只作为诊断，不作为本 Goal 的否决门禁。

## 5. Held-out 闭环结果

| 策略 | 场景 | 3D RMSE (m) | success | timeout | safety |
|---|---|---:|---:|---:|---:|
| baseline weighted BC | default | 7.4591 | 0.65 | 0.35 | 0.00 |
| targeted-DAgger BC | default | **6.6236** | **0.70** | 0.30 | 0.00 |
| baseline weighted BC | strict125 | 9.7894 | 0.45 | 0.55 | 0.00 |
| targeted-DAgger BC | strict125 | 10.5982 | 0.25 | 0.75 | 0.00 |
| baseline weighted BC | path-zero | 11.6752 | 0.05 | 0.95 | 0.00 |
| targeted-DAgger BC | path-zero | 12.8765 | 0.00 | 1.00 | 0.00 |

相对 baseline：

- default RMSE 改善 `0.8355 m`（`-11.20%`），success 提升 `0.05`；
- strict125 RMSE 恶化 `0.8089 m`（`+8.26%`），success 下降 `0.20`；
- path-zero 相对正常 path 的 RMSE 比值 `1.9440`，说明路径输入仍被显著使用。

## 6. 门禁判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| default RMSE `<= 7.0 m` | PASS | `6.6236` |
| default success `>= 0.60` | PASS | `0.70` |
| strict125 RMSE `<= 10.5 m` | FAIL | `10.5982` |
| strict125 success `>= 0.10` | PASS | `0.25` |
| safety 不恶化超过 `+0.02` | PASS | 三组均为 `0.00` |
| path-zero 相对 default 至少恶化 `5%` | PASS | ratio `1.9440` |

最终：**`FAIL_RMSE`**。

strict125 的失败不是单一离群值造成的：

- seed `11,15,16,18,19` 从 success 变成 timeout；
- seed `13` 从 timeout 变成 success；
- 净 success 数从 `9/20` 降到 `5/20`。

因此不能把 strict125 的 `0.0982 m` 超标解释成舍入或单 seed 噪声，也不能跳过
strict125 护栏直接进入 PPO。

## 7. 逐 seed 解释

default 的主要改善来自：

- seed 2：`12.6194 -> 1.2758`，timeout -> success；
- seed 3：`17.2968 -> 2.5552`，timeout -> success；
- seed 5：`9.8527 -> 2.3037`，timeout -> success。

但同时出现两个新的 default 回退：

- seed 10：`0.8808 -> 11.3370`，success -> timeout；
- seed 12：`3.1944 -> 14.0300`，success -> timeout。

这说明新 BC 并不是在整体上更稳定，而是把 default 的成功集重新分配了。
strict125 的 success 净减少进一步支持这个判断。

## 8. 决策

- 不冻结 targeted-DAgger BC；
- 不进入残差 PPO screen；
- 不启动正式 `3 training seeds × 50 eval seeds`；
- 不启动 Goal17；
- 不在本 Goal 内继续增加 DAgger 轮数、修改 seed 列表或放宽 strict125 门禁。

下一步应另开 Goal。基于本 Goal 的证据，优先级建议为：

1. 先分析 default 与 strict125 的分布差异，特别是当前策略在 strict125 下
   从 success 退化为 timeout 的 seed `11,15,16,18,19`；
2. 再决定 action chunking 是否针对真正瓶颈，而不是直接假定动作时间一致性是主因；
3. 若继续补数据，需要同时覆盖 default 与 strict125 的失败分布，并保留独立 test set。

## 9. 产物

服务器根目录：

```text
artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/
```

关键文件与 SHA256：

| 文件 | SHA256 |
|---|---|
| `stage_decision.json` | `3933730cd5ffa5a93453b4090e38e7b8807053e12c948a71500aadfb80a4505f` |
| `stage_decision.md` | `2a28e919536711ac26b4297be7a1464e5b9e10a0a9b516837a5e727052cec6db` |
| `run_manifest.json` | `bbb1d73304462fd91c9513604977a12bdcdc84798e886a545b9ab7449a134c4b` |
| `target_seeds.json` | `0ee12aa6fefdb42e3f6fd33946ba4296c07c8c21ae4928b502e08f28e84c5cd6` |
| `bc_satw_dagger2/pc_bc_actor.pt` | `acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3` |

日志目录：

```text
logs/pc_v0d_r2b_weighted_bc_dagger_20261107/
```

本地证据副本：

```text
artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/
```

## 10. 验证

- 服务器 targeted pytest：`25 passed`；
- 新 Python 脚本通过 `py_compile` 和 `ruff check`；
- 7 个采集进程均正常退出，日志无 traceback、NaN/Inf；
- 合并数据 episode id 唯一：`330` samples groups / `330` unique ids；
- 评估进程已退出，服务器当前没有 Goal19 runner 残留。
