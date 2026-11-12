# Decision：mapping 干净，capture frame 存在次级歧义

日期：2026-11-10<br>
关联：Goal83<br>
决定：`pivot`

## 判据

Goal83 只读复用 20 个 run、800 个 episode：

- H83-A 不支持：base timing、residual scale/rate、goal-distance/signed-offset
  投影、tangent norm 和 terminal replay 全部通过，没有 mapping 实现缺陷；
- H83-B 支持：done 时刻没有 p_nav/eta capture 或 protocol 标签分歧，但首次
  p_nav crossing 处出现 plane-sign 分歧；candidate default 还有 `4/15` 的
  crossing capture 分歧；
- true eta 的 tight-radius oracle 在四个 group 分别恢复 `1/1`、`2/6`、`8/15`、
  `1/7` lost，新增 false-success 为 `0.0063`、`0.0375`、`0.0187`、`0.0250`；
- H83-C 作为总体判定不支持，因为预注册要求 H83-A/B 都不支持；但 strict125
  出现次级 terminal-chain 信号：baseline/candidate heading tracking 分别改善
  `-0.1305` / `-0.1151`，terminal p_nav/eta 改善的 CI 跨零，true diff 为负；
- 没有 NoisyNet、residual 幅度、优化失败或样本效率失败的直接证据。

## 决定

- 不修改 reward、动作参数化、controller、checkpoint、seed 或 held-out；
- 不启动 SAC/REDQ/DroQ；
- 不把 true eta 直接写成正式 success；它只是 oracle upper bound；
- 下一个预注册 goal 只做 eval-only capture-frame 单因素审计，确认 frame 规则
  在 default/strict125 的 true recovery 与 additive false-success 是否稳定；
- 若 frame 规则不可接受，写 blocker 并进入 autoresearch loop；若可接受，再讨论
  可观测估计器或 terminal approach。

## 为什么不是算法问题

residual 已经按预期进入执行器，scale/rate 没有 clip，终止规则能逐行复现。
失败集中在 terminal capture frame 和 terminal approach，不是“网络输出没有生效”
或“NoisyNet 采样主导”。在 frame 尚未对齐时切换 SAC/REDQ/DroQ 会改变优化器、
采样和探索变量，无法判断收益来自算法还是来自更长的训练/更大的采样预算。

## 训练型 arm 的前置条件

1. 先做 `num_envs 128/256/512 × batch_size 256/512` 单卡吞吐基准；
2. 记录 env transitions/s、gradient steps/s、GPU 利用率、显存、CPU、wall；
3. 按 seed 分 8 张 GPU，一卡一个 seed；
4. 扩大采样/批量必须独立预注册，不能混进正式 Gate；
5. 只有出现“residual 方向正确、但探索/样本效率不足”的直接证据，才开 SAC
   同协议对照；REDQ/DroQ 只在样本效率证据明确时考虑。

## 证据

- `doc/goals/goal83_mapping_capture_audit_20261110.md`
- `doc/goals/goal83_result_20261110.md`
- `artifacts/lowlevel/goal83_mapping_capture_audit_20261110/goal83.json`
- `artifacts/lowlevel/goal83_mapping_capture_audit_20261110/goal83.md`
- `scripts/analyze_pc_goal83_mapping_capture.py`
- `tests/test_pc_goal83_mapping_capture.py`
