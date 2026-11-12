# Goal142 决策：terminal profile selector 不值得开新 Goal，转 C-line 课程

**日期**：2026-11-11<br>
**决策**：`pivot`<br>
**上游**：Goal141 terminal contract audit、Goal142 terminal profile ceiling audit

## 证据

1. P_D3 正式 trace 复现 BC proxy：strict125 `19/40`，zero-residual `20/40`；
   failure 全部 `GOAL_PLANE_MISS`，2 m 入口纵向误差中位数 `1.650 m`，
   success 仅 `0.517 m`。
2. B20 strict125 `24/40`，相对 zero gained `{29,31,33,56}`、lost `[]`；
   但 terminal 纵向误差只改善 `0.15–0.33 m`，seed 31/33 仍 `>= 1 m`。
3. B20、BP3、stern-025、zero 的 strict success 并集只有 `24/40`；
   加入 Goal133 Vm010 后也只有 `25/40`。
4. P_D3/B20 的 success 中仍有 `q >= 15 deg/s`，说明 pitch-rate 包络是
   secondary metric，不能单独作为 success 条件。

## 决策

停止 terminal profile selector、固定 bias、residual scale、channel cap 和
action gate 的继续扫描。不换 SAC/PPO，不引入 REDQ/DroQ，不打开 held-out。

## 下一步

按用户优先级转 Goal136 C-line：

1. 把 `CurriculumPromotionController` 接进 PPO/SAC 训练循环；
2. 每 100k transitions 做冻结 dev 评估，写 `curriculum_decisions.jsonl`；
3. 晋级/回退后切换 `CurriculumStageSampler` 的 mix；
4. 冻结 C0–C6 配置并跑 1M pilot；
5. C-line 裁决后才启动 Goal140 纵向 tracking reward。
