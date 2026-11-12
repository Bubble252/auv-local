# Research Ledger

这个目录把“遇到难关自己调研”外化成可检查的文件，而不是依赖模型记忆。

| 文件 | 作用 |
|---|---|
| `STATE.md` | 当前 goal、blocker、hypothesis、next experiment |
| `LEDGER.jsonl` | 每次 research 的决策记录，一行一条 |
| `blockers/` | 单个 blocker 的现象、证据、排除项 |
| `decisions/` | `pivot` / `stop` 的决策记录 |
| `paper_reading/` | 外部论文阅读笔记 |
| `AUTORESEARCH_LOOP.md` | 指标驱动的自主迭代、论文/代码检索、pivot 与停止条件 |

## LEDGER 字段

每条 JSONL 必须包含：

```text
date, goal, trigger, evidence, hypotheses, experiment, result, decision, next
```

`decision` 只能是 `continue` / `pivot` / `stop`，`hypotheses` 至少 2 条。

## 检查

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal25 \
  --goal-doc doc/goals/goal25_d2_pregoal_repair_20261108.md \
  --strict
```
