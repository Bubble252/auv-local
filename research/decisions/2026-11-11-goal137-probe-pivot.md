# Goal137 decision: pivot from encoder/freezing to PPO-head and closed-loop timing

## Evidence

- `z_t` linear probes on six held-out episodes: `R²=0.9979–0.9989` for vertical error, pitch error, q, w.
- BC residual Jacobian: vertical/pitch/q response per std `0.47/0.57/0.57`, key/reference input ratio `2.89×`.
- `head_only` vs `last_layer`: validation RMSE improvement `0.66%`, key sensitivity change `+0.73%`.

## Decision

`pivot`

The encoder/frozen-encoder hypothesis is not supported for the Goal86 residual BC proxy. Continue by checking the Goal95 PPO final residual head and then closed-loop action timing/authority; do not change encoder pretraining or add vertical skip based on this result alone.

## Guardrail

The local `P_D3` checkpoint is unavailable. The BC-head result is a proxy, not proof about the PPO head. No closed-loop success claim is made.
