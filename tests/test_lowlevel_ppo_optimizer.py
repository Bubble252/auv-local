from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_batch_lowlevel_ppo.py"
SPEC = importlib.util.spec_from_file_location("train_batch_lowlevel_ppo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Policy(torch.nn.Module):
    def __init__(self, *, dade_head: bool) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 3)
        self.critic = torch.nn.Linear(4, 1)
        self.std = torch.nn.Parameter(torch.full((3,), 0.1))
        if dade_head:
            self.actor_dade_head = torch.nn.Linear(4, 3)


def _runner(policy: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=None, learning_rate=None),
    )


def _training(*, dade_head_lr: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        base_mode="frozen_pc_bc_residual",
        algorithm={
            "learning_rate_actor": 1e-5,
            "learning_rate_critic": 3e-4,
            "learning_rate_dade_head": dade_head_lr,
            "critic_warmup_iterations": 5,
            "schedule": "fixed",
        },
    )


def test_frozen_residual_optimizer_keeps_head_without_separate_lr() -> None:
    policy = _Policy(dade_head=True)
    runner = _runner(policy)

    MODULE._configure_optimizer_learning_rates(
        runner,
        _training(dade_head_lr=None),
        is_pc=True,
    )

    trainable = {
        id(parameter)
        for parameter in policy.parameters()
        if parameter.requires_grad
    }
    optimized = [
        parameter
        for group in runner.alg.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimized) == len(trainable)
    assert {id(parameter) for parameter in optimized} == trainable
    assert [group["name"] for group in runner.alg.optimizer.param_groups] == [
        "actor",
        "critic",
    ]


def test_frozen_residual_optimizer_keeps_separate_head_group() -> None:
    policy = _Policy(dade_head=True)
    runner = _runner(policy)

    MODULE._configure_optimizer_learning_rates(
        runner,
        _training(dade_head_lr=5e-4),
        is_pc=True,
    )

    groups = runner.alg.optimizer.param_groups
    assert [group["name"] for group in groups] == ["actor", "critic", "dade_head"]
    head_group = next(group for group in groups if group["name"] == "dade_head")
    MODULE._apply_learning_rate_for_iteration(runner, 5)
    assert head_group["lr"] == 5e-4
    assert {id(parameter) for parameter in head_group["params"]} == {
        id(parameter) for parameter in policy.actor_dade_head.parameters()
    }
