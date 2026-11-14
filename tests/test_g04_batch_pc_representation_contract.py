from __future__ import annotations

import importlib


def test_pc_representation_aliases_preserve_module_and_public_object_identity() -> None:
    aliases = {
        "pc_normalization": "pc_representation.normalization",
        "pc_history_ablation": "pc_representation.history_ablation",
    }
    for old_name, new_name in aliases.items():
        old = importlib.import_module(f"auv_lowlevel.batch.{old_name}")
        new = importlib.import_module(f"auv_lowlevel.batch.{new_name}")
        assert old is new

        for public_name in old.__all__:
            assert getattr(old, public_name) is getattr(new, public_name)


def test_history_ablation_old_import_path_keeps_behavior() -> None:
    import torch

    from auv_lowlevel.batch.pc_history_ablation import ablate_pc_history
    from auv_lowlevel.batch.pc_policy import PCPolicyLayout

    layout = PCPolicyLayout(state_dim=8, path_dim=4)
    observation = torch.arange(layout.flat_dim, dtype=torch.float32).reshape(1, -1)
    result = ablate_pc_history(observation, layout, "history-zero")
    assert result.shape == observation.shape
    assert torch.equal(
        observation, torch.arange(layout.flat_dim, dtype=torch.float32).reshape(1, -1)
    )
