from __future__ import annotations

import importlib


def test_pc_bc_tools_aliases_preserve_module_and_public_object_identity() -> None:
    aliases = {
        "pc_bc_merge": "pc_bc_tools.bc_merge",
        "pc_bc_subset": "pc_bc_tools.bc_subset",
        "pc_residual_pretrain": "pc_bc_tools.residual_pretrain",
    }
    for old_name, new_name in aliases.items():
        old = importlib.import_module(f"auv_lowlevel.batch.{old_name}")
        new = importlib.import_module(f"auv_lowlevel.batch.{new_name}")
        assert old is new

        for public_name in old.__all__:
            assert getattr(old, public_name) is getattr(new, public_name)


def test_pc_bc_tools_package_exports_stable_public_names() -> None:
    from auv_lowlevel.batch.pc_bc_tools import (
        merge_pc_demonstration_datasets,
        sample_pc_demonstration_dataset,
        train_pc_supervised_residual,
    )

    assert callable(merge_pc_demonstration_datasets)
    assert callable(sample_pc_demonstration_dataset)
    assert callable(train_pc_supervised_residual)
