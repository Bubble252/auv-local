from __future__ import annotations

import importlib


def test_pc_residual_env_aliases_preserve_module_and_public_object_identity() -> None:
    aliases = {
        "pc_teacher": "pc_residual_env.teacher",
        "pc_controller_gain_residual": "pc_residual_env.controller_gain_residual",
        "pc_actuator_residual": "pc_residual_env.actuator_residual",
        "pc_frozen_residual_vec_env": "pc_residual_env.frozen_residual_vec_env",
    }
    for old_name, new_name in aliases.items():
        old = importlib.import_module(f"auv_lowlevel.batch.{old_name}")
        new = importlib.import_module(f"auv_lowlevel.batch.{new_name}")
        assert old is new

        for public_name in old.__all__:
            assert getattr(old, public_name) is getattr(new, public_name)


def test_old_residual_wrapper_import_remains_instantiable() -> None:
    from auv_lowlevel.batch.pc_controller_gain_residual import (
        BatchedPCControllerGainResidualEnv,
    )
    from auv_lowlevel.batch.pc_residual_env import (
        BatchedPCControllerGainResidualEnv as NewControllerGainResidualEnv,
    )

    assert BatchedPCControllerGainResidualEnv is NewControllerGainResidualEnv
