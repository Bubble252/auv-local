from __future__ import annotations

import importlib


def test_pc_support_aliases_preserve_module_and_public_object_identity() -> None:
    aliases = {
        "pc_calibration": "pc_support.calibration",
        "pc_diagnostics": "pc_support.diagnostics",
    }
    for old_name, new_name in aliases.items():
        old = importlib.import_module(f"auv_lowlevel.batch.{old_name}")
        new = importlib.import_module(f"auv_lowlevel.batch.{new_name}")
        assert old is new

        for public_name in old.__all__:
            assert getattr(old, public_name) is getattr(new, public_name)


def test_old_module_monkeypatch_targets_shared_namespace() -> None:
    old = importlib.import_module("auv_lowlevel.batch.pc_diagnostics")
    new = importlib.import_module("auv_lowlevel.batch.pc_support.diagnostics")
    original = old.policy_diagnostics
    sentinel = object()
    try:
        old.policy_diagnostics = sentinel
        assert new.policy_diagnostics is sentinel
    finally:
        old.policy_diagnostics = original
