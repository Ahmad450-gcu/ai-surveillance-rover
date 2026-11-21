"""Runs every script-style check in tests/ under pytest, so CI gets one pass/fail per check.

Each check module has a main() that prints [PASS]/[FAIL] lines and exits non-zero on failure.
They can still be run on their own, e.g.  python tests/test_learned.py
"""
import importlib

import pytest

CHECKS = ["test_learned", "test_patrol", "test_protocol", "test_theft_sim"]


@pytest.mark.parametrize("name", CHECKS)
def test_check_script(name):
    module = importlib.import_module(f"tests.{name}")
    try:
        module.main()
    except SystemExit as e:
        assert e.code in (0, None), f"{name} reported a failure (exit code {e.code})"
