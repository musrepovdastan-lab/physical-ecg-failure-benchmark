"""Regression coverage for the M2 trainer/finalizer hand-off."""
from pathlib import Path

import numpy as np
import pytest

from src import mitigation_m2_ningbo_v1 as workflow
from src.mitigation_m2_finalize_v1 import ece


def test_finalizer_starts_only_after_trainer_lock_is_released(tmp_path, monkeypatch):
    trainer_lock = tmp_path / "mitigation_m2_ningbo_v1.lock"
    trainer_lock.write_text('{"pid": 1}', encoding="utf8")
    calls = []
    monkeypatch.setattr(workflow, "status", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(workflow.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="trainer lock must be released"):
        workflow.launch_finalizer_after_trainer_release(trainer_lock)
    assert len(calls) == 0

    trainer_lock.unlink()
    workflow.launch_finalizer_after_trainer_release(trainer_lock)
    assert calls[0][0][0] == "FINALIZING"
    assert calls[1][0][0][1] == "src/mitigation_m2_finalize_v1.py"


def test_m2_finalizer_ece_uses_all_fifteen_bins_without_generator_scope_error():
    value = ece(np.asarray([0, 1]), np.asarray([0.1, 0.9]))
    assert np.isfinite(value)
    assert value == pytest.approx(0.1, abs=1e-12)
