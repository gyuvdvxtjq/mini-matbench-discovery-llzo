"""Unit tests for the audit framework core (no model downloads required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlip_audit.analysis import build_hull, cross_model_attribution, evaluate_stability
from mlip_audit.checkpoint import CheckpointStore, make_key, params_hash
from mlip_audit.config import load_config
from mlip_audit.stats import auroc, bootstrap_metric, safe_spearman


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(0.5, 0.1, size=50)})
    result = bootstrap_metric(frame, lambda d: d["x"].mean(), n_resamples=200)
    assert result is not None
    assert result["ci_low"] < result["point"] < result["ci_high"]


def test_bootstrap_too_small_returns_none():
    assert bootstrap_metric(pd.DataFrame({"x": [1.0, 2.0]}), lambda d: 1.0) is None


def test_auroc_perfect_and_degenerate():
    assert auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auroc([0.1, 0.2], [1, 1]) is None


def test_safe_spearman_constant_series():
    assert safe_spearman([1, 1, 1], [1, 2, 3]) is None
    assert safe_spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_checkpoint_key_tracks_params(tmp_path):
    store = CheckpointStore(tmp_path / "ckpt.jsonl")
    key_a = make_key("chgnet", "relax", {"fmax": 0.1}, "mp-1")
    key_b = make_key("chgnet", "relax", {"fmax": 0.05}, "mp-1")
    assert key_a != key_b
    store.append(key_a, {"status": "ok", "value": 1})
    assert store.get_ok(key_a)["value"] == 1
    assert store.get_ok(key_b) is None
    reloaded = CheckpointStore(tmp_path / "ckpt.jsonl")
    assert reloaded.get_ok(key_a)["value"] == 1


def test_params_hash_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_load_config(tmp_path):
    cfg_file = tmp_path / "audit.yaml"
    cfg_file.write_text(
        "name: t\nelements: [Li, O]\n"
        "models: [{name: m1, type: chgnet}]\n"
        "relaxation: {enabled: false}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.name == "t"
    assert cfg.elements == ["Li", "O"]
    assert cfg.relax.enabled is False
    assert cfg.models[0].device == "cpu"


def _synthetic_records(n: int = 6) -> list[dict]:
    # One-element toy space: Li polymorphs, first one stable, energies agree
    # exactly with the MP reference columns so every error term is zero.
    records = []
    for i in range(n):
        eah = 0.03 * i
        records.append(
            {
                "material_id": f"li-{i}",
                "formula": "Li",
                "chemsys": "Li",
                "nsites": 1,
                "status": "ok",
                "energy_per_atom": -1.9 + eah,
                "mp_energy_per_atom": -1.9 + eah,
                "mp_formation_energy_per_atom": eah,
                "mp_energy_above_hull": eah,
                "mp_is_stable": i == 0,
            }
        )
    return records


def test_hull_and_stability_metrics():
    frame = build_hull(pd.DataFrame(_synthetic_records()), "energy_per_atom", "sp")
    assert "sp_energy_above_hull" in frame.columns
    metrics = evaluate_stability(
        frame, "sp", candidate_window=0.05, bootstrap_resamples=50
    )
    assert metrics["n_success"] == 6
    # Model and MP energies agree exactly in this toy set.
    assert metrics["hull_mae_eV_per_atom"]["point"] == pytest.approx(0.0)
    assert metrics["stable_classification_f1"]["point"] == pytest.approx(1.0)


def test_cross_model_attribution_uses_model_hull_not_mp_reference():
    # Regression: the MP reference column also ends in "_energy_above_hull";
    # attribution must diff the model column against it, not against itself.
    frames = {}
    for model, offset in (("m1", 0.0), ("m2", 0.2)):
        records = []
        for i, r in enumerate(_synthetic_records()):
            r = dict(r)
            r["energy_per_atom"] = r["mp_energy_per_atom"] + (offset if i == 5 else 0.0)
            records.append(r)
        frames[model] = build_hull(pd.DataFrame(records), "energy_per_atom", "relax")
    report = cross_model_attribution(frames, top_k=2)
    assert report["n_shared_structures"] == 6
    # Only m2 moved li-5, so the worst-error sets must differ across models.
    assert report["top_error_overlap"]["n_shared_worst"] < 2
