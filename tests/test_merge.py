"""Tests for merging a split audit into figures and a summary table.

The important property is graceful degradation: an audit that only got through
one or two of the configured models must still produce figures and a table, and
must say which models are absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mlip_audit.merge import (
    hull_mae_by_chemsys,
    load_records,
    markdown_summary,
    metrics_table,
    model_frames,
    plot_hull_mae_by_chemsys,
    plot_hull_parity,
)
from mlip_audit.runner import _slug, run_audit

from .conftest import StubCalc, write_synthetic_snapshot

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "merge_and_plot.py"


def _run_one_model(tmp_path: Path, model: str) -> Path:
    """Produce a real single-model checkpoint, using the stub calculator."""
    from mlip_audit import runner

    cache = write_synthetic_snapshot(tmp_path / "mp.json")
    out_dir = tmp_path / f"run_{model}"
    original = runner.build_calculator
    runner.build_calculator = lambda spec: StubCalc(spec.name)
    try:
        runner.run_audit(
            _config_for(cache, out_dir, [model]),
        )
    finally:
        runner.build_calculator = original
    return out_dir


def _config_for(cache: Path, out_dir: Path, models: list[str]):
    from mlip_audit.config import AuditConfig, MDSpec, ModelSpec, RelaxSpec

    return AuditConfig(
        name="test",
        elements=["Li", "O"],
        models=[ModelSpec(name=name, type="chgnet") for name in models],
        relax=RelaxSpec(enabled=False),
        md=MDSpec(enabled=False),
        single_point=True,
        bootstrap_resamples=20,
        mp_cache=cache,
        out_dir=out_dir,
    )


def _load_cli(name: str):
    """Import scripts/merge_and_plot.py as a module so main() is callable."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_reads_a_single_model_checkpoint(tmp_path):
    run_dir = _run_one_model(tmp_path, "m-a")
    frames = model_frames(load_records(run_dir))
    assert list(frames) == ["m-a"]
    assert len(frames["m-a"]) == 8


def test_merge_survives_missing_models(tmp_path):
    """Only m-a ran; every downstream function must still work."""
    run_dir = _run_one_model(tmp_path, "m-a")
    frames = model_frames(load_records(run_dir))

    table = hull_mae_by_chemsys(frames)
    assert not table.empty
    assert list(table.columns) == ["m-a"]

    metrics = metrics_table(frames, candidate_window=0.05, bootstrap_resamples=20)
    assert "hull_mae_eV_per_atom" in metrics["m-a"]

    summary = markdown_summary(frames, metrics, expected=["m-a", "m-b"])
    assert "m-b" in summary
    assert "Missing" in summary
    assert "m-a" in summary

    assert plot_hull_mae_by_chemsys(table, tmp_path / "fig" / "chemsys.png")
    assert plot_hull_parity(frames, tmp_path / "fig" / "parity.png")


def test_merge_ignores_failed_records(tmp_path):
    """A crashed record is not silently counted as a result."""
    run_dir = _run_one_model(tmp_path, "m-a")
    path = run_dir / "checkpoints.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["status"] = "error"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frames = model_frames(load_records(run_dir))
    assert len(frames["m-a"]) == 7


def test_merge_rejects_unknown_protocol(tmp_path):
    run_dir = _run_one_model(tmp_path, "m-a")
    try:
        model_frames(load_records(run_dir), protocol="nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_plots_return_none_without_data(tmp_path):
    """No data must be a clean no-op, not a crash."""
    import pandas as pd

    empty = pd.DataFrame()
    assert plot_hull_mae_by_chemsys(empty, tmp_path / "x.png") is None
    assert plot_hull_parity({}, tmp_path / "y.png") is None
    assert hull_mae_by_chemsys({}).empty
    assert not (tmp_path / "x.png").exists()


def test_merge_cli_reports_absent_models(tmp_path, monkeypatch, capsys):
    run_dir = _run_one_model(tmp_path, "m-a")
    config_file = tmp_path / "audit.yaml"
    config_file.write_text(
        "name: t\n"
        "elements: [Li, O]\n"
        "mp_cache: mp.json\n"
        "out_dir: out\n"
        "models:\n"
        "  - {name: m-a, type: chgnet}\n"
        "  - {name: m-b, type: mace}\n",
        encoding="utf-8",
    )
    fig_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_and_plot.py",
            "--run-dir", str(run_dir),
            "--config", str(config_file),
            "--fig-dir", str(fig_dir),
        ],
    )
    assert _load_cli("merge_and_plot_cli").main() == 0

    out = capsys.readouterr().out
    assert "models absent" in out
    assert "m-b" in out
    assert (fig_dir / "model_summary.md").exists()
    assert (fig_dir / "hull_mae_by_chemsys_multimodel.png").exists()
    assert (fig_dir / "hull_parity_multimodel.png").exists()
    summary = (fig_dir / "model_summary.md").read_text(encoding="utf-8")
    assert "m-a" in summary and "m-b" in summary


def test_merge_cli_errors_without_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_and_plot.py", "--run-dir", str(tmp_path / "absent")],
    )
    assert _load_cli("merge_and_plot_cli2").main() == 1
    assert "no checkpoint" in capsys.readouterr().err


def test_merge_reports_agree_with_runner_metrics(tmp_path, stub_backend, stub_config):
    """The merged table must not drift from what run_audit recorded."""
    out_dir = tmp_path / "run"
    report = run_audit(stub_config(out_dir, ["m-a", "m-b"]))
    frames = model_frames(load_records(out_dir))
    metrics = metrics_table(frames, candidate_window=0.05, bootstrap_resamples=20)
    for model in ("m-a", "m-b"):
        assert (
            metrics[model]["hull_mae_eV_per_atom"]["point"]
            == report[model]["single_point"]["hull_mae_eV_per_atom"]["point"]
        )
        assert (
            metrics[model]["candidate_classification_f1"]["point"]
            == report[model]["single_point"]["candidate_classification_f1"]["point"]
        )


def test_merge_tables_hold_every_model(tmp_path, stub_backend, stub_config):
    out_dir = tmp_path / "run"
    run_audit(stub_config(out_dir, ["m-a", "m-b"]))
    frames = model_frames(load_records(out_dir))
    table = hull_mae_by_chemsys(frames)
    assert list(table.columns) == ["m-a", "m-b"]
    summary = markdown_summary(
        frames,
        metrics_table(frames, candidate_window=0.05, bootstrap_resamples=20),
    )
    assert "m-a" in summary and "m-b" in summary
    assert "Missing" not in summary


def test_slug_maps_model_names_onto_committed_directories():
    """These slugs name directories under data/runs/llzo/; they must not drift."""
    assert _slug("chgnet-0.3.0") == "chgnet_030"
    assert _slug("mace-mp-0-medium") == "mace_mp_0_medium"
    assert _slug("a b.c-d") == "a_bc_d"
