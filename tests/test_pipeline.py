"""Tests for the runnable path: device resolution, backend errors, and the
pass-by-pass workflow.

These use the stub calculator and synthetic MP snapshot from conftest.py, so
the whole config -> protocols -> checkpoint -> report pipeline is exercised
without downloading a model or touching the Materials Project API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_audit.config import ModelSpec, load_config
from mlip_audit.models import (
    BACKEND_SETUP,
    ModelUnavailable,
    build_calculator,
    detect_device,
    resolve_device,
)
from mlip_audit.runner import run_audit

from .conftest import checkpoint_counts


# ------------------------------------------------------- device resolution
def test_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert detect_device() == "cpu"
    assert resolve_device("auto") == "cpu"
    assert resolve_device(None) == "cpu"
    assert resolve_device("") == "cpu"


def test_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert detect_device() == "cuda"
    assert resolve_device("auto") == "cuda"


def test_device_override_and_explicit_config_win(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    # An explicit --device overrides the detected value...
    assert resolve_device("auto", "cuda") == "cuda"
    # ...and so does an explicit config value, which is never auto-detected.
    assert resolve_device("mps") == "mps"
    assert resolve_device("cuda") == "cuda"


def test_device_auto_survives_a_broken_cuda_probe(monkeypatch):
    def _explode():
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr("torch.cuda.is_available", _explode)
    assert detect_device() == "cpu"


def test_device_auto_does_not_leak_into_checkpoints(tmp_path, stub_backend, stub_config):
    """device is not part of the checkpoint key, so a CPU run stays valid on GPU."""
    out_dir = tmp_path / "run"
    run_audit(stub_config(out_dir, ["m-a"]), device_override="cpu")
    first = checkpoint_counts(out_dir)["m-a"]
    # Re-running on another device must resume, not recompute.
    run_audit(stub_config(out_dir, ["m-a"]), device_override="cuda")
    assert checkpoint_counts(out_dir)["m-a"] == first


def test_config_without_device_key_defaults_to_auto(tmp_path):
    cfg_file = tmp_path / "audit.yaml"
    cfg_file.write_text(
        "name: t\nelements: [Li, O]\nmodels: [{name: m1, type: chgnet}]\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.models[0].device == "auto"


def test_config_paths_resolve_against_the_repo_root(tmp_path):
    """Relative paths in a config must not depend on the caller's cwd."""
    cfg_file = tmp_path / "audit.yaml"
    cfg_file.write_text(
        "name: t\nelements: [Li]\nout_dir: data/runs/tmp-abc\nmodels: []\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.out_dir.is_absolute()
    assert cfg.out_dir.parts[-3:] == ("data", "runs", "tmp-abc")

    # Absolute paths (what tests use) pass through untouched.
    absolute = tmp_path / "somewhere"
    cfg_file.write_text(
        f"name: t\nelements: [Li]\nout_dir: {absolute}\nmodels: []\n",
        encoding="utf-8",
    )
    assert load_config(cfg_file).out_dir == absolute


SHIPPED_CONFIGS = ["configs/llzo.yaml", "configs/smoke.yaml"]


@pytest.mark.parametrize("relative", SHIPPED_CONFIGS)
def test_shipped_configs_pin_no_device(relative):
    """A hard-coded device is a regression: it breaks laptops and workers alike."""
    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((repo_root / relative).read_text(encoding="utf-8"))
    assert raw["models"], "config declares no models"
    for model in raw["models"]:
        assert "device" not in model, f"{relative}: {model['name']} pins a device"


@pytest.mark.parametrize("relative", SHIPPED_CONFIGS)
def test_shipped_configs_load_with_auto_devices(relative):
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / relative)
    assert cfg.models
    for spec in cfg.models:
        assert spec.device == "auto"
    assert cfg.mp_cache.is_absolute() and cfg.out_dir.is_absolute()


def test_llzo_config_declares_the_two_models():
    """The shipped config names exactly the models the reference ran."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "configs/llzo.yaml")
    by_name = {spec.name: spec for spec in cfg.models}
    assert set(by_name) == {
        "chgnet-0.3.0",
        "mace-mp-0-medium",
    }
    # Every declared backend type must have a documented setup script.
    for spec in by_name.values():
        assert spec.type in BACKEND_SETUP
        assert Path(repo_root / BACKEND_SETUP[spec.type][1]).is_file()


# ------------------------------------------------------------ backend errors
def test_skip_message_names_package_and_setup_script():
    """A skip must say what to install and which script installs it."""
    exc = ModelUnavailable(
        "mace-torch not installed: No module named 'mace'",
        package="mace-torch",
        setup_script="cloud/setup_mace.sh",
    )
    report = exc.report()
    assert "mace-torch" in report
    assert "cloud/setup_mace.sh" in report


def test_report_without_hints_is_just_the_message():
    assert ModelUnavailable("boom").report() == "boom"


@pytest.mark.parametrize(
    ("model_type", "package", "script"),
    [
        ("chgnet", "chgnet", "cloud/setup_base.sh"),
        ("mace", "mace-torch", "cloud/setup_mace.sh"),
    ],
)
def test_backend_setup_map_covers_every_model_type(model_type, package, script):
    assert BACKEND_SETUP[model_type] == (package, script)


def test_unknown_model_type_is_reported():
    with pytest.raises(ModelUnavailable, match="unknown model type"):
        build_calculator(ModelSpec(name="mystery", type="nope"))


def test_absent_backend_is_recorded_in_report(tmp_path, stub_config, monkeypatch):
    """An unavailable model lands in skipped_models with an actionable reason."""
    from .conftest import StubCalc

    out_dir = tmp_path / "run"
    config = stub_config(out_dir, ["m-a"])
    config.models.append(ModelSpec(name="m-missing", type="mace"))

    def _build(spec):
        if spec.type == "mace":
            raise ModelUnavailable(
                "mace-torch not installed: No module named 'mace'",
                package="mace-torch",
                setup_script="cloud/setup_mace.sh",
            )
        return StubCalc(spec.name)

    monkeypatch.setattr("mlip_audit.runner.build_calculator", _build)
    report = run_audit(config)
    assert "m-missing" in report["skipped_models"]
    assert "cloud/setup_mace.sh" in report["skipped_models"]["m-missing"]
    assert "m-a" in report, "the available model must still be reported"


# ------------------------------------------- pass-by-pass equivalence
def test_split_batches_match_a_single_run(tmp_path, stub_backend, stub_config):
    """Two passes over one config == one pass over both models.

    This is the property the runner depends on when a config is executed in
    passes (e.g. one model at a time) against the same checkpoint file.
    """
    models = ["m-a", "m-b"]

    single_dir = tmp_path / "single"
    run_audit(stub_config(single_dir, models))
    single = checkpoint_counts(single_dir)

    stub_backend.clear()
    split_dir = tmp_path / "split"
    config = stub_config(split_dir, models)
    run_audit(config, only_models=["m-a"])
    run_audit(config, only_models=["m-b"])
    split = checkpoint_counts(split_dir)

    assert set(single) == set(split) == set(models)
    for model in models:
        assert len(single[model]) == len(split[model]) > 0
        # Not just the count: the records themselves must be identical.
        assert [r["_key"] for r in single[model]] == [r["_key"] for r in split[model]]
        assert [r["energy_per_atom"] for r in single[model]] == [
            r["energy_per_atom"] for r in split[model]
        ]
    # The second pass resumed m-a instead of recomputing it.
    assert stub_backend.count("m-a") == 1


def test_split_runs_accumulate_in_one_report(tmp_path, stub_backend, stub_config):
    """report.json must be merged across passes, not overwritten by the last one."""
    out_dir = tmp_path / "split"
    config = stub_config(out_dir, ["m-a", "m-b"], relax=True)

    first = run_audit(config, only_models=["m-a"])
    assert set(first) >= {"m-a", "skipped_models"}
    assert "m-b" not in first

    merged = run_audit(config, only_models=["m-b"])
    assert "m-a" in merged, "the first batch's results were clobbered"
    assert "m-b" in merged
    # Cross-model attribution is computed from relaxation frames. m-a's frame
    # only exists as a CSV from the previous pass, so it must be read back.
    assert "cross_model_attribution" in merged
    assert set(merged["cross_model_attribution"]["models"]) == {"m-a", "m-b"}
    on_disk = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert set(on_disk) >= {"m-a", "m-b"}


def test_unknown_model_filter_selects_nothing(tmp_path, stub_backend, stub_config):
    report = run_audit(stub_config(tmp_path / "run", ["m-a"]), only_models=["nope"])
    assert stub_backend == []
    assert set(report) == {"skipped_models"}


def test_disabled_model_is_never_built(tmp_path, stub_backend, stub_config):
    config = stub_config(tmp_path / "run", ["m-a"])
    config.models[0].enabled = False
    report = run_audit(config)
    assert stub_backend == []
    assert set(report) == {"skipped_models"}


def test_audit_runs_from_any_working_directory(tmp_path, stub_backend, stub_config, monkeypatch):
    """Outputs land in the config's out_dir regardless of cwd."""
    import os

    out_dir = tmp_path / "elsewhere"
    monkeypatch.chdir(tmp_path)
    before = os.getcwd()
    run_audit(stub_config(out_dir, ["m-a"]))
    assert os.getcwd() == before
    assert (out_dir / "checkpoints.jsonl").exists()
    assert (out_dir / "report.json").exists()
