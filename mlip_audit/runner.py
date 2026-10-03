"""Orchestration: config -> per-model protocol runs -> metrics -> report.

The runner is deliberately restartable and additive. Because mace-torch and
deepmd-kit cannot share an environment (see cloud/setup_base.sh), the audit is
expected to be executed in several passes -- one per environment -- each
passing a subset of models. Every record is therefore addressed by
(model, protocol, params-hash, material_id) rather than by "this run", so:

* a second pass skips whatever an earlier pass already computed, and
* `report.json` is merged, never overwritten, so the halves add up.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .analysis import (
    build_hull,
    cross_model_attribution,
    evaluate_stability,
    write_metrics,
)
from .checkpoint import CheckpointStore
from .config import AuditConfig
from .md import analyze_md, run_md
from .models import ModelUnavailable, build_calculator, resolve_device
from .mp import load_phase_space


def _slug(name: str) -> str:
    """Filesystem-safe model name. Dots are dropped, other separators become
    underscores: 'chgnet-0.3.0' -> 'chgnet_030'. This maps the model names in
    configs/llzo.yaml onto the committed data/runs/llzo/ subdirectories, so it
    must not change without renaming those directories too."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name.replace(".", ""))


def _model_protocol_frames(
    spec_name: str,
    records: list[dict[str, Any]],
    config: AuditConfig,
    store: CheckpointStore,
    calculator,
) -> dict[str, pd.DataFrame]:
    from .protocols import run_relaxation, run_single_point

    frames: dict[str, pd.DataFrame] = {}
    if config.single_point:
        rows = run_single_point(records, calculator, store, spec_name)
        frame = build_hull(pd.DataFrame(rows), "energy_per_atom", "sp")
        frames["single_point"] = frame
    if config.relax.enabled:
        rows = run_relaxation(
            records,
            lambda: calculator,
            store,
            spec_name,
            fmax=config.relax.fmax,
            steps=config.relax.steps,
            relax_cell=config.relax.relax_cell,
        )
        frame = build_hull(pd.DataFrame(rows), "relaxed_e_per_atom", "relax")
        frames["relaxation"] = frame
    return frames


def _load_existing_report(path: Path) -> dict[str, Any]:
    """Previous passes' results, so a split run does not erase them."""
    if not path.exists():
        return {"skipped_models": {}}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"skipped_models": {}}
    if not isinstance(previous, dict):
        return {"skipped_models": {}}
    previous.setdefault("skipped_models", {})
    return previous


def _collect_relax_frames(
    out_dir: Path, config: AuditConfig, fresh: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Relaxation frames for every model with results on disk.

    Models computed in an earlier pass are read back from their per-model CSV,
    so cross-model attribution spans the whole audit rather than only the
    models that happened to run in this process. Frames computed in this pass
    always win over the on-disk copy.
    """
    frames: dict[str, pd.DataFrame] = dict(fresh)
    if not config.relax.enabled:
        return frames
    names_by_slug = {_slug(spec.name): spec.name for spec in config.models}
    for csv_path in sorted(out_dir.glob("*/relaxation.csv")):
        model = names_by_slug.get(csv_path.parent.name, csv_path.parent.name)
        if model in frames:
            continue
        try:
            frames[model] = pd.read_csv(csv_path)
        except (OSError, ValueError) as exc:
            print(f"  skip stale {csv_path}: {exc}", flush=True)
    return frames


def run_audit(
    config: AuditConfig,
    *,
    only_models: Iterable[str] | None = None,
    device_override: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_phase_space(config.mp_cache, config.elements)
    if limit:
        records = records[:limit]
    print(f"phase-space structures: {len(records)}", flush=True)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(out_dir / "checkpoints.jsonl")

    wanted = set(only_models) if only_models else None
    report = _load_existing_report(out_dir / "report.json")
    report.setdefault("skipped_models", {})
    relax_frames: dict[str, pd.DataFrame] = {}

    for spec in config.models:
        if not spec.enabled or (wanted is not None and spec.name not in wanted):
            continue
        spec = replace(spec, device=resolve_device(spec.device, device_override))
        try:
            calculator = build_calculator(spec)
        except ModelUnavailable as exc:
            print(f"skip {spec.name} (device={spec.device}): {exc.report()}", flush=True)
            report["skipped_models"][spec.name] = exc.report()
            continue

        model_dir = out_dir / _slug(spec.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_metrics: dict[str, Any] = {}
        for protocol, frame in _model_protocol_frames(
            spec.name, records, config, store, calculator
        ).items():
            prefix = "relax" if protocol == "relaxation" else "sp"
            frame.to_csv(
                model_dir / f"{protocol}.csv", index=False, encoding="utf-8-sig"
            )
            model_metrics[protocol] = evaluate_stability(
                frame,
                prefix,
                candidate_window=config.candidate_window_ev,
                bootstrap_resamples=config.bootstrap_resamples,
            )
            if protocol == "relaxation":
                relax_frames[spec.name] = frame
        if config.md.enabled:
            target = next(
                (r for r in records if r["material_id"] == config.md.material_id),
                None,
            )
            if target is None:
                print(f"md target {config.md.material_id} not in dataset", flush=True)
            else:
                md_records = run_md(
                    target,
                    lambda: build_calculator(spec),
                    store,
                    spec.name,
                    config.md,
                    model_dir,
                )
                model_metrics["md"] = analyze_md(md_records, config.md, model_dir)
        write_metrics(out_dir, _slug(spec.name), model_metrics)
        report[spec.name] = model_metrics

    attribution_frames = _collect_relax_frames(out_dir, config, relax_frames)
    if len(attribution_frames) >= 2:
        report["cross_model_attribution"] = cross_model_attribution(attribution_frames)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
