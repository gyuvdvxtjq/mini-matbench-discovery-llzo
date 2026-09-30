"""Orchestration: config -> per-model protocol runs -> metrics -> report."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import (
    build_hull,
    cross_model_attribution,
    evaluate_stability,
    write_metrics,
)
from .checkpoint import CheckpointStore
from .config import AuditConfig
from .data import load_phase_space
from .md import analyze_md, run_md
from .models import ModelUnavailable, build_calculator


def _slug(name: str) -> str:
    return name.replace(".", "").replace("-", "_").replace(" ", "_")


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


def run_audit(
    config: AuditConfig,
    *,
    only_model: str | None = None,
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

    all_metrics: dict[str, Any] = {"skipped_models": {}}
    relax_frames: dict[str, pd.DataFrame] = {}
    md_results: dict[str, Any] = {}

    for spec in config.models:
        if not spec.enabled or (only_model and spec.name != only_model):
            continue
        spec = replace(spec, device=device_override or spec.device)
        try:
            calculator = build_calculator(spec)
        except ModelUnavailable as exc:
            print(f"skip {spec.name}: {exc}", flush=True)
            all_metrics["skipped_models"][spec.name] = str(exc)
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
                md_results[spec.name] = analyze_md(md_records, config.md, model_dir)
                model_metrics["md"] = md_results[spec.name]
        write_metrics(out_dir, _slug(spec.name), model_metrics)
        all_metrics[spec.name] = model_metrics

    if len(relax_frames) >= 2:
        all_metrics["cross_model_attribution"] = cross_model_attribution(relax_frames)
    (out_dir / "report.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    return all_metrics
