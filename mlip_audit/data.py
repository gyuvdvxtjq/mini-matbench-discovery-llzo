"""Data layer: Materials Project snapshots for a chemical space, cached locally."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "structure",
    "nsites",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
]


def chemical_systems(elements: list[str]) -> list[str]:
    return [
        "-".join(sorted(combo))
        for size in range(1, len(elements) + 1)
        for combo in itertools.combinations(sorted(elements), size)
    ]


def require_api_key() -> str:
    load_dotenv()
    key = os.getenv("MP_API_KEY", "").strip()
    if not key or key == "replace_with_your_key":
        raise RuntimeError("MP_API_KEY is missing from .env.")
    return key


def fetch_phase_space(elements: list[str], cache_path: Path) -> list[dict[str, Any]]:
    from mp_api.client import MPRester

    records: list[dict[str, Any]] = []
    with MPRester(require_api_key()) as mpr:
        for chemsys in chemical_systems(elements):
            docs = mpr.materials.summary.search(
                chemsys=[chemsys], fields=MP_FIELDS, chunk_size=1000
            )
            print(f"{chemsys}: {len(docs)}")
            records.extend(
                {
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "chemsys": doc.chemsys,
                    "nsites": int(doc.nsites),
                    "mp_energy_per_atom": float(doc.energy_per_atom),
                    "mp_formation_energy_per_atom": float(doc.formation_energy_per_atom),
                    "mp_energy_above_hull": float(doc.energy_above_hull),
                    "mp_is_stable": bool(doc.is_stable),
                    "structure": doc.structure.as_dict(),
                }
                for doc in docs
            )
    records.sort(key=lambda row: row["material_id"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def load_phase_space(
    cache_path: Path, elements: list[str] | None = None
) -> list[dict[str, Any]]:
    """Load the cached MP snapshot, fetching it only if absent."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not elements:
        raise RuntimeError(f"MP cache missing and no elements given: {cache_path}")
    return fetch_phase_space(elements, cache_path)
