"""Materials Project access: API-key handling, snapshot fetch and record layout.

This is the single source of truth for how an MP document becomes a benchmark
record. The v0.1 pilot scripts in ``legacy/`` import from here instead of
carrying their own copies of the field list and the record assembly, so a
schema change cannot silently desynchronise the two generations.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

#: Fields requested from the MP summary endpoint. Enough for the energy/hull
#: analysis; the structure is needed to run a potential on it.
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

#: Value of MP_API_KEY in .env.example, i.e. "not filled in".
API_KEY_PLACEHOLDER = "replace_with_your_key"


def require_api_key() -> str:
    """Read MP_API_KEY from the environment or .env, or explain how to set it."""
    load_dotenv()
    key = os.getenv("MP_API_KEY", "").strip()
    if not key or key == API_KEY_PLACEHOLDER:
        raise RuntimeError(
            "MP_API_KEY is missing. Copy .env.example to .env and set it "
            "(https://next-gen.materialsproject.org/api)."
        )
    return key


def chemical_systems(elements: Iterable[str]) -> list[str]:
    """Every elemental, binary, ternary, ... subsystem of `elements`."""
    ordered = sorted(elements)
    return [
        "-".join(combo)
        for size in range(1, len(ordered) + 1)
        for combo in itertools.combinations(ordered, size)
    ]


def record_from_doc(doc: Any) -> dict[str, Any]:
    """Normalise one MP summary document into a benchmark record.

    The MP field names become ``mp_``-prefixed columns so they can never be
    confused with model-derived columns in the same table.
    """
    return {
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


def without_structure(record: dict[str, Any]) -> dict[str, Any]:
    """Record minus the (large, unserialisable) structure payload."""
    return {key: value for key, value in record.items() if key != "structure"}


def write_snapshot(records: list[dict[str, Any]], cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def fetch_chemsys(
    api_key: str, chemsys: str, *, max_sites: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    """Fetch every MP entry of one chemical system."""
    from mp_api.client import MPRester

    query: dict[str, Any] = {
        "chemsys": [chemsys],
        "fields": MP_FIELDS,
        "chunk_size": 1000,
    }
    if max_sites is not None:
        query["num_sites"] = max_sites
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**query)
    return [record_from_doc(doc) for doc in docs]


def fetch_phase_space(
    api_key: str, elements: list[str], cache_path: str | Path
) -> list[dict[str, Any]]:
    """Fetch and cache every subsystem of a chemical space."""
    records: list[dict[str, Any]] = []
    for chemsys in chemical_systems(elements):
        docs = fetch_chemsys(api_key, chemsys)
        print(f"{chemsys}: {len(docs)}")
        records.extend(docs)
    records.sort(key=lambda row: row["material_id"])
    write_snapshot(records, cache_path)
    return records


def load_phase_space(
    cache_path: str | Path, elements: list[str] | None = None
) -> list[dict[str, Any]]:
    """Load the cached MP snapshot, fetching it only if absent."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not elements:
        raise RuntimeError(f"MP cache missing and no elements given: {cache_path}")
    return fetch_phase_space(require_api_key(), elements, cache_path)
