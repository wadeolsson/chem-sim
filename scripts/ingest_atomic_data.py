"""
ChemSim data ingestion utility for atomic/property datasets.

Expected raw CSV columns (minimal):
    symbol,atomic_number,oxidation_states,valence_electrons,electron_configuration,
    covalent_radius_pm,vdw_radius_pm,electronegativity_pauling,
    first_ionization_energy_ev,electron_affinity_ev,status,citations

- `oxidation_states`, `citations`, and `hybridization_preferences` accept
  semicolon-separated values.
- Numerical columns are converted to floats when provided; blank cells become None.
- Provide `--base` to merge against an existing JSON (preserving metadata
  for elements not present in the raw file).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ELEMENT_TEMPLATE: Dict[str, Any] = {
    "symbol": "",
    "atomic_number": None,
    "oxidation_states": [],
    "valence_electrons": None,
    "electron_configuration": None,
    "covalent_radius_pm": None,
    "vdw_radius_pm": None,
    "electronegativity_pauling": None,
    "first_ionization_energy_ev": None,
    "electron_affinity_ev": None,
    "orbital_parameters": {
        "slater_exponents": [],
        "hybridization_preferences": [],
    },
    "metadata": {
        "notes": "Data ingested via scripts/ingest_atomic_data.py",
        "uncertainty": {},
        "citations": [],
    },
    "status": "pending",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize atomic property datasets.")
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        required=True,
        help="Path to raw atomic property data (CSV or JSON).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/periodic_table.json"),
        help="Destination for normalized periodic table JSON.",
    )
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=None,
        help="Optional existing periodic_table.json to merge with.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version label to apply to the output dataset.",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default=None,
        help="Human-readable citation for the dataset merge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and validate without writing the output file.",
    )
    return parser.parse_args()


def parse_semicolon_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Could not parse float from '{value}'") from exc


def load_raw_records(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                # Expect dict keyed by symbol or similar.
                for row in data.values():
                    yield row
            elif isinstance(data, list):
                for row in data:
                    yield row
            else:  # pragma: no cover
                raise TypeError("Unsupported JSON structure for raw data.")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_base_dataset(path: Optional[pathlib.Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"version": "0.0.0", "source": "", "elements": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_element_from_row(
    row: Dict[str, Any], base_element: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    element = deepcopy(base_element) if base_element else deepcopy(DEFAULT_ELEMENT_TEMPLATE)

    symbol = row.get("symbol")
    if not symbol:
        raise ValueError("Raw row missing required 'symbol' column.")
    symbol = symbol.strip()

    atomic_number = row.get("atomic_number")
    if atomic_number:
        element["atomic_number"] = int(atomic_number)
    elif element.get("atomic_number") is None:
        raise ValueError(f"Atomic number required for symbol '{symbol}'.")

    element["symbol"] = symbol
    element["oxidation_states"] = parse_semicolon_list(row.get("oxidation_states")) or element[
        "oxidation_states"
    ]
    if row.get("valence_electrons"):
        element["valence_electrons"] = int(row["valence_electrons"])
    if row.get("electron_configuration"):
        element["electron_configuration"] = row["electron_configuration"].strip()

    for key in (
        "covalent_radius_pm",
        "vdw_radius_pm",
        "electronegativity_pauling",
        "first_ionization_energy_ev",
        "electron_affinity_ev",
    ):
        value = parse_optional_float(row.get(key))
        if value is not None:
            element[key] = value

    if row.get("status"):
        element["status"] = row["status"].strip()
    else:
        # Upgrade to complete if we just ingested quantitative data.
        element["status"] = "complete"

    if row.get("hybridization_preferences"):
        element["orbital_parameters"]["hybridization_preferences"] = parse_semicolon_list(
            row["hybridization_preferences"]
        )

    citations = parse_semicolon_list(row.get("citations"))
    if citations:
        element["metadata"]["citations"] = citations

    if row.get("notes"):
        element["metadata"]["notes"] = row["notes"].strip()

    return element


def convert_raw_to_schema(
    raw_rows: Iterable[Dict[str, Any]],
    base_dataset: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    source_label: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = load_base_dataset(None) if base_dataset is None else deepcopy(base_dataset)
    element_map = {
        entry["symbol"]: entry for entry in dataset.get("elements", []) if "symbol" in entry
    }

    for row in raw_rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        symbol = symbol.strip()
        base_element = element_map.get(symbol)
        updated = update_element_from_row(row, base_element=base_element)
        element_map[symbol] = updated

    sorted_elements = sorted(
        element_map.values(), key=lambda item: item.get("atomic_number", float("inf"))
    )
    dataset["elements"] = sorted_elements
    if version:
        dataset["version"] = version
    if source_label:
        dataset["source"] = source_label
    return dataset


def main() -> None:
    args = parse_args()

    base_dataset = load_base_dataset(args.base or (args.out if args.out.exists() else None))
    raw_rows = list(load_raw_records(args.source))
    dataset = convert_raw_to_schema(
        raw_rows,
        base_dataset=base_dataset,
        version=args.version,
        source_label=args.source_label or base_dataset.get("source"),
    )

    if args.dry_run:
        print(json.dumps(dataset, indent=2))  # noqa: T201 (informational)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)
        handle.write(\"\\n\")


if __name__ == \"__main__\":
    main()
