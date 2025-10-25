"""
Sanity checks for core data assets.
"""

from __future__ import annotations

import json

import pytest


def test_periodic_table_contains_expected_elements(periodic_table_data) -> None:
    elements = periodic_table_data["elements"]
    symbols = {entry["symbol"] for entry in elements}
    for required in {"H", "O"}:
        assert required in symbols


def test_element_numerical_fields_are_present(periodic_table_data) -> None:
    hydrogen = next(entry for entry in periodic_table_data["elements"] if entry["symbol"] == "H")
    assert hydrogen["electronegativity_pauling"] is not None
    assert hydrogen["first_ionization_energy_ev"] > 0.0


def test_spectral_stub_has_normalized_entries(project_root) -> None:
    data_path = project_root / "data" / "spectra" / "water_ir_stub.json"
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    transitions = data["transitions"]
    assert transitions, "Expected at least one vibrational mode."
    assert max(t["intensity"] for t in transitions) == pytest.approx(1.0)


def test_periodic_table_status_fields(periodic_table_data) -> None:
    elements = periodic_table_data["elements"]
    for element in elements:
        assert "status" in element
    pending = [e for e in elements if e["status"] == "pending"]
    complete = [e for e in elements if e["status"] == "complete"]
    assert pending, "Expect pending placeholders for elements beyond current scope."
    assert complete, "Expect at least one fully populated element entry."
