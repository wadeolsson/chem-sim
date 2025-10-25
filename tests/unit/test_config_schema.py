"""
Schema-oriented tests for baseline configuration assets.

These tests provide early warnings if template structures change in ways
that downstream tooling is not expecting.
"""

from __future__ import annotations

from typing import Set


def test_config_template_sections(config_template) -> None:
    required_sections: Set[str] = {
        "metadata",
        "simulation",
        "thermostat",
        "force_fields",
        "system",
        "environment",
        "output",
        "ui",
    }
    assert required_sections.issubset(
        config_template
    ), f"Missing sections: {required_sections - set(config_template)}"


def test_config_atoms_have_core_fields(config_template) -> None:
    atoms = config_template["system"]["atoms"]
    for atom in atoms:
        assert {"id", "element", "position_angstrom", "velocity_angstrom_fs"} <= atom.keys()
        assert len(atom["position_angstrom"]) == 3
        assert len(atom["velocity_angstrom_fs"]) == 3


def test_example_scenario_references_template(example_scenario) -> None:
    assert "config_template" in example_scenario
    assert example_scenario["config_template"].endswith("config/template.yaml")

