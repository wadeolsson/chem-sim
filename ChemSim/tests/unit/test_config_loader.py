"""Tests for YAML simulation loader."""

from __future__ import annotations

from src.config_loader import load_simulation_from_yaml


def test_loads_water_cluster(project_root, periodic_table_data):
    path = project_root / "config" / "presets" / "water_cluster.yaml"
    metadata_map = {entry["symbol"]: entry for entry in periodic_table_data["elements"]}
    bundle = load_simulation_from_yaml(path, element_metadata=metadata_map)
    sim = bundle.simulation
    assert len(sim.particles) == 9
    assert bundle.metadata["name"] == "Water Cluster"
