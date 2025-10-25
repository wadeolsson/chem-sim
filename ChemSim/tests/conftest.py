"""
Shared pytest fixtures for ChemSim project scaffolding.

These fixtures expose parsed configuration/data structures so future
tests can build upon them without duplicating I/O logic.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict

import pytest

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Return repository root directory."""
    return pathlib.Path(__file__).resolve().parents[1]


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if yaml is None:
        pytest.skip("PyYAML is required to parse YAML configuration files.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def config_template(project_root: pathlib.Path) -> Dict[str, Any]:
    """Parsed representation of the default simulation config template."""
    config_path = project_root / "config" / "template.yaml"
    return _load_yaml(config_path)


@pytest.fixture(scope="session")
def example_scenario(project_root: pathlib.Path) -> Dict[str, Any]:
    """Parsed example scenario used for schema validation tests."""
    scenario_path = project_root / "tests" / "data" / "example_scenario.yaml"
    return _load_yaml(scenario_path)


@pytest.fixture(scope="session")
def periodic_table_data(project_root: pathlib.Path) -> Dict[str, Any]:
    """Periodic table metadata loaded from the JSON dataset."""
    data_path = project_root / "data" / "periodic_table.json"
    with data_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
