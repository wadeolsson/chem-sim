"""
Utilities for loading ChemSim simulations from YAML configuration files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from sim import (
    AngleParameters,
    BondStretchParameters,
    ForceField,
    ForceFieldParameters,
    Particle,
    Simulation,
    SimulationSettings,
)


@dataclass
class SimulationBundle:
    """Container returned by configuration loader."""

    simulation: Simulation
    metadata: Dict[str, Any]


def load_simulation_from_yaml(
    path: Path,
    *,
    element_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> SimulationBundle:
    """Load a Simulation object plus associated metadata from a YAML config."""
    data = _load_yaml(path)
    force_field = _build_force_field(data.get("force_fields", {}))
    settings = _build_settings(data.get("simulation", {}))
    particles = _build_particles(data.get("system", {}).get("atoms", []))

    simulation = Simulation(
        particles,
        force_field,
        settings,
        element_metadata=element_metadata,
    )
    bundle = SimulationBundle(simulation=simulation, metadata=data.get("metadata", {}))
    return bundle


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at the root.")
    return content


def _build_force_field(config: Dict[str, Any]) -> ForceField:
    pair_params: Dict[Tuple[str, str], ForceFieldParameters] = {}
    for entry in config.get("pairs", []):
        element_i = entry["element_i"]
        element_j = entry["element_j"]
        pair_params[(element_i, element_j)] = ForceFieldParameters(
            epsilon_kj_mol=float(entry["epsilon_kj_mol"]),
            sigma_angstrom=float(entry["sigma_angstrom"]),
            cutoff_angstrom=float(entry.get("cutoff_angstrom") or 12.0),
            coulomb_scale=float(entry.get("coulomb_scale", 1.0)),
        )
    force_field = ForceField(pair_params)

    for entry in config.get("bonds", []):
        force_field.register_bond(
            (entry["element_i"], entry["element_j"]),
            BondStretchParameters(
                k_kj_mol_nm2=float(entry["k_kj_mol_nm2"]),
                r0_angstrom=float(entry["r0_angstrom"]),
            ),
        )

    for entry in config.get("angles", []):
        force_field.register_angle(
            (entry["element_i"], entry["element_j"], entry["element_k"]),
            AngleParameters(
                k_kj_mol_rad2=float(entry["k_kj_mol_rad2"]),
                theta0_degrees=float(entry["theta0_degrees"]),
            ),
        )

    return force_field


def _build_settings(config: Dict[str, Any]) -> SimulationSettings:
    boundary = config.get("box_lengths_nm")
    box_lengths = tuple(boundary) if isinstance(boundary, Iterable) else None
    return SimulationSettings(
        timestep_fs=float(config.get("timestep_fs", 0.5)),
        box_lengths_nm=box_lengths,  # type: ignore[arg-type]
        periodic=bool(config.get("periodic", False)),
        target_temperature_k=float(config.get("target_temperature_k", 300.0)),
        thermostat_tau_fs=config.get("thermostat_tau_fs"),
    )


def _build_particles(atom_list: List[Dict[str, Any]]) -> List[Particle]:
    particles: List[Particle] = []
    for atom in atom_list:
        position = _parse_vector(atom, "position_nm", fallback_key="position_angstrom")
        velocity = _parse_vector(atom, "velocity_nm_fs", default=(0.0, 0.0, 0.0))
        if position is None:
            raise ValueError("Each atom requires position_nm or position_angstrom.")
        particles.append(
            Particle(
                id=int(atom["id"]),
                element=str(atom["element"]),
                mass_amu=float(atom.get("mass_amu", atom.get("mass", 1.0))),
                charge_e=float(atom.get("charge_e", atom.get("charge", 0.0))),
                position_nm=position,
                velocity_nm_fs=velocity,
                phase=str(atom.get("phase", "")).lower(),
            )
        )
    return particles


def _parse_vector(
    atom: Dict[str, Any],
    key: str,
    *,
    fallback_key: Optional[str] = None,
    default: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    if key in atom and atom[key] is not None:
        return _tuple3(atom[key])
    if fallback_key and fallback_key in atom and atom[fallback_key] is not None:
        values = _tuple3(atom[fallback_key])
        if fallback_key.endswith("angstrom"):
            return tuple(value * 0.1 for value in values)
        return values
    return default


def _tuple3(value: Any) -> Tuple[float, float, float]:
    if not isinstance(value, Iterable):
        raise ValueError("Vector field must be iterable with 3 numbers.")
    values = list(value)
    if len(values) != 3:
        raise ValueError("Vector field must contain exactly 3 entries.")
    return float(values[0]), float(values[1]), float(values[2])
