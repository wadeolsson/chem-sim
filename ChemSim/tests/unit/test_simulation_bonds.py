"""Tests for distance-based bonding heuristics in the Tier-1 simulation."""

from __future__ import annotations

import pytest

from sim import (
    AngleParameters,
    BondStretchParameters,
    ForceField,
    ForceFieldParameters,
    Particle,
    Simulation,
    SimulationSettings,
)

ELEMENT_METADATA = {
    "H": {"valence_electrons": 1, "category": "nonmetal", "phase": "gas"},
    "Cl": {"valence_electrons": 7, "category": "nonmetal", "phase": "gas"},
    "Na": {"valence_electrons": 1, "category": "alkali metal", "phase": "solid"},
    "O": {"valence_electrons": 6, "category": "nonmetal", "phase": "gas"},
}


def make_force_field() -> ForceField:
    params = {
        ("H", "H"): ForceFieldParameters(epsilon_kj_mol=0.1, sigma_angstrom=3.0, cutoff_angstrom=10.0),
        ("H", "Cl"): ForceFieldParameters(epsilon_kj_mol=0.1, sigma_angstrom=3.5, cutoff_angstrom=12.0),
        ("Na", "Cl"): ForceFieldParameters(epsilon_kj_mol=0.2, sigma_angstrom=3.8, cutoff_angstrom=12.0),
        ("O", "O"): ForceFieldParameters(epsilon_kj_mol=0.2, sigma_angstrom=3.2, cutoff_angstrom=10.0),
    }
    ff = ForceField(params)
    ff.register_bond(("H", "H"), BondStretchParameters(k_kj_mol_nm2=300.0, r0_angstrom=0.74))
    ff.register_bond(("Na", "Cl"), BondStretchParameters(k_kj_mol_nm2=150.0, r0_angstrom=2.4))
    ff.register_bond(("O", "O"), BondStretchParameters(k_kj_mol_nm2=400.0, r0_angstrom=1.21))
    ff.register_angle(("H", "O", "H"), AngleParameters(k_kj_mol_rad2=45.0, theta0_degrees=104.5))
    return ff


def make_particles(distance_nm: float) -> list[Particle]:
    return [
        Particle(
            id=1,
            element="H",
            mass_amu=1.008,
            charge_e=0.0,
            position_nm=(0.0, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
        Particle(
            id=2,
            element="H",
            mass_amu=1.008,
            charge_e=0.0,
            position_nm=(distance_nm, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
    ]


def create_simulation(particles: list[Particle]) -> Simulation:
    return Simulation(particles, make_force_field(), SimulationSettings(timestep_fs=1.0), element_metadata=ELEMENT_METADATA)


def test_covalent_bond_shares_valence() -> None:
    sim = create_simulation(make_particles(distance_nm=0.08))
    sim._update_bonds()  # type: ignore[attr-defined]
    assert len(sim.bonds) == 1
    bond = sim.bonds[0]
    assert bond.bond_type == "covalent"
    assert bond.shared_electrons == 2
    assert bond.order == pytest.approx(1.0)
    atom_a, atom_b = sim.particles
    assert atom_a.valence_electrons == 0
    assert atom_b.valence_electrons == 0


def test_no_bond_when_too_far() -> None:
    sim = create_simulation(make_particles(distance_nm=0.5))
    sim._update_bonds()  # type: ignore[attr-defined]
    assert not sim.bonds


def test_ionic_charge_transfer() -> None:
    particles = [
        Particle(
            id=1,
            element="Na",
            mass_amu=22.990,
            charge_e=0.0,
            position_nm=(0.0, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
        Particle(
            id=2,
            element="Cl",
            mass_amu=35.45,
            charge_e=0.0,
            position_nm=(0.09, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
    ]
    sim = create_simulation(particles)
    sim._update_bonds()  # type: ignore[attr-defined]
    assert len(sim.bonds) == 1
    bond = sim.bonds[0]
    assert bond.bond_type == "ionic"
    sodium = sim.particles[0]
    chlorine = sim.particles[1]
    assert sodium.charge_e == pytest.approx(1.0)
    assert chlorine.charge_e == pytest.approx(-1.0)
    assert sodium.valence_electrons <= sodium.base_valence_electrons - 1


def test_double_bond_allocation() -> None:
    particles = [
        Particle(
            id=1,
            element="O",
            mass_amu=15.999,
            charge_e=0.0,
            position_nm=(0.0, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
        Particle(
            id=2,
            element="O",
            mass_amu=15.999,
            charge_e=0.0,
            position_nm=(0.12, 0.0, 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        ),
    ]
    sim = create_simulation(particles)
    sim._update_bonds()  # type: ignore[attr-defined]
    assert len(sim.bonds) == 1
    bond = sim.bonds[0]
    assert bond.bond_type == "covalent"
    assert bond.order == pytest.approx(2.0)
    assert bond.shared_electrons == 4
    oxygen_a, oxygen_b = sim.particles
    assert oxygen_a.valence_electrons == 4
    assert oxygen_b.valence_electrons == 4
