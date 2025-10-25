"""
Preliminary Tier-1 molecular dynamics scaffold for ChemSim.

Unit conventions (subject to revision as the engine matures):
    - Positions: nanometres (nm)
    - Velocities: nm / femtosecond (fs)
    - Time step: femtoseconds
    - Mass: atomic mass units (amu)
    - Charges: elementary charge (e)
    - Lennard-Jones epsilon: kJ/mol
    - Lennard-Jones sigma: Å (converted to nm internally)
    - Force outputs: kJ/mol/nm

The numerical constants convert between kJ/mol and SI units so that
velocity-Verlet integration operates in nm/fs space. Future work will
revisit these conversions once benchmarking begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.chem_data import get_valence_rule


Vector = Tuple[float, float, float]

AVOGADRO = 6.022_140_76e23
AMU_TO_KG = 1.660_539_066_60e-27
ANGSTROM_TO_NM = 0.1
FS_TO_S = 1e-15
NM_TO_M = 1e-9
KJMOL_TO_J = 1000.0 / AVOGADRO
FORCE_KJMOL_NM_TO_NEWTON = KJMOL_TO_J / NM_TO_M
ACCEL_M_S2_TO_NM_FS2 = (1 / NM_TO_M) * (FS_TO_S**2)  # equals 1e-21
COULOMB_CONSTANT_KJMOL_NM = 138.935_456  # kJ mol^-1 nm e^-2
PM_TO_NM = 1e-3
GAS_EXPANSION_FACTOR = 0.02
LIQUID_PERSONAL_SPACE_NM = 0.25
LIQUID_REPULSION_FACTOR = 0.01
DEFAULT_BOND_K_KJMOL_NM2 = 200.0
DEFAULT_ANGLE_K_KJMOL_RAD2 = 20.0
BOLTZMANN_CONSTANT_J_PER_K = 1.380_649e-23
BOLTZMANN_CONSTANT_KJ_MOL_K = 0.008314462618
M_S_TO_NM_FS = 1e-6
BOND_CAPTURE_MULTIPLIER = 1.8
CAPTURE_FORCE_K = 50.0

DEFAULT_COVALENT_RADII_PM: Dict[str, float] = {
    "H": 31.0,
    "C": 76.0,
    "N": 71.0,
    "O": 66.0,
    "F": 57.0,
    "Na": 166.0,
    "Mg": 141.0,
    "Al": 121.0,
    "Si": 111.0,
    "P": 107.0,
    "S": 105.0,
    "Cl": 102.0,
    "K": 203.0,
    "Ca": 176.0,
    "Fe": 132.0,
    "Cu": 132.0,
    "Zn": 122.0,
    "Pb": 146.0,
}
BOND_TOLERANCE_PM = 40.0

NON_METAL_ELEMENTS = {
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
}

METAL_ELEMENTS = {
    "Li",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
}


def vector_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_scale(v: Vector, scalar: float) -> Vector:
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def vector_length(v: Vector) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vector_dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_zero() -> Vector:
    return (0.0, 0.0, 0.0)


def minimum_image(delta: Vector, box_lengths_nm: Optional[Vector]) -> Vector:
    if box_lengths_nm is None:
        return delta
    d0, d1, d2 = delta
    bx, by, bz = box_lengths_nm
    for idx, value in enumerate((d0, d1, d2)):
        box = (bx, by, bz)[idx]
        if box == 0:
            continue
        half_box = 0.5 * box
        if value > half_box:
            value -= box
        elif value < -half_box:
            value += box
        if idx == 0:
            d0 = value
        elif idx == 1:
            d1 = value
        else:
            d2 = value
    return (d0, d1, d2)


@dataclass
class Particle:
    id: int
    element: str
    mass_amu: float
    charge_e: float
    position_nm: Vector
    velocity_nm_fs: Vector
    force_kjmol_nm: Vector = field(default_factory=vector_zero)
    valence_electrons: int = 0
    base_valence_electrons: int = 0
    phase: str = ""
    bond_slots: int = 0
    max_bond_slots: int = 0
    valence_target: int = 0


@dataclass
class ForceFieldParameters:
    epsilon_kj_mol: float
    sigma_angstrom: float
    cutoff_angstrom: Optional[float] = None
    coulomb_scale: float = 1.0


@dataclass
class BondStretchParameters:
    k_kj_mol_nm2: float
    r0_angstrom: float


@dataclass
class AngleParameters:
    k_kj_mol_rad2: float
    theta0_degrees: float


class ForceField:
    """
    Stores Lennard-Jones/Coulomb parameters keyed by element pairs.
    Pairs are normalized so that ordering does not matter.
    """

    def __init__(self, parameters: Dict[Tuple[str, str], ForceFieldParameters]):
        normalized: Dict[Tuple[str, str], ForceFieldParameters] = {}
        for (a, b), params in parameters.items():
            key = tuple(sorted((a, b)))
            normalized[key] = params
        self._params = normalized
        self.bond_parameters: Dict[Tuple[str, str], BondStretchParameters] = {}
        self.angle_parameters: Dict[Tuple[str, str, str], AngleParameters] = {}

    def register_bond(self, element_pair: Tuple[str, str], params: "BondStretchParameters") -> None:
        key = tuple(sorted(element_pair))
        self.bond_parameters[key] = params

    def register_angle(self, elements: Tuple[str, str, str], params: "AngleParameters") -> None:
        self.angle_parameters[tuple(elements)] = params

    def get(self, element_a: str, element_b: str) -> Optional[ForceFieldParameters]:
        key = tuple(sorted((element_a, element_b)))
        return self._params.get(key)

    def get_bond(self, element_a: str, element_b: str) -> Optional["BondStretchParameters"]:
        key = tuple(sorted((element_a, element_b)))
        return self.bond_parameters.get(key)

    def get_angle(self, element_triplet: Tuple[str, str, str]) -> Optional["AngleParameters"]:
        return self.angle_parameters.get(element_triplet)


def lennard_jones_force(
    delta_nm: Vector, params: ForceFieldParameters
) -> Tuple[Vector, float]:
    sigma_nm = params.sigma_angstrom * ANGSTROM_TO_NM
    cutoff_nm = (
        params.cutoff_angstrom * ANGSTROM_TO_NM if params.cutoff_angstrom is not None else None
    )

    r = vector_length(delta_nm)
    if r == 0.0:
        return vector_zero(), 0.0
    if cutoff_nm is not None and r > cutoff_nm:
        return vector_zero(), 0.0

    sr = sigma_nm / r
    sr6 = sr**6
    sr12 = sr6**2

    force_magnitude = 24.0 * params.epsilon_kj_mol / r * (2.0 * sr12 - sr6)
    unit_vector = vector_scale(delta_nm, 1.0 / r)
    force_vector = vector_scale(unit_vector, force_magnitude * -1.0)
    potential = 4.0 * params.epsilon_kj_mol * (sr12 - sr6)
    return force_vector, potential


def coulomb_force(
    delta_nm: Vector, charge_a: float, charge_b: float, params: ForceFieldParameters
) -> Tuple[Vector, float]:
    r = vector_length(delta_nm)
    if r == 0.0 or charge_a == 0.0 or charge_b == 0.0 or params.coulomb_scale == 0.0:
        return vector_zero(), 0.0

    scale = params.coulomb_scale
    qc = charge_a * charge_b
    prefactor = scale * COULOMB_CONSTANT_KJMOL_NM * qc / (r * r)
    unit_vector = vector_scale(delta_nm, 1.0 / r)
    force_vector = vector_scale(unit_vector, -prefactor)
    potential = scale * COULOMB_CONSTANT_KJMOL_NM * qc / r
    return force_vector, potential


@dataclass
class SimulationSettings:
    timestep_fs: float
    box_lengths_nm: Optional[Vector] = None
    periodic: bool = False
    target_temperature_k: float = 300.0
    thermostat_tau_fs: Optional[float] = None


@dataclass
class AtomState:
    id: int
    element: str
    position_nm: Vector
    velocity_nm_fs: Vector
    valence_electrons: int
    bond_slots: int
    charge_e: float


@dataclass
class BondState:
    atom_i: int
    atom_j: int
    order: float = 1.0
    bond_type: str = "covalent"
    shared_electrons: int = 0
    charge_transfer: int = 0
    rest_length_nm: float = 0.0


@dataclass
class SimulationSnapshot:
    step_index: int
    time_fs: float
    atom_states: List[AtomState]
    bonds: List[BondState] = field(default_factory=list)


class Simulation:
    """
    Simple velocity-Verlet simulator for the Tier-1 classical layer.
    """

    def __init__(
        self,
        particles: Iterable[Particle],
        force_field: ForceField,
        settings: SimulationSettings,
        element_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.particles: List[Particle] = []
        self.force_field = force_field
        self.settings = settings
        self.element_metadata: Dict[str, Dict[str, Any]] = element_metadata or {}
        self._potential_energy = 0.0
        self.current_step: int = 0
        self.time_fs: float = 0.0
        self.bonds: List[BondState] = []
        self.random = random.Random()
        self._velocities_initialized = False
        for particle in particles:
            self.add_particle(particle)
        self._initialize_velocities()
        self._velocities_initialized = True

    def add_particle(self, particle: Particle) -> None:
        """Register a particle with metadata-derived properties."""
        self._initialize_particle(particle)
        if self._velocities_initialized:
            self._assign_random_velocity(particle)
        else:
            particle.velocity_nm_fs = vector_zero()
        self.particles.append(particle)

    def update_element_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        self.element_metadata = metadata
        for particle in self.particles:
            self._initialize_particle(particle, preserve_charge=True)

    def _initialize_particle(self, particle: Particle, preserve_charge: bool = False) -> None:
        metadata = self.element_metadata.get(particle.element, {})
        rule = get_valence_rule(particle.element) or {}
        valence_meta = metadata.get("valence_electrons", rule.get("target_electrons"))
        if valence_meta is not None:
            try:
                valence_value = int(valence_meta)
            except (TypeError, ValueError):
                valence_value = particle.valence_electrons or particle.base_valence_electrons
            if not preserve_charge or particle.base_valence_electrons == 0:
                particle.valence_electrons = valence_value
            if particle.base_valence_electrons == 0:
                particle.base_valence_electrons = valence_value
        else:
            if particle.base_valence_electrons == 0:
                particle.base_valence_electrons = particle.valence_electrons

        if particle.base_valence_electrons == 0:
            # fallback assumption for unknown elements
            particle.base_valence_electrons = particle.valence_electrons or 0

        particle.valence_target = int(metadata.get("valence_target", rule.get("target_electrons", max(2, min(8, particle.base_valence_electrons + 4)))))
        particle.max_bond_slots = int(metadata.get("max_bonds", rule.get("max_bonds", max(1, (particle.valence_target - particle.base_valence_electrons) // 2 or 1))))
        if particle.bond_slots == 0 or not preserve_charge:
            particle.bond_slots = particle.max_bond_slots

        phase_meta = metadata.get("phase", rule.get("phase"))
        if isinstance(phase_meta, str):
            particle.phase = phase_meta.lower()
        elif not particle.phase:
            particle.phase = "solid"

    def _acceleration_nm_fs2(self, force_component: float, mass_amu: float) -> float:
        force_newton = force_component * FORCE_KJMOL_NM_TO_NEWTON
        mass_kg = mass_amu * AMU_TO_KG
        accel_m_per_s2 = force_newton / mass_kg
        return accel_m_per_s2 * ACCEL_M_S2_TO_NM_FS2

    def compute_forces(self) -> None:
        for p in self.particles:
            p.force_kjmol_nm = vector_zero()
        self._potential_energy = 0.0

        for i, a in enumerate(self.particles):
            for j in range(i + 1, len(self.particles)):
                b = self.particles[j]
                params = self.force_field.get(a.element, b.element)
                if params is None:
                    continue
                delta = vector_sub(a.position_nm, b.position_nm)
                delta = minimum_image(delta, self.settings.box_lengths_nm if self.settings.periodic else None)
                lj_force, lj_pot = lennard_jones_force(delta, params)
                coul_force, coul_pot = coulomb_force(delta, a.charge_e, b.charge_e, params)

                total_force = vector_add(lj_force, coul_force)

                a.force_kjmol_nm = vector_add(a.force_kjmol_nm, total_force)
                b.force_kjmol_nm = vector_sub(b.force_kjmol_nm, total_force)

                self._potential_energy += lj_pot + coul_pot

        self._apply_bond_forces()
        self._apply_angle_forces()
        self._apply_capture_forces()

    def integrate(self, steps: int = 1) -> None:
        dt_fs = self.settings.timestep_fs
        for _ in range(steps):
            self.compute_forces()
            self._apply_phase_behaviors()

            # Velocity-Verlet integration
            half_dt = 0.5 * dt_fs
            accelerations: List[Vector] = []
            for particle in self.particles:
                ax = self._acceleration_nm_fs2(particle.force_kjmol_nm[0], particle.mass_amu)
                ay = self._acceleration_nm_fs2(particle.force_kjmol_nm[1], particle.mass_amu)
                az = self._acceleration_nm_fs2(particle.force_kjmol_nm[2], particle.mass_amu)
                accelerations.append((ax, ay, az))
                vx = particle.velocity_nm_fs[0] + ax * half_dt
                vy = particle.velocity_nm_fs[1] + ay * half_dt
                vz = particle.velocity_nm_fs[2] + az * half_dt
                particle.velocity_nm_fs = (vx, vy, vz)

            for particle, accel in zip(self.particles, accelerations):
                px = particle.position_nm[0] + particle.velocity_nm_fs[0] * dt_fs
                py = particle.position_nm[1] + particle.velocity_nm_fs[1] * dt_fs
                pz = particle.position_nm[2] + particle.velocity_nm_fs[2] * dt_fs
                particle.position_nm = (px, py, pz)

                if self.settings.periodic and self.settings.box_lengths_nm:
                    bx, by, bz = self.settings.box_lengths_nm
                    particle.position_nm = (
                        px % bx if bx else px,
                        py % by if by else py,
                        pz % bz if bz else pz,
                    )

            # Recompute forces using updated positions, then finalize velocities
            self.compute_forces()
            for particle, accel in zip(self.particles, accelerations):
                ax_new = self._acceleration_nm_fs2(particle.force_kjmol_nm[0], particle.mass_amu)
                ay_new = self._acceleration_nm_fs2(particle.force_kjmol_nm[1], particle.mass_amu)
                az_new = self._acceleration_nm_fs2(particle.force_kjmol_nm[2], particle.mass_amu)
                vx = particle.velocity_nm_fs[0] + ax_new * half_dt
                vy = particle.velocity_nm_fs[1] + ay_new * half_dt
                vz = particle.velocity_nm_fs[2] + az_new * half_dt
                particle.velocity_nm_fs = (vx, vy, vz)

            self._apply_thermostat(dt_fs)
            self._remove_center_of_mass_velocity()
            self._update_bonds()
            self.current_step += 1
            self.time_fs += dt_fs

    def total_kinetic_energy(self) -> float:
        energy = 0.0
        for particle in self.particles:
            vx, vy, vz = particle.velocity_nm_fs
            speed_sq = vx * vx + vy * vy + vz * vz
            mass_kg = particle.mass_amu * AMU_TO_KG
            speed_m_s = math.sqrt(speed_sq) * (NM_TO_M / FS_TO_S)
            energy_j = 0.5 * mass_kg * speed_m_s * speed_m_s
            energy += energy_j / KJMOL_TO_J
        return energy

    def total_potential_energy(self) -> float:
        return self._potential_energy

    def total_energy(self) -> float:
        return self.total_kinetic_energy() + self.total_potential_energy()

    def snapshot(self) -> SimulationSnapshot:
        atom_states = [
            AtomState(
                id=particle.id,
                element=particle.element,
                position_nm=particle.position_nm,
                velocity_nm_fs=particle.velocity_nm_fs,
                valence_electrons=particle.valence_electrons,
                bond_slots=self._bond_capacity(particle),
                charge_e=particle.charge_e,
            )
            for particle in self.particles
        ]
        bonds = [
            BondState(
                bond.atom_i,
                bond.atom_j,
                bond.order,
                bond.bond_type,
                bond.shared_electrons,
                bond.charge_transfer,
                bond.rest_length_nm,
            )
            for bond in self.bonds
        ]
        return SimulationSnapshot(
            step_index=self.current_step,
            time_fs=self.time_fs,
            atom_states=atom_states,
            bonds=bonds,
        )

    def _update_bonds(self) -> None:
        """Distance + valence driven bonding model."""
        existing = {tuple(sorted((bond.atom_i, bond.atom_j))): bond for bond in self.bonds}
        updated_bonds: List[BondState] = []

        for i, atom_a in enumerate(self.particles):
            for j in range(i + 1, len(self.particles)):
                atom_b = self.particles[j]
                if self._bond_capacity(atom_a) <= 0 or self._bond_capacity(atom_b) <= 0:
                    continue
                threshold_nm = self._bond_threshold_nm(atom_a.element, atom_b.element)
                if threshold_nm is None:
                    continue
                delta = vector_sub(atom_a.position_nm, atom_b.position_nm)
                delta = minimum_image(
                    delta, self.settings.box_lengths_nm if self.settings.periodic else None
                )
                distance = vector_length(delta)
                capture_radius = threshold_nm * BOND_CAPTURE_MULTIPLIER
                if distance <= capture_radius:
                    pair = tuple(sorted((atom_a.id, atom_b.id)))
                    if pair in existing:
                        updated_bonds.append(existing.pop(pair))
                    else:
                        bond = self._form_bond(atom_a, atom_b, threshold_nm)
                        if bond is not None:
                            if distance > threshold_nm:
                                bond.rest_length_nm = threshold_nm
                            updated_bonds.append(bond)

        # Break bonds that no longer meet distance criteria
        for bond in existing.values():
            atom_i = self._get_particle_by_id(bond.atom_i)
            atom_j = self._get_particle_by_id(bond.atom_j)
            if atom_i is None or atom_j is None:
                continue
            threshold_nm = self._bond_threshold_nm(atom_i.element, atom_j.element)
            if threshold_nm is None:
                self._break_bond(bond)
                continue
            capture_radius = threshold_nm * BOND_CAPTURE_MULTIPLIER
            distance = vector_length(self._vector_between(atom_i.position_nm, atom_j.position_nm))
            if distance > capture_radius:
                self._break_bond(bond)
            else:
                updated_bonds.append(bond)

        self.bonds = updated_bonds

    def _bond_threshold_nm(self, element_a: str, element_b: str) -> Optional[float]:
        radius_a_pm = DEFAULT_COVALENT_RADII_PM.get(element_a)
        radius_b_pm = DEFAULT_COVALENT_RADII_PM.get(element_b)
        if radius_a_pm is None or radius_b_pm is None:
            return None
        threshold_pm = radius_a_pm + radius_b_pm + BOND_TOLERANCE_PM
        return threshold_pm * PM_TO_NM

    def _covalent_rest_length(self, element_a: str, element_b: str, fallback_nm: float) -> float:
        params = self.force_field.get_bond(element_a, element_b)
        if params is not None:
            return params.r0_angstrom * ANGSTROM_TO_NM
        return fallback_nm * 0.9

    def _apply_phase_behaviors(self) -> None:
        if not self.particles:
            return
        count = len(self.particles)
        com = (
            sum(p.position_nm[0] for p in self.particles) / count,
            sum(p.position_nm[1] for p in self.particles) / count,
            sum(p.position_nm[2] for p in self.particles) / count,
        )

        for i, particle in enumerate(self.particles):
            phase = particle.phase
            offset = vector_sub(particle.position_nm, com)
            if phase == "gas":
                distance = vector_length(offset)
                if distance > 0:
                    scale = GAS_EXPANSION_FACTOR / max(distance, 0.1)
                    particle.velocity_nm_fs = vector_add(
                        particle.velocity_nm_fs,
                        vector_scale(offset, scale),
                    )
            elif phase == "liquid":
                for j, other in enumerate(self.particles):
                    if i == j:
                        continue
                    delta = vector_sub(particle.position_nm, other.position_nm)
                    distance = vector_length(delta)
                    if 0 < distance < LIQUID_PERSONAL_SPACE_NM:
                        repulsion = LIQUID_REPULSION_FACTOR / max(distance, 0.05)
                        particle.velocity_nm_fs = vector_add(
                            particle.velocity_nm_fs,
                            vector_scale(delta, repulsion),
                        )

    def _initialize_velocities(self) -> None:
        if not self.particles:
            return
        for particle in self.particles:
            self._assign_random_velocity(particle)
        self._remove_center_of_mass_velocity()

    def _assign_random_velocity(self, particle: Particle) -> None:
        sigma = self._thermal_velocity_sigma_nm_fs(particle.mass_amu)
        particle.velocity_nm_fs = (
            self.random.gauss(0.0, sigma),
            self.random.gauss(0.0, sigma),
            self.random.gauss(0.0, sigma),
        )

    def _remove_center_of_mass_velocity(self) -> None:
        total_mass = sum(p.mass_amu for p in self.particles)
        if total_mass == 0:
            return
        com_velocity = [0.0, 0.0, 0.0]
        for particle in self.particles:
            com_velocity[0] += particle.mass_amu * particle.velocity_nm_fs[0]
            com_velocity[1] += particle.mass_amu * particle.velocity_nm_fs[1]
            com_velocity[2] += particle.mass_amu * particle.velocity_nm_fs[2]
        com_velocity = [component / total_mass for component in com_velocity]
        for particle in self.particles:
            particle.velocity_nm_fs = (
                particle.velocity_nm_fs[0] - com_velocity[0],
                particle.velocity_nm_fs[1] - com_velocity[1],
                particle.velocity_nm_fs[2] - com_velocity[2],
            )

    def _thermal_velocity_sigma_nm_fs(self, mass_amu: float) -> float:
        mass_kg = mass_amu * AMU_TO_KG
        if mass_kg == 0.0:
            return 0.0
        sigma_m_s = math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * self.settings.target_temperature_k / mass_kg)
        return sigma_m_s * M_S_TO_NM_FS

    def _apply_thermostat(self, dt_fs: float) -> None:
        tau = self.settings.thermostat_tau_fs
        if tau is None or tau <= 0 or not self.particles:
            return
        kinetic_energy = self.total_kinetic_energy()
        if kinetic_energy <= 0.0:
            return
        dof = 3.0 * len(self.particles)
        current_temperature = (2.0 * kinetic_energy) / (dof * BOLTZMANN_CONSTANT_KJ_MOL_K)
        if current_temperature <= 0.0:
            return
        target = self.settings.target_temperature_k
        scale_factor = 1.0 + (dt_fs / tau) * (target / current_temperature - 1.0)
        if scale_factor <= 0.0:
            return
        scale = math.sqrt(scale_factor)
        for particle in self.particles:
            particle.velocity_nm_fs = vector_scale(particle.velocity_nm_fs, scale)

    def _apply_bond_forces(self) -> None:
        for bond in self.bonds:
            particle_i = self._get_particle_by_id(bond.atom_i)
            particle_j = self._get_particle_by_id(bond.atom_j)
            if particle_i is None or particle_j is None:
                continue
            delta = self._vector_between(particle_i.position_nm, particle_j.position_nm)
            distance = vector_length(delta)
            if distance == 0.0:
                continue
            rest_length = bond.rest_length_nm or distance
            params = self.force_field.get_bond(particle_i.element, particle_j.element)
            k = params.k_kj_mol_nm2 if params else DEFAULT_BOND_K_KJMOL_NM2
            dr = distance - rest_length
            force_magnitude = -k * dr
            unit_vector = vector_scale(delta, 1.0 / distance)
            force_vector = vector_scale(unit_vector, force_magnitude)
            particle_i.force_kjmol_nm = vector_add(particle_i.force_kjmol_nm, force_vector)
            particle_j.force_kjmol_nm = vector_sub(particle_j.force_kjmol_nm, force_vector)
            self._potential_energy += 0.5 * k * dr * dr

    def _apply_angle_forces(self) -> None:
        adjacency: Dict[int, List[int]] = {}
        for bond in self.bonds:
            if bond.bond_type != "covalent":
                continue
            adjacency.setdefault(bond.atom_i, []).append(bond.atom_j)
            adjacency.setdefault(bond.atom_j, []).append(bond.atom_i)

        for central_id, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            central_particle = self._get_particle_by_id(central_id)
            if central_particle is None:
                continue
            for idx in range(len(neighbors)):
                for jdx in range(idx + 1, len(neighbors)):
                    atom_i_id = neighbors[idx]
                    atom_k_id = neighbors[jdx]
                    particle_i = self._get_particle_by_id(atom_i_id)
                    particle_k = self._get_particle_by_id(atom_k_id)
                    if particle_i is None or particle_k is None:
                        continue
                    params = self._angle_parameters(particle_i.element, central_particle.element, particle_k.element)
                    if params is None:
                        continue
                    self._apply_single_angle_force(particle_i, central_particle, particle_k, params)

    def _angle_parameters(self, elem_i: str, elem_j: str, elem_k: str) -> Optional[AngleParameters]:
        params = self.force_field.get_angle((elem_i, elem_j, elem_k))
        if params is not None:
            return params
        return self.force_field.get_angle((elem_k, elem_j, elem_i))

    def _apply_single_angle_force(
        self,
        atom_i: Particle,
        atom_j: Particle,
        atom_k: Particle,
        params: AngleParameters,
    ) -> None:
        vec_ji = self._vector_between(atom_i.position_nm, atom_j.position_nm)
        vec_jk = self._vector_between(atom_k.position_nm, atom_j.position_nm)
        r_ji = vector_length(vec_ji)
        r_jk = vector_length(vec_jk)
        if r_ji == 0.0 or r_jk == 0.0:
            return
        u = vector_scale(vec_ji, 1.0 / r_ji)
        v = vector_scale(vec_jk, 1.0 / r_jk)
        cos_theta = max(-1.0, min(1.0, vector_dot(u, v)))
        theta = math.acos(cos_theta)
        theta0 = math.radians(params.theta0_degrees)
        sin_theta = max(1e-4, math.sqrt(1.0 - cos_theta * cos_theta))
        deviation = theta - theta0
        if deviation == 0.0:
            return
        prefactor = params.k_kj_mol_rad2 * deviation / sin_theta

        term_i = vector_sub(vector_scale(u, cos_theta), v)
        term_k = vector_sub(vector_scale(v, cos_theta), u)
        force_i = vector_scale(term_i, prefactor / r_ji)
        force_k = vector_scale(term_k, prefactor / r_jk)
        force_j = vector_scale(vector_add(force_i, force_k), -1.0)

        atom_i.force_kjmol_nm = vector_sub(atom_i.force_kjmol_nm, force_i)
        atom_k.force_kjmol_nm = vector_sub(atom_k.force_kjmol_nm, force_k)
        atom_j.force_kjmol_nm = vector_sub(atom_j.force_kjmol_nm, force_j)

        self._potential_energy += 0.5 * params.k_kj_mol_rad2 * deviation * deviation

    def _apply_capture_forces(self) -> None:
        for i, atom_a in enumerate(self.particles):
            capacity_a = self._bond_capacity(atom_a)
            if capacity_a <= 0:
                continue
            for j in range(i + 1, len(self.particles)):
                atom_b = self.particles[j]
                if self._bond_capacity(atom_b) <= 0:
                    continue
                if self._bond_exists(atom_a.id, atom_b.id):
                    continue
                threshold_nm = self._bond_threshold_nm(atom_a.element, atom_b.element)
                if threshold_nm is None:
                    continue
                capture_radius = threshold_nm * BOND_CAPTURE_MULTIPLIER
                delta = self._vector_between(atom_a.position_nm, atom_b.position_nm)
                distance = vector_length(delta)
                if distance == 0.0 or distance > capture_radius:
                    continue
                force_strength = CAPTURE_FORCE_K * (capture_radius - distance) / capture_radius
                unit = vector_scale(delta, 1.0 / distance)
                force = vector_scale(unit, force_strength)
                atom_a.force_kjmol_nm = vector_add(atom_a.force_kjmol_nm, force)
                atom_b.force_kjmol_nm = vector_sub(atom_b.force_kjmol_nm, force)

    def _vector_between(self, pos_a: Vector, pos_b: Vector) -> Vector:
        delta = vector_sub(pos_a, pos_b)
        if self.settings.periodic and self.settings.box_lengths_nm:
            delta = minimum_image(delta, self.settings.box_lengths_nm)
        return delta

    def _bond_exists(self, atom_i: int, atom_j: int) -> bool:
        pair = tuple(sorted((atom_i, atom_j)))
        for bond in self.bonds:
            if tuple(sorted((bond.atom_i, bond.atom_j))) == pair:
                return True
        return False

    def _form_bond(self, atom_a: Particle, atom_b: Particle, threshold_nm: float) -> Optional[BondState]:
        pair = tuple(sorted((atom_a.id, atom_b.id)))
        bond_type = self._determine_bond_type(atom_a.element, atom_b.element)
        if bond_type == "ionic":
            donor, acceptor = self._select_ionic_roles(atom_a, atom_b)
            if donor is None or acceptor is None:
                bond_type = "covalent"
            else:
                if donor.valence_electrons <= 0:
                    return None
                donor.valence_electrons = max(0, donor.valence_electrons - 1)
                donor.bond_slots = max(0, donor.bond_slots - 1)
                donor.charge_e += 1.0
                acceptor.valence_electrons += 1
                acceptor.charge_e -= 1.0
                acceptor.bond_slots = max(0, acceptor.bond_slots - 1)
                transfer_direction = 1 if donor.id == pair[0] else -1
                return BondState(
                    atom_i=pair[0],
                    atom_j=pair[1],
                    order=1.0,
                    bond_type="ionic",
                    shared_electrons=0,
                    charge_transfer=transfer_direction,
                    rest_length_nm=threshold_nm,
                )

        # Covalent sharing path
        available_a = min(atom_a.valence_electrons, self._bond_capacity(atom_a))
        available_b = min(atom_b.valence_electrons, self._bond_capacity(atom_b))
        shared_pairs = min(available_a, available_b)
        if shared_pairs <= 0:
            return None
        shared_pairs = max(1, min(shared_pairs, 3))
        atom_a.valence_electrons = max(0, atom_a.valence_electrons - shared_pairs)
        atom_b.valence_electrons = max(0, atom_b.valence_electrons - shared_pairs)
        atom_a.bond_slots = max(0, atom_a.bond_slots - shared_pairs)
        atom_b.bond_slots = max(0, atom_b.bond_slots - shared_pairs)
        shared_electrons = shared_pairs * 2
        rest_length = self._covalent_rest_length(atom_a.element, atom_b.element, threshold_nm)
        return BondState(
            atom_i=pair[0],
            atom_j=pair[1],
            order=float(shared_pairs),
            bond_type="covalent",
            shared_electrons=shared_electrons,
            charge_transfer=0,
            rest_length_nm=rest_length,
        )

    def _break_bond(self, bond: BondState) -> None:
        atom_i = self._get_particle_by_id(bond.atom_i)
        atom_j = self._get_particle_by_id(bond.atom_j)
        if atom_i is None or atom_j is None:
            return

        if bond.bond_type == "covalent" and bond.shared_electrons:
            to_restore = bond.shared_electrons // 2
            if to_restore > 0:
                atom_i.valence_electrons = min(
                    atom_i.base_valence_electrons, atom_i.valence_electrons + to_restore
                )
                atom_j.valence_electrons = min(
                    atom_j.base_valence_electrons, atom_j.valence_electrons + to_restore
                )
                atom_i.bond_slots = min(atom_i.max_bond_slots, atom_i.bond_slots + int(bond.order))
                atom_j.bond_slots = min(atom_j.max_bond_slots, atom_j.bond_slots + int(bond.order))
        elif bond.bond_type == "ionic" and bond.charge_transfer != 0:
            transfer = abs(bond.charge_transfer)
            if bond.charge_transfer > 0:
                donor, acceptor = atom_i, atom_j
            else:
                donor, acceptor = atom_j, atom_i
            acceptor.valence_electrons = max(0, acceptor.valence_electrons - transfer)
            donor.valence_electrons = min(
                donor.base_valence_electrons, donor.valence_electrons + transfer
            )
            donor.charge_e -= transfer
            acceptor.charge_e += transfer
            donor.bond_slots = min(donor.max_bond_slots, donor.bond_slots + transfer)
            acceptor.bond_slots = min(acceptor.max_bond_slots, acceptor.bond_slots + transfer)

    def _bond_capacity(self, atom: Particle) -> int:
        base_capacity = atom.bond_slots
        if base_capacity <= 0:
            base_capacity = 0
        used = sum(
            int(bond.order)
            for bond in self.bonds
            if bond.bond_type == "covalent"
            and (bond.atom_i == atom.id or bond.atom_j == atom.id)
        )
        return max(0, base_capacity - used)

    def _determine_bond_type(self, element_a: str, element_b: str) -> str:
        a_metal = self._is_metal(element_a)
        b_metal = self._is_metal(element_b)
        a_non = self._is_non_metal(element_a)
        b_non = self._is_non_metal(element_b)

        if a_non and b_non:
            return "covalent"
        if (a_metal and b_non) or (b_metal and a_non):
            return "ionic"
        return "metallic" if a_metal and b_metal else "covalent"

    def _select_ionic_roles(self, atom_a: Particle, atom_b: Particle) -> Tuple[Optional[Particle], Optional[Particle]]:
        a_metal = self._is_metal(atom_a.element)
        b_metal = self._is_metal(atom_b.element)
        a_non = self._is_non_metal(atom_a.element)
        b_non = self._is_non_metal(atom_b.element)
        if a_metal and b_non:
            return atom_a, atom_b
        if b_metal and a_non:
            return atom_b, atom_a
        return None, None

    def _is_metal(self, element: str) -> bool:
        if element in METAL_ELEMENTS:
            return True
        metadata = self.element_metadata.get(element, {})
        category = metadata.get("category")
        if isinstance(category, str) and "metal" in category.lower():
            return True
        return False

    def _is_non_metal(self, element: str) -> bool:
        if element in NON_METAL_ELEMENTS:
            return True
        metadata = self.element_metadata.get(element, {})
        category = metadata.get("category")
        if isinstance(category, str) and "nonmetal" in category.lower():
            return True
        return False

    def _get_particle_by_id(self, particle_id: int) -> Optional[Particle]:
        for particle in self.particles:
            if particle.id == particle_id:
                return particle
        return None
