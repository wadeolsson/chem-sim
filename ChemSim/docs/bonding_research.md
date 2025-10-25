# Bonding & Intermolecular Force Research Notes

## Reactive Bonding Approaches
- **Bond-Order Potentials (Tersoff, Brenner/REBO/AIREBO)**: bond strength evolves with local coordination; good for carbon allotropes and hydrocarbons, but requires fitted parameters per element pair.
- **ReaxFF / eReaxFF / COMB**: reactive force fields with charge equilibration (EEM/QEq) and bond order formalism; support bond formation/breaking, angles, torsions, van der Waals, and Coulomb terms. Implementations available in LAMMPS and GULP.
- **Machine-Learned Potentials (NequIP, ANI, SNAP, GAP)**: neural or kernel models that approximate ab initio PES; require large training sets but can replace traditional force fields in high-accuracy mode.
- **QM/MM Hybrids**: keep reactive center in DFT/semi-empirical region and embed in classical surroundings; useful for catalysis and enzymatic chemistry.

## Bonded Interaction Modeling
- Harmonic or Morse stretching for covalent bonds; Morse captures anharmonicity and bond dissociation.
- Angle terms (harmonic or cosine) enforce geometry (e.g., 104.5° in water, 109.5° sp³).
- Torsion/dihedral potentials (Ryckaert-Bellemans, OPLS cosine series) for rotational barriers.
- Improper torsions maintain planarity (e.g., aromatic systems, trigonal centers).
- Constraint algorithms (SHAKE/RATTLE/SETTLE) lock rigid bonds to allow longer timesteps.

## Intermolecular Nonbonded Forces
- Lennard-Jones (12-6) standard vdW; Buckingham (exp-6) and Mie (n-m) offer better short-range behavior.
- Polarizable models: Drude oscillators, inducible dipoles (AMOEBA) for accurate dielectric response.
- Electrostatics: PME/Ewald summation for periodic systems; multipole expansions or charge smearing for improved accuracy.
- Directional interactions: 
  - Hydrogen bonding via 12-10 or Morse potentials plus angle dependence.
  - π-π stacking and halogen bonding modeled with anisotropic LJ or off-center sites.
- Many-body dispersion (MBD) / Axilrod-Teller potential for large systems when using coarse-grained models.

## Long-Range Algorithms & Stability
- Verlet/neighbor lists with cell or Verlet buffer for O(N) updates.
- Nosé-Hoover chains, Langevin, Andersen thermostats; Berendsen/NH barostats for pressure control.
- Time-reversible integrators (velocity Verlet, RESPA) to handle multiple timescales; adaptively reduce timestep when bonds forming/breaking.

## Parameter Sources & Benchmarks
- GAFF/GAFF2, OPLS-AA/M, CHARMM, AMBER for organic/biomolecular systems.
- UFF, DREIDING, INTERFACE for generalized inorganic coverage.
- TIP3P/TIP4P/OPC water models; TraPPE and SAFT-based models for fluids.
- Datasets: NIST CCCBDB, ASE datasets, Materials Project, PubChemQC for fitting.

## Implementation Considerations
- Modular force-field registry supporting multiple bonded/angle/torsion functional forms per element pair or template.
- Charge equilibration (QEq, ACKS2) to update partial charges dynamically during reactions.
- Energy derivatives must be consistent with integration; accumulate per-term contributions for visualization.
- GPU paths: adapt to use CuPy/NumPy GPU or port to Vulkan/OpenCL for large systems.

## References
- J. R. K. Head-Gordon & M. J. T. "Reactive force fields" in Annual Review of Physical Chemistry (2016).
- Adri C. T. van Duin et al., "ReaxFF: A Reactive Force Field" J. Phys. Chem. A 2001.
- S. Plimpton et al., LAMMPS documentation (bond_style, angle_style, pair_style overviews).
- Leach, *Molecular Modelling: Principles and Applications* (2nd ed.).
- Frenkel & Smit, *Understanding Molecular Simulation* (2nd ed.).
