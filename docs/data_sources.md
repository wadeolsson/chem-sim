# Data Source Planning

## Atomic & Molecular Properties
- **Primary candidates**: NIST Atomic Spectra Database, NIST Chemistry WebBook, CRC Handbook of Chemistry and Physics.
- **Backup/augmentation**: PubChem, OpenQM datasets, Materials Project (for solid-state references).
- **Collection tasks**
  1. Compile atomic numbers, masses, electronegativities, ionization energies, electron affinities, covalent/van der Waals radii.
  2. Store uncertainties and temperature/pressure conditions when available.
  3. Track licensing/usage terms and cite references explicitly in JSON metadata.

## Force-Field Parameters
- **Classical (Tier 1)**: OPLS-AA, AMBER, CHARMM, UFF.
- **Reactive**: ReaxFF parameter sets for hydrocarbons, metal oxides, and generic organics.
- **Action items**
  - Evaluate compatibility with ChemSim unit system (SI) and convert as needed.
  - Maintain provenance notes (original publication, DOI) within `data/potentials`.
  - Flag parameters requiring redistribution permissions prior to bundling.

## Orbital & Quantum Data
- **Sources**: Slater-type orbital databases, STO-3G/6-31G basis sets, DFT-derived orbital coefficients from literature.
- **Uses**: Populate orbital overlap approximations, hybridization preferences, and electron density templates.
- **To-do**
  - Define minimal dataset required for Tier 2 bonding approximations.
  - Investigate open datasets (e.g., QM9) for machine-learned orbital properties.

## Spectroscopy References
- **IR/Raman**: HITRAN, NIST Vibrational Spectroscopy Data, literature compilations for common molecules.
- **UV-Vis**: NIST, PhotochemCAD, published spectra for conjugated systems.
- **Checklist**
  - Store frequencies/energies with units (cm⁻¹, nm, eV) and conversion factors.
  - Capture intensities/oscillator strengths and experimental conditions (solvent, temperature).
  - Provide cross-links to validation scenarios in `Tests_core_checks`.

## Data Integrity Workflow
1. Collect raw data into staging spreadsheets or scripts.
2. Normalize units and structure according to schemas defined in `mission.txt`.
3. Automate validation (schema checks, unit tests) before data promotion to `data/`.
4. Record provenance, licenses, and update history in `docs/data_sources.md`.
5. Implement automation in `scripts/ingest_atomic_data.py` (currently stubbed).

## Element Coverage Strategy
- **Initial scope**: populate high-confidence data for elements H through Pb (Z ≤ 82) to cover common organic, inorganic, and basic materials chemistry.
- **Beyond Pb**: retain placeholder entries with minimal metadata so the periodic table UI can display structure without values; mark these clearly as "data pending".
- **Prioritization**: focus first on H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, transition metals (Sc–Zn), and Pb to support early sandbox experiments.
- Update `data/periodic_table.json` iteratively; ensure tests flag missing critical fields for the covered subset while tolerating blanks for deferred elements.
