# ChemSim

ChemSim is an ambitious chemistry sandbox that grows from classical molecular dynamics toward quantum‑inspired simulations. The engine models atoms as particles, layers in heuristic bonding with reactive foresight, and visualizes the evolving system through an interactive pygame UI. The long‑term mission (see `mission.txt`) is to let bonds, reactions, spectra, and electrons emerge from physically grounded approximations while keeping the experience approachable.

## Current Capabilities
- **Tier‑1 classical MD core** with velocity‑Verlet integration, Lennard‑Jones/Coulomb nonbonded forces, harmonic bond & angle terms, and Berendsen‑style thermostatting/phase heuristics.
- **Valence-driven bonding heuristics** that aggressively seek duet/octet completion, classify covalent vs ionic interactions, transfer/share electrons, and render bond order (single/double/triple) directly in the viewport.
- **Scenario presets & loader** (`config/presets/*.yaml`) describing atoms, force‑field parameters, and simulation settings (temperature, timestep, etc.) so the sandbox boots into meaningful states.
- **pygame sandbox UI** featuring drag‑and‑drop atom placement, inspector readouts (valence, orbitals, speed), control dock with play/pause/step/speed controls, and scenario cycling.
- **Data & docs** scaffolding: periodic table metadata, bonding research notes, roadmap, setup instructions, and critical test checklist.

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m src.ui.app
```

Controls:
- Drag an element tile from the periodic table into the sandbox to spawn an atom.
- Use the control dock to run/pause (`Space`), step (`.`), reset, or cycle scenarios (`Next Scenario` button or `N` key).
- Inspector panel updates when you click an atom inside the sandbox.

## Project Structure
```
ChemSim/
├── config/                # YAML templates & scenario presets
├── data/                  # Periodic data, potentials, spectral stubs
├── docs/                  # Mission, bonding research, UI layout, setup notes
├── scripts/               # Data ingestion scaffolding
├── src/
│   ├── config_loader.py   # YAML → Simulation builder
│   └── ui/                # pygame app, panels, controllers, viewport
├── tests/                 # pytest suites (config, data, bonding, etc.)
├── mission.txt            # Full roadmap & philosophy
├── project_log.txt        # Versioned change log
├── Tests_core_checks.txt  # Critical regression checklist
└── requirements-dev.txt
```

## Development Workflow
1. Edit or create YAML scenarios under `config/presets/` to define atoms, force fields, and settings. The UI loads these via `src/config_loader.py`.
2. Run `pytest` (see `Tests_core_checks.txt` for must‑pass suites) to validate schema, data integrity, and bonding interactions.
3. Iterate on engine logic in `sim.py` (forces, bonding, thermostat) and on UI behavior in `src/ui/`.
4. Document changes in `project_log.txt` (increment the `v0.xx` entries) and keep mission/docs updated as scope evolves.

## Roadmap Snapshot
- **Phase 1 (in progress):** Complete classical MD backbone (neighbor lists, richer force fields, scenario presets, save/load) and solidify visualization hooks.
- **Phase 2:** Introduce quantum‑inspired bonding (tight‑binding, bond-order potentials) and advanced reaction logic.
- **Phase 3:** Model dynamic electron densities, photon interactions, and EM fields.
- **Phase 4+:** Deliver spectroscopy outputs, thermodynamics dashboards, GPU acceleration, workflow tooling, and stretch goals (quantum tunneling, photochemistry, collaborative modes).

See `mission.txt` and `docs/bonding_research.md` for detailed context and references.
