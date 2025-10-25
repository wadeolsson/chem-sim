# First Runnable Version (v0.1) Checklist

This document summarizes the minimum work required to deliver the initial interactive ChemSim build: a classical MD sandbox with pygame visualization and limited element coverage (H–Pb data priority).

## 1. Data & Assets
- Populate `data/periodic_table.json` with validated entries for priority elements (H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Sc–Zn, Pb) using sources listed in `docs/data_sources.md`.
- Add Lennard-Jones/Coulomb parameters for the same element pairs in `data/potentials`.
- Ensure spectral references exist for a small demo set (H₂, H₂O, CO₂) even if visualization is basic.
- Mark post-lead elements explicitly as `status: "pending"` to keep UI placeholders clean.

## 2. Core Simulation Engine (Tier 1 scope)
- Implement particle data structures (positions, velocities, masses, charges) and classical force calculations (LJ + Coulomb).
- Integrate velocity Verlet (with optional thermostat toggle) and periodic boundary conditions.
- Build neighbor list or cell spatial partition for efficiency; confirm with unit tests/benchmarks.
- Expose basic energy accounting (kinetic, potential, temperature) for analytics and testing.

## 3. Reactive Bond Approximation (Minimal)
- Define distance-based bonding heuristic configurable via `config/*.yaml`.
- Update simulation loop to refresh bond network each step/batch and produce bond listings for visualization.
- Store bond metadata (length, strength estimate) for display and future energy tracking.

## 4. pygame Visualization & Interaction
- Render atoms, bonds, basic HUD overlays in the sandbox viewport at target framerates.
- Implement periodic table panel with search, filter, and drag-and-drop atom placement into the sandbox.
- Wire up control dock (play/pause, step, speed, temperature slider) and inspector panel for selected atoms/molecules.
- Provide camera controls (zoom, pan) plus timeline scrubber stub (even if playback is linear initially).

## 5. Configuration & Persistence
- Finalize `config/template.yaml` structure; design future save/load format for sandbox scenes (`.yaml` or JSON).
- Allow runtime environment tweaks (temperature, EM toggle) via config updates or UI controls.
- Implement basic output writers (trajectory snapshots, energy log) to `output/`.

## 6. Testing & Tooling
- Install and wire CI-friendly dependencies (`pytest`, `PyYAML`).
- Expand unit/integration tests:
  - schema validation for data/configs,
  - deterministic short MD runs checking energy drift,
  - drag-and-drop interaction smoke tests (headless with mocked events).
- Establish manual test checklist for UI regressions and data integrity.

## 7. Documentation & UX Polish
- Provide quickstart guide describing installation, running the sandbox, and data limitations (elements beyond Pb pending).
- Add inline tooltips/help overlays explaining controls and data provenance links.
- Capture feedback notes for stretch features (quantum layers, spectroscopy visuals) to inform post-v0.1 roadmap.

## 8. Stretch (Optional for v0.1 if time allows)
- Simple temperature-based color grading for atoms.
- Basic screenshot capture/export feature.
- Configurable random seed to replay sessions.

Dependencies and detailed task breakdowns should roll into the main backlog (`mission.txt` milestones). Update this checklist as tasks close or scope shifts.
