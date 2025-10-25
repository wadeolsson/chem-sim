# Codex Context Artifact

Use this note to bootstrap future Codex sessions quickly. It summarizes the repo’s purpose, status, and where to find critical information.

## Mission Snapshot
- *Goal:* Build a chemistry sandbox where bonding, reactions, and electron behavior emerge from layered physics (see `mission.txt`).
- *Current tier:* Phase‑1 classical MD core with Lennard‑Jones/Coulomb nonbonded forces, harmonic bond/angle terms, heuristic covalent/ionic bonding, thermostats, and pygame visualization.
- *Next priorities:* neighbor lists, richer force-field library, charge equilibration, scenario save/load, and UI analytics.

## Key Files
- `mission.txt` – Complete roadmap and philosophy.
- `project_log.txt` – Versioned change log; bump `v0.xx` for each milestone.
- `Tests_core_checks.txt` – Must-pass tests/regressions.
- `docs/` – Data sources, bonding research, UI plans, setup notes, first-run plan.
- `config/presets/` – YAML scenarios consumed by `src/config_loader.py`.
- `sim.py` – Core engine (forces, bonding, thermostat, phase heuristics).
- `src/ui/` – pygame app, controllers, viewport, panels.
- `README.md` – Public-facing summary & quickstart instructions.

## Working Agreements
- Record each substantive change in `project_log.txt` with a new `v0.xx` entry.
- Keep `Tests_core_checks.txt` synchronized with real pytest coverage.
- Prefer editing tools (`apply_patch`) over ad-hoc replacements; avoid touching `.venv/`.
- For new features: update docs/mission, add tests, and mention in README if user-facing.

## Quick Start Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m src.ui.app     # run sandbox UI
pytest -q                # run tests
```

## Scenario Workflow
- Add presets under `config/presets/*.yaml`.
- UI “Next Scenario” button (or `N` key) cycles through presets.
- `src/config_loader.py` is the single source of truth for config schema.

## Reminder
If Codex resumes work later, skim this file plus `project_log.txt:latest entry` to reorient. Always check for outstanding TODOs in docs and mission backlog before implementing new phases.
