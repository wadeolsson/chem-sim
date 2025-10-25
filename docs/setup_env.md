# Development Environment Setup

ChemSim currently targets a Python 3.11+ environment. Install the following dependencies for the prototype build:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Required Packages (v0.x)
- `pygame` – 2D rendering and event handling for the sandbox UI.
- `PyYAML` – configuration/schema parsing for scenario files.
- `pytest` – unit testing harness (schema checks, data validation).

Optional (future):
- `numpy` – accelerated vector math once performance profiling begins.
- `mypy`/`ruff` – static analysis and linting once codebase grows.

## Notes
- The pygame scaffold lives in `src/ui/app.py`; running `python -m src.ui.app` launches the current mock UI.
- Install system-level SDL libraries if pygame build complains (varies by OS).
- Keep the virtual environment activated while running ingestion scripts or tests.

