"""Element metadata used for valence-driven bonding heuristics."""

from __future__ import annotations

VALENCE_RULES = {
    "H": {"target_electrons": 2, "max_bonds": 1, "electronegativity": 2.20, "phase": "gas"},
    "He": {"target_electrons": 2, "max_bonds": 0, "electronegativity": 0.0, "phase": "gas"},
    "Li": {"target_electrons": 2, "max_bonds": 1, "electronegativity": 0.98, "phase": "solid"},
    "Be": {"target_electrons": 4, "max_bonds": 2, "electronegativity": 1.57, "phase": "solid"},
    "B": {"target_electrons": 6, "max_bonds": 3, "electronegativity": 2.04, "phase": "solid"},
    "C": {"target_electrons": 8, "max_bonds": 4, "electronegativity": 2.55, "phase": "solid"},
    "N": {"target_electrons": 8, "max_bonds": 3, "electronegativity": 3.04, "phase": "gas"},
    "O": {"target_electrons": 8, "max_bonds": 2, "electronegativity": 3.44, "phase": "gas"},
    "F": {"target_electrons": 8, "max_bonds": 1, "electronegativity": 3.98, "phase": "gas"},
    "Ne": {"target_electrons": 8, "max_bonds": 0, "electronegativity": 0.0, "phase": "gas"},
    "Na": {"target_electrons": 2, "max_bonds": 1, "electronegativity": 0.93, "phase": "solid"},
    "Mg": {"target_electrons": 4, "max_bonds": 2, "electronegativity": 1.31, "phase": "solid"},
    "Al": {"target_electrons": 6, "max_bonds": 3, "electronegativity": 1.61, "phase": "solid"},
    "Si": {"target_electrons": 8, "max_bonds": 4, "electronegativity": 1.90, "phase": "solid"},
    "P": {"target_electrons": 8, "max_bonds": 3, "electronegativity": 2.19, "phase": "solid"},
    "S": {"target_electrons": 8, "max_bonds": 2, "electronegativity": 2.58, "phase": "solid"},
    "Cl": {"target_electrons": 8, "max_bonds": 1, "electronegativity": 3.16, "phase": "gas"},
    "Ar": {"target_electrons": 8, "max_bonds": 0, "electronegativity": 0.0, "phase": "gas"},
    "K": {"target_electrons": 2, "max_bonds": 1, "electronegativity": 0.82, "phase": "solid"},
    "Ca": {"target_electrons": 4, "max_bonds": 2, "electronegativity": 1.00, "phase": "solid"},
}


def get_valence_rule(symbol: str) -> dict | None:
    """Return metadata dict for symbol if known."""

    return VALENCE_RULES.get(symbol)
