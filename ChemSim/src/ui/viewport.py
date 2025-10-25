"""
Sandbox viewport rendering helpers for the pygame UI.

Rendering remains lightweight initially and will expand as simulation
data structures mature. The viewport consumes `SimulationSnapshot`
objects from `sim.py` (Tier-1 engine) and draws atoms/bonds using basic
pygame primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, TYPE_CHECKING

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from sim import SimulationSnapshot, AtomState, BondState

Color = Tuple[int, int, int]


DEFAULT_ELEMENT_COLORS: Dict[str, Color] = {
    "H": (255, 255, 255),
    "C": (80, 80, 80),
    "N": (48, 80, 248),
    "O": (255, 13, 13),
    "F": (144, 224, 80),
    "Na": (171, 92, 242),
    "Mg": (138, 255, 0),
    "Al": (191, 166, 166),
    "Si": (240, 200, 160),
    "P": (255, 128, 0),
    "S": (255, 255, 48),
    "Cl": (31, 240, 31),
    "K": (143, 64, 212),
    "Ca": (61, 255, 0),
    "Fe": (224, 102, 51),
    "Cu": (200, 128, 51),
    "Zn": (125, 128, 176),
    "Pb": (79, 79, 79),
}


@dataclass
class ViewportConfig:
    width: int
    height: int
    background_color: Color = (15, 15, 30)
    atom_radius_px: int = 12
    bond_color: Color = (180, 180, 180)
    selection_color: Color = (255, 255, 0)
    element_colors: Dict[str, Color] = field(default_factory=lambda: DEFAULT_ELEMENT_COLORS.copy())


class SandboxViewport:
    """
    Manages world-to-screen transforms and rendering calls for the sandbox.
    """

    def __init__(self, rect: "pygame.Rect", config: Optional[ViewportConfig] = None):
        if pygame is None:
            raise RuntimeError("pygame must be installed to use SandboxViewport.")
        self.rect = rect
        self.config = config or ViewportConfig(width=rect.width, height=rect.height)
        self.pan_offset_nm = (0.0, 0.0)
        self.zoom_nm_to_px = 30.0  # pixels per nm
        self.selected_atom_id: Optional[int] = None

    def world_to_screen(self, position_nm: Tuple[float, float, float]) -> Tuple[int, int]:
        x_nm, y_nm, _ = position_nm
        px = int((x_nm - self.pan_offset_nm[0]) * self.zoom_nm_to_px) + self.rect.centerx
        py = int((y_nm - self.pan_offset_nm[1]) * self.zoom_nm_to_px) + self.rect.centery
        return px, py

    def screen_to_world(self, position_px: Tuple[int, int]) -> Tuple[float, float]:
        x_px, y_px = position_px
        x_nm = (x_px - self.rect.centerx) / self.zoom_nm_to_px + self.pan_offset_nm[0]
        y_nm = (y_px - self.rect.centery) / self.zoom_nm_to_px + self.pan_offset_nm[1]
        return x_nm, y_nm

    def render(self, surface: "pygame.Surface", snapshot: "SimulationSnapshot") -> None:
        surface.fill(self.config.background_color)
        if pygame is None:
            return

        # Draw bonds first so atoms sit on top.
        for bond in snapshot.bonds:
            atom_i = next((a for a in snapshot.atom_states if a.id == bond.atom_i), None)
            atom_j = next((a for a in snapshot.atom_states if a.id == bond.atom_j), None)
            if not atom_i or not atom_j:
                continue
            start = self.world_to_screen(atom_i.position_nm)
            end = self.world_to_screen(atom_j.position_nm)
            self._draw_bond(surface, start, end, bond)

        # Draw atoms
        for atom in snapshot.atom_states:
            pos_px = self.world_to_screen(atom.position_nm)
            color = self.config.element_colors.get(atom.element, (200, 200, 200))
            pygame.draw.circle(surface, color, pos_px, self.config.atom_radius_px)
            if atom.id == self.selected_atom_id:
                pygame.draw.circle(
                    surface,
                    self.config.selection_color,
                    pos_px,
                    self.config.atom_radius_px + 3,
                    width=2,
                )

    def pan(self, delta_pixels: Tuple[int, int]) -> None:
        dx_nm = delta_pixels[0] / self.zoom_nm_to_px
        dy_nm = delta_pixels[1] / self.zoom_nm_to_px
        self.pan_offset_nm = (
            self.pan_offset_nm[0] - dx_nm,
            self.pan_offset_nm[1] - dy_nm,
        )

    def zoom(self, factor: float) -> None:
        self.zoom_nm_to_px = max(1.0, self.zoom_nm_to_px * factor)

    def set_selected_atom(self, atom_id: Optional[int]) -> None:
        self.selected_atom_id = atom_id

    def _draw_bond(self, surface: "pygame.Surface", start: Tuple[int, int], end: Tuple[int, int], bond) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        color = self._bond_color(bond.bond_type)
        order = max(1, int(round(bond.order)))
        if bond.bond_type == "ionic":
            self._draw_dashed_line(surface, start, end, color)
            return
        if order == 1:
            pygame.draw.line(surface, color, start, end, width=2)
            return

        ux = -dy / length
        uy = dx / length
        spacing = 4
        center_index = (order - 1) / 2.0
        for idx in range(order):
            offset = (idx - center_index) * spacing
            offset_start = (
                int(start[0] + ux * offset),
                int(start[1] + uy * offset),
            )
            offset_end = (
                int(end[0] + ux * offset),
                int(end[1] + uy * offset),
            )
            pygame.draw.line(surface, color, offset_start, offset_end, width=2)

    def _bond_color(self, bond_type: str) -> Color:
        if bond_type == "ionic":
            return (255, 180, 80)
        if bond_type == "metallic":
            return (150, 200, 255)
        return self.config.bond_color

    def _draw_dashed_line(
        self, surface: "pygame.Surface", start: Tuple[int, int], end: Tuple[int, int], color: Color
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        dash_length = 8
        gap = 6
        steps = max(1, int(length // (dash_length + gap)))
        ux = dx / length
        uy = dy / length
        cursor = 0.0
        for _ in range(steps):
            dash_start = (
                int(start[0] + ux * cursor),
                int(start[1] + uy * cursor),
            )
            dash_end = (
                int(start[0] + ux * (cursor + dash_length)),
                int(start[1] + uy * (cursor + dash_length)),
            )
            pygame.draw.line(surface, color, dash_start, dash_end, width=2)
            cursor += dash_length + gap
