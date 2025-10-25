"""
Controller scaffolding connecting pygame UI and simulation layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING, Dict, Any

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore

from .viewport import SandboxViewport
from .panels import PeriodicTablePanel, ControlDockPanel, InspectorPanel

if TYPE_CHECKING:  # pragma: no cover
    from sim import Simulation, SimulationSnapshot


@dataclass
class SimulationController:
    simulation: "Simulation"
    is_running: bool = False
    speed_multiplier: float = 1.0
    _step_accumulator: float = 0.0

    def toggle_running(self) -> None:
        self.is_running = not self.is_running

    def step(self, steps: int = 1) -> None:
        self.simulation.integrate(steps=steps)

    def reset(self) -> None:
        # Future: reload from initial config.
        self.is_running = False
        self._step_accumulator = 0.0

    def update(self, dt_seconds: float) -> None:
        if not self.is_running:
            return
        self._step_accumulator += self.speed_multiplier
        steps = int(self._step_accumulator)
        if steps >= 1:
            self.simulation.integrate(steps=steps)
            self._step_accumulator -= steps

    def snapshot(self) -> "SimulationSnapshot":
        return self.simulation.snapshot()


class UIController:
    """
    Routes pygame events to panels and handles drag/drop atom placement.
    """

    def __init__(
        self,
        simulation_controller: SimulationController,
        viewport: SandboxViewport,
        periodic_panel: PeriodicTablePanel,
        control_panel: ControlDockPanel,
        inspector_panel: InspectorPanel,
        spawn_atom_callback: Callable[[str, tuple[float, float]], None],
        element_metadata: Dict[str, Dict[str, Any]],
    ):
        if pygame is None:
            raise RuntimeError("pygame must be installed to use UIController.")
        self.simulation_controller = simulation_controller
        self.viewport = viewport
        self.periodic_panel = periodic_panel
        self.control_panel = control_panel
        self.inspector_panel = inspector_panel
        self.spawn_atom = spawn_atom_callback
        self.dragging_symbol: Optional[str] = None
        self.selected_atom_id: Optional[int] = None
        self._latest_snapshot: Optional["SimulationSnapshot"] = None
        self.element_metadata = element_metadata

    def handle_event(self, event: "pygame.event.Event") -> None:
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if hasattr(event, "pos"):
                pos = event.pos
                if self.periodic_panel.rect.collidepoint(pos):
                    self.periodic_panel.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        symbol = self.periodic_panel.drag_symbol
                        if symbol:
                            self.dragging_symbol = symbol
                else:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.viewport.rect.collidepoint(pos):
                        if not self.dragging_symbol:
                            self._select_atom_at(pos)
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging_symbol:
                        if self.viewport.rect.collidepoint(pos):
                            world_pos = self.viewport.screen_to_world(pos)
                            self.spawn_atom(self.dragging_symbol, world_pos)
                    if event.type == pygame.MOUSEBUTTONUP:
                        self.dragging_symbol = None

        self.control_panel.handle_event(event)

    def update(self, dt_seconds: float) -> "SimulationSnapshot":
        self.simulation_controller.update(dt_seconds)
        snapshot = self.simulation_controller.snapshot()
        self._latest_snapshot = snapshot
        self.control_panel.is_running = self.simulation_controller.is_running
        self.control_panel.speed_multiplier = self.simulation_controller.speed_multiplier
        self.viewport.set_selected_atom(self.selected_atom_id)
        self._update_inspector(snapshot)
        return snapshot

    def render(self, screen: "pygame.Surface", snapshot: "SimulationSnapshot") -> None:
        viewport_surface = screen.subsurface(self.viewport.rect)
        self.viewport.render(viewport_surface, snapshot)
        self.periodic_panel.render(screen)
        self.control_panel.render(screen)
        self.inspector_panel.render(screen)

        if self.dragging_symbol:
            mx, my = pygame.mouse.get_pos()
            label = self.periodic_panel.font.render(self.dragging_symbol, True, (255, 255, 255))
            rect = label.get_rect(center=(mx, my))
            screen.blit(label, rect)

    def select_atom_by_id(self, atom_id: Optional[int]) -> None:
        self.selected_atom_id = atom_id
        self.viewport.set_selected_atom(atom_id)
        if self._latest_snapshot:
            self._update_inspector(self._latest_snapshot)

    def _select_atom_at(self, position_px: tuple[int, int]) -> None:
        if self._latest_snapshot is None:
            return
        max_pick_distance = 24.0
        chosen_id: Optional[int] = None
        best_distance = float("inf")
        for atom in self._latest_snapshot.atom_states:
            atom_px = self.viewport.world_to_screen(atom.position_nm)
            dx = atom_px[0] - position_px[0]
            dy = atom_px[1] - position_px[1]
            distance = math.hypot(dx, dy)
            if distance < max_pick_distance and distance < best_distance:
                best_distance = distance
                chosen_id = atom.id
        self.select_atom_by_id(chosen_id)

    def _update_inspector(self, snapshot: "SimulationSnapshot") -> None:
        if self.selected_atom_id is None:
            self.inspector_panel.update_selection(None, {})
            return
        atom = next((a for a in snapshot.atom_states if a.id == self.selected_atom_id), None)
        if atom is None:
            self.selected_atom_id = None
            self.inspector_panel.update_selection(None, {})
            self.viewport.set_selected_atom(None)
            return
        vx, vy, vz = atom.velocity_nm_fs
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        element_info = self.element_metadata.get(atom.element, {})
        valence = element_info.get("valence_electrons")
        configuration = element_info.get("electron_configuration")
        oxidation_states = element_info.get("oxidation_states") or []
        shell_excerpt = "-"
        if isinstance(configuration, str):
            parts = configuration.split()
            if parts:
                shell_excerpt = parts[-1]
        info: Dict[str, str] = {
            "Element": atom.element,
            "Position (nm)": f"{atom.position_nm[0]:.2f}, {atom.position_nm[1]:.2f}, {atom.position_nm[2]:.2f}",
            "Speed (nm/fs)": f"{speed:.3f}",
        }
        if valence is not None:
            info["Valence electrons"] = str(valence)
        info["Remaining bond slots"] = str(atom.bond_slots)
        info["Charge (e)"] = f"{atom.charge_e:+.2f}"
        if configuration:
            info["Electron config"] = configuration
            info["Frontier shell"] = shell_excerpt
        if oxidation_states:
            info["Oxidation states"] = ", ".join(oxidation_states)
        self.inspector_panel.update_selection(atom.id, info)
