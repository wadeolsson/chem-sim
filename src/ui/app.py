"""
pygame application scaffold for ChemSim.

This module outlines the structure of the initial visualization loop.
Implementation remains TODO until the simulation core is ready to feed
renderable state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any


try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore

from sim import (
    AngleParameters,
    BondStretchParameters,
    ForceField,
    ForceFieldParameters,
    Particle,
    Simulation,
    SimulationSettings,
    SimulationSnapshot,
)  # type: ignore
from .viewport import SandboxViewport
from .panels import PeriodicTablePanel, ControlDockPanel, InspectorPanel
from .controllers import SimulationController, UIController


DEFAULT_ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "Pb": 207.2,
}

@dataclass
class AppConfig:
    width: int = 1280
    height: int = 720
    title: str = "ChemSim Sandbox"
    target_fps: int = 60
    enable_vsync: bool = False


@dataclass
class AppState:
    running: bool = True
    paused: bool = False
    drag_symbol: Optional[str] = None
    clock: Optional["pygame.time.Clock"] = field(default=None, repr=False)


class ChemSimApp:
    """
    High-level pygame application manager.

    Responsibilities (to be implemented):
      - Initialize pygame, surfaces, and panels.
      - Route events to controllers (sandbox, periodic table, control dock).
      - Invoke simulation steps when not paused.
      - Coordinate rendering using modular renderer classes.
    """

    def __init__(self, config: AppConfig | None = None, simulation: Optional[Simulation] = None):
        if pygame is None:
            raise RuntimeError("pygame is not installed. Install it to run the UI prototype.")
        self.config = config or AppConfig()
        self.state = AppState()
        self.screen: Optional["pygame.Surface"] = None
        self.simulation = simulation or self._create_default_simulation()
        self.sim_controller = SimulationController(self.simulation)
        self.viewport: Optional[SandboxViewport] = None
        self.periodic_panel: Optional[PeriodicTablePanel] = None
        self.control_panel: Optional[ControlDockPanel] = None
        self.inspector_panel: Optional[InspectorPanel] = None
        self.ui_controller: Optional[UIController] = None
        self._latest_snapshot: Optional[SimulationSnapshot] = None
        self._next_particle_id = (
            max((particle.id for particle in self.simulation.particles), default=0) + 1
        )
        self.element_metadata: Dict[str, Dict[str, Any]] = {}

    def setup(self) -> None:
        """Initialize pygame context and create root surfaces."""
        pygame.init()
        flags = pygame.SCALED if self.config.enable_vsync else 0
        self.screen = pygame.display.set_mode((self.config.width, self.config.height), flags)
        pygame.display.set_caption(self.config.title)
        self.state.clock = pygame.time.Clock()

        font = pygame.font.SysFont("Helvetica", 18)
        sidebar_width = 320
        inspector_height = 160

        viewport_rect = pygame.Rect(
            0, 0, self.config.width - sidebar_width, self.config.height - inspector_height
        )
        periodic_rect = pygame.Rect(
            self.config.width - sidebar_width, 0, sidebar_width, self.config.height - inspector_height
        )
        control_rect = pygame.Rect(
            0, self.config.height - inspector_height, self.config.width - sidebar_width, inspector_height
        )
        inspector_rect = pygame.Rect(
            self.config.width - sidebar_width,
            self.config.height - inspector_height,
            sidebar_width,
            inspector_height,
        )

        self.viewport = SandboxViewport(viewport_rect)
        symbols, metadata = self._load_periodic_table_metadata()
        self.element_metadata = metadata
        self.simulation.update_element_metadata(self.element_metadata)
        self.periodic_panel = PeriodicTablePanel(
            rect=periodic_rect,
            font=font,
            on_element_selected=self._on_element_selected,
            element_symbols=symbols,
            columns=4,
            tile_size=(68, 52),
            tile_margin=6,
        )
        self.control_panel = ControlDockPanel(
            rect=control_rect,
            font=font,
            on_toggle_run=self._toggle_run,
            on_step=self._step_once,
            on_reset=self._reset_simulation,
            on_speed_change=self._on_speed_change,
        )
        self.inspector_panel = InspectorPanel(rect=inspector_rect, font=font)
        self._rebuild_ui_controller()
        self._latest_snapshot = self.sim_controller.snapshot()

    def handle_event(self, event: "pygame.event.Event") -> None:
        """Dispatch a single pygame event."""
        if event.type == pygame.QUIT:
            self.state.running = False
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.sim_controller.toggle_running()
            elif event.key == pygame.K_PERIOD:
                self.sim_controller.step(1)

        if self.ui_controller:
            self.ui_controller.handle_event(event)

    def update(self, dt_seconds: float) -> None:
        """Advance simulation and UI state."""
        if self.ui_controller:
            self._latest_snapshot = self.ui_controller.update(dt_seconds)

    def render(self) -> None:
        """Render the current frame."""
        if self.screen is None or self._latest_snapshot is None or self.ui_controller is None:
            return
        self.screen.fill((10, 10, 30))
        self.ui_controller.render(self.screen, self._latest_snapshot)
        pygame.display.flip()

    def run(self) -> None:
        """Main loop entry point."""
        if self.screen is None or self.state.clock is None:
            self.setup()

        assert self.state.clock is not None
        while self.state.running:
            dt_ms = self.state.clock.tick(self.config.target_fps)
            dt_seconds = dt_ms / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt_seconds)
            self.render()

        pygame.quit()

    def spawn_atom(self, symbol: str, position_nm: tuple[float, float]) -> None:
        """Instantiate a new particle when the user drops an element."""
        mass = DEFAULT_ELEMENT_MASSES.get(symbol, 12.0)
        particle = Particle(
            id=self._next_particle_id,
            element=symbol,
            mass_amu=mass,
            charge_e=0.0,
            position_nm=(position_nm[0], position_nm[1], 0.0),
            velocity_nm_fs=(0.0, 0.0, 0.0),
        )
        self.simulation.add_particle(particle)
        self._next_particle_id += 1
        if self.ui_controller:
            self.ui_controller.select_atom_by_id(particle.id)

    def _toggle_run(self) -> None:
        self.sim_controller.toggle_running()

    def _step_once(self) -> None:
        self.sim_controller.step(1)

    def _reset_simulation(self) -> None:
        self._set_simulation(self._create_default_simulation())

    def _set_simulation(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.sim_controller = SimulationController(self.simulation)
        self._next_particle_id = (
            max((particle.id for particle in self.simulation.particles), default=0) + 1
        )
        if self.viewport:
            self.viewport.set_selected_atom(None)
        self._rebuild_ui_controller()
        self._latest_snapshot = self.sim_controller.snapshot()

    def _rebuild_ui_controller(self) -> None:
        if not all([self.viewport, self.periodic_panel, self.control_panel, self.inspector_panel]):
            return
        self.ui_controller = UIController(
            simulation_controller=self.sim_controller,
            viewport=self.viewport,
            periodic_panel=self.periodic_panel,
            control_panel=self.control_panel,
            inspector_panel=self.inspector_panel,
            spawn_atom_callback=self.spawn_atom,
            element_metadata=self.element_metadata,
        )
        if self.control_panel:
            self.control_panel.is_running = self.sim_controller.is_running
            self.control_panel.speed_multiplier = self.sim_controller.speed_multiplier

    def _load_periodic_table_metadata(self) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load element symbols and metadata from the periodic table dataset."""
        root = Path(__file__).resolve().parents[2]
        data_path = root / "data" / "periodic_table.json"
        if not data_path.exists():
            symbols = ["H", "C", "N", "O"]
            metadata = {symbol: {} for symbol in symbols}
            return symbols, metadata
        with data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        elements = data.get("elements", [])
        metadata: Dict[str, Dict[str, Any]] = {}
        symbols: List[str] = []
        for entry in elements:
            symbol = entry.get("symbol", "?")
            symbols.append(symbol)
            metadata[symbol] = entry
        return symbols, metadata

    def _on_element_selected(self, symbol: str) -> None:
        # Placeholder hook; drag-and-drop handled by UIController but we retain
        # callback for potential future use (tooltips, previews).
        pass

    def _on_speed_change(self, multiplier: float) -> None:
        self.sim_controller.speed_multiplier = multiplier

    def _create_default_simulation(self) -> Simulation:
        """Construct an empty simulation shell to drive the UI."""
        settings = SimulationSettings(
            timestep_fs=0.5,
            periodic=False,
            target_temperature_k=350.0,
            thermostat_tau_fs=50.0,
        )
        force_field = self._build_default_force_field()
        return Simulation([], force_field, settings)

    def _build_default_force_field(self) -> ForceField:
        params = {
            ("H", "H"): ForceFieldParameters(epsilon_kj_mol=0.1, sigma_angstrom=2.5, cutoff_angstrom=12.0),
            ("H", "O"): ForceFieldParameters(epsilon_kj_mol=0.16, sigma_angstrom=3.0, cutoff_angstrom=12.0),
            ("O", "O"): ForceFieldParameters(epsilon_kj_mol=0.21, sigma_angstrom=3.3, cutoff_angstrom=12.0),
            ("H", "C"): ForceFieldParameters(epsilon_kj_mol=0.12, sigma_angstrom=3.2, cutoff_angstrom=12.0),
            ("C", "C"): ForceFieldParameters(epsilon_kj_mol=0.22, sigma_angstrom=3.4, cutoff_angstrom=12.0),
        }
        force_field = ForceField(params)
        force_field.register_bond(("H", "H"), BondStretchParameters(k_kj_mol_nm2=320.0, r0_angstrom=0.74))
        force_field.register_bond(("H", "O"), BondStretchParameters(k_kj_mol_nm2=450.0, r0_angstrom=0.97))
        force_field.register_bond(("O", "O"), BondStretchParameters(k_kj_mol_nm2=400.0, r0_angstrom=1.21))
        force_field.register_bond(("C", "H"), BondStretchParameters(k_kj_mol_nm2=340.0, r0_angstrom=1.09))
        force_field.register_bond(("C", "C"), BondStretchParameters(k_kj_mol_nm2=600.0, r0_angstrom=1.54))
        force_field.register_angle(("H", "O", "H"), AngleParameters(k_kj_mol_rad2=60.0, theta0_degrees=104.5))
        force_field.register_angle(("H", "C", "H"), AngleParameters(k_kj_mol_rad2=50.0, theta0_degrees=109.5))
        return force_field


def main() -> None:
    app = ChemSimApp()
    app.run()


if __name__ == "__main__":
    main()
