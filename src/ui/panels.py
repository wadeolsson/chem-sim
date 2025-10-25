"""
Panel scaffolding for pygame UI components (periodic table, control dock, inspector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore

Color = Tuple[int, int, int]


@dataclass
class PeriodicTablePanel:
    rect: "pygame.Rect"
    font: "pygame.font.Font"
    on_element_selected: Callable[[str], None]
    element_symbols: List[str]
    columns: int = 10
    tile_size: Tuple[int, int] = (48, 48)
    tile_margin: int = 8
    hover_symbol: Optional[str] = None
    drag_symbol: Optional[str] = None

    def render(self, surface: "pygame.Surface") -> None:
        pygame.draw.rect(surface, (20, 20, 40), self.rect)
        if pygame is None:
            return
        x0, y0 = self.rect.topleft
        for index, symbol in enumerate(self.element_symbols):
            row = index // self.columns
            col = index % self.columns
            tile_rect = pygame.Rect(
                x0 + col * (self.tile_size[0] + self.tile_margin),
                y0 + row * (self.tile_size[1] + self.tile_margin),
                self.tile_size[0],
                self.tile_size[1],
            )
            if tile_rect.right > self.rect.right:
                continue  # skip overflow; layout tuning pending
            color = (60, 60, 90)
            if symbol == self.hover_symbol:
                color = (90, 90, 130)
            pygame.draw.rect(surface, color, tile_rect, border_radius=6)
            label = self.font.render(symbol, True, (220, 220, 240))
            label_rect = label.get_rect(center=tile_rect.center)
            surface.blit(label, label_rect)

            if self.drag_symbol == symbol:
                pygame.draw.rect(surface, (255, 255, 0), tile_rect, width=2, border_radius=6)

    def handle_event(self, event: "pygame.event.Event") -> None:
        if pygame is None:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hover_symbol = self._symbol_at_point(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            symbol = self._symbol_at_point(event.pos)
            if symbol:
                self.drag_symbol = symbol
                self.on_element_selected(symbol)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drag_symbol = None

    def _symbol_at_point(self, position: Tuple[int, int]) -> Optional[str]:
        x0, y0 = self.rect.topleft
        x, y = position
        if not self.rect.collidepoint(x, y):
            return None
        col_width = self.tile_size[0] + self.tile_margin
        row_height = self.tile_size[1] + self.tile_margin
        col = (x - x0) // col_width
        row = (y - y0) // row_height
        index = int(row * self.columns + col)
        if index < 0 or index >= len(self.element_symbols):
            return None
        return self.element_symbols[index]


@dataclass
class ControlDockPanel:
    rect: "pygame.Rect"
    font: "pygame.font.Font"
    on_toggle_run: Callable[[], None]
    on_step: Callable[[], None]
    on_reset: Callable[[], None]
    on_speed_change: Callable[[float], None]
    is_running: bool = False
    speed_multiplier: float = 1.0
    min_speed: float = 0.25
    max_speed: float = 4.0
    _button_rects: Dict[str, "pygame.Rect"] = field(default_factory=dict, init=False, repr=False)
    _slider_track: Optional["pygame.Rect"] = field(default=None, init=False, repr=False)
    _slider_handle: Optional["pygame.Rect"] = field(default=None, init=False, repr=False)
    _slider_drag_active: bool = field(default=False, init=False, repr=False)

    def render(self, surface: "pygame.Surface") -> None:
        if pygame is None:
            return
        pygame.draw.rect(surface, (25, 25, 45), self.rect)
        status_text = "Running" if self.is_running else "Paused"
        status_surf = self.font.render(f"Status: {status_text}", True, (230, 230, 240))
        surface.blit(status_surf, (self.rect.x + 16, self.rect.y + 16))
        button_labels = [
            ("run", "Pause" if self.is_running else "Run"),
            ("step", "Step"),
            ("reset", "Reset"),
        ]
        self._button_rects = {}
        for idx, (key, label) in enumerate(button_labels):
            rect = pygame.Rect(self.rect.x + 16 + idx * 112, self.rect.y + 48, 100, 36)
            pygame.draw.rect(surface, (45, 45, 70), rect, border_radius=6)
            text_surface = self.font.render(label, True, (240, 240, 255))
            surface.blit(text_surface, text_surface.get_rect(center=rect.center))
            self._button_rects[key] = rect

        # Speed slider
        track_y = self.rect.y + 108
        track_rect = pygame.Rect(self.rect.x + 16, track_y, 240, 12)
        pygame.draw.rect(surface, (60, 60, 90), track_rect, border_radius=6)
        self._slider_track = track_rect

        normalized = (self.speed_multiplier - self.min_speed) / (self.max_speed - self.min_speed)
        normalized = max(0.0, min(1.0, normalized))
        handle_x = track_rect.x + int(normalized * track_rect.width)
        handle_rect = pygame.Rect(handle_x - 6, track_rect.y - 4, 12, 20)
        pygame.draw.rect(surface, (200, 200, 255), handle_rect, border_radius=4)
        self._slider_handle = handle_rect

        speed_label = self.font.render(f"Speed: {self.speed_multiplier:.2f}x", True, (220, 220, 235))
        surface.blit(speed_label, (track_rect.x, track_rect.y - 28))

    def handle_event(self, event: "pygame.event.Event") -> None:
        if pygame is None:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hasattr(event, "pos"):
                pos = event.pos
                for key, rect in self._button_rects.items():
                    if rect.collidepoint(pos):
                        if key == "run":
                            self.on_toggle_run()
                        elif key == "step":
                            self.on_step()
                        elif key == "reset":
                            self.on_reset()
                        return
                if self._slider_track and self._slider_track.collidepoint(pos):
                    self._slider_drag_active = True
                    self._set_speed_from_position(pos[0])
                elif self._slider_handle and self._slider_handle.collidepoint(pos):
                    self._slider_drag_active = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._slider_drag_active = False
            if hasattr(event, "pos") and self._slider_track and self._slider_track.collidepoint(event.pos):
                self._set_speed_from_position(event.pos[0])

        elif event.type == pygame.MOUSEMOTION and self._slider_drag_active:
            self._set_speed_from_position(event.pos[0])

    def _set_speed_from_position(self, x_pos: int) -> None:
        if self._slider_track is None:
            return
        track = self._slider_track
        normalized = (x_pos - track.x) / track.width
        normalized = max(0.0, min(1.0, normalized))
        new_speed = self.min_speed + normalized * (self.max_speed - self.min_speed)
        self.speed_multiplier = new_speed
        self.on_speed_change(new_speed)


@dataclass
class InspectorPanel:
    rect: "pygame.Rect"
    font: "pygame.font.Font"
    selected_atom_id: Optional[int] = None
    selected_info: Dict[str, str] = field(default_factory=dict)

    def render(self, surface: "pygame.Surface") -> None:
        if pygame is None:
            return
        pygame.draw.rect(surface, (18, 18, 32), self.rect)
        title = self.font.render("Inspector", True, (200, 200, 210))
        surface.blit(title, (self.rect.x + 12, self.rect.y + 12))
        y = self.rect.y + 40
        for key, value in self.selected_info.items():
            label = self.font.render(f"{key}: {value}", True, (180, 180, 190))
            surface.blit(label, (self.rect.x + 12, y))
            y += 20

    def update_selection(self, atom_id: Optional[int], info: Dict[str, str]) -> None:
        self.selected_atom_id = atom_id
        self.selected_info = info
