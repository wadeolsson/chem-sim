# pygame Rendering Strategy

## Overview
ChemSim's initial visualization layer will be implemented with `pygame`, taking advantage of its lightweight event loop and surface management while keeping rendering abstractions modular for future migration to OpenGL or GPU pipelines.

## Core Concepts
- **Main loop**: single-threaded pygame loop that processes events, steps the simulation, and renders frames at a target 60 FPS (configurable).
- **Scene graph**: logical representation of visual objects (atoms, bonds, density overlays) decoupled from pygame primitives.
- **Controllers**: input handlers (mouse, keyboard, drag-and-drop) that translate pygame events into higher-level UI actions.
- **Renderer modules**: dedicated classes that draw specific entity types to surfaces with shared styling configuration.

## Module Breakdown
- `App`: initializes pygame, creates window/surfaces, manages loop lifecycle, and orchestrates subsystem updates.
- `Viewport`:
  - Maintains world-to-screen transforms (pan/zoom).
  - Provides hooks for rendering atoms, bonds, fields, and overlays.
  - Exposes helper methods for hit-testing sandbox coordinates.
- `PeriodicTablePanel`:
  - Renders element tiles using cached surfaces/text.
  - Handles hover/drag state, triggers atom spawn callbacks.
- `ControlDock`:
  - Uses simple UI widgets (sliders, buttons) implemented via pygame rectangles and text.
  - Dispatches actions to the simulation controller (play/pause, temperature changes).
- `InspectorPanel`:
  - Draws readouts for selected entities and interactive buttons for operations (excite, ionize).

## Rendering Flow (per frame)
1. Process pygame events → route to controllers (sandbox, periodic table, control dock, inspector).
2. Update simulation state if the run/pause toggle is active; integrate multiple substeps if needed to match real time.
3. Clear surfaces, then render:
   1. Sandbox background + grid (optional).
   2. Bonds (lines), atoms (discs with element-specific colors), selection highlights.
   3. Density overlays/field vectors (as textures or vector glyphs).
   4. HUD overlays (timeline, energy readouts).
4. Render UI panels (periodic table, control dock, inspector) on top.
5. Flip the display with `pygame.display.flip()` and enforce frame cap via `clock.tick(fps)`.

## Drag-and-Drop Implementation Notes
- Periodic table tiles initiate drag state when mouse button down occurs within tile bounds.
- During drag:
  - Show a semi-transparent ghost atom following cursor.
  - Clamp ghost to sandbox viewport coordinates.
- On drop:
  - Convert screen position → world coordinates via `Viewport`.
  - Emit an event to the simulation controller to spawn a new atom (queued for next simulation tick).
- Cancel drag on right-click or leaving the window; revert cursor.

## Performance Considerations
- Cache pre-rendered surfaces (element tiles, frequently used icons) to minimize per-frame text rendering.
- Batch drawing operations where possible (e.g., `pygame.draw.circle` for atoms using loops on a single surface).
- Defer expensive draws (electron densities) to lower-frequency updates, blitting cached textures in between.
- Provide configuration for reduced detail (skip density, reduce update rate) to keep frame times stable.

## Future Migration Path
- Abstract the renderer behind interfaces so that replacing pygame with OpenGL, moderngl, or a GPU-centered engine requires minimal UI logic changes.
- Keep simulation stepping decoupled from frame rate to allow headless runs and offline rendering.
- Explore using `pygame.surfarray` or OpenGL contexts for volumetric electron density visuals once performance requirements increase.
