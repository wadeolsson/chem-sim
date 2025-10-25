# UI Layout & Interaction Notes

## Window Composition
- **Sandbox View (primary area)**  
  - Displays atoms, bonds, fields, electron densities.  
  - Supports zoom/pan, time scrubber, selection highlighting.  
  - Layered rendering: atoms > bonds > density overlays > field vectors.  
  - Context HUD showing local properties of hovered selection.

- **Wireframe Sketch**

```
┌─────────────────────────────── Sandbox View ────────────────────────────────┐
│                                                                              │
│   [Timeline scrubber]                                                        │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │   Atom sprites / bonds / density overlays render here                │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
┌────────────── Periodic Table ──────────────┐┌────────── Control Dock ───────┐
│ ┌───────────────┐  ┌───────────────┐       ││ Temperature: 300 K  [slider]  │
│ │ 1 H │ 2 He ...│  │13 Al│14 Si ...│       ││ Pressure:    1 atm [slider]   │
│ │ ... │ ...     │  │...  │ ...     │       ││ EM Field:    Off   [toggle]   │
│ └───────────────┘  └───────────────┘       ││ Photon Src:  532 nm [picker]  │
│ [Search ▢] [Filters ▽]                     ││                                │
└────────────────────────────────────────────┘│ [Play][Pause][Step][Reset]     │
                                              │ Presets ▽  | Save | Load       │
                                              └────────────────────────────────┘
┌────────────────────────── Inspector (toggle) ───────────────────────────────┐
│ Selected: H₂O        Charge: 0.0 e    HOMO: -12.6 eV   LUMO: -0.5 eV         │
│ Bond lengths: O-H 0.96 Å (2)                                              ▽ │
│ Actions: [Excite e⁻] [Ionize] [Lock Bond] [Freeze]                        ▢ │
└────────────────────────────────────────────────────────────────────────────┘
```

- **Periodic Table Panel (right column)**  
  - Interactive element grid grouped by periodic trends.  
  - Hover tooltips with quick facts; click to spawn/place atoms or open inspector.  
  - Filters for states (gas/liquid/solid) and categories (alkali metals, halogens, etc.).  
  - Search bar for symbol/name lookup.

- **Control Dock (right/bottom)**  
  - Environment sliders (temperature, pressure, EM fields).  
  - Photon source selector with wavelength presets and manual entry.  
  - Simulation controls: play/pause, step, reset, speed multiplier.  
  - Scenario presets list and save/load buttons.

- **Inspector Panel (toggle, bottom or overlay)**  
  - Shows selected atom/molecule properties: orbital occupation, partial charges, bond partners.  
  - Reveals valence electron counts, frontier shell occupancy, and oxidation states sourced from periodic-table metadata.  
  - Provides actions: excite electron, ionize, adjust spin, apply constraints.  
  - Displays analytics plots (energy contributions, spectra previews) on demand.

Bond Rendering
- Single, double, and triple bonds render as 1/2/3 parallel lines, colored by bond type (covalent vs ionic vs metallic).
- Ionic bonds adopt orange hues to highlight charge transfer; metallic bonds use cool blues for future expansion.

## Interaction Patterns
- Left-click select, shift-click multi-select, drag to box-select region.  
- Right-click context menu with operations (bond lock, label, freeze, delete).  
- Keyboard shortcuts for simulation control (`Space` toggle run, `.` step, numbers for speed).  
- Tool palette for placing atoms, measuring distances/angles, and drawing EM field regions.
- Drag-and-drop: click an element tile in the periodic table, drag into the sandbox to instantiate an atom at the drop location; preview ghosted atom outlines during drag.

## Visualization Styling
- Color scheme tied to periodic categories with accessible contrast.  
- Temperature/energy overlays use perceptually uniform gradients (e.g., Viridis).  
- Electron density shown via semi-transparent isosurfaces with dynamic threshold slider.  
- Field vectors animated with fading trails to indicate direction/magnitude.

## Performance Considerations
- Batch draw atoms/bonds, update density textures selectively to maintain framerate.  
- Defer expensive recalculations (spectra, analytics) to background threads or scheduled ticks.  
- Provide adjustable detail levels (LOD) for large systems: simplified atom sprites, aggregated density.

## Future Enhancements
- Multi-window support for analyst vs sandbox views.  
- VR/AR exploration options.  
- Collaborative mode with shared state synchronization.
