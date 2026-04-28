# Fast Splatting

High-performance 3D Gaussian Splatting viewer embedded in Blender's viewport using GPU instanced rendering with custom GLSL shaders.

## Features

- **Real-time Splat Rendering** — Billboard splats with proper 3D covariance projected to screen space, alpha-blended in correct order
- **Spatial Block Indexing** — Automatic grid-based spatial partitioning for efficient frustum culling, supports per-block sorting for correct transparency
- **Incremental View-Dependent Sorting** — Detects camera movement and incrementally re-sorts splats far-to-near each frame; visible blocks processed first
- **Color Adjustments** — Real-time tint, exposure, gamma, hue shift, and saturation controls
- **Animation Export** — Export 3D viewport frames as PNG image sequences with optional per-frame sorting
- **GPU Instancing** — All splats batched into per-block GPU batches with precomputed 3D covariance on CPU

## Requirements

- Blender 4.2.0+
- `numpy` (bundled with Blender)

## Installation

This addon is distributed as a Blender extension (`.toml` manifest).

1. Clone or download this repository
2. In Blender, go to **Edit → Preferences → Get Extensions → Install from Disk**
3. Select the `blender_manifest.toml` file
4. Enable the addon in **Preferences → Add-ons** (search "Fast Splatting")

Or install as a legacy addon by placing the folder in Blender's `scripts/addons/` directory and enabling it in Preferences.

## Usage

### Quick Start

1. **Import a PLY model** — `File → Import → PLY (.ply)`, select a Gaussian Splatting PLY file (e.g., exported from 3D Gaussian Splatting or PostShot)
2. **Select the imported mesh** — Select the mesh in the 3D viewport, then in the sidebar (`N` key → **FastSplatting** tab) click the object icon button to set it as target
3. **Start Render** — Click **Start Render** to initialize GPU buffers and begin rendering
4. **Stop Render** — Click **Stop Render** to release GPU resources

### Controls

| Control | Description |
|---|---|
| Block Size | Grid cell size for spatial partitioning (requires restart render) |
| Sort Blocks per Frame | Number of blocks sorted per frame during auto-sort; higher = faster convergence, lower = smoother fps |
| Tint | RGB color multiplier |
| Exposure | Brightness multiplier |
| Gamma | Gamma correction |
| Hue | Hue shift (-1 to 1) |
| Saturation | Saturation (0 = grayscale, 1 = original) |
| Splat Scale | Uniform scale multiplier for splat size |

### Animation Export

1. Set **Start/End** frame range
2. Choose **Output** directory
3. Enable **Force Sorting Every Frame** to maintain correct transparency during camera animation
4. Click **Export Frames** — the viewport enters fullscreen mode and exports a PNG sequence
5. Cancel the operation by pressing **Esc** key

### Statistics

While rendering, the panel shows total and displayed block/splat counts.

## Architecture

```
Fast Splatting/
├── __init__.py           # Plugin metadata, registration
├── operators.py          # Blender operators (render control, sort, export animation)
├── panels.py             # UI sidebar panel, scene properties
├── splatting_data.py     # Data loading, spatial index, state management, sorting
├── gpu_renderer.py       # GPU batch building, GLSL shader, draw loop
└── blender_manifest.toml # Blender extension manifest
```

## Performance Notes

- Splat count and block size determine memory and draw-call overhead. Larger block size = fewer blocks = fewer draw calls but coarser culling
- The initial sort after loading or large camera moves may take several frames to converge; **Sort Blocks per Frame** controls the tradeoff

## License

GPL-3.0-or-later
