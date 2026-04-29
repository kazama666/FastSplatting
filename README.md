
![banner](assets/banner.png)
High-performance 3D Gaussian Splatting viewer embedded in Blender's viewport using GPU instanced rendering with custom GLSL shaders.

## Features

- **Real-time Splat Rendering** — Billboard splats with proper 3D covariance projected to screen space, alpha-blended in correct order
- **Spatial Block Indexing** — Automatic grid-based spatial partitioning for efficient frustum culling, supports per-block sorting for correct transparency
- **Incremental View-Dependent Sorting** — Detects camera movement and incrementally re-sorts splats far-to-near each frame; visible blocks processed first
- **Color Adjustments** — Real-time tint, exposure, gamma, hue shift, and saturation controls
- **Animation Export** — Export 3D viewport frames as PNG image sequences with optional per-frame sorting
- **GPU Instancing** — All splats batched into per-block GPU batches with precomputed 3D covariance on CPU


## Installation
This addon is distributed as a Blender extension.

1. download zip.
2. drop the zip into blender's viewport.
3. confirm install.

## Usage

### Getting Splat Data
Fast Splatting works with Gaussian Splat files generated from external tools.

You can generate splats using:

- Tencent Hunyuan (single-image world generation)
- Any Gaussian Splatting pipeline that exports .ply
- Scanned data (Polycam, RealityScan, etc.)

### Quick Start

1. **Import a PLY model** — `File → Import → PLY (.ply)`, select a Gaussian Splatting PLY file
2. **Select the imported mesh** — Select the mesh in the 3D viewport, then in the sidebar (`N` key → **FastSplatting** tab) click the object icon button to set it as target
3. **Start Render** — Click **Start Render** to initialize GPU buffers and begin rendering
4. **Stop Render** — Click **Stop Render** to release GPU resources

### Controls

| Control | Description |
|---|---|
| Block Size | Grid cell size for spatial partitioning (requires restart render) |
| Tint | RGB color multiplier |
| Exposure | Brightness multiplier |
| Gamma | Gamma correction |
| Hue | Hue shift (-1 to 1) |
| Saturation | Saturation (0 = grayscale, 1 = original) |
| Splat Scale | Uniform scale multiplier for splat size |

### Animation Export
Fast Spatting does not support for render pipline, so if you want to export animation, you need to use the animation export function to snapshot the viewport to image sequence.

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
├── operators.py          # Blender operators
├── panels.py             # UI sidebar panel, scene properties
├── splatting_data.py     # Data loading, spatial index, state management, sorting
├── gpu_renderer.py       # GPU batch building, GLSL shader, draw loop
└── blender_manifest.toml # Blender extension manifest
```

## Performance Notes

- Splat count and block size determine memory and draw-call overhead. Larger block size = fewer blocks = fewer draw calls but coarser culling
- The initial sort after loading or large camera moves may take several frames to converge;

## License
GPL-3.0
