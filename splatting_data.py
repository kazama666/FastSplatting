import concurrent.futures
import os
import time
import numpy as np
import bpy
from bpy import types
from gpu.types import (
    GPUBatch,
    GPUVertBuf,
    GPUVertFormat,
    GPUIndexBuf,
)


# ---------------------------------------------------------------------------
# Per-instance splat data
# ---------------------------------------------------------------------------
class SplattingState:
    """Holds all data for one splat mesh instance."""

    def __init__(self):
        self.target_mesh = None
        self.point_count = 0
        self.block_count = 0

        # Core splatting data
        self.positions = None      # (N, 3) float32
        self.raw_dc = None         # (N, 3) float32 — pre-sigmoid SH DC
        self.opacities = None      # (N, 1) float32
        self.scales = None         # (N, 3) float32
        self.rotations = None      # (N, 4) float32 (quaternion)

        # SH higher-degree coefficients
        self.sh_coeffs = None      # (N, n_rest) float32 or None
        self.sh_degree = 0         # 0, 1, 2, or 3

        # Spatial indexing
        self.block_indices = None
        self.grid_dims = None    # (nx, ny, nz) from build_spatial_index
        self.block_bounds = None   # (M, 2, 3) min/max for each block
        self.block_centers = None  # (M, 3) center of each block
        self.block_radii = None    # (M,) bounding sphere radius
        self.block_splat_indices = None  # list of np.ndarray

        # Per-frame stats (updated by renderer during draw)
        self.displayed_block_count = 0
        self.displayed_splat_count = 0

        # Auto-sort state (alpha blend incremental sorting)
        self.sorted_up_to = 0
        self._sort_active = False
        self._camera_pos_np = None
        self._prev_view_matrix = None
        self._vp_world = None
        self._proj_00 = 0.0
        self._proj_11 = 0.0

        # Original block data (frozen at render start for delta-based transform)
        self._orig_block_centers = None
        self._orig_block_radii = None
        self._orig_block_bounds = None
        self._orig_model_matrix = None
        self._orig_model_inv = None
        # Frozen world-space AABB at render start (for grid overlay during rendering)
        self._frozen_grid_min = None
        self._frozen_grid_max = None
        # Tighter AABB from actual splat positions (for blue wireframe)
        self._frozen_positions_min = None
        self._frozen_positions_max = None

        # GPU batch cache (per-instance)
        self._batch_cache = {}

        # Pre-computed display color for SH0 fast path (sigmoid + gamma 2.2).
        self.display_colors = None   # (N, 3) float32

    def clear(self):
        self.target_mesh = None
        self.point_count = 0
        self.block_count = 0
        self.positions = None
        self.raw_dc = None
        self.opacities = None
        self.scales = None
        self.rotations = None
        self.sh_coeffs = None
        self.sh_degree = 0
        self.block_indices = None
        self.grid_dims = None
        self.block_bounds = None
        self.block_centers = None
        self.block_radii = None
        self.block_splat_indices = None
        self.sorted_up_to = 0
        self._sort_active = False
        self._camera_pos_np = None
        self._prev_view_matrix = None
        self._vp_world = None
        self._proj_00 = 0.0
        self._proj_11 = 0.0
        self._batch_cache = {}
        self.display_colors = None
        self._orig_block_centers = None
        self._orig_block_radii = None
        self._orig_block_bounds = None
        self._orig_model_matrix = None
        self._orig_model_inv = None
        self._frozen_grid_min = None
        self._frozen_grid_max = None
        self._frozen_positions_min = None
        self._frozen_positions_max = None

    # ------------------------------------------------------------------
    # GPU batch building (moved from gpu_renderer.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_cov3d(scales, rotations):
        """Vectorized 3D covariance from scales and rotations (GLSL column-major)"""
        r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        s0, s1, s2 = scales[:, 0], scales[:, 1], scales[:, 2]

        R00 = 1.0 - 2.0 * (y*y + z*z)
        R01 = 2.0 * (x*y + r*z)
        R02 = 2.0 * (x*z - r*y)

        R10 = 2.0 * (x*y - r*z)
        R11 = 1.0 - 2.0 * (x*x + z*z)
        R12 = 2.0 * (y*z + r*x)

        R20 = 2.0 * (x*z + r*y)
        R21 = 2.0 * (y*z - r*x)
        R22 = 1.0 - 2.0 * (x*x + y*y)

        s0_2, s1_2, s2_2 = s0*s0, s1*s1, s2*s2

        cov_a = np.column_stack([
            s0_2*R00*R00 + s1_2*R10*R10 + s2_2*R20*R20,
            s0_2*R00*R01 + s1_2*R10*R11 + s2_2*R20*R21,
            s0_2*R00*R02 + s1_2*R10*R12 + s2_2*R20*R22,
        ])
        cov_b = np.column_stack([
            s0_2*R01*R01 + s1_2*R11*R11 + s2_2*R21*R21,
            s0_2*R01*R02 + s1_2*R11*R12 + s2_2*R21*R22,
            s0_2*R02*R02 + s1_2*R12*R12 + s2_2*R22*R22,
        ])
        return cov_a, cov_b

    def _get_or_create_block_batch(self, block_idx):
        """Get cached batch for a block."""
        cache_key = ('b', block_idx)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        block_splat_indices = self.block_splat_indices[block_idx]
        if len(block_splat_indices) == 0:
            return None

        positions = self.positions[block_splat_indices]
        opacities = self.opacities[block_splat_indices]
        scales = self.scales[block_splat_indices]
        rotations = self.rotations[block_splat_indices]
        colors = self.display_colors[block_splat_indices]

        # SH data for the full shader (VBO attrs, no texture)
        sh_dc = self.raw_dc[block_splat_indices] if self.raw_dc is not None else None
        sh_rest = self.sh_coeffs[block_splat_indices] if self.sh_coeffs is not None else None

        batch = self._build_billboard_batch_for(
            positions, opacities, scales, rotations, colors, sh_dc=sh_dc, sh_rest=sh_rest)
        self._batch_cache[cache_key] = batch
        return batch

    def _get_or_create_fallback_batch(self):
        """Fallback single batch — top 1/16 per block"""
        cache_key = ('fallback', 0)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        fallback_indices = []
        for block_idx in range(self.block_count):
            indices = self.block_splat_indices[block_idx]
            count = max(1, len(indices) // 16)
            fallback_indices.extend(indices[:count])
        if len(fallback_indices) == 0:
            return None

        positions = self.positions[fallback_indices]
        opacities = self.opacities[fallback_indices]
        scales = self.scales[fallback_indices]
        rotations = self.rotations[fallback_indices]

        sh_dc = self.raw_dc[fallback_indices] if self.raw_dc is not None else None
        sh_rest = self.sh_coeffs[fallback_indices] if self.sh_coeffs is not None else None

        batch = self._build_billboard_batch_for(
            positions, opacities, scales, rotations,
            self.display_colors[fallback_indices], sh_dc=sh_dc, sh_rest=sh_rest)
        self._batch_cache[cache_key] = batch
        return batch

    def _build_billboard_batch_for(self, positions, opacities, scales, rotations, display_colors,
                                     sh_dc=None, sh_rest=None):
        """Build a single-VBO batch. SH0 display color comes from ``display_colors``
        (CPU sigmoid+gamma).  SH1+ coefficients come from ``sh_dc`` (raw DC) and
        ``sh_rest`` (N×9 float32) packed as VBO attributes — no texture needed."""
        N = len(positions)
        if N == 0:
            return None

        opacity_ok = opacities.ravel() >= 0.005
        scale_ok = scales.max(axis=1) >= 0.0003
        mask = opacity_ok & scale_ok
        if not mask.all():
            positions = positions[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            rotations = rotations[mask]
            display_colors = display_colors[mask]
            if sh_dc is not None:
                sh_dc = sh_dc[mask]
            if sh_rest is not None:
                sh_rest = sh_rest[mask]
            N = len(positions)
            if N == 0:
                return None

        quad_coords = np.array([
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        ], dtype=np.float32)

        base = np.arange(N, dtype=np.uint32) * 4
        indices = np.empty(N * 6, dtype=np.uint32)
        indices[0::6] = base
        indices[1::6] = base + 1
        indices[2::6] = base + 2
        indices[3::6] = base
        indices[4::6] = base + 2
        indices[5::6] = base + 3

        cov_a, cov_b = self._compute_cov3d(scales, rotations)

        fmt = GPUVertFormat()
        fmt.attr_add(id="quad_coord", comp_type='F32', len=2, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_position", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_color", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_opacity", comp_type='F32', len=1, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_cov_a", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_cov_b", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_sh_0", comp_type='F32', len=4, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_sh_1", comp_type='F32', len=4, fetch_mode='FLOAT')
        fmt.attr_add(id="inst_sh_2", comp_type='F32', len=4, fetch_mode='FLOAT')

        zero4 = np.zeros((N * 4, 4), dtype=np.float32)

        vbo = GPUVertBuf(fmt, len=N * 4)
        vbo.attr_fill(id="quad_coord", data=np.tile(quad_coords, (N, 1)))
        vbo.attr_fill(id="inst_position", data=np.repeat(positions, 4, axis=0))
        vbo.attr_fill(id="inst_color", data=np.repeat(display_colors, 4, axis=0))
        vbo.attr_fill(id="inst_opacity", data=np.repeat(opacities.ravel(), 4))
        vbo.attr_fill(id="inst_cov_a", data=np.repeat(cov_a, 4, axis=0))
        vbo.attr_fill(id="inst_cov_b", data=np.repeat(cov_b, 4, axis=0))
        if sh_dc is not None and sh_rest is not None and sh_rest.shape[1] >= 9:
            pack0 = np.column_stack([sh_dc, sh_rest[:, 0:1]])        # (N,4): dc.xyz, rest_0
            pack1 = sh_rest[:, 1:5]                                   # (N,4): rest_1..4
            pack2 = sh_rest[:, 5:9]                                   # (N,4): rest_5..8
            vbo.attr_fill(id="inst_sh_0", data=np.repeat(pack0, 4, axis=0))
            vbo.attr_fill(id="inst_sh_1", data=np.repeat(pack1, 4, axis=0))
            vbo.attr_fill(id="inst_sh_2", data=np.repeat(pack2, 4, axis=0))
        else:
            vbo.attr_fill(id="inst_sh_0", data=zero4)
            vbo.attr_fill(id="inst_sh_1", data=zero4)
            vbo.attr_fill(id="inst_sh_2", data=zero4)

        ibo = GPUIndexBuf(type='TRIS', seq=indices)
        return GPUBatch(type='TRIS', buf=vbo, elem=ibo)

    def clear_cache(self):
        """Clear batch cache (call after splat index reordering)."""
        self._batch_cache = {}

    def clear_block_cache(self, block_indices):
        """Clear cached batches only for specific blocks."""
        for block_idx in block_indices:
            self._batch_cache.pop(('b', block_idx), None)
        self._batch_cache.pop(('fallback', 0), None)


# ---------------------------------------------------------------------------
# UI property group for the instance list
# ---------------------------------------------------------------------------
class SplattingInstanceItem(types.PropertyGroup):
    mesh_name: bpy.props.StringProperty(
        name="Mesh",
        description="Mesh object containing splatting PLY data",
        default="",
    )
    mesh_uid: bpy.props.IntProperty(
        name="Mesh UID",
        description="Runtime session_uid for tracking renames",
        default=0,
    )
    enabled: bpy.props.BoolProperty(
        name="Enabled",
        description="Include this instance when rendering",
        default=True,
    )
    # Per-instance render adjustments
    color_tint: bpy.props.FloatVectorProperty(default=(1.0, 1.0, 1.0), min=0.0, max=2.0,
        subtype='COLOR', size=3, description="Overall color tint (multiply)")
    color_brightness: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Brightness multiplier")
    color_gamma: bpy.props.FloatProperty(default=1.0, min=0.0, max=5.0, step=0.1,
        description="Gamma correction")
    color_hue: bpy.props.FloatProperty(default=0.0, min=-1.0, max=1.0, step=0.01,
        description="Hue shift (-1..1)")
    color_saturation: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Saturation (0=gray, 1=original)")
    quad_scale: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0,
        description="Uniform scale multiplier for splat size")


# ---------------------------------------------------------------------------
# Light Probe data (stored per-object)
# ---------------------------------------------------------------------------
class ProbePoint(types.PropertyGroup):
    """A single light probe point with SH2 coefficients."""
    location: bpy.props.FloatVectorProperty(size=3, subtype='TRANSLATION',
        description="World-space position of this probe point")
    sh_r: bpy.props.FloatVectorProperty(size=9,
        description="SH2 red channel coefficients (9 bands)")
    sh_g: bpy.props.FloatVectorProperty(size=9,
        description="SH2 green channel coefficients (9 bands)")
    sh_b: bpy.props.FloatVectorProperty(size=9,
        description="SH2 blue channel coefficients (9 bands)")


# ---------------------------------------------------------------------------
# Scene — container for all instances + global render state
# ---------------------------------------------------------------------------
class SplattingScene:
    def __init__(self):
        self.instances = []          # list of SplattingState (parallel to collection)
        self.is_rendering = False
        self._draw_handle = None
        self._redraw_timer = None
        self._preferred_area = None  # the 3D view area that was active at render start

        # Aggregated stats (set by renderer each frame)
        self.displayed_block_count = 0
        self.displayed_splat_count = 0

    def clear(self):
        """Clear splat data. Does NOT remove draw handlers/timers — callers manage them."""
        for inst in self.instances:
            inst.clear()
        self.instances.clear()
        self.is_rendering = False
        self._preferred_area = None
        self.displayed_block_count = 0
        self.displayed_splat_count = 0


_scene = SplattingScene()


def get_state():
    """Return the global SplattingScene."""
    return _scene


# ---------------------------------------------------------------------------
# Probe placement state (runtime only, not persisted)
# ---------------------------------------------------------------------------
_irradiance_probe_positions = None  # np.ndarray (N, 3) or None
_envmap_probe_positions = None      # np.ndarray (N, 3) or None
_probe_last_change_time = 0.0       # time.time() of last param change

def mark_probe_preview_dirty():
    """Called when user changes probe params — records timestamp for debounce."""
    import time
    global _probe_last_change_time
    _probe_last_change_time = time.time()

def get_probe_positions(probe_type='envmap'):
    """Return cached probe positions, or None if not yet computed."""
    global _irradiance_probe_positions, _envmap_probe_positions
    if probe_type == 'irradiance':
        return _irradiance_probe_positions
    return _envmap_probe_positions


def _farthest_point_sample(candidates, n, bb_lo, bb_hi, min_spacing=0.0):
    """Farthest-point sampling from candidate positions.

    Returns (n, 3) float32 array. The first point is always the
    bounding-box center so that count=1 places a probe at center.
    Subsequent points are farthest-point-sampled from candidates.
    """
    center = (bb_lo + bb_hi) * 0.5

    if n <= 1:
        return center.reshape(1, 3)

    k = min(n - 1, len(candidates))  # number of farthest-point candidates to select
    dists = np.linalg.norm(candidates - center, axis=1)
    selected = [int(np.argmax(dists))]
    selected_mask = np.zeros(len(candidates), dtype=bool)
    selected_mask[selected[0]] = True

    min_dists = np.full(len(candidates), np.inf, dtype=np.float32)
    for _ in range(k - 1):
        d = np.linalg.norm(candidates - candidates[selected[-1]], axis=1)
        np.minimum(min_dists, d, out=min_dists)
        if min_spacing > 0.0:
            too_close = d < min_spacing
            min_dists[too_close] = -1.0
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] <= 0:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True

    return np.vstack([center.reshape(1, 3), candidates[selected]])


def compute_probe_positions(context):
    """Compute both irradiance and envmap probe positions.

    Uses ``bake_area_center`` ± ``bake_area_offset`` as the bounding box,
    builds a density volume from splat positions, generates 64³ candidates,
    filters by density threshold, and runs separate farthest-point sampling
    for each probe type.

    Returns (irradiance_positions, envmap_positions) or (None, None).
    Caches both in module-level globals.
    """
    import numpy as np
    import time
    global _irradiance_probe_positions, _envmap_probe_positions

    props = context.scene.splatting_properties
    n_irr = props.irradiance_probe_count
    n_env = props.envmap_probe_count

    # Bake volume from center + size
    center = np.array(props.bake_area_center, dtype=np.float32)
    half = np.array(props.bake_area_size, dtype=np.float32) * 0.5
    lo = center - half
    hi = center + half
    dims = np.array(props.bake_area_size, dtype=np.float32)
    dims = np.maximum(dims, 1.0)

    # Collect world-space splat positions from all enabled instances
    all_positions = []
    for item in context.scene.splatting_instances:
        if not item.enabled:
            continue
        obj = bpy.data.objects.get(item.mesh_name)
        if not obj or obj.type != 'MESH' or len(obj.data.vertices) == 0:
            continue

        n = len(obj.data.vertices)
        pos = np.empty(n * 3, dtype=np.float32)
        obj.data.vertices.foreach_get('co', pos)
        pos = pos.reshape(-1, 3)

        mat = np.array(obj.matrix_world, dtype=np.float32)
        ones = np.ones((n, 1), dtype=np.float32)
        pos_world = (np.concatenate([pos, ones], axis=1) @ mat.T)[:, :3]
        all_positions.append(pos_world)

    if not all_positions:
        _irradiance_probe_positions = None
        _envmap_probe_positions = None
        return None, None

    positions = np.concatenate(all_positions, axis=0)

    # Density volume: 32³ voxels covering the bake volume
    VOXEL_RES = 32
    voxel_size = dims / VOXEL_RES
    vox_ij = np.floor((positions - lo) / voxel_size).astype(np.int32)
    in_bounds = ((vox_ij >= 0) & (vox_ij < VOXEL_RES)).all(axis=1)
    vox_ij = vox_ij[in_bounds]
    if len(vox_ij) == 0:
        # No splats inside bake volume — spread evenly
        _irradiance_probe_positions = _farthest_point_sample(
            _generate_uniform_candidates(lo, hi, 64), n_irr, lo, hi)
        _envmap_probe_positions = _farthest_point_sample(
            _generate_uniform_candidates(lo, hi, 64), n_env, lo, hi)
        return _irradiance_probe_positions, _envmap_probe_positions

    flat_idx = (vox_ij[:, 0] * VOXEL_RES * VOXEL_RES
                + vox_ij[:, 1] * VOXEL_RES + vox_ij[:, 2])
    density = np.bincount(flat_idx, minlength=VOXEL_RES**3,
                           ).astype(np.float32).reshape(VOXEL_RES, VOXEL_RES, VOXEL_RES)

    # 3×3×3 box blur
    density_smooth = np.zeros_like(density)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                s = np.roll(np.roll(np.roll(density, dx, 0), dy, 1), dz, 2)
                density_smooth += s
    density_smooth /= 27.0

    # Candidate points: 64³ grid within the bake volume
    candidates = _generate_uniform_candidates(lo, hi, 64)

    # Look up smoothed density at each candidate
    dv_ij = np.floor((candidates - lo) / voxel_size).astype(np.int32)
    dv_ij = np.clip(dv_ij, 0, VOXEL_RES - 1)
    cand_density = density_smooth[dv_ij[:, 0], dv_ij[:, 1], dv_ij[:, 2]]

    # Keep bottom 30% density (open space)
    max_n = max(n_irr, n_env)
    threshold = max(np.percentile(cand_density, 30), 1.0)
    keep = cand_density <= threshold
    if keep.sum() < max_n:
        threshold = np.percentile(cand_density, 50)
        keep = cand_density <= threshold
    if keep.sum() < max_n:
        keep = np.ones(len(candidates), dtype=bool)

    candidates = candidates[keep]
    if len(candidates) == 0:
        candidates = _generate_uniform_candidates(lo, hi, 64)

    # Farthest-point for each probe type (always via _farthest_point_sample
    # so the first point is always the bake-area center)
    _irradiance_probe_positions = _farthest_point_sample(
        candidates, n_irr, lo, hi) if n_irr > 0 else None
    _envmap_probe_positions = _farthest_point_sample(
        candidates, n_env, lo, hi) if n_env > 0 else None

    return _irradiance_probe_positions, _envmap_probe_positions


def _generate_uniform_candidates(lo, hi, res):
    """Generate a ``res``×``res``×``res`` uniform grid of candidate points."""
    dims = hi - lo
    cell = dims / res
    cx = lo[0] + cell[0] * (np.arange(res) + 0.5)
    cy = lo[1] + cell[1] * (np.arange(res) + 0.5)
    cz = lo[2] + cell[2] * (np.arange(res) + 0.5)
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def try_update_probe_preview(context):
    """Debounced preview update — called from draw handler.

    If the debounce timer (1 s since last ``mark_probe_preview_dirty``)
    has expired, recompute both probe position sets.
    Returns True if positions are available (cached or fresh).
    """
    import time
    global _probe_last_change_time
    if _probe_last_change_time > 0 and time.time() - _probe_last_change_time >= 1.0:
        _probe_last_change_time = 0.0
        compute_probe_positions(context)
    return (_irradiance_probe_positions is not None
            or _envmap_probe_positions is not None)


# ---------------------------------------------------------------------------
# Irradiance probe interpolation (PRE_VIEW draw handler)
# ---------------------------------------------------------------------------
_irradiance_objects = []        # object names with "irradiance_from_splats" modifier
_irradiance_objects_dirty = True  # re-scan needed
_irradiance_update_idx = 0      # round-robin counter
_irradiance_pre_handle = None   # PRE_VIEW draw handler handle
_probe_cache_valid = False
_probe_cache_grids = None   # list of dicts, one per instance with probe grid data


def _extract_scattered(probes):
    """From a probe_points collection, build scattered probe data for IDW.

    Probes are now placed via farthest-point sampling (not a grid),
    so we skip grid deduction and store raw positions + SH data.
    """
    if len(probes) < 1:
        return None
    positions = np.array([list(p.location) for p in probes], dtype=np.float32)
    sh_r = np.array([list(p.sh_r) for p in probes], dtype=np.float32)
    sh_g = np.array([list(p.sh_g) for p in probes], dtype=np.float32)
    sh_b = np.array([list(p.sh_b) for p in probes], dtype=np.float32)
    return {
        'positions': positions,
        'sh_r': sh_r, 'sh_g': sh_g, 'sh_b': sh_b,
    }


def _idw_interpolate(scattered, local_pos, k=6, eps=1e-6):
    """Gaussian RBF interpolation from k nearest probes with sorted distances.

    Uses Gaussian weighting exp(-d²/(2σ²)) with sigma = 0.4 × k-th-neighbor
    distance so the farthest included probe contributes ≈0.04, making
    entering/exiting the k-set invisible.  k indices are sorted so sigma
    uses the true k-th distance.
    Returns 9 RGB tuples (never None — falls back to first probe).
    """
    pos = scattered['positions']
    n = len(pos)
    k = min(k, n)
    d = np.linalg.norm(pos - local_pos, axis=1)
    # k smallest, then sort by distance so d_k[-1] is truly the farthest
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    d_k = d[idx]
    sigma = max(d_k[-1] * 0.4, 0.01) if k >= 2 else 1.0
    w = np.exp(-d_k**2 / (2.0 * sigma**2))
    w += eps
    w /= w.sum()
    sh_r = (scattered['sh_r'][idx] * w[:, None]).sum(axis=0)
    sh_g = (scattered['sh_g'][idx] * w[:, None]).sum(axis=0)
    sh_b = (scattered['sh_b'][idx] * w[:, None]).sum(axis=0)
    return [(float(sh_r[b]), float(sh_g[b]), float(sh_b[b])) for b in range(9)]


def _refresh_probe_cache():
    """Rebuild per-instance scattered probe data cache."""
    global _probe_cache_grids, _probe_cache_valid
    _probe_cache_grids = []
    for inst in _scene.instances:
        if inst.target_mesh and hasattr(inst.target_mesh, 'probe_points'):
            probes = inst.target_mesh.probe_points
            if len(probes) > 0:
                g = _extract_scattered(probes)
                if g is not None:
                    g['target_mesh'] = inst.target_mesh
                    _probe_cache_grids.append(g)
    _probe_cache_valid = True


def collect_irradiance_objects():
    """Scan scene for meshes with a geometry node named 'irradiance_from_splats'.

    Result is cached — only re-scans when _irradiance_objects_dirty is set.
    """
    global _irradiance_objects, _irradiance_objects_dirty
    if not _irradiance_objects_dirty:
        return
    _irradiance_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type == 'NODES' and mod.node_group and "irradiance_from_splats" in mod.node_group.name:
                _irradiance_objects.append(obj.name)
                break
    _irradiance_objects_dirty = False


def invalidate_irradiance_cache():
    """Force collect_irradiance_objects to re-scan on next call."""
    global _irradiance_objects_dirty, _probe_cache_valid
    _irradiance_objects_dirty = True
    _probe_cache_valid = False


def _irradiance_pre_draw():
    """PRE_VIEW handler: for one irradiance receiver object per frame,
    trilinearly interpolate probe SH coefficients and write to its geometry node modifier."""
    if not _scene.is_rendering:
        return

    collect_irradiance_objects()
    if not _irradiance_objects:
        return

    if not _probe_cache_valid:
        _refresh_probe_cache()
    if not _probe_cache_grids:
        return

    # Round-robin: update one object per frame
    global _irradiance_update_idx
    idx = _irradiance_update_idx % len(_irradiance_objects)
    _irradiance_update_idx += 1

    obj_name = _irradiance_objects[idx]
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        return

    # Find the modifier (still present?)
    target_mod = None
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group and "irradiance_from_splats" in mod.node_group.name:
            target_mod = mod
            break
    if not target_mod:
        return

    # Use the first probe grid, transform receiver → target local space
    g = _probe_cache_grids[0]
    target = g['target_mesh']
    try:
        mat = np.array(target.matrix_world, dtype=np.float32)
        inv = np.linalg.inv(mat)
        center = np.array(obj.matrix_world.translation, dtype=np.float32)
        local_pos = (inv @ np.array([center[0], center[1], center[2], 1.0], dtype=np.float32))[:3]
    except (ReferenceError, np.linalg.LinAlgError):
        return
    bands = _idw_interpolate(g, local_pos)

    # Write 9 Color inputs (SH_Band_0 … SH_Band_8) to the modifier
    try:
        # Map display names → socket identifiers from the node group interface
        socket_ids = {}
        if target_mod.node_group:
            for item in target_mod.node_group.interface.items_tree:
                if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                    socket_ids[item.name] = item.identifier
        for bi in range(9):
            name = f"SH_Band_{bi}"
            key = socket_ids.get(name, name)
            v = bands[bi]
            try:
                arr = target_mod[key]
                arr[0] = float(v[0])
                arr[1] = float(v[1])
                arr[2] = float(v[2])
            except (KeyError, TypeError):
                target_mod[key] = [float(v[0]), float(v[1]), float(v[2])]
    except Exception as e:
        print(f"[Splatting] Irradiance write failed for '{obj_name}': {e}")
        return

    # Tag object for update so geometry node re-evaluates next frame
    obj.update_tag()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class SplattingProperties(types.PropertyGroup):
    is_rendering: bpy.props.BoolProperty(default=False)
    point_count: bpy.props.IntProperty(default=0)
    block_count: bpy.props.IntProperty(default=0)
    sort_near_to_far: bpy.props.BoolProperty(default=False)
    block_size: bpy.props.FloatProperty(default=1.0, min=0.1, max=20.0, step=0.1,
        description="Spatial block size for culling. Larger = fewer blocks, fewer draw calls. Requires restart")
    clip_alpha: bpy.props.FloatProperty(
        name="Clip Alpha",
        description="Discard splats with opacity below this threshold on init. Reduces splat count and clutter.",
        default=0.1, min=0.0, max=1.0, step=0.01,
    )
    clip_size: bpy.props.FloatProperty(
        name="Clip Size",
        description="Discard splats with average scale below this threshold on init.",
        default=0.0, min=0.0, max=0.1, step=0.001,
    )
    show_block_grid: bpy.props.BoolProperty(
        name="Display Grid",
        description="Show block grid overlay in the viewport for previewing block boundaries.",
        default=False,
    )
    block_offset: bpy.props.FloatVectorProperty(
        name="Block Offset",
        description="Offset the grid origin to shift block boundaries. Adjustable in real-time.",
        default=(0.0, 0.0, 0.0), size=3,
    )
    grid_color: bpy.props.FloatVectorProperty(
        name="Grid Color",
        default=(1.0, 0.5, 0.0), size=3, subtype='COLOR', min=0, max=1,
    )
    grid_alpha: bpy.props.FloatProperty(
        name="Grid Alpha",
        default=0.5, min=0.0, max=1.0,
    )
    active_instance_index: bpy.props.IntProperty(default=0,
        description="Active index in the splat instances list")

    # Scene-level defaults for per-instance render adjustments
    default_color_tint: bpy.props.FloatVectorProperty(default=(1.0, 1.0, 1.0), min=0.0, max=2.0,
        subtype='COLOR', size=3, description="Default color tint for new instances")
    default_color_brightness: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Default brightness for new instances")
    default_color_gamma: bpy.props.FloatProperty(default=1.0, min=0.0, max=5.0, step=0.1,
        description="Default gamma for new instances")
    default_color_hue: bpy.props.FloatProperty(default=0.0, min=-1.0, max=1.0, step=0.01,
        description="Default hue shift for new instances")
    default_color_saturation: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Default saturation for new instances")
    default_quad_scale: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0,
        description="Default splat scale multiplier for new instances")
    bake_gain: bpy.props.FloatProperty(
        name="Bake Gain",
        description="Expand splat color range before baking probes. Higher values = more dynamic range in indirect lighting.",
        default=3.0, min=1.0, max=10.0, step=0.1,
    )
    bake_gain_start: bpy.props.FloatProperty(
        name="Bake Gain Start",
        description="Threshold where non-linear expansion begins. Colors below this stay nearly unchanged.",
        default=0.7, min=0.5, max=0.95, step=0.01,
    )
    anim_start_frame: bpy.props.IntProperty(default=1,
        description="First frame of animation export range")
    anim_end_frame: bpy.props.IntProperty(default=250,
        description="Last frame of animation export range")
    anim_output_path: bpy.props.StringProperty(default="//", subtype='DIR_PATH',
        description="Directory to save exported frames")
    anim_force_sort: bpy.props.BoolProperty(default=True,
        description="Sort blocks far-to-near every frame when camera changes")
    default_irradiance_color: bpy.props.FloatVectorProperty(
        name="Default Irradiance",
        description="Default ambient color for receivers with no nearby probes",
        default=(0.2, 0.2, 0.2), subtype='COLOR', min=0, max=1, size=3,
    )
    ui_export_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle export animation section")
    ui_color_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle color adjustment section")
    ui_stats_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle statistics section")
    ui_meshes_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle splat meshes section")
    ui_settings_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle settings section")

    # --- Bake area & probes ---
    bake_area_center: bpy.props.FloatVectorProperty(
        name="Bake Center",
        description="Center of the bake volume. Probes are placed within center ± offset.",
        default=(0.0, 0.0, 0.0), size=3, subtype='TRANSLATION',
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )
    bake_area_size: bpy.props.FloatVectorProperty(
        name="Bake Size",
        description="Full dimensions of the bake volume. Probes are placed within center ± size/2.",
        default=(10.0, 10.0, 10.0), size=3, subtype='TRANSLATION',
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )

    # Irradiance (SH) probes
    irradiance_probe_count: bpy.props.IntProperty(
        name="Irradiance Probes",
        description="Number of irradiance (SH) probes to place",
        default=16, min=1, max=4096,
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )
    irradiance_show_preview: bpy.props.BoolProperty(
        name="Show Irradiance Probes",
        description="Preview irradiance probe positions as green dots",
        default=False,
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )

    # Envmap probes
    envmap_probe_count: bpy.props.IntProperty(
        name="Envmap Probes",
        description="Number of environment map probes to place (1-128)",
        default=16, min=1, max=128,
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )
    envmap_show_preview: bpy.props.BoolProperty(
        name="Show Envmap Probes",
        description="Preview envmap probe positions as green dots",
        default=False,
        update=lambda self, ctx: mark_probe_preview_dirty(),
    )
    envmap_probe_resolution: bpy.props.IntProperty(
        name="Probe Resolution",
        description="Resolution of each equirectangular probe (width)",
        default=512, min=128, max=1024, step=128,
    )

    ui_bake_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle bake section")


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# msgbus: track object renames to keep mesh_name in sync
# ---------------------------------------------------------------------------
_msgbus_owner = object()


def _on_id_renamed():
    """Update mesh_name in instance items when an object is renamed."""
    changed = False
    for scn in bpy.data.scenes:
        for item in scn.splatting_instances:
            if item.mesh_uid == 0:
                continue
            obj = next((o for o in bpy.data.objects if o.session_uid == item.mesh_uid), None)
            if obj and obj.name != item.mesh_name:
                item.mesh_name = obj.name
                changed = True
    if changed:
        _tag_view3d_redraw()


@bpy.app.handlers.persistent
def _on_load_post(dummy):
    """Recover mesh_uid after loading a .blend file (session_uid not persisted)."""
    for scn in bpy.data.scenes:
        # Reset rendering state — runtime resources (GPU, handlers) don't survive file load
        scn.splatting_properties.is_rendering = False
        scn.splatting_properties.point_count = 0
        scn.splatting_properties.block_count = 0
        for item in scn.splatting_instances:
            obj = bpy.data.objects.get(item.mesh_name)
            if obj:
                item.mesh_uid = obj.session_uid


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """Remove instances whose mesh objects have been deleted from the scene."""
    instances = scene.splatting_instances
    if not instances:
        return
    removed = False
    for i in range(len(instances) - 1, -1, -1):
        item = instances[i]
        if item.mesh_name and bpy.data.objects.get(item.mesh_name) is None:
            instances.remove(i)
            removed = True
    if removed:
        props = scene.splatting_properties
        if props.active_instance_index >= len(instances):
            props.active_instance_index = max(0, len(instances) - 1)


def register():
    bpy.utils.register_class(SplattingProperties)
    bpy.utils.register_class(SplattingInstanceItem)
    bpy.utils.register_class(ProbePoint)
    bpy.types.Scene.splatting_properties = bpy.props.PointerProperty(type=SplattingProperties)
    bpy.types.Scene.splatting_instances = bpy.props.CollectionProperty(type=SplattingInstanceItem)
    bpy.types.Object.probe_points = bpy.props.CollectionProperty(type=ProbePoint)
    bpy.types.Object.bake_bbox_corners = bpy.props.FloatVectorProperty(
        name="Bake BBox Corners",
        description="8 local-space corners of the bake bounding box (24 floats)",
        default=(0.0,) * 24,
        size=24,
    )
    bpy.types.Object.bake_matrix_world = bpy.props.FloatVectorProperty(
        name="Bake Matrix World",
        description="Object's matrix_world at bake time (16 floats row-major)",
        default=(1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0),
        size=16,
    )

    bpy.types.Object.envmap_atlas = bpy.props.StringProperty(
        name="Envmap Atlas",
        description="Name of the envmap atlas image for this object",
        default="",
    )

    # Subscribe to Object name changes
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.Object, "name"),
        owner=_msgbus_owner,
        args=(),
        notify=_on_id_renamed,
        options={'PERSISTENT'},
    )
    # Recover uids after .blend load
    bpy.app.handlers.load_post.append(_on_load_post)
    # Track object deletion to auto-remove instances
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)

    from .gpu_renderer import register_grid_draw
    register_grid_draw()


def unregister():
    from .gpu_renderer import unregister_grid_draw
    unregister_grid_draw()

    _stop_redraw_timer()
    if _scene._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_scene._draw_handle, 'WINDOW')
        except Exception:
            pass
        _scene._draw_handle = None
    _unregister_irradiance_pre_draw()
    try:
        from .gpu_renderer import release_renderer
        release_renderer()
    except Exception:
        pass
    _scene.is_rendering = False
    _scene.clear()

    # Clean up msgbus subscription and handlers
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)

    if hasattr(bpy.types.Scene, "splatting_instances"):
        del bpy.types.Scene.splatting_instances
    bpy.utils.unregister_class(SplattingInstanceItem)
    if hasattr(bpy.types.Object, "probe_points"):
        del bpy.types.Object.probe_points
    if hasattr(bpy.types.Object, "bake_bbox_corners"):
        del bpy.types.Object.bake_bbox_corners
    if hasattr(bpy.types.Object, "bake_matrix_world"):
        del bpy.types.Object.bake_matrix_world
    if hasattr(bpy.types.Object, "envmap_atlas"):
        del bpy.types.Object.envmap_atlas
    bpy.utils.unregister_class(ProbePoint)
    if hasattr(bpy.types.Scene, "splatting_properties"):
        del bpy.types.Scene.splatting_properties
    bpy.utils.unregister_class(SplattingProperties)


# ---------------------------------------------------------------------------
# PLY mesh reading
# ---------------------------------------------------------------------------
def read_ply_attributes(mesh):
    """Read Gaussian Splatting attributes from mesh vertex data.

    Returns (positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree).
    ``colors`` is the DC-only view-independent color (sigmoid + gamma-encoded).
    ``raw_dc`` is the pre-sigmoid SH DC coefficient. ``sh_coeffs`` is (N, n_rest) or None.
    """
    vertex_count = len(mesh.vertices)

    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", positions.ravel())

    attributes = mesh.attributes
    attr_names = [attr.name for attr in attributes]

    def find_attr(prefix):
        for name in attr_names:
            if name.startswith(prefix):
                return name
        return None

    # Raw SH DC coefficients (pre-sigmoid, used for GPU SH evaluation)
    raw_dc = np.zeros((vertex_count, 3), dtype=np.float32)
    dc0 = find_attr('f_dc_0')
    dc1 = find_attr('f_dc_1')
    dc2 = find_attr('f_dc_2')
    if dc0 and dc1 and dc2:
        raw_dc[:, 0] = np.array([v.value for v in attributes[dc0].data], dtype=np.float32)
        raw_dc[:, 1] = np.array([v.value for v in attributes[dc1].data], dtype=np.float32)
        raw_dc[:, 2] = np.array([v.value for v in attributes[dc2].data], dtype=np.float32)

    # Processed color (sigmoid + gamma) for fallback / backward compat
    linear = 1.0 / (1.0 + np.exp(-raw_dc))
    colors = np.power(linear, 2.2)

    # SH higher-degree coefficients (f_rest) — capped at degree 1 (9 coeffs).
    # Degree 2/3 from input PLY are discarded; these require significantly more
    # GPU shader work for diminishing visual returns.
    rest_names = sorted(
        (name for name in attr_names if name.startswith('f_rest_')),
        key=lambda n: int(n.split('_')[-1])
    )
    if len(rest_names) >= 9:
        sh_degree = 1
        n_coeffs = 9
    else:
        sh_degree = 0
        n_coeffs = 0

    if n_coeffs > 0:
        sh_coeffs = np.zeros((vertex_count, n_coeffs), dtype=np.float32)
        for i in range(n_coeffs):
            sh_coeffs[:, i] = np.array(
                [v.value for v in attributes[rest_names[i]].data], dtype=np.float32)
        print(f"[Splatting]  SH degree {sh_degree} ({n_coeffs} coefficients)")
    else:
        sh_coeffs = None

    # Opacity
    opacities = np.ones((vertex_count, 1), dtype=np.float32) * 0.5
    opacity_attr = find_attr('opacity')
    if opacity_attr:
        opacities[:, 0] = np.array([v.value for v in attributes[opacity_attr].data], dtype=np.float32)
        opacities[:, 0] = 1.0 / (1.0 + np.exp(-opacities[:, 0]))

    # Scales
    scales = np.ones((vertex_count, 3), dtype=np.float32) * 0.01
    scale0 = find_attr('scale_0')
    scale1 = find_attr('scale_1')
    scale2 = find_attr('scale_2')
    if scale0 and scale1 and scale2:
        scales[:, 0] = np.exp(np.array([v.value for v in attributes[scale0].data], dtype=np.float32))
        scales[:, 1] = np.exp(np.array([v.value for v in attributes[scale1].data], dtype=np.float32))
        scales[:, 2] = np.exp(np.array([v.value for v in attributes[scale2].data], dtype=np.float32))

    # Rotations
    rotations = np.zeros((vertex_count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    rot0 = find_attr('rot_0')
    rot1 = find_attr('rot_1')
    rot2 = find_attr('rot_2')
    rot3 = find_attr('rot_3')
    if rot0 and rot1 and rot2 and rot3:
        rotations[:, 0] = np.array([v.value for v in attributes[rot0].data], dtype=np.float32)
        rotations[:, 1] = np.array([v.value for v in attributes[rot1].data], dtype=np.float32)
        rotations[:, 2] = np.array([v.value for v in attributes[rot2].data], dtype=np.float32)
        rotations[:, 3] = np.array([v.value for v in attributes[rot3].data], dtype=np.float32)

    return positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree


# ---------------------------------------------------------------------------
# Spatial index
# ---------------------------------------------------------------------------
def _process_block_batch(args):
    """Process a batch of blocks: compute center, radius, bounds (runs in worker pool)"""
    positions, indices_list = args
    results = []
    for indices in indices_list:
        block_pos = positions[indices]
        center = block_pos.mean(axis=0)
        radii = np.linalg.norm(block_pos - center, axis=1)
        results.append((center, radii.max(), block_pos.min(axis=0), block_pos.max(axis=0)))
    return results


def build_spatial_index(positions, block_size=1.0, origin_offset=None, use_parallel=True):
    """Build spatial index for fast culling.

    Args:
        positions: (N, 3) float32 positions in world space.
        block_size: grid cell size.
        origin_offset: shift the grid origin by this amount (grid lines at
                       min_coords + offset + k*block_size).
    """
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    dimensions = max_coords - min_coords
    grid_dims = np.ceil(dimensions / block_size).astype(int) + 1

    origin = min_coords if origin_offset is None else min_coords + np.asarray(origin_offset, dtype=np.float32)
    block_idx = np.floor((positions - origin) / block_size).astype(int)
    block_ids = (block_idx[:, 0] + block_idx[:, 1] * grid_dims[0] +
                 block_idx[:, 2] * grid_dims[0] * grid_dims[1])

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    block_count = len(unique_blocks)
    sorted_idx = np.argsort(inverse)
    split_pts = np.where(np.diff(inverse[sorted_idx]) != 0)[0] + 1
    block_splat_indices_np = np.split(sorted_idx, split_pts)
    block_splat_indices = [idx.astype(np.int32) for idx in block_splat_indices_np]

    block_centers = np.zeros((block_count, 3), dtype=np.float32)
    block_radii = np.zeros(block_count, dtype=np.float32)
    block_bounds = np.zeros((block_count, 2, 3), dtype=np.float32)

    if use_parallel and block_count > 200:
        n_jobs = min(os.cpu_count() or 4, 8)
        batches = np.array_split(np.arange(block_count), n_jobs)
        batch_args = [(positions, [block_splat_indices[i] for i in batch])
                      for batch in batches]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for batch, results in zip(batches, pool.map(_process_block_batch, batch_args)):
                for i, (center, radius, bmin, bmax) in zip(batch, results):
                    block_centers[i] = center
                    block_radii[i] = radius
                    block_bounds[i, 0] = bmin
                    block_bounds[i, 1] = bmax
    else:
        for i in range(block_count):
            block_pos = positions[block_splat_indices[i]]
            center = block_pos.mean(axis=0)
            block_centers[i] = center
            block_radii[i] = np.linalg.norm(block_pos - center, axis=1).max()
            block_bounds[i, 0] = block_pos.min(axis=0)
            block_bounds[i, 1] = block_pos.max(axis=0)

    return {
        'block_indices': block_ids,
        'unique_blocks': unique_blocks,
        'block_centers': block_centers,
        'block_radii': block_radii,
        'block_bounds': block_bounds,
        'block_splat_indices': block_splat_indices,
        'grid_dims': grid_dims,
        'min_coords': min_coords,
        'block_size': block_size,
    }




# ---------------------------------------------------------------------------
# Draw handler — render splats in exactly one 3D view
# ---------------------------------------------------------------------------
# We store _preferred_area at render start (the 3D view where the user clicked
# "Start Render") and only draw when that area's handler fires.  If the
# preferred area gets freed (new file load) we adopt the first 3D view seen.
def _draw_handler():
    try:
        context = bpy.context
        area = context.area
        if not (area and area.type == 'VIEW_3D'):
            return

        pref = _scene._preferred_area
        if pref is not None:
            try:
                _ = pref.type
            except ReferenceError:
                _scene._preferred_area = None
                pref = None

        if pref is not None:
            if area == pref:
                from .gpu_renderer import draw_splatting
                draw_splatting(context)
                area.tag_redraw()
        else:
            # No preferred area (freed on new file load) — adopt the first
            # 3D view encountered so splats stay visible.
            _scene._preferred_area = area
            from .gpu_renderer import draw_splatting
            draw_splatting(context)
            area.tag_redraw()
    except ReferenceError:
        _emergency_stop()


def _tag_view3d_redraw():
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            screen = win.screen
            if not screen:
                continue
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
                    return

# ---------------------------------------------------------------------------
# Start / Stop rendering
# ---------------------------------------------------------------------------
def start_render(context):
    """Initialize splatting data from all instances and start rendering."""
    global _scene

    if _scene.is_rendering:
        stop_render(context)

    # Ensure all objects are in Object Mode — custom attribute data access
    # (read_ply_attributes) returns empty arrays in Edit/Weight Paint mode.
    raw_mode = bpy.context.mode
    # Normalize: bpy.context.mode returns 'EDIT_MESH' etc., but
    # bpy.ops.object.mode_set() expects the short form 'EDIT'.
    prev_mode = 'EDIT' if raw_mode.startswith('EDIT_') else raw_mode
    if prev_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    splatting_props = context.scene.splatting_properties
    block_size = splatting_props.block_size

    # Collect valid meshes from the UI collection
    items = context.scene.splatting_instances
    meshes_to_load = []
    for item in items:
        obj = bpy.data.objects.get(item.mesh_name)
        if obj and obj.type == 'MESH' and len(obj.data.vertices) > 0:
            meshes_to_load.append(obj)
        elif obj:
            print(f"[Splatting] Skipping '{item.mesh_name}': not a mesh or empty")
        else:
            print(f"[Splatting] Skipping '{item.mesh_name}': object not found")

    if not meshes_to_load:
        print("[Splatting] No valid meshes to load")
        return

    # Create per-instance state for each mesh
    total_points = 0
    total_blocks = 0
    _scene.clear()

    for obj in meshes_to_load:
        mesh = obj.data
        print(f"[Splatting] Loading '{obj.name}' ({len(mesh.vertices)} vertices)")

        positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree = read_ply_attributes(mesh)

        # Clip low-opacity splats
        clip_val = splatting_props.clip_alpha
        if clip_val > 0.0:
            mask = opacities.ravel() >= clip_val
            kept = mask.sum()
            if kept < len(positions):
                positions = positions[mask]
                colors = colors[mask]
                opacities = opacities[mask]
                scales = scales[mask]
                rotations = rotations[mask]
                raw_dc = raw_dc[mask]
                if sh_coeffs is not None:
                    sh_coeffs = sh_coeffs[mask]
                print(f"[Splatting]  Clipped {len(mask) - kept} splats below alpha {clip_val}")

        # Clip small splats by average scale
        size_val = splatting_props.clip_size
        if size_val > 0.0:
            mask = scales.mean(axis=1) >= size_val
            kept = mask.sum()
            if kept < len(positions):
                positions = positions[mask]
                colors = colors[mask]
                opacities = opacities[mask]
                scales = scales[mask]
                rotations = rotations[mask]
                raw_dc = raw_dc[mask]
                if sh_coeffs is not None:
                    sh_coeffs = sh_coeffs[mask]
                print(f"[Splatting]  Clipped {len(mask) - kept} splats below size {size_val}")

        inst = SplattingState()
        inst.positions = positions
        inst.display_colors = colors
        inst.raw_dc = raw_dc
        inst.sh_coeffs = sh_coeffs
        inst.sh_degree = sh_degree
        inst.opacities = opacities
        inst.scales = scales
        inst.rotations = rotations
        inst.point_count = len(positions)
        inst.target_mesh = obj

        # Build spatial index in world space with optional grid offset
        mat = np.array(obj.matrix_world, dtype=np.float32)
        ones = np.ones((len(positions), 1), dtype=np.float32)
        positions_h = np.concatenate([positions, ones], axis=1)
        positions_world = (positions_h @ mat.T)[:, :3]
        # Frozen AABB from bound_box (same source as pre-render grid) so the
        # grid overlay exactly matches the pre-render view at render start.
        bbox_local = np.array(obj.bound_box, dtype=np.float32)
        ones_8 = np.ones((8, 1), dtype=np.float32)
        corners_h = np.concatenate([bbox_local, ones_8], axis=1)
        corners_w = (corners_h @ mat.T)[:, :3]
        inst._frozen_grid_min = corners_w.min(axis=0).copy()
        inst._frozen_grid_max = corners_w.max(axis=0).copy()
        inst._frozen_positions_min = positions_world.min(axis=0).copy()
        inst._frozen_positions_max = positions_world.max(axis=0).copy()
        offset = splatting_props.block_offset
        # Adjust block offset so blocks share the same spatial origin as the
        # grid (which is based on bound_box AABB).  Without this the block
        # division and grid lines drift apart under rotation since
        # positions_world.min ≠ (bound_box @ mat.T).min.
        grid_ref = corners_w.min(axis=0)  # bound_box world AABB min
        pos_ref = positions_world.min(axis=0)
        offset_arr = np.array(offset, dtype=np.float32)
        adjusted = grid_ref + offset_arr - pos_ref
        spatial = build_spatial_index(positions_world, block_size=block_size,
                                      origin_offset=adjusted, use_parallel=True)
        inst.block_indices = spatial['block_indices']
        inst.grid_dims = spatial['grid_dims']
        inst.block_centers = spatial['block_centers']
        inst.block_radii = spatial['block_radii']
        inst.block_bounds = spatial['block_bounds']
        inst.block_splat_indices = spatial['block_splat_indices']
        inst.block_count = len(spatial['unique_blocks'])

        # Freeze originals for delta-based transform during rendering
        inst._orig_block_centers = spatial['block_centers'].copy()
        inst._orig_block_radii = spatial['block_radii'].copy()
        inst._orig_block_bounds = spatial['block_bounds'].copy()
        inst._orig_model_matrix = np.array(obj.matrix_world, dtype=np.float32)
        inst._orig_model_inv = np.linalg.inv(inst._orig_model_matrix)

        # SH coefficients stay in CPU arrays for VBO batch building
        pass

        total_points += inst.point_count
        total_blocks += inst.block_count
        _scene.instances.append(inst)

    # Restore the mode that was active before we forced OBJECT mode
    if prev_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=prev_mode)

    # Initialize GPU renderer (shared shader)
    from .gpu_renderer import init_renderer
    success = init_renderer()
    if not success:
        _scene.clear()
        return

    # Register draw handler, remembering which 3D view area was active
    _scene._preferred_area = context.area
    _scene._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_handler, (), 'WINDOW', 'POST_VIEW'
    )

    # Mark as rendering
    _scene.is_rendering = True
    splatting_props.is_rendering = True
    splatting_props.point_count = total_points
    splatting_props.block_count = total_blocks

    _tag_view3d_redraw()

    # --- Irradiance probe interpolation setup ---
    global _irradiance_objects_dirty
    _irradiance_objects_dirty = True
    collect_irradiance_objects()
    _refresh_probe_cache()
    global _irradiance_pre_handle, _irradiance_update_idx
    _irradiance_update_idx = 0
    if _irradiance_pre_handle is None:
        _irradiance_pre_handle = bpy.types.SpaceView3D.draw_handler_add(
            _irradiance_pre_draw, (), 'WINDOW', 'PRE_VIEW'
        )


def _unregister_irradiance_pre_draw():
    """Remove the PRE_VIEW irradiance draw handler if registered."""
    global _irradiance_pre_handle
    if _irradiance_pre_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_irradiance_pre_handle, 'WINDOW')
        except Exception:
            pass
        _irradiance_pre_handle = None


def stop_render(context):
    """Stop rendering and clear all splatting data."""
    global _scene

    _stop_redraw_timer()

    # Unregister POST_VIEW draw handler
    if _scene._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_scene._draw_handle, 'WINDOW')
        except Exception:
            pass
        _scene._draw_handle = None

    # Unregister PRE_VIEW irradiance handler
    _unregister_irradiance_pre_draw()

    # Release GPU resources
    from .gpu_renderer import release_renderer
    release_renderer()

    _scene.is_rendering = False
    _scene.clear()
    context.scene.splatting_properties.is_rendering = False
    context.scene.splatting_properties.point_count = 0
    context.scene.splatting_properties.block_count = 0

    _tag_view3d_redraw()


def _emergency_stop():
    """Safe cleanup when draw handler detects a freed Area (e.g. new file loaded)."""
    global _scene
    _stop_redraw_timer()
    if _scene._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_scene._draw_handle, 'WINDOW')
        except Exception:
            pass
        _scene._draw_handle = None
    _unregister_irradiance_pre_draw()
    _scene.is_rendering = False
    _scene.clear()
    try:
        from .gpu_renderer import release_renderer
        release_renderer()
    except Exception:
        pass
    # Defer scene property reset — may be called from draw context where
    # writing to ID properties is not allowed.
    if not bpy.app.timers.is_registered(_deferred_reset_props):
        bpy.app.timers.register(_deferred_reset_props, first_interval=0.0)


def _deferred_reset_props():
    """One-shot timer to reset scene properties outside of draw context."""
    for scn in bpy.data.scenes:
        scn.splatting_properties.is_rendering = False
        scn.splatting_properties.point_count = 0
        scn.splatting_properties.block_count = 0
    return None


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------
def _sort_blocks(inst, cp_np, block_range):
    """Sort splats in given blocks far-to-near by distance² plus scale sum."""
    pos = inst.positions
    for i in block_range:
        indices = inst.block_splat_indices[i]
        if len(indices) < 2:
            continue
        diff = pos[indices] - cp_np
        dists = np.sum(diff * diff, axis=1)
        # dists = dists + inst.scales[indices].sum(axis=1)
        order = np.argsort(dists)[::-1]
        inst.block_splat_indices[i] = indices[order]


def sort_blocks_far_to_near(camera_pos):
    """Sort all instances' blocks far-to-near.

    camera_pos is in world space (Vector or 3-tuple) — transformed to each
    instance's local space.  Returns True if any sorting was performed.
    """
    from mathutils import Vector
    if not isinstance(camera_pos, Vector):
        camera_pos = Vector(camera_pos)
    global _scene
    if not _scene.instances:
        return False

    t0 = time.perf_counter()
    for inst in _scene.instances:
        if inst.point_count == 0:
            continue
        # Transform world camera position to instance local space
        try:
            if inst.target_mesh:
                local_cam = inst.target_mesh.matrix_world.inverted() @ camera_pos
                cp = np.array([local_cam[0], local_cam[1], local_cam[2]], dtype=np.float32)
            else:
                cp = np.array([camera_pos[0], camera_pos[1], camera_pos[2]], dtype=np.float32)
        except ReferenceError:
            continue
        _sort_blocks(inst, cp, range(inst.block_count))
    t = time.perf_counter() - t0
    print(f"[Splatting] Sort far→near: {t*1000:.1f}ms")
    return True


def check_view_changed(state_obj, view_matrix, threshold=0.0001):
    """Check if view changed from previous frame. Used for auto-sort trigger."""
    if state_obj._prev_view_matrix is None:
        state_obj._prev_view_matrix = view_matrix.copy()
        return True
    diff = 0.0
    for i in range(4):
        for j in range(4):
            d = view_matrix[i][j] - state_obj._prev_view_matrix[i][j]
            diff += d * d
    state_obj._prev_view_matrix = view_matrix.copy()
    return diff > threshold


def sort_next_batch(inst, context):
    """Sort next batch of blocks for one instance — visible blocks first."""
    if not inst._sort_active or inst.sorted_up_to >= inst.block_count:
        inst._sort_active = False
        return

    cp_np = inst._camera_pos_np
    if cp_np is None:
        inst._sort_active = False
        return

    _start_redraw_timer()

    SORT_BUDGET_MS = 3.0
    block_range = np.arange(inst.sorted_up_to, inst.block_count, dtype=np.intp)

    # Prioritise visible blocks via vectorised frustum test
    vp = inst._vp_world
    if vp is not None:
        centers = inst.block_centers[block_range]
        radii = inst.block_radii[block_range]
        ones = np.ones((len(block_range), 1), dtype=np.float32)
        centers_h = np.concatenate([centers, ones], axis=1)
        vp_np = np.asarray(vp, dtype=np.float32)
        clip = centers_h @ vp_np.T

        behind = clip[:, 3] <= 0
        margin_x = radii * inst._proj_00
        margin_y = radii * inst._proj_11
        in_frustum = ~behind & (
            (np.abs(clip[:, 0]) < clip[:, 3] + margin_x) &
            (np.abs(clip[:, 1]) < clip[:, 3] + margin_y)
        )
        block_range = np.concatenate([block_range[in_frustum], block_range[~in_frustum]])

    t_start = time.perf_counter()
    sorted_blocks = []
    for idx in block_range:
        _sort_blocks(inst, cp_np, np.array([idx], dtype=np.intp))
        sorted_blocks.append(int(idx))
        if (time.perf_counter() - t_start) * 1000 >= SORT_BUDGET_MS:
            break

    inst.sorted_up_to = sorted_blocks[-1] + 1 if sorted_blocks else inst.sorted_up_to

    inst.clear_block_cache(sorted_blocks)

    if inst.sorted_up_to >= inst.block_count:
        inst._sort_active = False


def _find_first_3dview():
    """Return the first VIEW_3D area across all windows, or None."""
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            for a in win.screen.areas:
                if a.type == 'VIEW_3D':
                    return a
    return None


def _start_redraw_timer():
    if _scene._redraw_timer is not None:
        return
    _scene._redraw_timer = bpy.app.timers.register(
        _redraw_tick, first_interval=1.0 / 60.0)


def _stop_redraw_timer():
    if _scene._redraw_timer is not None:
        try:
            bpy.app.timers.unregister(_scene._redraw_timer)
        except Exception:
            pass
        _scene._redraw_timer = None


def _try_stop_redraw_timer():
    """Stop the redraw timer only if no instance needs sorting."""
    if _scene._redraw_timer is None:
        return
    for inst in _scene.instances:
        if inst._sort_active and inst.sorted_up_to < inst.block_count:
            return
    _stop_redraw_timer()


def _redraw_tick():
    global _scene
    if not _scene.is_rendering:
        _scene._redraw_timer = None
        return None

    # If the preferred area is no longer among any visible 3D view
    # (e.g. user maximized a different window), reassign to the first one.
    if _scene._preferred_area is not None:
        try:
            _ = _scene._preferred_area.type
        except ReferenceError:
            _scene._preferred_area = None

        if _scene._preferred_area is not None:
            found = any(
                a == _scene._preferred_area
                for wm in bpy.data.window_managers
                for win in wm.windows
                for a in win.screen.areas
                if a.type == 'VIEW_3D'
            )
            if not found:
                _scene._preferred_area = _find_first_3dview()

    if _scene._preferred_area is None:
        _scene._preferred_area = _find_first_3dview()

    # Check if any instance still needs sorting
    for inst in _scene.instances:
        if inst._sort_active and inst.sorted_up_to < inst.block_count:
            for wm in bpy.data.window_managers:
                for win in wm.windows:
                    for area in win.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
            return 1.0 / 24.0
    # No sorting needed, stop the timer
    _scene._redraw_timer = None
    return None
