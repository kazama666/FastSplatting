import concurrent.futures
import os
import time
import numpy as np
import bpy
from bpy import types
import gpu
from gpu.types import (
    GPUBatch,
    GPUVertBuf,
    GPUVertFormat,
    GPUIndexBuf,
)


# LOD configuration (disabled)
LOD_LEVELS = 3
LOD_THRESHOLDS = [100, 30, 10]


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
        self.colors = None         # (N, 3) float32
        self.opacities = None      # (N, 1) float32
        self.scales = None         # (N, 3) float32
        self.rotations = None      # (N, 4) float32 (quaternion)

        # Spatial indexing
        self.block_indices = None
        self.block_bounds = None   # (M, 2, 3) min/max for each block
        self.block_centers = None  # (M, 3) center of each block
        self.block_radii = None    # (M,) bounding sphere radius
        self.block_splat_indices = None  # list of np.ndarray

        # LOD data (disabled)
        self.lod_data = [None] * LOD_LEVELS

        # Per-frame stats (updated by renderer during draw)
        self.displayed_block_count = 0
        self.displayed_splat_count = 0

        # Auto-sort state (alpha blend incremental sorting)
        self.sorted_up_to = 0
        self._sort_active = False
        self._camera_pos_np = None
        self._prev_view_matrix = None
        self._vp_matrix = None
        self._proj_00 = 0.0
        self._proj_11 = 0.0

        # GPU batch cache (per-instance)
        self._batch_cache = {}

    def clear(self):
        self.target_mesh = None
        self.point_count = 0
        self.block_count = 0
        self.positions = None
        self.colors = None
        self.opacities = None
        self.scales = None
        self.rotations = None
        self.block_indices = None
        self.block_bounds = None
        self.block_centers = None
        self.block_radii = None
        self.block_splat_indices = None
        self.lod_data = [None] * LOD_LEVELS
        self.sorted_up_to = 0
        self._sort_active = False
        self._camera_pos_np = None
        self._prev_view_matrix = None
        self._vp_matrix = None
        self._proj_00 = 0.0
        self._proj_11 = 0.0
        self._batch_cache = {}

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

    def _get_or_create_block_batch(self, block_idx, lod_level):
        """Get cached batch for a block at given LOD level"""
        cache_key = ('b', block_idx, lod_level)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        block_splat_indices = self.block_splat_indices[block_idx]
        lod_factor = 4 ** lod_level
        count = max(1, len(block_splat_indices) // lod_factor)
        lod_indices = block_splat_indices[:count]
        if len(lod_indices) == 0:
            return None

        positions = self.positions[lod_indices]
        colors = self.colors[lod_indices]
        opacities = self.opacities[lod_indices]
        scales = self.scales[lod_indices]
        rotations = self.rotations[lod_indices]

        batch = self._build_billboard_batch_for(positions, colors, opacities, scales, rotations)
        self._batch_cache[cache_key] = batch
        return batch

    def _get_or_create_fallback_batch(self):
        """Fallback single batch — top 1/16 per block"""
        cache_key = ('fallback', 0)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        lod_indices = []
        for block_idx in range(self.block_count):
            indices = self.block_splat_indices[block_idx]
            count = max(1, len(indices) // 16)
            lod_indices.extend(indices[:count])
        if len(lod_indices) == 0:
            return None

        positions = self.positions[lod_indices]
        colors = self.colors[lod_indices]
        opacities = self.opacities[lod_indices]
        scales = self.scales[lod_indices]
        rotations = self.rotations[lod_indices]

        batch = self._build_billboard_batch_for(positions, colors, opacities, scales, rotations)
        self._batch_cache[cache_key] = batch
        return batch

    def _build_billboard_batch_for(self, positions, colors, opacities, scales, rotations):
        """Build a batch, filtering invisible splats."""
        N = len(positions)
        if N == 0:
            return None
        # Filter out low-opacity and tiny splats (view-independent)
        opacity_ok = opacities.ravel() >= 0.005
        scale_ok = scales.max(axis=1) >= 0.0003
        mask = opacity_ok & scale_ok
        if not mask.all():
            positions = positions[mask]
            colors = colors[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            rotations = rotations[mask]
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

        vbo = GPUVertBuf(fmt, len=N * 4)
        vbo.attr_fill(id="quad_coord", data=np.tile(quad_coords, (N, 1)))
        vbo.attr_fill(id="inst_position", data=np.repeat(positions, 4, axis=0))
        vbo.attr_fill(id="inst_color", data=np.repeat(colors, 4, axis=0))
        vbo.attr_fill(id="inst_opacity", data=np.repeat(opacities.ravel(), 4))
        vbo.attr_fill(id="inst_cov_a", data=np.repeat(cov_a, 4, axis=0))
        vbo.attr_fill(id="inst_cov_b", data=np.repeat(cov_b, 4, axis=0))

        ibo = GPUIndexBuf(type='TRIS', seq=indices)
        return GPUBatch(type='TRIS', buf=vbo, elem=ibo)

    def clear_cache(self):
        """Clear batch cache (call after splat index reordering)."""
        self._batch_cache = {}

    def clear_block_cache(self, block_indices):
        """Clear cached batches only for specific blocks."""
        for block_idx in block_indices:
            for lod in range(LOD_LEVELS):
                self._batch_cache.pop(('b', block_idx, lod), None)
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
# Scene — container for all instances + global render state
# ---------------------------------------------------------------------------
class SplattingScene:
    def __init__(self):
        self.instances = []          # list of SplattingState (parallel to collection)
        self.is_rendering = False
        self._draw_handle = None
        self._target_area = None
        self._target_space = None     # for background color restore
        self._redraw_timer = None

        # Aggregated stats (set by renderer each frame)
        self.displayed_block_count = 0
        self.displayed_splat_count = 0

    def clear(self):
        for inst in self.instances:
            inst.clear()
        self.instances.clear()
        self.is_rendering = False
        self._draw_handle = None
        self._target_area = None
        self._target_space = None
        self._redraw_timer = None


_scene = SplattingScene()


def get_state():
    """Return the global SplattingScene."""
    return _scene


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class SplattingProperties(types.PropertyGroup):
    is_rendering: bpy.props.BoolProperty(default=False)
    point_count: bpy.props.IntProperty(default=0)
    block_count: bpy.props.IntProperty(default=0)
    lod_enabled: bpy.props.BoolProperty(default=False)
    lod_bias: bpy.props.FloatProperty(default=1.0, min=0.25, max=4.0, step=0.25,
        description="LOD threshold multiplier. Higher = more aggressive LOD = faster but lower quality")
    sort_near_to_far: bpy.props.BoolProperty(default=False)
    block_size: bpy.props.FloatProperty(default=0.5, min=0.1, max=5.0, step=0.1,
        description="Spatial block size for culling. Larger = fewer blocks, fewer draw calls. Requires restart")
    active_instance_index: bpy.props.IntProperty(default=0,
        description="Active index in the splat instances list")
    anim_start_frame: bpy.props.IntProperty(default=1,
        description="First frame of animation export range")
    anim_end_frame: bpy.props.IntProperty(default=250,
        description="Last frame of animation export range")
    anim_output_path: bpy.props.StringProperty(default="//", subtype='DIR_PATH',
        description="Directory to save exported frames")
    anim_force_sort: bpy.props.BoolProperty(default=True,
        description="Sort blocks far-to-near every frame when camera changes")
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


def _on_load_post(dummy):
    """Recover mesh_uid after loading a .blend file (session_uid not persisted)."""
    for scn in bpy.data.scenes:
        for item in scn.splatting_instances:
            obj = bpy.data.objects.get(item.mesh_name)
            if obj:
                item.mesh_uid = obj.session_uid


def register():
    bpy.utils.register_class(SplattingProperties)
    bpy.utils.register_class(SplattingInstanceItem)
    bpy.types.Scene.splatting_properties = bpy.props.PointerProperty(type=SplattingProperties)
    bpy.types.Scene.splatting_instances = bpy.props.CollectionProperty(type=SplattingInstanceItem)

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


def unregister():
    _stop_redraw_timer()
    if _scene._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_scene._draw_handle, 'WINDOW')
        except Exception:
            pass
        _scene._draw_handle = None
    try:
        from .gpu_renderer import release_renderer
        release_renderer()
    except Exception:
        pass
    _scene.clear()

    # Clean up msgbus subscription and handlers
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

    if hasattr(bpy.types.Scene, "splatting_instances"):
        del bpy.types.Scene.splatting_instances
    bpy.utils.unregister_class(SplattingInstanceItem)
    if hasattr(bpy.types.Scene, "splatting_properties"):
        del bpy.types.Scene.splatting_properties
    bpy.utils.unregister_class(SplattingProperties)


# ---------------------------------------------------------------------------
# PLY mesh reading
# ---------------------------------------------------------------------------
def read_ply_attributes(mesh):
    """Read Gaussian Splatting attributes from mesh vertex data"""
    vertex_count = len(mesh.vertices)

    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    for i, vert in enumerate(mesh.vertices):
        positions[i] = vert.co

    attributes = mesh.attributes
    attr_names = [attr.name for attr in attributes]

    def find_attr(prefix):
        for name in attr_names:
            if name.startswith(prefix):
                return name
        return None

    # Colors
    colors = np.zeros((vertex_count, 3), dtype=np.float32)
    dc0 = find_attr('f_dc_0')
    dc1 = find_attr('f_dc_1')
    dc2 = find_attr('f_dc_2')
    if dc0 and dc1 and dc2:
        raw_colors = np.zeros((vertex_count, 3), dtype=np.float32)
        raw_colors[:, 0] = np.array([v.value for v in attributes[dc0].data], dtype=np.float32)
        raw_colors[:, 1] = np.array([v.value for v in attributes[dc1].data], dtype=np.float32)
        raw_colors[:, 2] = np.array([v.value for v in attributes[dc2].data], dtype=np.float32)
        linear = 1.0 / (1.0 + np.exp(-raw_colors))
        colors = np.power(linear, 2.2)
    else:
        colors[:, :] = [1.0, 1.0, 1.0]

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

    return positions, colors, opacities, scales, rotations


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


def build_spatial_index(positions, block_size=1.0, use_parallel=True):
    """Build spatial index for fast culling."""
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    dimensions = max_coords - min_coords
    grid_dims = np.ceil(dimensions / block_size).astype(int) + 1

    block_idx = ((positions - min_coords) / block_size).astype(int)
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
# Draw handler
# ---------------------------------------------------------------------------
def _draw_handler():
    context = bpy.context
    area = context.area
    if area and area.type == 'VIEW_3D' and area == _scene._target_area:
        from .gpu_renderer import draw_splatting
        draw_splatting(context)
        area.tag_redraw()


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
    _scene.instances.clear()

    for obj in meshes_to_load:
        mesh = obj.data
        print(f"[Splatting] Loading '{obj.name}' ({len(mesh.vertices)} vertices)")

        positions, colors, opacities, scales, rotations = read_ply_attributes(mesh)

        inst = SplattingState()
        inst.positions = positions
        inst.colors = colors
        inst.opacities = opacities
        inst.scales = scales
        inst.rotations = rotations
        inst.point_count = len(positions)
        inst.target_mesh = obj

        # Build spatial index
        spatial = build_spatial_index(positions, block_size=block_size, use_parallel=True)
        inst.block_indices = spatial['block_indices']
        inst.block_centers = spatial['block_centers']
        inst.block_radii = spatial['block_radii']
        inst.block_bounds = spatial['block_bounds']
        inst.block_splat_indices = spatial['block_splat_indices']
        inst.block_count = len(spatial['unique_blocks'])
        inst.lod_data = None

        total_points += inst.point_count
        total_blocks += inst.block_count
        _scene.instances.append(inst)

    # Initialize GPU renderer (shared shader)
    from .gpu_renderer import init_renderer
    success = init_renderer()
    if not success:
        _scene.instances.clear()
        return

    # Find the first 3D view area
    _scene._target_area = None
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            _scene._target_area = area
            break

    # Register draw handler
    _scene._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_handler, (), 'WINDOW', 'POST_VIEW'
    )

    # Mark as rendering
    _scene.is_rendering = True
    splatting_props.is_rendering = True
    splatting_props.point_count = total_points
    splatting_props.block_count = total_blocks

    _tag_view3d_redraw()


def stop_render(context):
    """Stop rendering and clear all splatting data."""
    global _scene

    _stop_redraw_timer()

    # Unregister draw handler
    if _scene._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_scene._draw_handle, 'WINDOW')
        except Exception:
            pass
        _scene._draw_handle = None

    # Release GPU resources
    from .gpu_renderer import release_renderer
    release_renderer()

    _scene.clear()
    context.scene.splatting_properties.is_rendering = False
    context.scene.splatting_properties.point_count = 0
    context.scene.splatting_properties.block_count = 0

    _tag_view3d_redraw()


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
        dists = dists + inst.scales[indices].sum(axis=1)
        order = np.argsort(dists)[::-1]
        inst.block_splat_indices[i] = indices[order]


def sort_blocks_far_to_near(camera_pos):
    """Sort all instances' blocks far-to-near.

    camera_pos is in world space — transformed to each instance's local space.
    Returns True if any sorting was performed.
    """
    global _scene
    if not _scene.instances:
        return False

    t0 = time.perf_counter()
    for inst in _scene.instances:
        if inst.point_count == 0:
            continue
        # Transform world camera position to instance local space
        if inst.target_mesh:
            local_cam = inst.target_mesh.matrix_world.inverted() @ camera_pos
            cp = np.array([local_cam[0], local_cam[1], local_cam[2]], dtype=np.float32)
        else:
            cp = np.array([camera_pos[0], camera_pos[1], camera_pos[2]], dtype=np.float32)
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

    SORT_BUDGET_MS = 5.0
    block_range = np.arange(inst.sorted_up_to, inst.block_count, dtype=np.intp)

    # Prioritise visible blocks via vectorised frustum test
    vp = inst._vp_matrix
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
        _try_stop_redraw_timer()


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
    # Check if any instance still needs sorting
    any_active = False
    for inst in _scene.instances:
        if inst._sort_active and inst.sorted_up_to < inst.block_count:
            any_active = True
            break
    if not any_active:
        _scene._redraw_timer = None
        return None
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    return 1.0 / 24.0
