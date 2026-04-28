import concurrent.futures
import os
import time
import numpy as np
import bpy
from bpy import types


# LOD configuration
LOD_LEVELS = 3  # 0=full, 1=medium, 2=low
LOD_THRESHOLDS = [100, 30, 10]  # Screen pixels - draw if block > threshold


# Global splatting state
class SplattingState:
    def __init__(self):
        self.is_rendering = False
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
        self.block_radii = None    # (M,) bounding sphere radius for each block
        self.block_splat_indices = None  # list of np.ndarray, splat indices per block

        # LOD data: for each LOD level, store (positions, colors, opacities, scales, rotations) per block
        self.lod_data = [None] * LOD_LEVELS

        # Per-frame stats (updated by renderer during draw)
        self.displayed_block_count = 0
        self.displayed_splat_count = 0

        # Auto-sort state (alpha blend incremental sorting)
        self.sorted_up_to = 0
        self._sort_active = False
        self._camera_pos_np = None
        self._prev_view_matrix = None
        self._vp_matrix = None        # current VP matrix from renderer (for visibility)
        self._proj_00 = 0.0
        self._proj_11 = 0.0
        self._redraw_timer = None

        # Handle for removal
        self._draw_handle = None

        # Restrict drawing to the first 3D view only
        self._target_area = None

    def clear(self):
        self.is_rendering = False
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
        self._target_area = None
        self._redraw_timer = None
        self._draw_handle = None


_state = SplattingState()


class SplattingProperties(types.PropertyGroup):
    is_rendering: bpy.props.BoolProperty(default=False)
    point_count: bpy.props.IntProperty(default=0)
    block_count: bpy.props.IntProperty(default=0)
    lod_enabled: bpy.props.BoolProperty(default=False)
    lod_bias: bpy.props.FloatProperty(default=1.0, min=0.25, max=4.0, step=0.25,
        description="LOD threshold multiplier. Higher = more aggressive LOD = faster but lower quality")
    sort_near_to_far: bpy.props.BoolProperty(default=False)
    quad_scale: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0)
    block_size: bpy.props.FloatProperty(default=0.5, min=0.1, max=5.0, step=0.1,
        description="Spatial block size for culling. Larger = fewer blocks, fewer draw calls. Requires restart")
    sort_blocks_per_frame: bpy.props.IntProperty(default=32, min=0, max=200,
        description="Number of blocks to sort per frame during auto-sort (0 = disable)")
    color_gamma: bpy.props.FloatProperty(default=1.0, min=0.0, max=5.0, step=0.1,
        description="Gamma correction")
    color_hue: bpy.props.FloatProperty(default=0.0, min=-1.0, max=1.0, step=0.01,
        description="Hue shift (-1..1)")
    color_saturation: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Saturation (0=gray, 1=original)")
    color_brightness: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0, step=0.01,
        description="Brightness multiplier")
    color_tint: bpy.props.FloatVectorProperty(default=(1.0, 1.0, 1.0), min=0.0, max=2.0,
        subtype='COLOR', size=3, description="Overall color tint (multiply)")
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
    ui_stats_expanded: bpy.props.BoolProperty(default=True,
        description="Toggle statistics section")


def register():
    bpy.utils.register_class(SplattingProperties)
    bpy.types.Scene.splatting_properties = bpy.props.PointerProperty(type=SplattingProperties)


def unregister():
    # Clean up any active handlers/resources before module is reloaded
    _stop_redraw_timer()
    if _state._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_state._draw_handle, 'WINDOW')
        except Exception:
            pass
        _state._draw_handle = None
    try:
        from .gpu_renderer import release_renderer
        release_renderer()
    except Exception:
        pass
    _state.clear()

    if hasattr(bpy.types.Scene, "splatting_properties"):
        del bpy.types.Scene.splatting_properties
    bpy.utils.unregister_class(SplattingProperties)


def read_ply_attributes(mesh):
    """Read Gaussian Splatting attributes from mesh vertex data"""
    vertex_count = len(mesh.vertices)

    # Get base position
    positions = np.zeros((vertex_count, 3), dtype=np.float32)
    for i, vert in enumerate(mesh.vertices):
        positions[i] = vert.co

    # Parse attribute names to find splatting data
    attributes = mesh.attributes
    attr_names = [attr.name for attr in attributes]

    def find_attr(prefix):
        """Find attribute matching prefix (e.g., 'f_dc_0')"""
        for name in attr_names:
            if name.startswith(prefix):
                return name
        return None

    # Extract colors (f_dc_0, f_dc_1, f_dc_2)
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

    # Extract opacity
    opacities = np.ones((vertex_count, 1), dtype=np.float32) * 0.5
    opacity_attr = find_attr('opacity')
    if opacity_attr:
        opacities[:, 0] = np.array([v.value for v in attributes[opacity_attr].data], dtype=np.float32)
        opacities[:, 0] = 1.0 / (1.0 + np.exp(-opacities[:, 0]))  # sigmoid
    else:
        pass

    # Extract scales (scale_0, scale_1, scale_2)
    scales = np.ones((vertex_count, 3), dtype=np.float32) * 0.01
    scale0 = find_attr('scale_0')
    scale1 = find_attr('scale_1')
    scale2 = find_attr('scale_2')

    if scale0 and scale1 and scale2:
        scales[:, 0] = np.exp(np.array([v.value for v in attributes[scale0].data], dtype=np.float32))
        scales[:, 1] = np.exp(np.array([v.value for v in attributes[scale1].data], dtype=np.float32))
        scales[:, 2] = np.exp(np.array([v.value for v in attributes[scale2].data], dtype=np.float32))
    else:
        pass

    # Extract rotations (rot_0, rot_1, rot_2, rot_3) - quaternion
    rotations = np.zeros((vertex_count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w = 1 (identity quaternion)
    rot0 = find_attr('rot_0')
    rot1 = find_attr('rot_1')
    rot2 = find_attr('rot_2')
    rot3 = find_attr('rot_3')

    if rot0 and rot1 and rot2 and rot3:
        rotations[:, 0] = np.array([v.value for v in attributes[rot0].data], dtype=np.float32)
        rotations[:, 1] = np.array([v.value for v in attributes[rot1].data], dtype=np.float32)
        rotations[:, 2] = np.array([v.value for v in attributes[rot2].data], dtype=np.float32)
        rotations[:, 3] = np.array([v.value for v in attributes[rot3].data], dtype=np.float32)
    else:
        pass

    return positions, colors, opacities, scales, rotations


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
    """
    Build spatial index for fast culling - vectorized numpy + parallel block stats.

    Uses ThreadPoolExecutor for parallelism: numpy C kernels release GIL,
    so threads run in parallel on compute-bound operations. On Linux/Windows
    this can be swapped to ProcessPoolExecutor for true multi-process.
    """
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    dimensions = max_coords - min_coords
    grid_dims = np.ceil(dimensions / block_size).astype(int) + 1

    # Assign each point to a block (fully vectorized)
    block_idx = ((positions - min_coords) / block_size).astype(int)
    block_ids = (block_idx[:, 0] + block_idx[:, 1] * grid_dims[0] +
                 block_idx[:, 2] * grid_dims[0] * grid_dims[1])

    # Vectorized block grouping via argsort (replaces O(N) Python loop)
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    block_count = len(unique_blocks)
    sorted_idx = np.argsort(inverse)
    split_pts = np.where(np.diff(inverse[sorted_idx]) != 0)[0] + 1
    block_splat_indices_np = np.split(sorted_idx, split_pts)
    block_splat_indices = [idx.astype(np.int32) for idx in block_splat_indices_np]

    # Per-block stats: center, radius, bounds
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


def build_lod_data(positions, colors, opacities, scales, rotations, block_splat_indices):
    """Build LOD data for each block"""
    lod_data = []

    # LOD 0: Full detail
    lod0_data = {
        'positions': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations,
        'block_splat_indices': block_splat_indices,
    }
    lod_data.append(lod0_data)

    # LOD 1: 1/4 detail (every 4th splat)
    for stride in [4, 16]:
        lod_positions = positions[::stride]
        lod_colors = colors[::stride]
        lod_opacities = opacities[::stride]
        lod_scales = scales[::stride]
        lod_rotations = rotations[::stride]

        lod_splat_indices = [list(range(0, len(idx), stride)) for idx in block_splat_indices]

        lod_data.append({
            'positions': lod_positions,
            'colors': lod_colors,
            'opacities': lod_opacities,
            'scales': lod_scales,
            'rotations': lod_rotations,
            'block_splat_indices': lod_splat_indices,
        })

    return lod_data


def _draw_handler():
    """Single POST_VIEW callback — draws splats only in the first 3D view."""
    context = bpy.context
    area = context.area
    if area and area.type == 'VIEW_3D' and area == _state._target_area:
        from .gpu_renderer import draw_splatting
        draw_splatting(context)
        area.tag_redraw()


def _tag_view3d_redraw():
    """Find the 3D viewport and force a redraw."""
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            screen = win.screen
            if not screen:
                continue
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
                    return


def start_render(context):
    """Initialize splatting data from selected mesh and start rendering"""
    global _state

    # Clear previous state first
    if _state.is_rendering:
        stop_render(context)

    mesh_name = context.scene.splatting_target_mesh
    if not mesh_name:
        return

    obj = bpy.data.objects.get(mesh_name)
    if not obj or obj.type != 'MESH':
        return

    mesh = obj.data
    if not mesh:
        return

    positions, colors, opacities, scales, rotations = read_ply_attributes(mesh)

    # Store in state
    _state.positions = positions
    _state.colors = colors
    _state.opacities = opacities
    _state.scales = scales
    _state.rotations = rotations
    _state.point_count = len(positions)
    _state.target_mesh = obj

    # Build spatial index (uses multiprocessing for block stats)
    block_size = context.scene.splatting_properties.block_size
    spatial = build_spatial_index(positions, block_size=block_size, use_parallel=True)

    _state.block_indices = spatial['block_indices']
    _state.block_centers = spatial['block_centers']
    _state.block_radii = spatial['block_radii']
    _state.block_bounds = spatial['block_bounds']
    _state.block_splat_indices = spatial['block_splat_indices']
    _state.block_count = len(spatial['unique_blocks'])

    # LOD branching disabled: size-sorting for LOD is unnecessary
    # splat_size = scales.max(axis=1)
    # for i in range(_state.block_count):
    #     indices = _state.block_splat_indices[i]
    #     if len(indices) > 1:
    #         order = np.argsort(splat_size[np.array(indices)])[::-1]
    #         _state.block_splat_indices[i] = np.array(indices)[order].tolist()

    # Build LOD data — disabled
    # _state.lod_data = build_lod_data(positions, colors, opacities, scales, rotations, _state.block_splat_indices)
    _state.lod_data = None

    # Initialize GPU renderer
    from .gpu_renderer import init_renderer
    success = init_renderer(positions, colors, opacities, scales, rotations)

    if not success:
        return

    # Find the first 3D view area — only draw in this one
    _state._target_area = None
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            _state._target_area = area
            break

    # Single POST_VIEW handler — draws only in the target area
    _state._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_handler, (), 'WINDOW', 'POST_VIEW'
    )

    # Mark as rendering
    _state.is_rendering = True
    context.scene.splatting_properties.is_rendering = True
    context.scene.splatting_properties.point_count = _state.point_count
    context.scene.splatting_properties.block_count = _state.block_count

    # Force 3D view to redraw so splats appear immediately
    _tag_view3d_redraw()


def stop_render(context):
    """Stop rendering and clear splatting data"""
    global _state

    _stop_redraw_timer()

    # Unregister draw handler
    if _state._draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_state._draw_handle, 'WINDOW')
        except:
            pass
        _state._draw_handle = None

    # Release GPU resources
    from .gpu_renderer import release_renderer
    release_renderer()

    _state.clear()
    context.scene.splatting_properties.is_rendering = False
    context.scene.splatting_properties.point_count = 0
    context.scene.splatting_properties.block_count = 0

    # Force 3D view to redraw so splats disappear immediately
    _tag_view3d_redraw()


def get_state():
    """Get the current splatting state"""
    return _state


def get_lod_thresholds():
    """Get LOD thresholds"""
    return LOD_THRESHOLDS


def get_lod_levels():
    """Get number of LOD levels"""
    return LOD_LEVELS


def sort_blocks_far_to_near(camera_pos):
    """Sort splats inside each block by distance from camera (far to near).
    Returns True if sorting was performed."""
    global _state
    if _state.point_count == 0:
        return False

    t0 = time.perf_counter()
    pos = _state.positions
    cp = np.array([camera_pos[0], camera_pos[1], camera_pos[2]], dtype=np.float32)
    for i in range(_state.block_count):
        indices = _state.block_splat_indices[i]
        if len(indices) < 2:
            continue
        diff = pos[indices] - cp
        dists = np.sum(diff * diff, axis=1)
        order = np.argsort(dists)[::-1]  # far to near
        _state.block_splat_indices[i] = indices[order]

    # Rebuild LOD data — disabled
    # _state.lod_data = build_lod_data(
    #     _state.positions, _state.colors, _state.opacities,
    #     _state.scales, _state.rotations, _state.block_splat_indices)

    t = time.perf_counter() - t0
    print(f"[Splatting] Sort far→near: {t*1000:.1f}ms")
    return True


def check_view_changed(view_matrix, threshold=0.0001):
    """Check if view (position + rotation) changed from previous frame. Used for auto-sort trigger."""
    global _state
    if _state._prev_view_matrix is None:
        _state._prev_view_matrix = view_matrix.copy()
        return True
    diff = 0.0
    for i in range(4):
        for j in range(4):
            d = view_matrix[i][j] - _state._prev_view_matrix[i][j]
            diff += d * d
    _state._prev_view_matrix = view_matrix.copy()
    return diff > threshold


def sort_next_batch(context):
    """Sort next batch of blocks — visible blocks first, then culled ones."""
    global _state
    if not _state._sort_active or _state.sorted_up_to >= _state.block_count:
        _state._sort_active = False
        _stop_redraw_timer()
        return

    per_frame = context.scene.splatting_properties.sort_blocks_per_frame
    if per_frame <= 0:
        _state._sort_active = False
        _stop_redraw_timer()
        return

    cp_np = _state._camera_pos_np
    if cp_np is None:
        _state._sort_active = False
        _stop_redraw_timer()
        return

    _start_redraw_timer()

    end = min(_state.sorted_up_to + per_frame, _state.block_count)
    pos = _state.positions
    block_range = np.arange(_state.sorted_up_to, end, dtype=np.intp)

    # Prioritise visible blocks via vectorised frustum test
    vp = _state._vp_matrix
    if vp is not None:
        centers = _state.block_centers[block_range]
        radii = _state.block_radii[block_range]
        ones = np.ones((len(block_range), 1), dtype=np.float32)
        centers_h = np.concatenate([centers, ones], axis=1)
        vp_np = np.asarray(vp, dtype=np.float32)
        clip = centers_h @ vp_np.T  # (N, 4)

        behind = clip[:, 3] <= 0
        margin_x = radii * _state._proj_00
        margin_y = radii * _state._proj_11
        in_frustum = ~behind & (
            (np.abs(clip[:, 0]) < clip[:, 3] + margin_x) &
            (np.abs(clip[:, 1]) < clip[:, 3] + margin_y)
        )
        block_range = np.concatenate([block_range[in_frustum], block_range[~in_frustum]])

    for i in block_range:
        indices = _state.block_splat_indices[i]
        if len(indices) < 2:
            continue
        diff = pos[indices] - cp_np
        dists = np.sum(diff * diff, axis=1)
        order = np.argsort(dists)[::-1]
        _state.block_splat_indices[i] = indices[order]

    _state.sorted_up_to = int(end)

    from .gpu_renderer import get_renderer
    get_renderer().clear_block_cache(block_range.tolist())

    if _state.sorted_up_to >= _state.block_count:
        _state._sort_active = False
        _stop_redraw_timer()


def _start_redraw_timer():
    """Start a lightweight timer to keep viewport refreshing during auto-sort."""
    if _state._redraw_timer is not None:
        return
    _state._redraw_timer = bpy.app.timers.register(
        _redraw_tick, first_interval=1.0 / 24.0)


def _stop_redraw_timer():
    """Stop the redraw timer."""
    if _state._redraw_timer is not None:
        try:
            bpy.app.timers.unregister(_state._redraw_timer)
        except:
            pass
        _state._redraw_timer = None


def _redraw_tick():
    """Timer: keep 3D view refreshing while auto-sort is active. Does NO sort work."""
    global _state
    if not _state._sort_active or not _state.is_rendering:
        _state._redraw_timer = None
        return None
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    return 1.0 / 24.0
