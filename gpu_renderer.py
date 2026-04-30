import bpy
import gpu
import numpy as np
from gpu.types import GPUUniformBuf, GPUVertBuf, GPUVertFormat, GPUBatch
from gpu import state
from mathutils import Matrix


class SplattingRenderer:
    def __init__(self):
        self.shader = None
        self.initialized = False
        self._matrices_ubo = None

    def init_buffers(self):
        """Create shared shader and UBO (no per-instance data)."""
        if self.initialized:
            return True

        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "v_coordxy")
        vert_out.smooth('VEC3', "v_color")
        vert_out.smooth('FLOAT', "v_opacity")
        vert_out.smooth('VEC3', "v_conic")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.typedef_source("""
struct SplattingMatrices {
    mat4 u_VPMatrix;
    mat4 u_ViewMatrix;
};
""")
        shader_info.uniform_buf(0, 'SplattingMatrices', 'u_Matrices')
        shader_info.push_constant('VEC3', "u_CameraPos")
        shader_info.push_constant('VEC2', "u_FocalParams")
        shader_info.push_constant('VEC2', "u_ViewportSize")
        shader_info.push_constant('FLOAT', "u_QuadScale")
        shader_info.push_constant('FLOAT', "u_Gamma")
        shader_info.push_constant('FLOAT', "u_Hue")
        shader_info.push_constant('FLOAT', "u_Saturation")
        shader_info.push_constant('FLOAT', "u_Brightness")
        shader_info.push_constant('VEC3', "u_Tint")

        shader_info.vertex_in(0, 'VEC2', "quad_coord")
        shader_info.vertex_in(1, 'VEC3', "inst_position")
        shader_info.vertex_in(2, 'VEC3', "inst_color")
        shader_info.vertex_in(3, 'FLOAT', "inst_opacity")
        shader_info.vertex_in(4, 'VEC3', "inst_cov_a")
        shader_info.vertex_in(5, 'VEC3', "inst_cov_b")

        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "FragColor")

        shader_info.vertex_source("""
vec3 computeCov2D(vec4 pos_view, float focal, mat3 cov3d_in, mat3 view_rot) {
    float tz = -pos_view.z;
    float limx = 1.3 * 10.0;
    float limy = 1.3 * 10.0;
    float txtz = -pos_view.x / tz;
    float tytz = -pos_view.y / tz;
    vec4 t = pos_view;
    t.x = min(limx, max(-limx, txtz)) * tz;
    t.y = min(limy, max(-limy, tytz)) * tz;

    mat3 J = mat3(
        focal / tz, 0.0, -(focal * t.x) / (tz * tz),
        0.0, focal / tz, -(focal * t.y) / (tz * tz),
        0.0, 0.0, 0.0
    );

    mat3 W = view_rot;
    mat3 T = W * J;
    mat3 cov = transpose(T) * cov3d_in * T;
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main() {
    v_color = inst_color;
    v_opacity = inst_opacity;

    // Color adjustments: tint -> HSL -> gamma
    v_color *= u_Tint;
    {
        const vec3 k = vec3(0.57735, 0.57735, 0.57735);
        float cosH = cos(u_Hue * 3.14159);
        float sinH = sin(u_Hue * 3.14159);
        v_color = v_color * cosH + cross(k, v_color) * sinH + k * dot(k, v_color) * (1.0 - cosH);

        float lum = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        v_color = mix(vec3(lum), v_color, u_Saturation);

        v_color *= u_Brightness;

        v_color = pow(max(v_color, vec3(0.0)), vec3(1.0 / u_Gamma));
    }

    vec4 pos_view = u_Matrices.u_ViewMatrix * vec4(inst_position, 1.0);
    vec4 pos_clip = u_Matrices.u_VPMatrix * vec4(inst_position, 1.0);

    if (pos_view.z >= -0.001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    float cxx = inst_cov_a.x, cxy = inst_cov_a.y, cxz = inst_cov_a.z;
    float cyy = inst_cov_b.x, cyz = inst_cov_b.y, czz = inst_cov_b.z;
    mat3 cov3D = mat3(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz);
    mat3 view_rot = transpose(mat3(u_Matrices.u_ViewMatrix));
    float focal = u_FocalParams.x;
    vec3 cov2D = computeCov2D(pos_view, focal, cov3D, view_rot);

    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.00001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    float det_inv = 1.0 / det;
    v_conic = vec3(cov2D.z * det_inv, -cov2D.y * det_inv, cov2D.x * det_inv);

    float quad_w = 3.0 * sqrt(max(det / cov2D.z, 1e-6));
    float quad_h = 3.0 * sqrt(max(det / cov2D.x, 1e-6));

    vec2 quad_ndc = vec2(quad_w, quad_h) / u_ViewportSize * 2.0 * u_QuadScale;
    pos_clip.xyz = pos_clip.xyz / pos_clip.w;
    pos_clip.xy += quad_coord * quad_ndc;
    pos_clip.w = 1.0;

    gl_Position = pos_clip;
    v_coordxy = quad_coord * vec2(quad_w, quad_h);
}
""")

        shader_info.fragment_source("""
void main() {
    float d2 = v_conic.x * v_coordxy.x * v_coordxy.x +
               v_conic.z * v_coordxy.y * v_coordxy.y +
               v_conic.y * v_coordxy.x * v_coordxy.y;
    d2 = -0.5*d2;
    float opacity = v_opacity * exp(d2);
    float r2 = -2.0*d2/9.0;
    opacity *= max(0.0, 1.0 - r2 * r2);
    FragColor = vec4(v_color * opacity, opacity);
}
""")

        try:
            self.shader = gpu.shader.create_from_info(shader_info)
        except Exception as e:
            print(f"[Splatting] Shader creation failed: {e}")
            return False

        self._matrices_ubo = GPUUniformBuf(bytearray([0]) * 128)

        del vert_out
        del shader_info
        self.initialized = True
        return True

    def _update_matrices_ubo(self, vp_matrix, view_matrix):
        """Pack matrices into UBO in column-major order for GLSL"""
        if self._matrices_ubo is None:
            return
        data = np.zeros(32, dtype=np.float32)
        idx = 0
        for col in range(4):
            for row in range(4):
                data[idx] = vp_matrix[row][col]
                idx += 1
        for col in range(4):
            for row in range(4):
                data[idx] = view_matrix[row][col]
                idx += 1
        self._matrices_ubo.update(data)

    def draw(self, context):
        """Draw all splat instances with global block-level sorting for correct
        alpha blending across objects."""
        if not self.initialized or self.shader is None:
            return

        # --- Viewport / projection setup ---
        vp_matrix = Matrix.Identity(4)
        view_matrix = Matrix.Identity(4)
        camera = None
        region3d = None

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                region3d = area.spaces.active.region_3d
                is_camera_view = region3d.view_perspective == 'CAMERA'
                camera = context.scene.camera if is_camera_view else None
                break
        viewport = state.viewport_get()
        vp_w, vp_h = viewport[2], viewport[3]
        if vp_w <= 0 or vp_h <= 0:
            return
        focal = vp_w * 0.5

        if camera:
            depsgraph = context.evaluated_depsgraph_get()
            view_matrix = camera.matrix_world.inverted()
            proj_matrix = camera.calc_matrix_camera(
                depsgraph,
                x=context.scene.render.resolution_x,
                y=context.scene.render.resolution_y,
            )
            vp_matrix = proj_matrix @ view_matrix
            focal = proj_matrix[0][0] * vp_w * 0.5
        elif region3d:
            view_matrix = region3d.view_matrix
            proj_matrix = region3d.window_matrix
            vp_matrix = proj_matrix @ view_matrix
            focal = proj_matrix[0][0] * vp_w * 0.5
        else:
            return

        proj_00 = proj_matrix[0][0]
        proj_11 = proj_matrix[1][1]

        gpu.state.viewport_set(0, 0, vp_w, vp_h)

        from .splatting_data import get_state
        scene = get_state()

        if not scene.instances:
            return

        splatting_props = context.scene.splatting_properties

        # --- State setup ---
        state.blend_set('ALPHA_PREMULT')
        state.depth_mask_set(False)
        state.depth_test_set('LESS')

        self.shader.bind()
        self.shader.uniform_float("u_FocalParams", (focal, focal))
        self.shader.uniform_float("u_ViewportSize", (vp_w, vp_h))

        near_to_far = splatting_props.sort_near_to_far

        # --- World-space camera position (same for all instances) ---
        if camera:
            camera_pos_world = camera.matrix_world.translation
        else:
            view_matrix_inv = region3d.view_matrix.inverted()
            camera_pos_world = view_matrix_inv.translation
        camera_world_np = np.array(
            [camera_pos_world[0], camera_pos_world[1], camera_pos_world[2]], dtype=np.float32)

        # =====================================================================
        # Phase 1: Collect visible blocks from all instances
        # =====================================================================
        draw_entries = []      # (inst_idx, block_idx, sort_key)
        inst_data = {}         # inst_idx -> per-instance draw info
        inst_drawn_blocks = {}
        inst_drawn_splats = {}
        total_drawn_blocks = 0
        total_drawn_splats = 0

        for inst_idx, inst in enumerate(scene.instances):
            if inst.point_count == 0 or inst.target_mesh is None:
                continue

            # Check UI toggle
            ui_item = None
            for item in context.scene.splatting_instances:
                if item.mesh_name == inst.target_mesh.name:
                    ui_item = item
                    break
            if ui_item is None or not ui_item.enabled:
                continue

            # Per-instance matrices
            model_matrix = inst.target_mesh.matrix_world.copy()
            inst_vp = vp_matrix @ model_matrix
            inst_view = view_matrix @ model_matrix
            camera_pos_local = inst_view.inverted().translation

            # Camera positions for auto-sort
            inst._camera_pos_np = np.array(
                [camera_pos_local[0], camera_pos_local[1], camera_pos_local[2]], dtype=np.float32)
            inst._camera_world_np = camera_world_np
            inst._vp_world = vp_matrix
            inst._proj_00 = proj_00
            inst._proj_11 = proj_11

            # Auto-sort
            from .splatting_data import check_view_changed, sort_next_batch
            if check_view_changed(inst, inst_view):
                inst._sort_active = True
                if inst.sorted_up_to >= inst.block_count:
                    inst.sorted_up_to = 0
            sort_next_batch(inst, context)

            # Block ordering (by near-end distance from camera, world space)
            distances = np.linalg.norm(inst.block_centers - camera_world_np, axis=1)
            near_end = distances - inst.block_radii
            block_order = np.argsort(near_end)
            if not near_to_far:
                block_order = block_order[::-1]

            # Vectorized frustum culling (world space)
            count = len(inst.block_centers)
            ones = np.ones((count, 1), dtype=np.float32)
            centers_h = np.concatenate([inst.block_centers, ones], axis=1)
            vp_np = np.asarray(vp_matrix, dtype=np.float32)
            clip_pos = centers_h @ vp_np.T
            visible_mask = (clip_pos[:, 3] > 0) & \
                (np.abs(clip_pos[:, 0]) < clip_pos[:, 3] + inst.block_radii * proj_00) & \
                (np.abs(clip_pos[:, 1]) < clip_pos[:, 3] + inst.block_radii * proj_11)

            # Collect visible block entries
            sort_sign = 1.0 if near_to_far else -1.0
            for idx in block_order:
                if visible_mask[idx]:
                    draw_entries.append((inst_idx, idx, near_end[idx] * sort_sign))

            inst_data[inst_idx] = {
                'vp': inst_vp,
                'view': inst_view,
                'cam_local': camera_pos_local,
                'ui': ui_item,
                'inst': inst,
            }
            inst_drawn_blocks[inst_idx] = 0
            inst_drawn_splats[inst_idx] = 0

        # =====================================================================
        # Phase 2: Global sort by distance across all instances
        # =====================================================================
        draw_entries.sort(key=lambda x: x[2])

        # =====================================================================
        # Phase 3: Draw with per-instance uniform switching
        # =====================================================================
        current_inst_idx = -1
        for inst_idx, block_idx, _ in draw_entries:
            data = inst_data[inst_idx]

            if inst_idx != current_inst_idx:
                # Rebind per-instance uniforms and matrices
                self.shader.uniform_float("u_QuadScale", data['ui'].quad_scale)
                self.shader.uniform_float("u_Gamma", data['ui'].color_gamma)
                self.shader.uniform_float("u_Hue", data['ui'].color_hue)
                self.shader.uniform_float("u_Saturation", data['ui'].color_saturation)
                self.shader.uniform_float("u_Brightness", data['ui'].color_brightness)
                self.shader.uniform_float("u_Tint", data['ui'].color_tint)

                self._update_matrices_ubo(data['vp'], data['view'])
                self.shader.uniform_block('u_Matrices', self._matrices_ubo)
                self.shader.uniform_float("u_CameraPos", data['cam_local'])
                current_inst_idx = inst_idx

            batch = data['inst']._get_or_create_block_batch(block_idx)
            if batch:
                batch.draw(self.shader)
                inst_drawn_blocks[inst_idx] += 1
                splats = max(1, len(data['inst'].block_splat_indices[block_idx]))
                inst_drawn_splats[inst_idx] += splats
                total_drawn_blocks += 1
                total_drawn_splats += splats

        # Fallback for instances with no visible blocks
        for inst_idx, data in inst_data.items():
            inst = data['inst']
            if inst_drawn_blocks[inst_idx] == 0 and inst.point_count > 0:
                batch = inst._get_or_create_fallback_batch()
                if batch:
                    batch.draw(self.shader)
                    inst_drawn_blocks[inst_idx] = inst.block_count
                    inst_drawn_splats[inst_idx] = max(1, inst.point_count // 16)
                    total_drawn_blocks += inst_drawn_blocks[inst_idx]
                    total_drawn_splats += inst_drawn_splats[inst_idx]

        # Update per-instance stats
        for inst_idx, data in inst_data.items():
            data['inst'].displayed_block_count = inst_drawn_blocks[inst_idx]
            data['inst'].displayed_splat_count = inst_drawn_splats[inst_idx]

        # Update scene-level stats
        scene.displayed_block_count = total_drawn_blocks
        scene.displayed_splat_count = total_drawn_splats

        # --- Reset blend ---
        state.blend_set('NONE')

    def release(self):
        """Release GPU resources"""
        self.shader = None
        self.initialized = False
        self._matrices_ubo = None


_renderer = SplattingRenderer()


def get_renderer():
    return _renderer


def init_renderer():
    """Initialize shared shader and UBO (no per-instance data)."""
    return _renderer.init_buffers()


def release_renderer():
    _renderer.release()


def draw_splatting(context):
    _renderer.draw(context)


def _steps(start, end, step):
    """Generate evenly spaced values from start to end (inclusive) at given step."""
    if step <= 0:
        return [start]
    pos, vals = start, []
    while pos <= end + 1e-6:
        vals.append(pos)
        pos += step
    return vals


# ---------------------------------------------------------------------------
# Block grid overlay (POST_VIEW draw handler)
# ---------------------------------------------------------------------------
_grid_draw_handle = None


def _grid_draw():
    """Draw block grid overlay in 3D viewports."""
    try:
        context = bpy.context
        scene = context.scene
        props = scene.splatting_properties
    except AttributeError:
        return

    if not props.show_block_grid:
        return

    block_size = props.block_size
    if block_size <= 0:
        return

    offset = np.array(props.block_offset, dtype=np.float32)

    # Compute world-space bounding box from mesh bound_box (fast — 8 corners)
    items = scene.splatting_instances
    if not items:
        return

    min_coords = None
    max_coords = None
    for item in items:
        if not item.enabled:
            continue
        obj = bpy.data.objects.get(item.mesh_name)
        if obj and obj.type == 'MESH' and len(obj.data.vertices) > 0:
            bbox_local = np.array(obj.bound_box, dtype=np.float32)
            mat = np.array(obj.matrix_world, dtype=np.float32)
            ones = np.ones((8, 1), dtype=np.float32)
            corners_h = np.concatenate([bbox_local, ones], axis=1)
            corners_w = (corners_h @ mat.T)[:, :3]
            obj_min = corners_w.min(axis=0)
            obj_max = corners_w.max(axis=0)
            if min_coords is None:
                min_coords = obj_min.copy()
                max_coords = obj_max.copy()
            else:
                min_coords = np.minimum(min_coords, obj_min)
                max_coords = np.maximum(max_coords, obj_max)

    if min_coords is None:
        return

    origin = min_coords + offset
    bs = block_size

    xs = _steps(origin[0], max_coords[0], bs)
    ys = _steps(origin[1], max_coords[1], bs)
    zs = _steps(origin[2], max_coords[2], bs)

    if len(xs) < 2 and len(ys) < 2 and len(zs) < 2:
        return

    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    z_min, z_max = zs[0], zs[-1]

    lines = []

    # Parallel to X: at each (y,z) grid intersection
    for y in ys:
        for z in zs:
            lines.append([x_min, y, z])
            lines.append([x_max, y, z])

    # Parallel to Y: at each (x,z) grid intersection
    for x in xs:
        for z in zs:
            lines.append([x, y_min, z])
            lines.append([x, y_max, z])

    # Parallel to Z: at each (x,y) grid intersection
    for x in xs:
        for y in ys:
            lines.append([x, y, z_min])
            lines.append([x, y, z_max])

    if not lines:
        return

    verts = np.array(lines, dtype=np.float32)
    fmt = GPUVertFormat()
    fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
    vbo = GPUVertBuf(fmt, len(verts))
    vbo.attr_fill(id="pos", data=verts)
    batch = GPUBatch(type='LINES', buf=vbo)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    gpu.state.depth_test_set('ALWAYS')
    gpu.state.blend_set('ALPHA')

    srgb = np.array(props.grid_color, dtype=np.float32)
    linear = np.sqrt(srgb)
    color = (*linear, props.grid_alpha)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS')


def register_grid_draw():
    global _grid_draw_handle
    if _grid_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_grid_draw_handle, 'WINDOW')
        except Exception:
            pass
    _grid_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _grid_draw, (), 'WINDOW', 'POST_VIEW'
    )


def unregister_grid_draw():
    global _grid_draw_handle
    if _grid_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_grid_draw_handle, 'WINDOW')
        except Exception:
            pass
        _grid_draw_handle = None
