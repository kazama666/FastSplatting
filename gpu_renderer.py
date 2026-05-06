import bpy
import gpu
import numpy as np
from gpu.types import GPUUniformBuf, GPUVertBuf, GPUVertFormat, GPUBatch, GPUIndexBuf
from gpu import state
from mathutils import Matrix


# ---------------------------------------------------------------------------
# Shared GLSL source fragments
# ---------------------------------------------------------------------------

_COV2D_SOURCE = """
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
"""

_COLOR_ADJUST_SOURCE = """
    v_opacity = min(inst_opacity * 2.0, 1.0);

    // Forward brightness gain (pre-color-adjust pseudo-HDR expansion)
    {
        vec3 t = clamp((v_color - u_BrightnessGainStart) / max(1.0 - u_BrightnessGainStart, 1e-6), 0.0, 1.0);
        v_color *= (1.0 + t * t * max(u_BrightnessGain - 1.0, 0.0));
    }

    // User color adjustments
    v_color *= u_Tint;
    {
        const vec3 k = vec3(0.57735, 0.57735, 0.57735);
        float cosH = cos(u_Hue * 3.14159);
        float sinH = sin(u_Hue * 3.14159);
        v_color = v_color * cosH + cross(k, v_color) * sinH + k * dot(k, v_color) * (1.0 - cosH);

        float lum = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        v_color = mix(vec3(lum), v_color, u_Saturation);

        v_color *= 1.2*u_Brightness;

        v_color = pow(max(v_color, vec3(0.0)), vec3(1.0 / (0.65*u_Gamma)));
    }

    // Inverse brightness gain (luminance-only, post-color-adjust)
    // Restores relative bright/dark separation without amplifying saturation.
    // Skipped when u_ApplyInverseGain == 0 (e.g. during envmap offscreen bake)
    if (u_ApplyInverseGain > 0.5) {
        float l = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        float tl = clamp((l - u_BrightnessGainStart) / max(1.0 - u_BrightnessGainStart, 1e-6), 0.0, 1.0);
        v_color *= (1.0 + tl * tl * max(u_BrightnessGain - 1.0, 0.0));
    }
"""

_RENDER_SOURCE = """
    vec4 pos_view = u_Matrices.u_ViewMatrix * vec4(inst_position, 1.0);
    vec4 pos_clip = u_Matrices.u_VPMatrix * vec4(inst_position, 1.0);

    // Near clip (behind camera)
    if (pos_view.z >= -0.001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    // Far clip — clip-space check handles both near & far automatically
    if (abs(pos_clip.z) >= pos_clip.w) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    // XY frustum culling — discard if center is >1.4× frustum bounds
    // (matching Spark's clipXY = 1.4 default, accounting for wide splats)
    float clip_margin = 1.4 * pos_clip.w;
    if (abs(pos_clip.x) > clip_margin || abs(pos_clip.y) > clip_margin) {
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

    // Compute quad half-extent in pixels, clamped to 512 max radius
    // (matching Spark's maxPixelRadius = 512 default)
    float qx = 3.0 * sqrt(max(cov2D.x, 1e-6));
    float qy = 3.0 * sqrt(max(cov2D.z, 1e-6));
    qx = min(qx, 512.0);
    qy = min(qy, 512.0);

    vec2 quad_ndc = vec2(qx, qy) / u_ViewportSize * 2.0 * 1.0;
    pos_clip.xyz = pos_clip.xyz / pos_clip.w;
    pos_clip.xy += quad_coord * quad_ndc;
    pos_clip.w = 1.0;

    gl_Position = pos_clip;
    v_coordxy = quad_coord * vec2(qx, qy);
}
"""

_FRAGMENT_SOURCE = """
void main() {
    float d2 = -0.5 * (v_conic.x * v_coordxy.x * v_coordxy.x +
                       v_conic.z * v_coordxy.y * v_coordxy.y) -
                       v_conic.y * v_coordxy.x * v_coordxy.y;
    float opacity = v_opacity * exp(d2);
    float r2 = -2.0*d2/9.0;
    opacity *= max(0.0, 1.0 - r2 * r2);
    FragColor = vec4(v_color * opacity, opacity);
}
"""


_SIMPLE_VERTEX_SOURCE = _COV2D_SOURCE + """
void main() {
    v_color = inst_color;
""" + _COLOR_ADJUST_SOURCE + _RENDER_SOURCE

_FULL_VERTEX_SOURCE = _COV2D_SOURCE + """
void main() {
    // SH1 from 3x VEC4 VBO attributes — no texture needed
    vec3 dir = normalize(inst_position - u_CameraPos);
    float x = dir.x, y = dir.y, z = dir.z;
    const float C1 = 0.4886025119029199;
    vec3 result = inst_sh_0.xyz                             // DC
        + vec3(inst_sh_0.w, inst_sh_1.xy) * (-C1 * y)       // Y1-
        + vec3(inst_sh_1.zw, inst_sh_2.x) * (C1 * z)        // Y10
        + inst_sh_2.yzw * (-C1 * x);                         // Y11
    result = 1.0 / (1.0 + exp(-result));
    v_color = pow(max(result, vec3(0.0)), vec3(2.2));
""" + _COLOR_ADJUST_SOURCE + _RENDER_SOURCE

# ---------------------------------------------------------------------------
# Orthographic vertex shader variants
# ---------------------------------------------------------------------------

_ORTHO_COV2D_SOURCE = """
vec3 computeCov2D_ortho(vec4 pos_view, float focal, mat3 cov3d_in, mat3 view_rot) {
    // Orthographic Jacobian = [[focal, 0, 0], [0, focal, 0]] maps view-space
    // coordinates to screen pixels (no depth dependence).  Apply it:
    //   Σ'_2D = J * (W * Σ * W^T) * J^T  →  scale by focal²
    // The 0.3 bias is applied after scaling (same as perspective path).
    mat3 cov_view = transpose(view_rot) * cov3d_in * view_rot;
    float f2 = focal * focal;
    return vec3(cov_view[0][0] * f2 + 0.3, cov_view[0][1] * f2, cov_view[1][1] * f2 + 0.3);
}
"""

_ORTHO_RENDER_SOURCE = """
    vec4 pos_view = u_Matrices.u_ViewMatrix * vec4(inst_position, 1.0);
    vec4 pos_clip = u_Matrices.u_VPMatrix * vec4(inst_position, 1.0);

    // Near clip (behind camera)
    if (pos_view.z >= -0.001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    // XY frustum culling (ortho: pos_clip.w = 1.0, clip-space == NDC)
    float clip_margin = 1.4 * pos_clip.w;
    if (abs(pos_clip.x) > clip_margin || abs(pos_clip.y) > clip_margin) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    float cxx = inst_cov_a.x, cxy = inst_cov_a.y, cxz = inst_cov_a.z;
    float cyy = inst_cov_b.x, cyz = inst_cov_b.y, czz = inst_cov_b.z;
    mat3 cov3D = mat3(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz);
    mat3 view_rot = transpose(mat3(u_Matrices.u_ViewMatrix));
    float focal = u_FocalParams.x;
    vec3 cov2D = computeCov2D_ortho(pos_view, focal, cov3D, view_rot);

    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.00001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    float det_inv = 1.0 / det;
    v_conic = vec3(cov2D.z * det_inv, -cov2D.y * det_inv, cov2D.x * det_inv);

    float qx = 3.0 * sqrt(max(cov2D.x, 1e-6));
    float qy = 3.0 * sqrt(max(cov2D.z, 1e-6));
    qx = min(qx, 512.0);
    qy = min(qy, 512.0);

    vec2 quad_ndc = vec2(qx, qy) / u_ViewportSize * 2.0 * 1.0;
    // Ortho: pos_clip is already in NDC (w = 1.0)
    pos_clip.xy += quad_coord * quad_ndc;

    gl_Position = pos_clip;
    v_coordxy = quad_coord * vec2(qx, qy);
}
"""

_SIMPLE_ORTHO_VERTEX_SOURCE = _ORTHO_COV2D_SOURCE + """
void main() {
    v_color = inst_color;
""" + _COLOR_ADJUST_SOURCE + _ORTHO_RENDER_SOURCE

_FULL_ORTHO_VERTEX_SOURCE = _ORTHO_COV2D_SOURCE + """
void main() {
    vec3 dir = normalize(inst_position - u_CameraPos);
    float x = dir.x, y = dir.y, z = dir.z;
    const float C1 = 0.4886025119029199;
    vec3 result = inst_sh_0.xyz
        + vec3(inst_sh_0.w, inst_sh_1.xy) * (-C1 * y)
        + vec3(inst_sh_1.zw, inst_sh_2.x) * (C1 * z)
        + inst_sh_2.yzw * (-C1 * x);
    result = 1.0 / (1.0 + exp(-result));
    v_color = pow(max(result, vec3(0.0)), vec3(2.2));
""" + _COLOR_ADJUST_SOURCE + _ORTHO_RENDER_SOURCE


# ---------------------------------------------------------------------------
# Shared typedef source
# ---------------------------------------------------------------------------

_TYPEDEF_SOURCE = """
struct SplattingMatrices {
    mat4 u_VPMatrix;
    mat4 u_ViewMatrix;
};
"""


def _build_shader(vert_out, with_sh, ortho=False):
    """Build and return a shader, with or without SH evaluation (texture).
    If *ortho* is True the vertex shader uses the orthographic cov2D path."""
    info = gpu.types.GPUShaderCreateInfo()
    info.typedef_source(_TYPEDEF_SOURCE)
    info.uniform_buf(0, 'SplattingMatrices', 'u_Matrices')

    # Push constants (shared)
    info.push_constant('VEC2', "u_FocalParams")
    info.push_constant('VEC2', "u_ViewportSize")
    info.push_constant('FLOAT', "u_Gamma")
    info.push_constant('FLOAT', "u_Hue")
    info.push_constant('FLOAT', "u_Saturation")
    info.push_constant('FLOAT', "u_Brightness")
    info.push_constant('FLOAT', "u_BrightnessGain")
    info.push_constant('FLOAT', "u_BrightnessGainStart")
    info.push_constant('FLOAT', "u_ApplyInverseGain")
    info.push_constant('VEC3', "u_Tint")

    if with_sh:
        info.push_constant('VEC3', "u_CameraPos")

    # Vertex inputs (same VBO layout for both shaders)
    info.vertex_in(0, 'VEC2', "quad_coord")
    info.vertex_in(1, 'VEC3', "inst_position")
    info.vertex_in(2, 'VEC3', "inst_color")
    info.vertex_in(3, 'FLOAT', "inst_opacity")
    info.vertex_in(4, 'VEC3', "inst_cov_a")
    info.vertex_in(5, 'VEC3', "inst_cov_b")
    info.vertex_in(6, 'VEC4', "inst_sh_0")
    info.vertex_in(7, 'VEC4', "inst_sh_1")
    info.vertex_in(8, 'VEC4', "inst_sh_2")

    info.vertex_out(vert_out)
    info.fragment_out(0, 'VEC4', "FragColor")

    if ortho:
        vs_source = _FULL_ORTHO_VERTEX_SOURCE if with_sh else _SIMPLE_ORTHO_VERTEX_SOURCE
    else:
        vs_source = _FULL_VERTEX_SOURCE if with_sh else _SIMPLE_VERTEX_SOURCE
    info.vertex_source(vs_source)
    info.fragment_source(_FRAGMENT_SOURCE)

    return gpu.shader.create_from_info(info)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SplattingRenderer:
    def __init__(self):
        self.shader_simple = None    # SH0 path, no texture
        self.shader_full = None      # SH1+ path, with texture + evaluateSH
        self.shader_simple_ortho = None
        self.shader_full_ortho = None
        self.initialized = False
        self._matrices_ubo = None
        self._skip_inverse_gain = False  # set by draw_offscreen for envmap bake

    def init_buffers(self):
        """Create both shaders and shared UBO."""
        if self.initialized:
            return True

        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "v_coordxy")
        vert_out.smooth('VEC3', "v_color")
        vert_out.smooth('FLOAT', "v_opacity")
        vert_out.smooth('VEC3', "v_conic")

        # Simple shader (SH0) — no sampler, no evaluateSH
        try:
            self.shader_simple = _build_shader(vert_out, with_sh=False)
            self.shader_simple_ortho = _build_shader(vert_out, with_sh=False, ortho=True)
        except Exception as e:
            print(f"[Splatting] Simple shader creation failed: {e}")
            return False

        # Full shader (SH1+) — with sampler + evaluateSH
        try:
            self.shader_full = _build_shader(vert_out, with_sh=True)
            self.shader_full_ortho = _build_shader(vert_out, with_sh=True, ortho=True)
        except Exception as e:
            print(f"[Splatting] Full shader creation failed: {e}")
            return False

        self._matrices_ubo = GPUUniformBuf(bytearray([0]) * 128)

        del vert_out
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

    def _bind_instance_uniforms(self, shader, data, is_sh):
        """Set per-instance uniforms common to both shader variants."""
        shader.uniform_float("u_Gamma", data['ui'].color_gamma)
        shader.uniform_float("u_Hue", data['ui'].color_hue)
        shader.uniform_float("u_Saturation", data['ui'].color_saturation)
        shader.uniform_float("u_Brightness", data['ui'].color_brightness)
        shader.uniform_float("u_BrightnessGain", data['ui'].brightness_gain)
        shader.uniform_float("u_BrightnessGainStart", data['ui'].brightness_gain_start)
        shader.uniform_float("u_ApplyInverseGain", 0.0 if self._skip_inverse_gain else 1.0)
        shader.uniform_float("u_Tint", data['ui'].color_tint)

        self._update_matrices_ubo(data['vp'], data['view'])
        shader.uniform_block('u_Matrices', self._matrices_ubo)

        if is_sh:
            shader.uniform_float("u_CameraPos", data['cam_local'])

    def _render_to_framebuffer(self, context, vp_w, vp_h,
                                view_matrix, proj_matrix,
                                camera_pos_world, is_ortho):
        """Core rendering: block culling, sorting, drawing.

        GPU state (viewport, blend, depth) and active framebuffer must be
        set by the caller beforehand.
        """
        vp_matrix = proj_matrix @ view_matrix
        focal = proj_matrix[0][0] * vp_w * 0.5
        proj_00 = proj_matrix[0][0]
        proj_11 = proj_matrix[1][1]

        from .splatting_data import get_state
        scene = get_state()

        if not scene.instances:
            return

        splatting_props = context.scene.splatting_properties
        near_to_far = splatting_props.sort_near_to_far
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

        # Build UI lookup dict (avoids O(nÂ²) linear scan per instance)
        ui_lookup = {}
        for item in context.scene.splatting_instances:
            ui_lookup[item.mesh_name] = item

        for inst_idx, inst in enumerate(scene.instances):
            if inst.point_count == 0 or inst.target_mesh is None:
                continue

            # Check UI toggle (O(1) dict lookup)
            ui_item = ui_lookup.get(inst.target_mesh.name)
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
            inst._vp_world = vp_matrix
            inst._proj_00 = proj_00
            inst._proj_11 = proj_11

            # Apply delta transform to block centers for animated objects
            if inst._orig_model_matrix is not None:
                model_np = np.asarray(model_matrix, dtype=np.float32)
                delta = model_np @ inst._orig_model_inv
                ones = np.ones((len(inst._orig_block_centers), 1), dtype=np.float32)
                centers_h = np.concatenate([inst._orig_block_centers, ones], axis=1)
                inst.block_centers[:] = (centers_h @ delta.T)[:, :3]
                s = max(np.linalg.norm(delta[:3, i]) for i in range(3))
                inst.block_radii[:] = inst._orig_block_radii * s
                lo = inst._orig_block_bounds[:, 0, :]
                hi = inst._orig_block_bounds[:, 1, :]
                half = (hi - lo) * 0.5 * s
                center = (lo + hi) * 0.5
                ones_c = np.ones((len(center), 1), dtype=np.float32)
                center_h = np.concatenate([center, ones_c], axis=1)
                new_center = (center_h @ delta.T)[:, :3]
                inst.block_bounds[:, 0, :] = new_center - half
                inst.block_bounds[:, 1, :] = new_center + half

            # Auto-sort (camera motion only)
            from .splatting_data import check_view_changed, sort_next_batch
            if check_view_changed(inst, view_matrix):
                inst._sort_active = True
                if inst.sorted_up_to >= inst.block_count:
                    inst.sorted_up_to = 0
            sort_next_batch(inst, context)

            # Block ordering: midpoint of nearest and farthest view-space Z
            view_np = np.asarray(view_matrix, dtype=np.float32)
            vz = view_np[2:3, :3]
            lo = inst.block_bounds[:, 0, :]
            hi = inst.block_bounds[:, 1, :]
            near_pick = np.where(vz > 0, hi, lo)
            far_pick = np.where(vz > 0, lo, hi)
            near_end = -((near_pick * vz).sum(axis=1) + view_np[2, 3])
            far_end = -((far_pick * vz).sum(axis=1) + view_np[2, 3])
            sort_key = (near_end + far_end) * 0.5
            block_order = np.argsort(sort_key)
            if not near_to_far:
                block_order = block_order[::-1]

            # Vectorized frustum culling (world space)
            count = len(inst.block_centers)
            ones = np.ones((count, 1), dtype=np.float32)
            centers_h = np.concatenate([inst.block_centers, ones], axis=1)
            vp_np = np.asarray(vp_matrix, dtype=np.float32)
            clip_pos = centers_h @ vp_np.T
            visible_mask = (clip_pos[:, 3] > 0) &                 (np.abs(clip_pos[:, 0]) < clip_pos[:, 3] + inst.block_radii * proj_00) &                 (np.abs(clip_pos[:, 1]) < clip_pos[:, 3] + inst.block_radii * proj_11)

            # Collect visible block entries
            sort_sign = 1.0 if near_to_far else -1.0
            for idx in block_order:
                if visible_mask[idx]:
                    draw_entries.append((inst_idx, idx, sort_key[idx] * sort_sign))

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
        # Phase 3: Draw with per-instance shader + uniform switching
        # =====================================================================
        current_inst_idx = -1
        bound_shader = None

        for draw_idx, (inst_idx, block_idx, _) in enumerate(draw_entries):
            data = inst_data[inst_idx]

            if inst_idx != current_inst_idx:
                inst = data['inst']
                is_sh = inst.sh_degree > 0
                if is_ortho:
                    shader = self.shader_full_ortho if is_sh else self.shader_simple_ortho
                else:
                    shader = self.shader_full if is_sh else self.shader_simple

                if shader is not bound_shader:
                    shader.bind()
                    bound_shader = shader
                    shader.uniform_float("u_FocalParams", (focal, focal))
                    shader.uniform_float("u_ViewportSize", (vp_w, vp_h))

                self._bind_instance_uniforms(shader, data, is_sh)
                current_inst_idx = inst_idx

            batch = data['inst']._get_or_create_block_batch(block_idx)
            if batch:
                batch.draw(bound_shader)
                inst_drawn_blocks[inst_idx] += 1
                splats = max(1, len(data['inst'].block_splat_indices[block_idx]))
                inst_drawn_splats[inst_idx] += splats
                total_drawn_blocks += 1
                total_drawn_splats += splats

        # Fallback for instances with no visible blocks
        for inst_idx, data in inst_data.items():
            inst = data['inst']
            if inst_drawn_blocks[inst_idx] == 0 and inst.point_count > 0:
                is_sh = inst.sh_degree > 0
                if is_ortho:
                    shader = self.shader_full_ortho if is_sh else self.shader_simple_ortho
                else:
                    shader = self.shader_full if is_sh else self.shader_simple
                if shader is not bound_shader:
                    shader.bind()
                    bound_shader = shader
                    shader.uniform_float("u_FocalParams", (focal, focal))
                    shader.uniform_float("u_ViewportSize", (vp_w, vp_h))

                self._bind_instance_uniforms(shader, data, is_sh)
                batch = inst._get_or_create_fallback_batch()
                if batch:
                    batch.draw(shader)
                    inst_drawn_blocks[inst_idx] = inst.block_count
                    inst_drawn_splats[inst_idx] = max(1, inst.point_count // 16)
                    total_drawn_blocks += inst_drawn_blocks[inst_idx]
                    total_drawn_splats += inst_drawn_splats[inst_idx]

        # Update per-instance stats
        for inst_idx, data in inst_data.items():
            data['inst'].displayed_block_count = inst_drawn_blocks[inst_idx]
            data['inst'].displayed_splat_count = inst_drawn_splats[inst_idx]

        scene.displayed_block_count = total_drawn_blocks
        scene.displayed_splat_count = total_drawn_splats

    def draw(self, context):
        """Draw all splat instances into the current viewport."""
        if not self.initialized:
            return
        if self.shader_simple is None and self.shader_full is None:
            return

        # --- Viewport / projection setup ---
        vp_matrix = Matrix.Identity(4)
        view_matrix = Matrix.Identity(4)
        camera = None
        region3d = None
        is_ortho = False

        area = context.area
        if area and area.type == 'VIEW_3D':
            region3d = area.spaces.active.region_3d
            if region3d:
                is_camera_view = region3d.view_perspective == 'CAMERA'
                camera = context.scene.camera if is_camera_view else None
                if is_camera_view:
                    is_ortho = camera and camera.data.type == 'ORTHO'
                else:
                    is_ortho = region3d.view_perspective == 'ORTHO'
        viewport = state.viewport_get()
        vp_w, vp_h = viewport[2], viewport[3]
        if vp_w <= 0 or vp_h <= 0:
            return

        if camera:
            depsgraph = context.evaluated_depsgraph_get()
            view_matrix = camera.matrix_world.inverted()
            proj_matrix = camera.calc_matrix_camera(
                depsgraph,
                x=context.scene.render.resolution_x,
                y=context.scene.render.resolution_y,
            )
            camera_pos_world = camera.matrix_world.translation
        elif region3d:
            view_matrix = region3d.view_matrix
            proj_matrix = region3d.window_matrix
            view_matrix_inv = view_matrix.inverted()
            camera_pos_world = view_matrix_inv.translation
        else:
            return

        # --- State setup ---
        state.blend_set('ALPHA_PREMULT')
        state.depth_mask_set(False)
        state.depth_test_set('LESS')
        gpu.state.viewport_set(0, 0, vp_w, vp_h)

        self._render_to_framebuffer(
            context, vp_w, vp_h,
            view_matrix, proj_matrix,
            camera_pos_world, is_ortho,
        )

        state.blend_set('NONE')

    def draw_offscreen(self, context, width, height,
                       view_matrix, proj_matrix,
                       camera_pos_world, skip_sort=False, skip_inverse_gain=True):
        """Render to the currently-bound off-screen framebuffer and return pixels.

        Performs a full sort before rendering (blocks sorted far-to-near from
        the given camera position), unless ``skip_sort`` is True (caller has
        already sorted).  Returns a (height, width, 4) float32 numpy array
        in RGBA.

        When *skip_inverse_gain* is True (default, used for envmap baking) the
        inverse gain pass is skipped — forward gain still runs so the cubemap
        faces have expanded dynamic range, but the post-color-adjust inverse
        pass is not needed since there's no color-adjust in the bake path.

        The caller must bind a GPUOffScreen before calling this method.
        """
        gpu.state.viewport_set(0, 0, width, height)
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))

        state.blend_set('ALPHA_PREMULT')
        state.depth_mask_set(False)
        # Offscreen has no depth buffer — disable depth test to avoid
        # undefined behavior on GPUs that reject fragments when missing.
        state.depth_test_set('NONE')

        if not skip_sort:
            from .splatting_data import sort_blocks_far_to_near
            sort_blocks_far_to_near((
                camera_pos_world[0], camera_pos_world[1], camera_pos_world[2]
            ))

        self._skip_inverse_gain = skip_inverse_gain
        self._render_to_framebuffer(
            context, width, height,
            view_matrix, proj_matrix,
            camera_pos_world, is_ortho=False,
        )
        self._skip_inverse_gain = False

        buffer = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
        buffer.dimensions = width * height * 4
        pixels = np.asarray(buffer, dtype=np.float32).reshape(height, width, 4)

        state.blend_set('NONE')
        return pixels

    def release(self):
        """Release GPU resources"""
        self.shader_simple = None
        self.shader_full = None
        self.shader_simple_ortho = None
        self.shader_full_ortho = None
        self.initialized = False
        self._matrices_ubo = None


_renderer = SplattingRenderer()


def get_renderer():
    return _renderer


def init_renderer():
    """Initialize both shaders and shared UBO."""
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
def _aabb_wireframe(lo, hi=None):
    """Generate 24 vertices (12 LINES × 2) for an AABB/OBB wireframe
    from min/max or from an (8,3) corner array."""
    if isinstance(lo, np.ndarray) and lo.shape == (8, 3):
        corners = lo
        idx = [0, 1, 1, 2, 2, 3, 3, 0,
               4, 5, 5, 6, 6, 7, 7, 4,
               0, 4, 1, 5, 2, 6, 3, 7]
        return corners[idx]
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array([
        [x0, y0, z0], [x1, y0, z0],
        [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],
        [x0, y1, z0], [x0, y0, z0],
        [x0, y0, z1], [x1, y0, z1],
        [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],
        [x0, y1, z1], [x0, y0, z1],
        [x0, y0, z0], [x0, y0, z1],
        [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],
        [x0, y1, z0], [x0, y1, z1],
    ], dtype=np.float32)


_grid_draw_handle = None
_round_point_shader = None
_sh_sphere_shader = None
_sh_sphere_vbo = None
_sh_sphere_ibo = None


def _get_sh_sphere_shader():
    """Lazy-create custom shader: 9 vec3 SH uniforms, sigmoid reconstruction."""
    global _sh_sphere_shader
    if _sh_sphere_shader is not None:
        return _sh_sphere_shader
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('MAT4', "u_model")
    for i in range(9):
        info.push_constant('VEC3', f"u_sh_{i}")
    vert_out = gpu.types.GPUStageInterfaceInfo("sh_sphere_iface")
    vert_out.smooth('VEC4', "v_color")
    info.vertex_out(vert_out)
    info.vertex_source("""
    vec3 eval_sh(vec3 n) {
        float x = n.x, y = n.y, z = n.z;
        return u_sh_0 * 0.282095
            + u_sh_1 * (0.488603 * y) + u_sh_2 * (0.488603 * z) + u_sh_3 * (0.488603 * x)
            + u_sh_4 * (1.092548 * x * y) + u_sh_5 * (1.092548 * y * z)
            + u_sh_6 * (0.315392 * (3.0*z*z - 1.0))
            + u_sh_7 * (1.092548 * x * z)
            + u_sh_8 * (0.546274 * (x*x - y*y));
    }
    void main() {
        vec3 world_n = normalize(mat3(u_model) * normalize(pos));
        vec3 raw = eval_sh(world_n);
        v_color = vec4(clamp(raw/3.14, 0.0, 1.0), 1.0);
        gl_Position = u_mvp * u_model * vec4(pos, 1.0);
    }
    """)
    info.fragment_out(0, 'VEC4', "FragColor")
    info.fragment_source("void main() { FragColor = v_color; }")
    _sh_sphere_shader = gpu.shader.create_from_info(info)
    return _sh_sphere_shader



_envmap_shader = None


def _get_envmap_shader():
    """Lazy-create shader: reflective sphere sampling equirect envmap atlas."""
    global _envmap_shader
    if _envmap_shader is not None:
        return _envmap_shader
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('MAT4', "u_model")
    info.push_constant('VEC3', "u_cam_pos")
    info.sampler(0, 'FLOAT_2D', "u_envmap")
    vert_out = gpu.types.GPUStageInterfaceInfo("envmap_iface")
    vert_out.smooth('VEC3', "v_refl")
    info.vertex_out(vert_out)
    info.vertex_source("""
    void main() {
        vec3 world_pos = (u_model * vec4(pos, 1.0)).xyz;
        vec3 world_norm = normalize(mat3(u_model) * normalize(pos));
        vec3 view_dir = normalize(u_cam_pos - world_pos);
        v_refl = reflect(-view_dir, world_norm);
        gl_Position = u_mvp * u_model * vec4(pos, 1.0);
    }
    """)
    info.fragment_out(0, 'VEC4', "FragColor")
    info.fragment_source("""
    void main() {
        vec3 d = normalize(v_refl);
        float lon = atan(d.x, d.y);
        float lat = asin(clamp(d.z, -1.0, 1.0));
        vec2 uv = vec2(0.5 + 0.5 * lon / 3.14159, 0.5 - lat / 3.14159);
        uv.y = 1.0 - uv.y;
        uv.y = uv.y*256.0/504.0;
                         
        FragColor = texture(u_envmap, uv);
    }
    """)
    _envmap_shader = gpu.shader.create_from_info(info)
    return _envmap_shader


def _get_round_point_shader():
    """Lazy-create a shader for round GL_POINTS (circular discard in fragment)."""
    global _round_point_shader
    if _round_point_shader is not None:
        return _round_point_shader
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_color")
    info.vertex_source(
        "void main() { gl_Position = u_mvp * vec4(pos, 1.0); }")
    info.fragment_out(0, 'VEC4', "FragColor")
    info.fragment_source(
        "void main() {"
        "  vec2 c = gl_PointCoord - 0.5;"
        "  if (dot(c, c) > 0.25) discard;"
        "  FragColor = u_color;"
        "}")
    _round_point_shader = gpu.shader.create_from_info(info)
    return _round_point_shader


def _grid_draw():
    """Draw block grid overlay + probe previews in 3D viewports."""
    try:
        context = bpy.context
        scene = context.scene
        props = scene.splatting_properties
    except AttributeError:
        return

    from .splatting_data import get_state
    _ss = get_state()

    offset = np.array(props.block_offset, dtype=np.float32)
    block_size = props.block_size

    # ------------------------------------------------------------------
    # Collect world-space data
    # ------------------------------------------------------------------
    # Collect grid lines (only when show_block_grid is on)
    _all_world_verts = []
    _all_baked_probe_verts = []  # baked probe markers

    if props.show_block_grid and not _ss.is_rendering:
        # Before rendering: world-space AABB from object bound_box × matrix_world.
        items = scene.splatting_instances

        # Combined world-space AABB across all enabled instances
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

        if min_coords is not None:
            # Generate grid lines directly in world space
            origin = min_coords + offset
            bs = block_size

            xs = _steps(origin[0], origin[0] + int(np.ceil((max_coords[0] - origin[0]) / bs)) * bs, bs)
            ys = _steps(origin[1], origin[1] + int(np.ceil((max_coords[1] - origin[1]) / bs)) * bs, bs)
            zs = _steps(origin[2], origin[2] + int(np.ceil((max_coords[2] - origin[2]) / bs)) * bs, bs)

            if len(xs) >= 2 or len(ys) >= 2 or len(zs) >= 2:
                x_min, x_max = xs[0], xs[-1]
                y_min, y_max = ys[0], ys[-1]
                z_min, z_max = zs[0], zs[-1]

                lines = []
                for y in ys:
                    for z in zs:
                        lines.append([x_min, y, z])
                        lines.append([x_max, y, z])
                for x in xs:
                    for z in zs:
                        lines.append([x, y_min, z])
                        lines.append([x, y_max, z])
                for x in xs:
                    for y in ys:
                        lines.append([x, y, z_min])
                        lines.append([x, y, z_max])

                if lines:
                    _all_world_verts.append(np.array(lines, dtype=np.float32))

    # ------------------------------------------------------------------
    # Collect probe point positions from baked data (local → world via current matrix_world)
    # ------------------------------------------------------------------
    for item in scene.splatting_instances:
        if not item.enabled:
            continue
        obj = bpy.data.objects.get(item.mesh_name)
        if obj and obj.type == 'MESH' and len(obj.probe_points) > 0:
            mat = np.array(obj.matrix_world, dtype=np.float32)
            pts_local = np.array([list(pt.location) for pt in obj.probe_points], dtype=np.float32)
            ones = np.ones((len(pts_local), 1), dtype=np.float32)
            pts_world = (np.concatenate([pts_local, ones], axis=1) @ mat.T)[:, :3]
            _all_baked_probe_verts.append(pts_world)

    # Early exit if nothing to draw and preview is off
    if not _all_world_verts and not _all_baked_probe_verts:
        if not props.show_bake_preview:
            return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('ALWAYS')
    gpu.state.blend_set('ALPHA')

    # Grid lines
    if _all_world_verts:
        verts = np.concatenate(_all_world_verts, axis=0)
        fmt = GPUVertFormat()
        fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = GPUVertBuf(fmt, len(verts))
        vbo.attr_fill(id="pos", data=verts)
        batch = GPUBatch(type='LINES', buf=vbo)
        srgb = np.array(props.grid_color, dtype=np.float32)
        linear = np.sqrt(srgb)
        color = (*linear, props.grid_alpha)
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    # ---- show_bake_preview controls all probe/bake visualization ----
    if not props.show_bake_preview:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS')
        return

    # ------------------------------------------------------------------
    # Compute MVP / view matrix (shared by all preview types)
    # ------------------------------------------------------------------
    mvp = Matrix.Identity(4)
    view_matrix = None
    area = context.area
    if area and area.type == 'VIEW_3D':
        r3d = area.spaces.active.region_3d
        if r3d:
            mvp = r3d.window_matrix @ r3d.view_matrix
            view_matrix = r3d.view_matrix

    # ------------------------------------------------------------------
    # Baked SH probe spheres (when light probes exist)
    # ------------------------------------------------------------------
    if _all_baked_probe_verts and view_matrix is not None:
        sphere_scale = 0.075

        # Build sphere geometry once, cache for reuse
        global _sh_sphere_vbo, _sh_sphere_ibo
        if _sh_sphere_vbo is None:
            segs_s, rings_s = 16, 8
            s_verts = []
            s_verts.append([0.0, 0.0, 1.0])
            for i in range(1, rings_s):
                theta = i * np.pi / rings_s
                for j in range(segs_s):
                    phi = j * 2.0 * np.pi / segs_s
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                    s_verts.append([x, y, z])
            s_verts.append([0.0, 0.0, -1.0])
            unit_verts = np.array(s_verts, dtype=np.float32)

            s_idx = []
            for j in range(segs_s):
                a, b = 1 + j, 1 + (j + 1) % segs_s
                s_idx.extend([0, b, a])
            for i in range(rings_s - 2):
                for j in range(segs_s):
                    a = 1 + i * segs_s + j
                    b = 1 + i * segs_s + (j + 1) % segs_s
                    c = 1 + (i + 1) * segs_s + j
                    d = 1 + (i + 1) * segs_s + (j + 1) % segs_s
                    s_idx.extend([a, b, c])
                    s_idx.extend([b, d, c])
            last_pole = 1 + (rings_s - 1) * segs_s
            for j in range(segs_s):
                a = 1 + (rings_s - 2) * segs_s + j
                b = 1 + (rings_s - 2) * segs_s + (j + 1) % segs_s
                s_idx.extend([a, b, last_pole])
            sphere_indices = np.array(s_idx, dtype=np.int32)
            _sh_sphere_ibo = GPUIndexBuf(type='TRIS', seq=sphere_indices)

            fmt_pos = GPUVertFormat()
            fmt_pos.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
            _sh_sphere_vbo = GPUVertBuf(fmt_pos, len(unit_verts))
            _sh_sphere_vbo.attr_fill(id="pos", data=unit_verts)

        sphere_shader = _get_sh_sphere_shader()
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS')
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('FRONT')

        batch = GPUBatch(type='TRIS', buf=_sh_sphere_vbo, elem=_sh_sphere_ibo)

        for item in scene.splatting_instances:
            if not item.enabled:
                continue
            obj = bpy.data.objects.get(item.mesh_name)
            if not obj or obj.type != 'MESH' or len(obj.probe_points) == 0:
                continue
            mat = np.array(obj.matrix_world, dtype=np.float32)
            rot_mat = mat[:3, :3]

            for pt in obj.probe_points:
                lp = np.array(pt.location, dtype=np.float32)
                wp = rot_mat @ lp + mat[:3, 3]

                model = np.eye(4, dtype=np.float32)
                model[:3, :3] = rot_mat * sphere_scale
                model[:3, 3] = wp

                sh_r = np.array(pt.sh_r[:], dtype=np.float32)
                sh_g = np.array(pt.sh_g[:], dtype=np.float32)
                sh_b = np.array(pt.sh_b[:], dtype=np.float32)

                sphere_shader.bind()
                sphere_shader.uniform_float("u_mvp", mvp)
                sphere_shader.uniform_float("u_model", model.T.ravel().tolist())
                for i in range(9):
                    sphere_shader.uniform_float(f"u_sh_{i}",
                        (float(sh_r[i]), float(sh_g[i]), float(sh_b[i])))
                batch.draw(sphere_shader)

        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('ALWAYS')

    # ------------------------------------------------------------------
    # Bake area bounding box (only when no baked probes)
    # ------------------------------------------------------------------
    if not _all_baked_probe_verts:
        center = np.array(props.bake_area_center, dtype=np.float32)
        half = np.array(props.bake_area_size, dtype=np.float32) * 0.5
        bb_lo = center - half
        bb_hi = center + half
        bake_bbox = _aabb_wireframe(bb_lo, bb_hi)
        bbox_fmt = GPUVertFormat()
        bbox_fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        bbox_vbo = GPUVertBuf(bbox_fmt, len(bake_bbox))
        bbox_vbo.attr_fill(id="pos", data=bake_bbox)
        bbox_batch = GPUBatch(type='LINES', buf=bbox_vbo)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.6, 0.0, 0.5))
        bbox_batch.draw(shader)

    # ------------------------------------------------------------------
    # Auto-placed probe previews
    # ------------------------------------------------------------------
    from .splatting_data import try_update_probe_preview, get_probe_positions
    try_update_probe_preview(context)

    # Green dots — irradiance preview (only when no baked light probes)
    if not _all_baked_probe_verts:
        irradiance_pos = get_probe_positions('irradiance')
        if irradiance_pos is not None and len(irradiance_pos) > 0:
            fmt = GPUVertFormat()
            fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
            vbo = GPUVertBuf(fmt, len(irradiance_pos))
            vbo.attr_fill(id="pos", data=np.asarray(irradiance_pos, dtype=np.float32))
            batch = GPUBatch(type='POINTS', buf=vbo)
            rs = _get_round_point_shader()
            rs.bind()
            rs.uniform_float("u_mvp", mvp)
            rs.uniform_float("u_color", (0.2, 0.8, 0.2, 0.9))
            gpu.state.point_size_set(14)
            batch.draw(rs)

    # Blue dot — envmap center (only when no baked envmap atlas exists)
    has_envmap = any(
        getattr(bpy.data.objects.get(item.mesh_name), 'envmap_atlas', '')
        for item in scene.splatting_instances if item.enabled
    )
    if not has_envmap:
        envmap_pos = get_probe_positions('envmap')
        if envmap_pos is not None and len(envmap_pos) > 0:
            fmt = GPUVertFormat()
            fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
            vbo = GPUVertBuf(fmt, len(envmap_pos))
            vbo.attr_fill(id="pos", data=np.asarray(envmap_pos, dtype=np.float32))
            batch = GPUBatch(type='POINTS', buf=vbo)
            rs = _get_round_point_shader()
            rs.bind()
            rs.uniform_float("u_mvp", mvp)
            rs.uniform_float("u_color", (0.2, 0.4, 1.0, 0.9))
            gpu.state.point_size_set(28)
            gpu.state.depth_test_set('LESS')
            gpu.state.depth_mask_set(True)
            batch.draw(rs)

    gpu.state.point_size_set(1)

    # ------------------------------------------------------------------
    # Envmap reflection sphere (if any instance has an envmap atlas)
    # ------------------------------------------------------------------
    envmap_img_name = None
    for item in scene.splatting_instances:
        if not item.enabled:
            continue
        obj = bpy.data.objects.get(item.mesh_name)
        if obj and obj.type == 'MESH' and getattr(obj, 'envmap_atlas', ''):
            envmap_img_name = obj.envmap_atlas
            break
    if envmap_img_name and view_matrix is not None:
        env_segs, env_rings = 32, 16
        ev = [[0.0, 0.0, 1.0]]
        for i in range(1, env_rings):
            theta = i * np.pi / env_rings
            for j in range(env_segs):
                phi = j * 2.0 * np.pi / env_segs
                ev.append([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])
        ev.append([0.0, 0.0, -1.0])
        env_unit_verts = np.array(ev, dtype=np.float32)

        ei = []
        for j in range(env_segs):
            a, b = 1 + j, 1 + (j + 1) % env_segs
            ei.extend([0, b, a])
        for i in range(env_rings - 2):
            for j in range(env_segs):
                a = 1 + i * env_segs + j
                b = 1 + i * env_segs + (j + 1) % env_segs
                c = 1 + (i + 1) * env_segs + j
                d = 1 + (i + 1) * env_segs + (j + 1) % env_segs
                ei.extend([a, b, c])
                ei.extend([b, d, c])
        last_pole = 1 + (env_rings - 1) * env_segs
        for j in range(env_segs):
            a = 1 + (env_rings - 2) * env_segs + j
            b = 1 + (env_rings - 2) * env_segs + (j + 1) % env_segs
            ei.extend([b, a, last_pole])
        env_ibo = GPUIndexBuf(type='TRIS', seq=np.array(ei, dtype=np.int32))
        env_fmt = GPUVertFormat()
        env_fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        env_vbo = GPUVertBuf(env_fmt, len(env_unit_verts))
        env_vbo.attr_fill(id="pos", data=env_unit_verts)
        env_sphere_scale = 0.3  # 4x of SH sphere scale (0.075)

        img = bpy.data.images.get(envmap_img_name)
        if img:
            tex = gpu.texture.from_image(img)
            cam_pos = view_matrix.inverted().translation
            center_wp = np.array(props.bake_area_center, dtype=np.float32)
            env_model = np.eye(4, dtype=np.float32)
            env_model[:3, :3] = np.eye(3, dtype=np.float32) * env_sphere_scale
            env_model[:3, 3] = center_wp
            env_shader = _get_envmap_shader()
            env_shader.bind()
            env_shader.uniform_float("u_mvp", mvp)
            env_shader.uniform_float("u_model", env_model.T.ravel().tolist())
            env_shader.uniform_float("u_cam_pos", (cam_pos.x, cam_pos.y, cam_pos.z))
            env_shader.uniform_sampler("u_envmap", tex)
            gpu.state.depth_test_set('LESS')
            gpu.state.depth_mask_set(True)
            gpu.state.face_culling_set('FRONT')
            batch = GPUBatch(type='TRIS', buf=env_vbo, elem=env_ibo)
            batch.draw(env_shader)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS')
    gpu.state.face_culling_set('NONE')


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
