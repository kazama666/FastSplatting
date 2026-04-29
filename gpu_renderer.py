import gpu
import numpy as np
from gpu.types import (
    GPUBatch,
    GPUVertBuf,
    GPUVertFormat,
    GPUIndexBuf,
)
from gpu import state
from mathutils import Matrix, Vector


LOD_THRESHOLDS = [100, 30, 10]
LOD_LEVELS = 3


class SplattingRenderer:
    def __init__(self):
        self.shader = None
        self.lod_batches = {}
        self.initialized = False
        self._batch_cache = {}
        self._matrices_ubo = None

    def init_buffers(self, positions, colors, opacities, scales, rotations):
        """Initialize GPU buffers"""
        if len(positions) == 0:
            return False

        N = len(positions)

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
        // Hue rotation around (1,1,1) axis in RGB space
        const vec3 k = vec3(0.57735, 0.57735, 0.57735);
        float cosH = cos(u_Hue * 3.14159);
        float sinH = sin(u_Hue * 3.14159);
        v_color = v_color * cosH + cross(k, v_color) * sinH + k * dot(k, v_color) * (1.0 - cosH);

        // Saturation
        float lum = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        v_color = mix(vec3(lum), v_color, u_Saturation);

        // Brightness
        v_color *= u_Brightness;

        // Gamma
        v_color = pow(max(v_color, vec3(0.0)), vec3(1.0 / u_Gamma));
    }

    vec4 pos_view = u_Matrices.u_ViewMatrix * vec4(inst_position, 1.0);
    vec4 pos_clip = u_Matrices.u_VPMatrix * vec4(inst_position, 1.0);

    if (pos_view.z >= -0.001) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }

    // Reconstruct 3D covariance from precomputed attributes
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
    // Squared Mahalanobis distance via conic matrix
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

        # Create UBO for matrices (2 × mat4 = 128 bytes)
        self._matrices_ubo = gpu.types.GPUUniformBuf(bytearray([0]) * 128)

        del vert_out
        del shader_info
        self.initialized = True
        return True

    @staticmethod
    def _compute_cov3d(scales, rotations):
        """Vectorized 3D covariance from scales and rotations (GLSL column-major)"""
        r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        s0, s1, s2 = scales[:, 0], scales[:, 1], scales[:, 2]

        # Rotation matrix in GLSL column-major: R[row][col]
        R00 = 1.0 - 2.0 * (y*y + z*z)
        R01 = 2.0 * (x*y + r*z)
        R02 = 2.0 * (x*z - r*y)

        R10 = 2.0 * (x*y - r*z)
        R11 = 1.0 - 2.0 * (x*x + z*z)
        R12 = 2.0 * (y*z + r*x)

        R20 = 2.0 * (x*z + r*y)
        R21 = 2.0 * (y*z - r*x)
        R22 = 1.0 - 2.0 * (x*x + y*y)

        # cov = (S*R)^T * (S*R) → cov[i][j] = sum_k s_k^2 * R[k][i] * R[k][j]
        s0_2, s1_2, s2_2 = s0*s0, s1*s1, s2*s2

        cov_a = np.column_stack([
            s0_2*R00*R00 + s1_2*R10*R10 + s2_2*R20*R20,  # xx
            s0_2*R00*R01 + s1_2*R10*R11 + s2_2*R20*R21,  # xy
            s0_2*R00*R02 + s1_2*R10*R12 + s2_2*R20*R22,  # xz
        ])
        cov_b = np.column_stack([
            s0_2*R01*R01 + s1_2*R11*R11 + s2_2*R21*R21,  # yy
            s0_2*R01*R02 + s1_2*R11*R12 + s2_2*R21*R22,  # yz
            s0_2*R02*R02 + s1_2*R12*R12 + s2_2*R22*R22,  # zz
        ])
        return cov_a, cov_b

    def _build_billboard_batch(self, positions, colors, opacities, scales, rotations):
        """Build billboard batch with precomputed 3D covariance"""
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

        # Precompute 3D covariance on CPU (view-independent)
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

    def _get_or_create_block_batch(self, block_idx, lod_level, state_obj):
        """Get cached batch for a block at given LOD level"""
        cache_key = ('b', block_idx, lod_level)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        block_splat_indices = state_obj.block_splat_indices[block_idx]
        # LOD: keep top N largest splats per block (sorted by size in preprocessing)
        lod_factor = 4 ** lod_level
        count = max(1, len(block_splat_indices) // lod_factor)
        lod_indices = block_splat_indices[:count]
        if len(lod_indices) == 0:
            return None

        batch = self._build_billboard_batch(
            state_obj.positions[lod_indices],
            state_obj.colors[lod_indices],
            state_obj.opacities[lod_indices],
            state_obj.scales[lod_indices],
            state_obj.rotations[lod_indices],
        )
        self._batch_cache[cache_key] = batch
        return batch

    def _get_or_create_fallback_batch(self, state_obj):
        """Fallback single batch when nothing is visible - top 1/16 per block"""
        cache_key = ('fallback', 0)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        lod_indices = []
        for block_idx in range(state_obj.block_count):
            indices = state_obj.block_splat_indices[block_idx]
            count = max(1, len(indices) // 16)
            lod_indices.extend(indices[:count])
        if len(lod_indices) == 0:
            return None

        batch = self._build_billboard_batch(
            state_obj.positions[lod_indices],
            state_obj.colors[lod_indices],
            state_obj.opacities[lod_indices],
            state_obj.scales[lod_indices],
            state_obj.rotations[lod_indices],
        )
        self._batch_cache[cache_key] = batch
        return batch

    @staticmethod
    def _compute_screen_size(center_clip, block_center, block_radius, vp_matrix):
        """Compute screen-space diameter in pixels from pre-computed center_clip"""
        if center_clip[3] <= 0:
            return 0

        edge_clip = vp_matrix @ Vector((block_center[0] + block_radius, block_center[1], block_center[2], 1.0))
        if edge_clip[3] <= 0:
            return float('inf')

        viewport = state.viewport_get()
        vp_w, vp_h = viewport[2], viewport[3]

        center_ndc = (center_clip[0] / center_clip[3], center_clip[1] / center_clip[3])
        edge_ndc = (edge_clip[0] / edge_clip[3], edge_clip[1] / edge_clip[3])

        center_sx = (center_ndc[0] + 1) * 0.5 * vp_w
        center_sy = (center_ndc[1] + 1) * 0.5 * vp_h
        edge_sx = (edge_ndc[0] + 1) * 0.5 * vp_w
        edge_sy = (edge_ndc[1] + 1) * 0.5 * vp_h

        return ((center_sx - edge_sx)**2 + (center_sy - edge_sy)**2) ** 0.5 * 2

    @staticmethod
    def _extract_frustum_planes(vp_matrix):
        """Extract 6 frustum planes from VP matrix (row-major).
        Returns list of (normal, distance) tuples: left, right, bottom, top, near, far."""
        rows = [vp_matrix[i] for i in range(4)]
        signs = [(3, 0, 1), (3, 0, -1), (3, 1, 1), (3, 1, -1), (3, 2, 1), (3, 2, -1)]
        planes = []
        for r1, r2, s in signs:
            if s > 0:
                plane = rows[r1] + rows[r2]
            else:
                plane = rows[r1] - rows[r2]
            n = plane.to_3d()
            d = plane[3]
            length = n.length
            if length > 1e-8:
                n /= length
                d /= length
            planes.append((n, d))
        return planes

    @staticmethod
    def _compute_lod_level(screen_size, lod_bias=1.0):
        """Determine LOD level from screen size and bias"""
        t2 = LOD_THRESHOLDS[2] * lod_bias
        t1 = LOD_THRESHOLDS[1] * lod_bias
        t0 = LOD_THRESHOLDS[0] * lod_bias
        if screen_size < t2:
            return 2
        elif screen_size < t1:
            return 1
        elif screen_size < t0:
            return 1
        return 0

    def _update_matrices_ubo(self, vp_matrix, view_matrix):
        """Pack matrices into UBO in column-major order for GLSL"""
        if self._matrices_ubo is None:
            return
        data = np.zeros(32, dtype=np.float32)
        # Convert row-major (Blender) → column-major (GLSL mat4)
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
        """Draw splats with proper covariance"""
        if not self.initialized:
            return

        if self.shader is None:
            return

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
        state_obj = get_state()

        if state_obj.point_count == 0:
            return

        if state_obj.target_mesh:
            model_matrix = state_obj.target_mesh.matrix_world.copy()
            vp_matrix = vp_matrix @ model_matrix
            view_matrix = view_matrix @ model_matrix

        camera_pos = view_matrix.inverted().translation
        state_obj._camera_pos_np = np.array(
            [camera_pos[0], camera_pos[1], camera_pos[2]], dtype=np.float32)

        # Store VP matrix + projection params for sort visibility check
        state_obj._vp_matrix = vp_matrix
        state_obj._proj_00 = proj_00
        state_obj._proj_11 = proj_11

        splatting_props = context.scene.splatting_properties
        # Auto-sort: detect view change and do one batch per frame
        from .splatting_data import check_view_changed, sort_next_batch
        if check_view_changed(view_matrix):
            state_obj._sort_active = True
            if state_obj.sorted_up_to >= state_obj.block_count:
                state_obj.sorted_up_to = 0
        sort_next_batch(context)
        # lod_enabled = splatting_props.lod_enabled
        # lod_bias = splatting_props.lod_bias
        near_to_far = splatting_props.sort_near_to_far

        # ----- State setup (alpha blend) -----
        state.blend_set('ALPHA_PREMULT')
        # Vectorized block distance computation
        block_centers = state_obj.block_centers
        block_radii = state_obj.block_radii
        cam_pos_np = state_obj._camera_pos_np
        distances = np.linalg.norm(block_centers - cam_pos_np, axis=1)
        near_end = distances - block_radii
        block_order = np.argsort(near_end)
        if not near_to_far:
            block_order = block_order[::-1]
        state.depth_mask_set(False)
        state.depth_test_set('LESS')

        # ----- Shader bind + uniforms -----
        self._update_matrices_ubo(vp_matrix, view_matrix)
        self.shader.bind()
        self.shader.uniform_block('u_Matrices', self._matrices_ubo)
        self.shader.uniform_float("u_CameraPos", camera_pos)
        self.shader.uniform_float("u_FocalParams", (focal, focal))
        self.shader.uniform_float("u_ViewportSize", (vp_w, vp_h))
        self.shader.uniform_float("u_QuadScale", splatting_props.quad_scale)
        self.shader.uniform_float("u_Gamma", splatting_props.color_gamma)
        self.shader.uniform_float("u_Hue", splatting_props.color_hue)
        self.shader.uniform_float("u_Saturation", splatting_props.color_saturation)
        self.shader.uniform_float("u_Brightness", splatting_props.color_brightness)
        self.shader.uniform_float("u_Tint", splatting_props.color_tint)

        # ----- Block rendering loop -----
        # Vectorized frustum culling
        count = len(block_centers)
        ones = np.ones((count, 1), dtype=np.float32)
        centers_h = np.concatenate([block_centers, ones], axis=1)
        vp_np = np.asarray(vp_matrix, dtype=np.float32)
        clip_pos = centers_h @ vp_np.T  # (N, 4)

        visible_mask = (clip_pos[:, 3] > 0) & \
            (np.abs(clip_pos[:, 0]) < clip_pos[:, 3] + block_radii * proj_00) & \
            (np.abs(clip_pos[:, 1]) < clip_pos[:, 3] + block_radii * proj_11)

        total_drawn = 0
        total_splats = 0
        for idx in block_order:
            if not visible_mask[idx]:
                continue

            lod_level = 0  # LOD disabled — always full detail
            # LOD branching disabled:
            # if lod_enabled:
            #     screen_size = self._compute_screen_size(
            #         center_clip, bc, br, vp_matrix)
            #     lod_level = self._compute_lod_level(screen_size, lod_bias)

            batch = self._get_or_create_block_batch(idx, lod_level, state_obj)
            if batch:
                batch.draw(self.shader)
                total_drawn += 1
                lod_factor = 4 ** lod_level
                total_splats += max(1, len(state_obj.block_splat_indices[idx]) // lod_factor)

        # Fallback if nothing was drawn
        if total_drawn == 0 and state_obj.point_count > 0:
            batch = self._get_or_create_fallback_batch(state_obj)
            if batch:
                batch.draw(self.shader)
                total_drawn = state_obj.block_count
                total_splats = max(1, state_obj.point_count // 16)

        state_obj.displayed_block_count = total_drawn
        state_obj.displayed_splat_count = total_splats or state_obj.point_count

        # ----- Reset state -----
        state.depth_test_set('LESS')
        state.depth_mask_set(False)
        state.blend_set('NONE')

    def release(self):
        """Release GPU resources"""
        self.lod_batches = {}
        self._batch_cache = {}
        self.shader = None
        self.initialized = False
        self._matrices_ubo = None

    def clear_cache(self):
        """Clear batch cache (call after splat index reordering)"""
        self._batch_cache = {}

    def clear_block_cache(self, block_indices):
        """Clear cached batches only for specific blocks (incremental sort)."""
        for block_idx in block_indices:
            for lod in range(LOD_LEVELS):
                self._batch_cache.pop(('b', block_idx, lod), None)
        self._batch_cache.pop(('fallback', 0), None)


_renderer = SplattingRenderer()


def get_renderer():
    return _renderer


def init_renderer(positions, colors, opacities, scales, rotations):
    return _renderer.init_buffers(positions, colors, opacities, scales, rotations)


def release_renderer():
    _renderer.release()


def draw_splatting(context):
    _renderer.draw(context)