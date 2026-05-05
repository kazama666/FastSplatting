import os
import bpy
from bpy import types

# ---------------------------------------------------------------------------
# POST_VIEW draw handler — captures the 3D viewport framebuffer AFTER splats
# have been drawn.  Communication with the export modal operator is through
# module-level globals.
# ---------------------------------------------------------------------------
_capture_handler = None
_capture_requested = False
_capture_ready = False


def _post_view_capture():
    global _capture_requested, _capture_ready
    if not _capture_requested:
        return
    _capture_requested = False
    _capture_ready = True


def _register_capture_handler():
    global _capture_handler
    if _capture_handler is None:
        _capture_handler = bpy.types.SpaceView3D.draw_handler_add(
            _post_view_capture, (), 'WINDOW', 'POST_VIEW')


def _unregister_capture_handler():
    global _capture_handler
    if _capture_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_capture_handler, 'WINDOW')
        _capture_handler = None


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class SPLATTING_OT_start_render(types.Operator):
    bl_idname = "splatting.start_render"
    bl_label = "Start Render"
    bl_description = "Initialize and start Gaussian Splatting render"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from . import splatting_data
        splatting_data.start_render(context)
        return {'FINISHED'}


class SPLATTING_OT_stop_render(types.Operator):
    bl_idname = "splatting.stop_render"
    bl_label = "Stop Render"
    bl_description = "Stop Gaussian Splatting render"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from . import splatting_data
        splatting_data.stop_render(context)
        return {'FINISHED'}


class SPLATTING_OT_add_instance(types.Operator):
    bl_idname = "splatting.add_instance"
    bl_label = "Add Splat Instance"
    bl_description = "Add the selected mesh object as a splat instance"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Please select a mesh object first")
            return {'CANCELLED'}

        # Check for duplicates
        for item in context.scene.splatting_instances:
            if item.mesh_name == obj.name:
                self.report({'INFO'}, f"'{obj.name}' is already in the list")
                return {'CANCELLED'}

        item = context.scene.splatting_instances.add()
        item.mesh_name = obj.name
        item.mesh_uid = obj.session_uid
        item.enabled = True
        _init_instance_defaults(item, context)
        context.scene.splatting_properties.active_instance_index = len(context.scene.splatting_instances) - 1
        self.report({'INFO'}, f"Added '{obj.name}'")
        return {'FINISHED'}


def _init_instance_defaults(item, context):
    """Copy scene-level render defaults onto a new SplattingInstanceItem."""
    d = context.scene.splatting_properties
    item.color_tint = d.default_color_tint
    item.color_brightness = d.default_color_brightness
    item.color_gamma = d.default_color_gamma
    item.color_hue = d.default_color_hue
    item.color_saturation = d.default_color_saturation
    item.brightness_gain = d.default_brightness_gain
    item.brightness_gain_start = d.default_brightness_gain_start


class SPLATTING_OT_remove_instance(types.Operator):
    bl_idname = "splatting.remove_instance"
    bl_label = "Remove Splat Instance"
    bl_description = "Remove this splat instance from the list"

    def execute(self, context):
        props = context.scene.splatting_properties
        idx = props.active_instance_index
        instances = context.scene.splatting_instances
        if 0 <= idx < len(instances):
            item = instances[idx]
            name = item.mesh_name
            instances.remove(idx)
            n = len(instances)
            props.active_instance_index = min(idx, n - 1)
            self.report({'INFO'}, f"Removed '{name}'")
        return {'FINISHED'}


class SPLATTING_OT_move_instance(types.Operator):
    bl_idname = "splatting.move_instance"
    bl_label = "Move Instance"
    bl_description = "Change render order of this instance"

    direction: bpy.props.EnumProperty(
        items=[
            ('UP', 'Up', 'Move earlier in the list'),
            ('DOWN', 'Down', 'Move later in the list'),
        ],
    )

    def execute(self, context):
        props = context.scene.splatting_properties
        idx = props.active_instance_index
        items = context.scene.splatting_instances
        n = len(items)
        if n < 2:
            return {'CANCELLED'}
        new_idx = idx
        if self.direction == 'UP' and idx > 0:
            items.move(idx, idx - 1)
            new_idx = idx - 1
        elif self.direction == 'DOWN' and idx < n - 1:
            items.move(idx, idx + 1)
            new_idx = idx + 1
        else:
            return {'CANCELLED'}
        props.active_instance_index = new_idx

        # Reorder runtime instances too
        from .splatting_data import get_state
        scene = get_state()
        if scene.is_rendering and len(scene.instances) == n:
            a, b = new_idx, idx
            scene.instances[a], scene.instances[b] = scene.instances[b], scene.instances[a]

        return {'FINISHED'}


class SPLATTING_OT_toggle_instance(types.Operator):
    bl_idname = "splatting.toggle_instance"
    bl_label = "Toggle Instance Visibility"
    bl_description = "Show/hide this splat instance"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        instances = context.scene.splatting_instances
        if 0 <= self.index < len(instances):
            instances[self.index].enabled = not instances[self.index].enabled
        # Tag redraw so the toggle updates immediately
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                break
        return {'FINISHED'}


class SPLATTING_OT_export_animation(bpy.types.Operator):
    bl_idname = "splatting.export_animation"
    bl_label = "Export Viewport Animation"
    bl_description = "Export 3D viewport (with splats) as image sequence"

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------
    def invoke(self, context, event):
        props = context.scene.splatting_properties
        self._current = props.anim_start_frame
        self._end = props.anim_end_frame
        self._output_dir = bpy.path.abspath(props.anim_output_path)
        self._orig_frame = context.scene.frame_current

        if not os.path.isdir(self._output_dir):
            try:
                os.makedirs(self._output_dir, exist_ok=True)
            except Exception:
                self.report({'ERROR'}, f"Cannot create directory: {self._output_dir}")
                return {'CANCELLED'}

        # Find 3D view area and its WINDOW region
        self._view3d_area = None
        self._view3d_region = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                self._view3d_area = area
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        self._view3d_region = reg
                        break
                break
        if self._view3d_area is None or self._view3d_region is None:
            self.report({'ERROR'}, "No 3D viewport found")
            return {'CANCELLED'}

        # Save overlays state and disable during export
        self._space3d = self._view3d_area.spaces.active
        overlay = self._space3d.overlay
        self._orig_show_overlays = overlay.show_overlays
        overlay.show_overlays = False

        # Camera-change-triggered sorting setting
        self._force_sort = props.anim_force_sort

        # Maximise 3D view to fill the entire window (hide all UI/panels)
        with context.temp_override(area=self._view3d_area):
            bpy.ops.screen.screen_full_area(use_hide_panels=True)
        self._needs_restore = True

        # Register the POST_VIEW capture handler
        _register_capture_handler()

        # Delay start so full-area has time to settle
        self._delay_ticks = 20
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # modal
    # ------------------------------------------------------------------
    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._delay_ticks > 0:
            self._delay_ticks -= 1
            return {'RUNNING_MODAL'}

        if not hasattr(self, '_started'):
            self._started = True
            context.scene.frame_set(self._current)
            if self._force_sort:
                self._maybe_sort(context)
            self._request_capture()
            context.window_manager.progress_begin(self._current, self._end)
            return {'RUNNING_MODAL'}

        if self._current > self._end:
            self._finish(context)
            self.report({'INFO'}, f"Exported to {self._output_dir}")
            return {'FINISHED'}

        global _capture_ready

        if _capture_ready:
            filepath = os.path.join(self._output_dir, f"frame_{self._current:04d}.png")
            self._save_capture(filepath)
            context.window_manager.progress_update(self._current)

            _capture_ready = False

            self._current += 1
            if self._current > self._end:
                self._finish(context)
                self.report({'INFO'}, f"Exported to {self._output_dir}")
                return {'FINISHED'}

            context.scene.frame_set(self._current)
            if self._force_sort:
                self._maybe_sort(context)
            self._request_capture()
        else:
            if not _capture_requested:
                self._request_capture()
            else:
                self._view3d_area.tag_redraw()

        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _maybe_sort(self, context):
        """Sort all instances far-to-near if camera has moved since last capture."""
        from . import splatting_data

        r3d = None
        area = context.area
        if area and area.type == 'VIEW_3D' and area.spaces.active:
            r3d = area.spaces.active.region_3d
        if r3d is None:
            for a in context.screen.areas:
                if a.type == 'VIEW_3D' and a.spaces.active:
                    r3d = a.spaces.active.region_3d
                    break
        if r3d is None:
            return

        if r3d.view_perspective == 'CAMERA' and context.scene.camera:
            cam_pos = context.scene.camera.matrix_world.translation
        else:
            cam_pos = r3d.view_matrix.inverted().translation

        # Only re-sort when camera moves more than one block
        threshold = context.scene.splatting_properties.block_size
        if hasattr(self, '_last_cam_pos') and self._last_cam_pos is not None:
            if (cam_pos - self._last_cam_pos).length < threshold:
                return

        self._last_cam_pos = cam_pos.copy()

        # Reset sort state and sort all instances
        scene = splatting_data.get_state()
        for inst in scene.instances:
            inst._sort_active = False
            inst.clear_cache()
        splatting_data.sort_blocks_far_to_near(cam_pos)

    def _request_capture(self):
        global _capture_requested
        _capture_requested = True
        self._view3d_area.tag_redraw()

    def _save_capture(self, filepath):
        bpy.ops.screen.screenshot(filepath=filepath)

    def _finish(self, context):
        if hasattr(self, '_orig_show_overlays') and self._space3d:
            try:
                self._space3d.overlay.show_overlays = self._orig_show_overlays
            except Exception:
                pass

        if getattr(self, '_needs_restore', False):
            bpy.ops.screen.screen_full_area(use_hide_panels=True)
            self._needs_restore = False
        context.scene.frame_set(self._orig_frame)
        context.window_manager.progress_end()
        if hasattr(self, '_timer'):
            context.window_manager.event_timer_remove(self._timer)
        _unregister_capture_handler()

        global _capture_ready
        _capture_ready = False




class SPLATTING_OT_import_spz(types.Operator):
    bl_idname = "import_scene.spz"
    bl_label = "Import Spark SPZ (.spz)"
    bl_description = "Import a Niantic Spark SPZ file as a splat instance"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default='*.spz', options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Switch to OBJECT mode for mesh operations / selection
        raw_mode = bpy.context.mode
        prev_mode = 'EDIT' if raw_mode.startswith('EDIT_') else raw_mode
        if prev_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        return self._execute_impl(context)

    def _execute_impl(self, context):
        import numpy as np
        from .load_spz import load_spz

        spz_path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(spz_path):
            self.report({'ERROR'}, f"File not found: {spz_path}")
            return {'CANCELLED'}

        try:
            positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree = \
                load_spz(spz_path)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to decode SPZ: {e}")
            return {'CANCELLED'}

        N = len(positions)
        if N == 0:
            self.report({'ERROR'}, "No splats found")
            return {'CANCELLED'}

        name = os.path.splitext(os.path.basename(spz_path))[0]
        mesh = bpy.data.meshes.new(name)
        verts = [(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        mesh.from_pydata(verts, [], [])

        # Build attribute list matching the PLY schema
        sh_n = sh_coeffs.shape[1] if sh_coeffs is not None else 0
        attr_list = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                     'scale_0', 'scale_1', 'scale_2',
                     'rot_0', 'rot_1', 'rot_2', 'rot_3']
        if sh_n > 0:
            attr_list += [f'f_rest_{i}' for i in range(sh_n)]
        for attr_name in attr_list:
            mesh.attributes.new(attr_name, 'FLOAT', 'POINT')

        # f_dc (pre-sigmoid SH DC).
        # SPZ stores display-oriented color centered at 0.5; invert sigmoid
        # so that read_ply_attributes() → sigmoid(f_dc) recovers the color.
        eps = 1e-6
        dc_clip = np.clip(raw_dc, eps, 1.0 - eps)
        raw_dc_logit = np.log(dc_clip / (1.0 - dc_clip))
        mesh.attributes['f_dc_0'].data.foreach_set('value', raw_dc_logit[:, 0].astype(np.float32))
        mesh.attributes['f_dc_1'].data.foreach_set('value', raw_dc_logit[:, 1].astype(np.float32))
        mesh.attributes['f_dc_2'].data.foreach_set('value', raw_dc_logit[:, 2].astype(np.float32))

        # SH rest (capped at degree 1)
        if sh_n > 0 and sh_degree >= 1:
            rest_count = min(sh_n, 9)
            for i in range(rest_count):
                mesh.attributes[f'f_rest_{i}'].data.foreach_set(
                    'value', sh_coeffs[:, i].astype(np.float32))

        # Opacity logit (inverse of sigmoid, matching PLY convention)
        eps = 1e-6
        op_clip = np.clip(opacities.ravel(), eps, 1.0 - eps)
        opacity_logit = np.log(op_clip / (1.0 - op_clip))
        mesh.attributes['opacity'].data.foreach_set('value', opacity_logit)

        # Scale log (matching PLY convention — mesh stores log(scale))
        mesh.attributes['scale_0'].data.foreach_set(
            'value', np.log(np.clip(scales[:, 0], 1e-10, None)).astype(np.float32))
        mesh.attributes['scale_1'].data.foreach_set(
            'value', np.log(np.clip(scales[:, 1], 1e-10, None)).astype(np.float32))
        mesh.attributes['scale_2'].data.foreach_set(
            'value', np.log(np.clip(scales[:, 2], 1e-10, None)).astype(np.float32))

        # Rotations (quaternion, wxyz)
        mesh.attributes['rot_0'].data.foreach_set('value', rotations[:, 0])
        mesh.attributes['rot_1'].data.foreach_set('value', rotations[:, 1])
        mesh.attributes['rot_2'].data.foreach_set('value', rotations[:, 2])
        mesh.attributes['rot_3'].data.foreach_set('value', rotations[:, 3])

        obj = bpy.data.objects.new(name, mesh)
        obj.rotation_euler.x = -1.5708  # -90° in radians
        context.collection.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Add to splatting instance list
        for item in context.scene.splatting_instances:
            if item.mesh_name == obj.name:
                self.report({'INFO'}, f"Imported '{obj.name}' ({N:,} splats)")
                return {'FINISHED'}
        item = context.scene.splatting_instances.add()
        item.mesh_name = obj.name
        item.mesh_uid = obj.session_uid
        item.enabled = True
        _init_instance_defaults(item, context)
        context.scene.splatting_properties.active_instance_index = \
            len(context.scene.splatting_instances) - 1

        self.report({'INFO'}, f"Imported '{obj.name}' ({N:,} splats)")
        return {'FINISHED'}


class SPLATTING_OT_set_default_render(types.Operator):
    bl_idname = "splatting.set_default_render"
    bl_label = "Set as Default"
    bl_description = "Copy current instance's render adjustments to scene defaults for new instances"

    def execute(self, context):
        instances = context.scene.splatting_instances
        idx = context.scene.splatting_properties.active_instance_index
        if idx < 0 or idx >= len(instances):
            self.report({'WARNING'}, "No instance selected")
            return {'CANCELLED'}
        src = instances[idx]
        d = context.scene.splatting_properties
        d.default_color_tint = src.color_tint
        d.default_color_brightness = src.color_brightness
        d.default_color_gamma = src.color_gamma
        d.default_color_hue = src.color_hue
        d.default_color_saturation = src.color_saturation
        d.default_brightness_gain = src.brightness_gain
        d.default_brightness_gain_start = src.brightness_gain_start
        self.report({'INFO'}, "Render adjustments saved as default")
        return {'FINISHED'}


class SPLATTING_OT_bake_lightprobe(types.Operator):
    bl_idname = "splatting.bake_lightprobe"
    bl_label = "Bake Light Probe"
    bl_description = "Bake SH2 light probes at block grid intersections for this splat instance"

    def execute(self, context):
        import numpy as np

        props = context.scene.splatting_properties
        idx = props.active_instance_index
        instances = context.scene.splatting_instances
        if idx < 0 or idx >= len(instances):
            self.report({'WARNING'}, "No instance selected")
            return {'CANCELLED'}

        item = instances[idx]
        obj = bpy.data.objects.get(item.mesh_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Instance mesh not found")
            return {'CANCELLED'}

        mesh = obj.data
        if len(mesh.vertices) == 0:
            self.report({'ERROR'}, "Mesh has no vertices")
            return {'CANCELLED'}

        from .splatting_data import read_ply_attributes
        positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree = \
            read_ply_attributes(mesh)

        N = len(positions)
        if N == 0:
            self.report({'ERROR'}, "No splat data found")
            return {'CANCELLED'}

        # World-space positions
        mat = np.array(obj.matrix_world, dtype=np.float32)
        ones = np.ones((N, 1), dtype=np.float32)
        positions_h = np.concatenate([positions, ones], axis=1)
        positions_world = (positions_h @ mat.T)[:, :3]

        # Auto-generate probe positions
        from .splatting_data import compute_probe_positions
        irr_positions, _ = compute_probe_positions(context)
        if irr_positions is None or len(irr_positions) == 0:
            self.report({'ERROR'}, "No probe positions could be generated")
            return {'CANCELLED'}
        probe_positions = np.asarray(irr_positions, dtype=np.float32)
        num_probes = len(probe_positions)

        # Block-grid dimensions for splat filtering only (not probe placement)
        bbox_local = np.array(obj.bound_box, dtype=np.float32)
        ones_8 = np.ones((8, 1), dtype=np.float32)
        corners_h = np.concatenate([bbox_local, ones_8], axis=1)
        corners_w = (corners_h @ mat.T)[:, :3]
        min_coords = corners_w.min(axis=0)
        max_coords = corners_w.max(axis=0)
        block_size = props.block_size
        offset = np.array(props.block_offset, dtype=np.float32)
        origin = min_coords + offset
        ncx = max(1, int(np.ceil((max_coords[0] - origin[0]) / block_size)))
        ncy = max(1, int(np.ceil((max_coords[1] - origin[1]) / block_size)))
        ncz = max(1, int(np.ceil((max_coords[2] - origin[2]) / block_size)))
        num_blocks = ncx * ncy * ncz

        # Clear existing probes
        obj.probe_points.clear()

        # SH constants
        C0 = 0.28209479177387814   # 1/(2*sqrt(pi))
        C1 = 0.4886025119029199    # sqrt(3/(4*pi))
        C2a = 1.0925484305920792   # 1/2 * sqrt(15/pi)
        C2b = 0.31539156525252005  # 1/4 * sqrt(5/pi)
        C2c = 0.5462742152960396   # 1/4 * sqrt(15/pi)

        flat_dc = 1.0 / (1.0 + np.exp(-raw_dc))  # (N, 3) linear color
        splat_opacity = np.minimum(opacities[:, 0] * 2.0, 1.0)  # (N,) match shader

        # ------------------------------------------------------------------
        # Block-based filtering: subdivide each block into 3×3×3 sub-blocks
        # and keep 1 brightest + 1 largest per sub-block. This ensures
        # selected splats are spatially distributed within each block.
        # ------------------------------------------------------------------
        SUBDIV = 3  # sub-blocks per axis

        brightness = flat_dc.max(axis=1)
        splat_size = scales.max(axis=1)

        if num_blocks > 0 and N > 2 * SUBDIV**3 * num_blocks:
            bs = block_size
            bix = np.floor((positions_world[:, 0] - origin[0]) / bs).astype(np.int32)
            biy = np.floor((positions_world[:, 1] - origin[1]) / bs).astype(np.int32)
            biz = np.floor((positions_world[:, 2] - origin[2]) / bs).astype(np.int32)
            bix = np.clip(bix, 0, ncx - 1)
            biy = np.clip(biy, 0, ncy - 1)
            biz = np.clip(biz, 0, ncz - 1)
            block_key = bix * (ncy * ncz) + biy * ncz + biz

            # Sub-block index within each block
            local_x = (positions_world[:, 0] - origin[0]) / bs - bix
            local_y = (positions_world[:, 1] - origin[1]) / bs - biy
            local_z = (positions_world[:, 2] - origin[2]) / bs - biz
            sub_ix = np.floor(local_x * SUBDIV).astype(np.int32)
            sub_iy = np.floor(local_y * SUBDIV).astype(np.int32)
            sub_iz = np.floor(local_z * SUBDIV).astype(np.int32)
            sub_ix = np.clip(sub_ix, 0, SUBDIV - 1)
            sub_iy = np.clip(sub_iy, 0, SUBDIV - 1)
            sub_iz = np.clip(sub_iz, 0, SUBDIV - 1)
            sub_key = sub_ix * SUBDIV * SUBDIV + sub_iy * SUBDIV + sub_iz
            combined_key = block_key * SUBDIV**3 + sub_key

            selected = np.zeros(N, dtype=bool)
            selected_bright = np.zeros(N, dtype=bool)
            for ck in np.unique(combined_key):
                mask = combined_key == ck
                indices = np.where(mask)[0]
                # 1 brightest
                top_b = indices[np.argmax(brightness[indices])]
                selected[top_b] = True
                selected_bright[top_b] = True
                # 1 largest (may be same as brightest, dedup via set True)
                top_l = indices[np.argmax(splat_size[indices])]
                selected[top_l] = True

            filtered_positions = positions_world[selected]
            filtered_colors = flat_dc[selected]
            filtered_opacities = splat_opacity[selected]
            M = len(filtered_positions)
            # block_avg from brightest splats only (ignore "largest" picks)
            block_avg = flat_dc[selected_bright].mean(axis=0)
            print(f"[Splatting Bake] block_avg (bright only) = R={block_avg[0]:.4f}  G={block_avg[1]:.4f}  B={block_avg[2]:.4f}")
        else:
            filtered_positions = positions_world
            filtered_colors = flat_dc
            filtered_opacities = splat_opacity
            M = N
            block_avg = filtered_colors.mean(axis=0)
            print(f"[Splatting Bake] block_avg (all splats) = R={block_avg[0]:.4f}  G={block_avg[1]:.4f}  B={block_avg[2]:.4f}")

        # Apply the same gain + color adjust to block_avg so the fallback
        # ambient color matches what the SH probes store.
        ba = block_avg.copy()
        gs_ba = item.brightness_gain_start
        gain_ba = item.brightness_gain
        t_ba = np.clip((ba - gs_ba) / max(1.0 - gs_ba, 1e-6), 0, 1)
        ba *= (1.0 + t_ba * t_ba * max(gain_ba - 1.0, 0.0))
        ba *= np.array(item.color_tint[:3], dtype=np.float32)
        cosH = np.float32(np.cos(item.color_hue * 3.14159))
        sinH = np.float32(np.sin(item.color_hue * 3.14159))
        k_ba = np.array([0.57735, 0.57735, 0.57735], dtype=np.float32)
        dk_ba = np.dot(ba, k_ba)
        ba = ba * cosH + np.cross(k_ba, ba) * sinH + k_ba * dk_ba * (1.0 - cosH)
        lum_ba = np.dot(ba, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        ba = lum_ba * (1.0 - item.color_saturation) + ba * item.color_saturation
        ba *= 1.2 * item.color_brightness
        ba = np.maximum(ba, 0.0) ** (1.0 / max(0.65 * item.color_gamma, 0.001))
        bpy.context.scene.splatting_properties.default_irradiance_color = (
            float(np.clip(ba[0], 0, 1)),
            float(np.clip(ba[1], 0, 1)),
            float(np.clip(ba[2], 0, 1)),
        )

        # ------------------------------------------------------------------
        # Precompute 64 uniform sphere directions (Fibonacci spiral)
        # for directional nearest-splat search.
        # ------------------------------------------------------------------
        NUM_DIR_SAMPLES = 64
        golden_angle = np.pi * (3 - np.sqrt(5))
        i = np.arange(NUM_DIR_SAMPLES, dtype=np.float64)
        theta = golden_angle * i
        z = np.linspace(1 - 1/NUM_DIR_SAMPLES, 1/NUM_DIR_SAMPLES - 1, NUM_DIR_SAMPLES)
        r = np.sqrt(1 - z*z)
        sample_dirs = np.column_stack([r * np.cos(theta), r * np.sin(theta), z]).astype(np.float32)

        sx, sy, sz = sample_dirs[:, 0], sample_dirs[:, 1], sample_dirs[:, 2]
        sx2, sy2, sz2 = sx*sx, sy*sy, sz*sz
        sample_basis = np.column_stack([
            np.full(NUM_DIR_SAMPLES, C0),
            C1 * sy, C1 * sz, C1 * sx,
            C2a * sx * sy, C2a * sy * sz,
            C2b * (3.0*sz2 - 1.0),
            C2a * sx * sz,
            C2c * (sx2 - sy2),
        ]).astype(np.float32)

        context.window_manager.progress_begin(0, num_probes)
        num_stored = 0
        gain = item.brightness_gain
        inv_mat = np.linalg.inv(mat)

        for pi in range(num_probes):
            if pi % 5 == 0 or pi == num_probes - 1:
                context.window_manager.progress_update(pi)

            pp = probe_positions[pi]
            dv = filtered_positions - pp
            dists = np.linalg.norm(dv, axis=1)
            ndirs = dv / (dists[:, None] + 1e-8)

            # --------------------------------------------------------------
            # Cone-based accumulation: assign each splat to its closest
            # direction (of 64), then alpha-composite from near to far.
            # --------------------------------------------------------------
            M_local = len(ndirs)
            BATCH = 200000
            cos_sim = np.empty((M_local, NUM_DIR_SAMPLES), dtype=np.float32)
            for start in range(0, M_local, BATCH):
                end = min(start + BATCH, M_local)
                cos_sim[start:end] = ndirs[start:end] @ sample_dirs.T

            # Each splat → its closest sample direction
            splat_dir = np.argmax(cos_sim, axis=1).astype(np.int32)  # (M,)

            # Per-direction alpha composite, fallback to block_avg
            sample_colors = np.tile(block_avg, (NUM_DIR_SAMPLES, 1))
            for di in range(NUM_DIR_SAMPLES):
                mask = splat_dir == di
                n_in_cone = mask.sum()
                if n_in_cone == 0:
                    continue

                sc = filtered_colors[mask]          # (n, 3)
                sd = dists[mask]                     # (n,)
                sa = filtered_opacities[mask]        # (n,)

                # Sort near→far
                order = np.argsort(sd)
                sc = sc[order]
                sa = sa[order]

                acc_color = np.zeros(3, dtype=np.float32)
                acc_alpha = np.float32(0.0)
                for j in range(len(sc)):
                    contrib = sa[j] * (np.float32(1.0) - acc_alpha)
                    acc_color += sc[j] * contrib
                    acc_alpha += contrib
                    # stop later because the splats in there is so less
                    if acc_alpha >= np.float32(1.5):
                        break

                if acc_alpha > np.float32(1e-6):
                    sample_colors[di] = acc_color

            # DEBUG: first probe stats
            if pi == 0:
                w_avg = (sample_colors.max(axis=1) - sample_colors.min(axis=1)).mean()
                print(f"[Splatting Bake] Probe 0 cone-accum: colors min={sample_colors.min():.4f} "
                      f"max={sample_colors.max():.4f} mean={sample_colors.mean():.4f} "
                      f"sat_avg={w_avg:.4f}")

            # Inverse tone mapping (matching shader forward gain exactly)
            gs = item.brightness_gain_start
            t = np.clip((sample_colors - gs) / max(1.0 - gs, 1e-6), 0, 1)
            sample_colors *= (1.0 + t * t * max(gain - 1.0, 0.0))

            # Color adjust — baked into SH so probe receivers get the
            # per-instance look without double-processing at render time.
            # Tint
            sample_colors *= np.array(item.color_tint[:3], dtype=np.float32)

            # Hue rotation
            cosH = np.float32(np.cos(item.color_hue * 3.14159))
            sinH = np.float32(np.sin(item.color_hue * 3.14159))
            k = np.array([0.57735, 0.57735, 0.57735], dtype=np.float32)
            dot_k = np.dot(sample_colors, k)
            sample_colors = (sample_colors * cosH +
                             np.cross(k, sample_colors) * sinH +
                             k[np.newaxis, :] * dot_k[:, np.newaxis] * (1.0 - cosH))

            # Saturation
            lum = np.dot(sample_colors, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
            sample_colors = (lum[:, np.newaxis] * (1.0 - item.color_saturation) +
                             sample_colors * item.color_saturation)

            # Brightness (1.2x factor matches the shader)
            sample_colors *= 1.2 * item.color_brightness

            # Gamma
            sample_colors = np.maximum(sample_colors, 0.0) ** (1.0 / max(0.65 * item.color_gamma, 0.001))

            # SH2 projection with uniform directional sampling
            uniform_scale = 4.0 * np.pi / NUM_DIR_SAMPLES
            sh_r = (sample_colors[:, 0:1] * sample_basis).sum(axis=0) * uniform_scale
            sh_g = (sample_colors[:, 1:2] * sample_basis).sum(axis=0) * uniform_scale
            sh_b = (sample_colors[:, 2:3] * sample_basis).sum(axis=0) * uniform_scale

            # Pre-convolution window: scale down higher bands to prevent
            # negative-ringing artifacts in irradiance reconstruction.
            window = np.array([1.0, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            sh_r *= window
            sh_g *= window
            sh_b *= window

            pt = obj.probe_points.add()
            # Store position in local space
            local_pp = inv_mat @ np.append(pp, 1.0)
            pt.location = (float(local_pp[0]), float(local_pp[1]), float(local_pp[2]))
            pt.sh_r = sh_r.tolist()
            pt.sh_g = sh_g.tolist()
            pt.sh_b = sh_b.tolist()
            num_stored += 1

        context.window_manager.progress_end()
        # Invalidate probe cache so runtime picks up the new bake immediately
        from . import splatting_data
        splatting_data.invalidate_irradiance_cache()

        # Save bake bbox (8 local-space corners) + bake matrix.
        if num_stored > 0:
            bc = np.array(props.bake_area_center, dtype=np.float32)
            bh = np.array(props.bake_area_size, dtype=np.float32) * 0.5
            corners_w = np.array([
                [bc[0] - bh[0], bc[1] - bh[1], bc[2] - bh[2], 1.0],
                [bc[0] + bh[0], bc[1] - bh[1], bc[2] - bh[2], 1.0],
                [bc[0] + bh[0], bc[1] + bh[1], bc[2] - bh[2], 1.0],
                [bc[0] - bh[0], bc[1] + bh[1], bc[2] - bh[2], 1.0],
                [bc[0] - bh[0], bc[1] - bh[1], bc[2] + bh[2], 1.0],
                [bc[0] + bh[0], bc[1] - bh[1], bc[2] + bh[2], 1.0],
                [bc[0] + bh[0], bc[1] + bh[1], bc[2] + bh[2], 1.0],
                [bc[0] - bh[0], bc[1] + bh[1], bc[2] + bh[2], 1.0],
            ], dtype=np.float32)
            local_corners = (corners_w @ inv_mat.T)[:, :3]
            obj.bake_bbox_corners = tuple(local_corners.ravel())
            obj.bake_matrix_world = tuple(mat.ravel())

        self.report({'INFO'}, f"Baked {num_stored} light probes on '{obj.name}'")
        return {'FINISHED'}


class SPLATTING_OT_remove_baked_lighting(types.Operator):
    bl_idname = "splatting.remove_baked_lighting"
    bl_label = "Remove Baked Lighting"
    bl_description = "Remove all baked lighting data (light probes + envmap) from this splat instance"

    def execute(self, context):
        props = context.scene.splatting_properties
        idx = props.active_instance_index
        instances = context.scene.splatting_instances
        if idx < 0 or idx >= len(instances):
            self.report({'WARNING'}, "No instance selected")
            return {'CANCELLED'}

        item = instances[idx]
        obj = bpy.data.objects.get(item.mesh_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Instance mesh not found")
            return {'CANCELLED'}

        # Remove light probes
        obj.probe_points.clear()
        obj.bake_bbox_corners = (0.0,) * 24
        bake_mat_default = (1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0)
        obj.bake_matrix_world = bake_mat_default

        # Remove envmap atlas image
        # Must unlink from node groups first to avoid GPU crash.
        atlas_name = obj.envmap_atlas
        if atlas_name:
            img = bpy.data.images.get(atlas_name)
            if img:
                # Unlink from FS_PBR_BRDF EnvImage nodes
                ng = bpy.data.node_groups.get('FS_PBR_BRDF')
                if ng:
                    for node in ng.nodes:
                        if node.type == 'TEX_IMAGE' and node.image is img:
                            node.image = None
                bpy.data.images.remove(img, do_unlink=True)
            obj.envmap_atlas = ""

        from . import splatting_data
        splatting_data.invalidate_irradiance_cache()
        self.report({'INFO'}, f"Removed baked lighting from '{obj.name}'")
        return {'FINISHED'}


class SPLATTING_OT_bake_envmap(types.Operator):
    bl_idname = "splatting.bake_envmap"
    bl_label = "Bake Envmap"
    bl_description = "Bake environment map probes at automatically placed positions and write to atlas texture"

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------
    def invoke(self, context, event):
        import numpy as np
        from mathutils import Matrix

        props = context.scene.splatting_properties
        idx = props.active_instance_index
        instances = context.scene.splatting_instances
        if idx < 0 or idx >= len(instances):
            self.report({'WARNING'}, "No instance selected")
            return {'CANCELLED'}

        item = instances[idx]
        obj = bpy.data.objects.get(item.mesh_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Instance mesh not found")
            return {'CANCELLED'}

        # Single envmap probe at bake area center
        from .splatting_data import compute_probe_positions
        _, env_positions = compute_probe_positions(context)
        if env_positions is None or len(env_positions) == 0:
            self.report({'ERROR'}, "No envmap probe positions could be generated")
            return {'CANCELLED'}

        self._obj = obj
        self._item = item  # SplattingInstanceItem with per-mesh brightness props
        self._probe_pos = env_positions[0]  # center probe
        from .envmap_utils import _FACE_NAMES
        self._face_names = _FACE_NAMES

        self._probe_res = 512
        self._probe_w = 512

        # ------------------------------------------------------------------
        # Initialize renderer & load splat data if not already rendering
        # ------------------------------------------------------------------
        from .splatting_data import get_state, read_ply_attributes, build_spatial_index, SplattingState
        scene_state = get_state()
        self._was_rendering = scene_state.is_rendering
        init_ok = True
        if not self._was_rendering:
            init_ok = self._init_renderer_data(context, obj, props, scene_state,
                                               read_ply_attributes, build_spatial_index, SplattingState)
        if not init_ok:
            self.report({'ERROR'}, "Failed to initialize renderer data")
            return {'CANCELLED'}

        # Build perspective projection matrix manually (90° FOV, 1:1 aspect)
        # OpenGL convention: cot(fov/2) on diagonal, standard frustum remapping
        near, far = 0.1, 1000.0
        f = 1.0  # cot(45°) for 90° vertical FOV
        self._proj_matrix = Matrix((
            (f, 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        ))

        # Face forward/up vectors (must match envmap_utils._FACE_PARAMS)
        face_lookat = {
            'px': ('RIGHT', (0.0, -1.0, 0.0)),
            'nx': ('LEFT', (0.0, -1.0, 0.0)),
            'py': ('UP', (0.0, 0.0, 1.0)),
            'ny': ('DOWN', (0.0, 0.0, -1.0)),
            'pz': ('FRONT', (0.0, -1.0, 0.0)),
            'nz': ('BACK', (0.0, -1.0, 0.0)),
        }
        forward_vectors = {
            'RIGHT': (1.0, 0.0, 0.0),
            'LEFT': (-1.0, 0.0, 0.0),
            'UP': (0.0, 1.0, 0.0),
            'DOWN': (0.0, -1.0, 0.0),
            'FRONT': (0.0, 0.0, 1.0),
            'BACK': (0.0, 0.0, -1.0),
        }

        self._view_matrices = {}
        for name, (fwd_key, up) in face_lookat.items():
            fwd_vec = forward_vectors[fwd_key]
            # Manual LookAt at origin: build view matrix from forward/up vectors
            import mathutils
            fwd_v = mathutils.Vector(fwd_vec)
            up_v = mathutils.Vector(up)
            right_v = fwd_v.cross(up_v).normalized()
            up_v = right_v.cross(fwd_v).normalized()  # re-orthogonalize
            rot_view = mathutils.Matrix((
                (right_v.x, right_v.y, right_v.z, 0.0),
                (up_v.x, up_v.y, up_v.z, 0.0),
                (-fwd_v.x, -fwd_v.y, -fwd_v.z, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ))
            self._view_matrices[name] = rot_view

        # State
        self._face_images = {}
        self._phase = 'faces'
        self._face_idx = 0
        self._convolve_level = 0
        self._mip_chain = []
        self._fully_sorted = False
        self._offscreen = None

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.001, window=context.window)
        context.window_manager.progress_begin(0, 12)
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # Renderer init (when not already rendering)
    # ------------------------------------------------------------------
    def _init_renderer_data(self, context, obj, props, scene_state,
                            read_ply_attributes, build_spatial_index, SplattingState):
        """Load splat data into the renderer state so draw_offscreen works.
        Returns True on success, False on failure."""
        import numpy as np

        mesh = obj.data
        if len(mesh.vertices) == 0:
            return False

        positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree = \
            read_ply_attributes(mesh)

        # Alpha clip
        clip_val = props.clip_alpha
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

        # Size clip
        size_val = props.clip_size
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

        # Build spatial index in world space
        mat = np.array(obj.matrix_world, dtype=np.float32)
        ones = np.ones((len(positions), 1), dtype=np.float32)
        positions_h = np.concatenate([positions, ones], axis=1)
        positions_world = (positions_h @ mat.T)[:, :3]

        bbox_local = np.array(obj.bound_box, dtype=np.float32)
        ones_8 = np.ones((8, 1), dtype=np.float32)
        corners_h = np.concatenate([bbox_local, ones_8], axis=1)
        corners_w = (corners_h @ mat.T)[:, :3]

        offset = props.block_offset
        grid_ref = corners_w.min(axis=0)
        pos_ref = positions_world.min(axis=0)
        offset_arr = np.array(offset, dtype=np.float32)
        adjusted = grid_ref + offset_arr - pos_ref

        spatial = build_spatial_index(positions_world, block_size=props.block_size,
                                      origin_offset=adjusted, use_parallel=True)
        inst.block_indices = spatial['block_indices']
        inst.grid_dims = spatial['grid_dims']
        inst.block_centers = spatial['block_centers']
        inst.block_radii = spatial['block_radii']
        inst.block_bounds = spatial['block_bounds']
        inst.block_splat_indices = spatial['block_splat_indices']
        inst.block_count = len(spatial['unique_blocks'])

        # Freeze originals for delta-based transform
        inst._orig_block_centers = spatial['block_centers'].copy()
        inst._orig_block_radii = spatial['block_radii'].copy()
        inst._orig_block_bounds = spatial['block_bounds'].copy()
        inst._orig_model_matrix = np.array(obj.matrix_world, dtype=np.float32)
        inst._orig_model_inv = np.linalg.inv(inst._orig_model_matrix)

        # Clear & add to scene state
        scene_state.clear()
        scene_state.instances.append(inst)

        # Initialize GPU renderer
        from .gpu_renderer import init_renderer
        success = init_renderer()
        if not success:
            scene_state.clear()
            self.report({'ERROR'}, "GPU renderer initialization failed")
            return False

        # Sort for the probe position
        pp = self._probe_pos
        from mathutils import Vector
        from .splatting_data import sort_blocks_far_to_near
        sort_blocks_far_to_near(Vector((float(pp[0]), float(pp[1]), float(pp[2]))))
        self._fully_sorted = True
        return True

    # ------------------------------------------------------------------
    # modal
    # ------------------------------------------------------------------
    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._phase == 'faces':
            self._render_next_face(context)
        elif self._phase == 'convolve':
            self._convolve_next_level(context)
        elif self._phase == 'pack':
            self._pack_and_store(context)
            self._cleanup(context)
            self.report({'INFO'}, f"Envmap atlas stored on '{self._obj.name}'")
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # Phase A: Render 6 cubemap faces
    # ------------------------------------------------------------------
    def _render_next_face(self, context):
        import gpu
        from gpu.types import GPUOffScreen
        from mathutils import Matrix

        probe_pos = self._probe_pos
        name = self._face_names[self._face_idx]
        res = self._probe_res

        if self._offscreen is None:
            self._offscreen = GPUOffScreen(res, res)

        # Build view matrix with probe position as eye:
        #   V = R * T(-probe_pos)  → first translate, then rotate
        pp = (float(probe_pos[0]), float(probe_pos[1]), float(probe_pos[2]))
        vm = self._view_matrices[name] @ Matrix.Translation((-pp[0], -pp[1], -pp[2]))
        camera_pos_world = pp

        with self._offscreen.bind():
            from .gpu_renderer import get_renderer
            renderer = get_renderer()
            pixels = renderer.draw_offscreen(
                context, res, res,
                vm, self._proj_matrix,
                camera_pos_world,
                skip_sort=self._fully_sorted,
                skip_inverse_gain=True,
            )
            self._face_images[name] = pixels

        self._fully_sorted = True
        self._face_idx += 1
        context.window_manager.progress_update(self._face_idx)

        if self._face_idx >= 6:
            self._offscreen.free()
            self._offscreen = None
            self._phase = 'convolve'

    # ------------------------------------------------------------------
    # Phase B: Per-level specular convolution (one level per tick)
    # ------------------------------------------------------------------
    def _convolve_next_level(self, context):
        from .envmap_utils import convolve_specular_level

        level = self._convolve_level
        eq = convolve_specular_level(level, 6, self._face_images, self._probe_w, 256)
        self._mip_chain.append(eq)

        self._convolve_level += 1
        context.window_manager.progress_update(6 + self._convolve_level)

        if self._convolve_level >= 6:
            self._face_images = None
            self._phase = 'pack'

    # ------------------------------------------------------------------
    # Phase D: Pack atlas & store
    # ------------------------------------------------------------------
    def _pack_and_store(self, context):
        import bpy
        from .envmap_utils import pack_mip_atlas

        atlas_data, (atlas_w, atlas_h) = pack_mip_atlas(self._mip_chain)

        # Create image in Blender
        name = f"{self._obj.name}_envmap_atlas"
        existing = bpy.data.images.get(name)
        if existing:
            bpy.data.images.remove(existing)

        img = bpy.data.images.new(name, width=atlas_w, height=atlas_h,
                                  float_buffer=True, alpha=True)
        img.pixels = atlas_data.ravel()

        # Store reference on the object
        self._obj.envmap_atlas = name

        # Auto-update FS_PBR_BRDF node group's EnvImage nodes
        ng = bpy.data.node_groups.get('FS_PBR_BRDF')
        if ng:
            updated = 0
            for node in ng.nodes:
                if node.type == 'TEX_IMAGE' and node.label == 'EnvImage':
                    node.image = img
                    updated += 1
            if updated:
                print(f"[Envmap] Updated {updated} EnvImage node(s) in FS_PBR_BRDF")
            else:
                print(f"[Envmap] FS_PBR_BRDF found but no EnvImage nodes")
        else:
            print(f"[Envmap] FS_PBR_BRDF node group not found, skip auto-update")

        self._mip_chain = None

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def _cleanup(self, context):
        context.window_manager.progress_end()
        if self._offscreen is not None:
            self._offscreen.free()
            self._offscreen = None
        if hasattr(self, '_timer'):
            context.window_manager.event_timer_remove(self._timer)

        # If we temporarily initialized the renderer, clean up
        if not getattr(self, '_was_rendering', True):
            from .splatting_data import get_state
            from .gpu_renderer import release_renderer
            release_renderer()
            get_state().clear()


_operators = [
    SPLATTING_OT_start_render,
    SPLATTING_OT_stop_render,
    SPLATTING_OT_add_instance,
    SPLATTING_OT_remove_instance,
    SPLATTING_OT_move_instance,
    SPLATTING_OT_toggle_instance,
    SPLATTING_OT_export_animation,
    SPLATTING_OT_import_spz,
    SPLATTING_OT_set_default_render,
    SPLATTING_OT_bake_lightprobe,
    SPLATTING_OT_remove_baked_lighting,
    SPLATTING_OT_bake_envmap,
]


# ---------------------------------------------------------------------------
# depsgraph_update_post handler — set rotation X to -90° on splat meshes
# ---------------------------------------------------------------------------
# Covers both .blend loading (via load_post) and PLY imports (via object
# creation), without polling every frame.
_SPLAT_ROTATE_HANDLER = None
_SPLAT_ROTATE_LOAD_PRE = None
_known_object_count = -1


def _check_splat_rotation(scene, depsgraph=None):
    """When object count changes, find new meshes with splatting attributes
    and set rotation_euler.x = -90°."""
    global _known_object_count
    n = len(bpy.data.objects)
    if n == _known_object_count:
        return
    _known_object_count = n
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.data.attributes.get("f_dc_0") is None:
            continue
        if abs(obj.rotation_euler.x + 1.5708) > 0.001:
            obj.rotation_euler.x = -1.5708


def _reset_object_counter(_dummy):
    global _known_object_count
    _known_object_count = -1


def _register_rotate_handlers():
    global _SPLAT_ROTATE_HANDLER, _SPLAT_ROTATE_LOAD_PRE
    if _SPLAT_ROTATE_HANDLER is None:
        _SPLAT_ROTATE_HANDLER = bpy.app.handlers.depsgraph_update_post.append(
            _check_splat_rotation)
    if _SPLAT_ROTATE_LOAD_PRE is None:
        _SPLAT_ROTATE_LOAD_PRE = bpy.app.handlers.load_pre.append(
            _reset_object_counter)


def _unregister_rotate_handlers():
    global _SPLAT_ROTATE_HANDLER, _SPLAT_ROTATE_LOAD_PRE
    if _SPLAT_ROTATE_HANDLER is not None:
        bpy.app.handlers.depsgraph_update_post.remove(_SPLAT_ROTATE_HANDLER)
        _SPLAT_ROTATE_HANDLER = None
    if _SPLAT_ROTATE_LOAD_PRE is not None:
        bpy.app.handlers.load_pre.remove(_SPLAT_ROTATE_LOAD_PRE)
        _SPLAT_ROTATE_LOAD_PRE = None


def register():
    for op in _operators:
        bpy.utils.register_class(op)

    # File → Import menu
    def menu_import(self, context):
        self.layout.operator(
            SPLATTING_OT_import_spz.bl_idname, text="Spark SPZ (.spz)")
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    SPLATTING_OT_import_spz._menu_import = menu_import

    _register_rotate_handlers()


def unregister():
    _unregister_capture_handler()
    global _capture_ready, _capture_requested
    _capture_ready = False
    _capture_requested = False

    for op in reversed(_operators):
        bpy.utils.unregister_class(op)

    menu_import = getattr(SPLATTING_OT_import_spz, '_menu_import', None)
    if menu_import:
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)

    _unregister_rotate_handlers()
