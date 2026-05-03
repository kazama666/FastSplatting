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
_capture_vp_w = 0
_capture_vp_h = 0


def _post_view_capture():
    global _capture_requested, _capture_ready
    global _capture_vp_w, _capture_vp_h
    if not _capture_requested:
        return
    import gpu
    _, _, w, h = gpu.state.viewport_get()
    _capture_vp_w = w
    _capture_vp_h = h
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
    item.quad_scale = d.default_quad_scale


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


class SPLATTING_OT_sort_blocks(types.Operator):
    bl_idname = "splatting.sort_blocks"
    bl_label = "Sort Blocks Far→Near"
    bl_description = "Sort splats inside each block by distance from camera (far to near) for correct alpha blending"

    def execute(self, context):
        camera_pos = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                region3d = area.spaces.active.region_3d
                is_camera_view = region3d.view_perspective == 'CAMERA'
                if is_camera_view and context.scene.camera:
                    camera_pos = context.scene.camera.matrix_world.translation
                else:
                    camera_pos = region3d.view_matrix.inverted().translation
                break

        if camera_pos is None:
            self.report({'WARNING'}, "No 3D viewport found")
            return {'CANCELLED'}

        from . import splatting_data

        # Reset sort state for all instances
        scene = splatting_data.get_state()
        for inst in scene.instances:
            inst._sort_active = False
            inst.clear_cache()

        if splatting_data.sort_blocks_far_to_near(camera_pos):
            self.report({'INFO'}, "Blocks sorted far→near")
        else:
            self.report({'WARNING'}, "No splatting data loaded")
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


class SPLATTING_OT_debug_generate(types.Operator):
    bl_idname = "splatting.debug_generate"
    bl_label = "Generate Debug Splat"
    bl_description = "Generate a single-row mesh with debug splat attributes for texture coordinate verification"

    splat_count: bpy.props.IntProperty(
        default=16, min=2, max=256,
        description="Number of debug splats to generate",
    )

    def execute(self, context):
        import numpy as np

        n = self.splat_count

        # Create mesh: vertices in a row along X, centered at origin
        mesh = bpy.data.meshes.new("DebugSplats")
        verts = [(i * 0.5 - (n - 1) * 0.25, 0.0, 0.0) for i in range(n)]
        mesh.from_pydata(verts, [], [])

        # Add float attributes matching PLY schema
        for name in ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                      'scale_0', 'scale_1', 'scale_2',
                      'rot_0', 'rot_1', 'rot_2', 'rot_3']:
            mesh.attributes.new(name, 'FLOAT', 'POINT')

        # Inverse of: display_color = pow(sigmoid(raw_dc), 2.2)
        def raw_from_display(r, g, b):
            eps = 1e-6
            linear = np.power(np.clip([r, g, b], eps, 1.0 - eps), 1.0 / 2.2)
            return [float(np.log(c / (1.0 - c))) for c in linear]

        # 8 clearly distinct colors — if sampling is right, pattern is (0,1,2,3,4,5,6,7,0,1,...)
        palette = [
            (1.0, 0.0, 0.0),  # 0: Red
            (0.0, 1.0, 0.0),  # 1: Green
            (0.0, 0.0, 1.0),  # 2: Blue
            (1.0, 1.0, 0.0),  # 3: Yellow
            (0.0, 1.0, 1.0),  # 4: Cyan
            (1.0, 0.0, 1.0),  # 5: Magenta
            (1.0, 0.5, 0.0),  # 6: Orange
            (0.5, 0.0, 1.0),  # 7: Purple
        ]

        for i in range(n):
            r, g, b = palette[i % 8]
            raw = raw_from_display(r, g, b)
            mesh.attributes["f_dc_0"].data[i].value = raw[0]
            mesh.attributes["f_dc_1"].data[i].value = raw[1]
            mesh.attributes["f_dc_2"].data[i].value = raw[2]

            # Opacity: sigmoid → ~0.99
            mesh.attributes["opacity"].data[i].value = float(np.log(0.99 / 0.01))
            # Scale: exp → 0.15
            mesh.attributes["scale_0"].data[i].value = float(np.log(0.15))
            mesh.attributes["scale_1"].data[i].value = float(np.log(0.15))
            mesh.attributes["scale_2"].data[i].value = float(np.log(0.15))
            # Rotation: identity quaternion
            mesh.attributes["rot_0"].data[i].value = 1.0
            mesh.attributes["rot_1"].data[i].value = 0.0
            mesh.attributes["rot_2"].data[i].value = 0.0
            mesh.attributes["rot_3"].data[i].value = 0.0

        # Create object at 3D cursor, select it
        obj = bpy.data.objects.new("DebugSplats", mesh)
        obj.location = context.scene.cursor.location
        context.collection.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Auto-add to splatting instance list
        for item in context.scene.splatting_instances:
            if item.mesh_name == obj.name:
                self.report({'INFO'}, f"Debug mesh '{obj.name}' already in list")
                return {'FINISHED'}
        item = context.scene.splatting_instances.add()
        item.mesh_name = obj.name
        item.mesh_uid = obj.session_uid
        item.enabled = True
        _init_instance_defaults(item, context)
        context.scene.splatting_properties.active_instance_index = len(context.scene.splatting_instances) - 1

        self.report({'INFO'}, f"Generated debug mesh '{obj.name}' with {n} splats")
        return {'FINISHED'}


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
        d.default_quad_scale = src.quad_scale
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

        # Grid origin aligned with block division (bound_box-based, same
        # as _frozen_grid_min/max used in the block grid overlay).
        bbox_local = np.array(obj.bound_box, dtype=np.float32)
        ones_8 = np.ones((8, 1), dtype=np.float32)
        corners_h = np.concatenate([bbox_local, ones_8], axis=1)
        corners_w = (corners_h @ mat.T)[:, :3]
        min_coords = corners_w.min(axis=0)
        max_coords = corners_w.max(axis=0)

        block_size = props.block_size
        offset = np.array(props.block_offset, dtype=np.float32)
        origin = min_coords + offset

        # Grid points at block intersections, excluding outermost layer.
        nx = max(1, int(np.ceil((max_coords[0] - origin[0]) / block_size)) + 1)
        ny = max(1, int(np.ceil((max_coords[1] - origin[1]) / block_size)) + 1)
        nz = max(1, int(np.ceil((max_coords[2] - origin[2]) / block_size)) + 1)
        xs = origin[0] + np.arange(nx) * block_size
        ys = origin[1] + np.arange(ny) * block_size
        zs = origin[2] + np.arange(nz) * block_size

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
        # Interior mask: keep only points not on the outer shell
        ig_x, ig_y, ig_z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        interior = (ig_x > 0) & (ig_x < nx - 1) & (ig_y > 0) & (ig_y < ny - 1) & (ig_z > 0) & (ig_z < nz - 1)
        probe_positions = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        probe_positions = probe_positions[interior.ravel()]

        num_probes = len(probe_positions)
        if num_probes == 0:
            self.report({'ERROR'}, "No grid positions generated")
            return {'CANCELLED'}

        # Clear existing probes
        obj.probe_points.clear()

        # SH constants
        C0 = 0.28209479177387814   # 1/(2*sqrt(pi))
        C1 = 0.4886025119029199    # sqrt(3/(4*pi))
        C2a = 1.0925484305920792   # 1/2 * sqrt(15/pi)
        C2b = 0.31539156525252005  # 1/4 * sqrt(5/pi)
        C2c = 0.5462742152960396   # 1/4 * sqrt(15/pi)

        flat_dc = 1.0 / (1.0 + np.exp(-raw_dc))  # (N, 3) linear color

        # ------------------------------------------------------------------
        # Block-based filtering: subdivide each block into 3×3×3 sub-blocks
        # and keep 1 brightest + 1 largest per sub-block. This ensures
        # selected splats are spatially distributed within each block.
        # ------------------------------------------------------------------
        SUBDIV = 3  # sub-blocks per axis
        ncx = max(1, nx - 1)
        ncy = max(1, ny - 1)
        ncz = max(1, nz - 1)
        num_blocks = ncx * ncy * ncz

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
            M = len(filtered_positions)
            # block_avg from brightest splats only (ignore "largest" picks)
            block_avg = flat_dc[selected_bright].mean(axis=0)
            print(f"[Splatting Bake] block_avg (bright only) = R={block_avg[0]:.4f}  G={block_avg[1]:.4f}  B={block_avg[2]:.4f}")
        else:
            filtered_positions = positions_world
            filtered_colors = flat_dc
            M = N
            block_avg = filtered_colors.mean(axis=0)
            print(f"[Splatting Bake] block_avg (all splats) = R={block_avg[0]:.4f}  G={block_avg[1]:.4f}  B={block_avg[2]:.4f}")

        # Use block_avg as the default irradiance fallback for receivers outside grids
        bpy.context.scene.splatting_properties.default_irradiance_color = (
            float(np.clip(block_avg[0], 0, 1)),
            float(np.clip(block_avg[1], 0, 1)),
            float(np.clip(block_avg[2], 0, 1)),
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
        gain = props.bake_gain

        for pi in range(num_probes):
            if pi % 5 == 0 or pi == num_probes - 1:
                context.window_manager.progress_update(pi)

            pp = probe_positions[pi]
            dv = filtered_positions - pp
            dists = np.linalg.norm(dv, axis=1)
            ndirs = dv / (dists[:, None] + 1e-8)
            ncolors = filtered_colors

            # Batched matmul: nearest splat per direction
            M_local = len(ndirs)
            BATCH = 200000
            cos_sim = np.empty((M_local, NUM_DIR_SAMPLES), dtype=np.float32)
            for start in range(0, M_local, BATCH):
                end = min(start + BATCH, M_local)
                cos_sim[start:end] = ndirs[start:end] @ sample_dirs.T
            nearest_idx = np.argmax(cos_sim, axis=0).astype(np.intp)
            max_cos = cos_sim[nearest_idx, np.arange(NUM_DIR_SAMPLES)]

            sample_colors = ncolors[nearest_idx]

            # Combined weight: angular confidence × distance-saturation.
            # Good match + close → use splat color.
            # Poor match or far → blend to block average.
            angular_weight = np.clip((max_cos - 0.3) / 0.5, 0, 1)
            sample_dists = dists[nearest_idx]
            sat = sample_colors.max(axis=1) - sample_colors.min(axis=1)
            dist_weight = 1.0 / (1.0 + sat * (sample_dists / block_size) ** 2)
            w = (angular_weight * dist_weight)[:, None]
            sample_colors = sample_colors * w + block_avg[None, :] * (1.0 - w)

            # DEBUG: first probe stats
            if pi == 0:
                print(f"[Splatting Bake] Probe 0: colors min={sample_colors.min():.4f} max={sample_colors.max():.4f} "
                      f"mean={sample_colors.mean():.4f}  w min={w.min():.4f} max={w.max():.4f} "
                      f"angular min={angular_weight.min():.4f} dist min={dist_weight.min():.4f}")

            # Inverse tone mapping
            gs = props.bake_gain_start
            t = np.clip((sample_colors - gs) / (1.0 - gs), 0, 1)
            sample_colors *= (1.0 + t * t * (gain - 1.0))

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
            pt.location = (float(pp[0]), float(pp[1]), float(pp[2]))
            pt.sh_r = sh_r.tolist()
            pt.sh_g = sh_g.tolist()
            pt.sh_b = sh_b.tolist()
            num_stored += 1

        context.window_manager.progress_end()
        # Invalidate probe cache so runtime picks up the new bake immediately
        from . import splatting_data
        splatting_data._probe_cache_valid = False
        self.report({'INFO'}, f"Baked {num_stored} light probes on '{obj.name}'")
        return {'FINISHED'}


class SPLATTING_OT_remove_lightprobe(types.Operator):
    bl_idname = "splatting.remove_lightprobe"
    bl_label = "Remove Light Probes"
    bl_description = "Remove all light probe data from this splat instance"

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

        n = len(obj.probe_points)
        obj.probe_points.clear()
        self.report({'INFO'}, f"Removed {n} light probes from '{obj.name}'")
        return {'FINISHED'}


_operators = [
    SPLATTING_OT_start_render,
    SPLATTING_OT_stop_render,
    SPLATTING_OT_add_instance,
    SPLATTING_OT_remove_instance,
    SPLATTING_OT_move_instance,
    SPLATTING_OT_toggle_instance,
    SPLATTING_OT_sort_blocks,
    SPLATTING_OT_export_animation,
    SPLATTING_OT_debug_generate,
    SPLATTING_OT_import_spz,
    SPLATTING_OT_set_default_render,
    SPLATTING_OT_bake_lightprobe,
    SPLATTING_OT_remove_lightprobe,
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
