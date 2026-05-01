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

    def execute(self, context):
        from . import splatting_data
        splatting_data.start_render(context)
        return {'FINISHED'}


class SPLATTING_OT_stop_render(types.Operator):
    bl_idname = "splatting.stop_render"
    bl_label = "Stop Render"
    bl_description = "Stop Gaussian Splatting render"

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
        context.scene.splatting_properties.active_instance_index = len(context.scene.splatting_instances) - 1
        self.report({'INFO'}, f"Added '{obj.name}'")
        return {'FINISHED'}


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
        context.scene.splatting_properties.active_instance_index = len(context.scene.splatting_instances) - 1

        self.report({'INFO'}, f"Generated debug mesh '{obj.name}' with {n} splats")
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
]


def register():
    for op in _operators:
        bpy.utils.register_class(op)


def unregister():
    _unregister_capture_handler()
    global _capture_ready, _capture_requested
    _capture_ready = False
    _capture_requested = False

    for op in reversed(_operators):
        bpy.utils.unregister_class(op)
