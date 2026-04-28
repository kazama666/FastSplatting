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
_capture_vp_w = 0          # physical pixel dimensions from viewport_get
_capture_vp_h = 0


def _post_view_capture():
    """Called inside Blender's draw cycle after splats are drawn."""
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


class SPLATTING_OT_select_mesh(types.Operator):
    bl_idname = "splatting.select_mesh"
    bl_label = "Select Mesh"
    bl_description = "Select mesh object containing Gaussian Splatting PLY data"

    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH':
            context.scene.splatting_target_mesh = obj.name
            self.report({'INFO'}, f"Selected: {obj.name}")
        else:
            self.report({'WARNING'}, "Please select a mesh object first")
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
        from .gpu_renderer import get_renderer

        splatting_data.get_state()._sort_active = False

        if splatting_data.sort_blocks_far_to_near(camera_pos):
            get_renderer().clear_cache()
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
        self._delay_ticks = 20   # ~1s at 20 Hz
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

        # Wait for full-area to settle before capturing first frame
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
            # Save current frame
            filepath = os.path.join(self._output_dir, f"frame_{self._current:04d}.png")
            self._save_capture(filepath)
            context.window_manager.progress_update(self._current)

            _capture_ready = False

            # Advance
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
            # Draw not yet completed — re-tag and wait
            if not _capture_requested:
                self._request_capture()
            else:
                self._view3d_area.tag_redraw()

        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _maybe_sort(self, context):
        """Sort blocks far-to-near if camera has moved since last capture."""
        from . import splatting_data
        from .gpu_renderer import get_renderer

        # Get current camera position from the 3D view
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

        # Sort all blocks for the new camera position
        splatting_data.get_state()._sort_active = False
        if splatting_data.sort_blocks_far_to_near(cam_pos):
            get_renderer().clear_cache()

    def _request_capture(self):
        global _capture_requested
        _capture_requested = True
        self._view3d_area.tag_redraw()

    def _save_capture(self, filepath):
        # 3D view is maximised to full window — screenshot IS the viewport.
        bpy.ops.screen.screenshot(filepath=filepath)

    def _finish(self, context):
        # Restore overlays
        if hasattr(self, '_orig_show_overlays') and self._space3d:
            try:
                self._space3d.overlay.show_overlays = self._orig_show_overlays
            except:
                pass

        # Restore the 3D view UI
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


_operators = [
    SPLATTING_OT_start_render,
    SPLATTING_OT_stop_render,
    SPLATTING_OT_select_mesh,
    SPLATTING_OT_sort_blocks,
    SPLATTING_OT_export_animation,
]


def register():
    for op in _operators:
        bpy.utils.register_class(op)


def unregister():
    # Clean up the export capture handler if still active
    _unregister_capture_handler()
    global _capture_ready, _capture_requested
    _capture_ready = False
    _capture_requested = False

    for op in reversed(_operators):
        bpy.utils.unregister_class(op)
