import bpy
from bpy import types


# ---------------------------------------------------------------------------
# UIList for splat instances (draggable, Bone-Collection style)
# ---------------------------------------------------------------------------
class SPLATTING_UL_instances(types.UIList):
    use_drag_reorder = True

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type not in {'DEFAULT', 'COMPACT'}:
            return
        row = layout.row(align=True)
        op = row.operator("splatting.toggle_instance", text="", icon='HIDE_OFF' if item.enabled else 'HIDE_ON', emboss=False)
        op.index = index
        obj = bpy.data.objects.get(item.mesh_name)
        if obj and obj.type == 'MESH':
            row.label(text=obj.name)
        else:
            row.label(text=item.mesh_name or "(missing)", icon='ERROR')


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class SPLATTING_PT_panel(types.Panel):
    bl_label = "Fast Splatting"
    bl_idname = "SPLATTING_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Fast Splatting"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        splatting_props = scene.splatting_properties

        # Stale rendering state from file load (splatting_props persisted,
        # but runtime _scene was reset) — auto-correct via deferred timer since
        # writing to ID properties is not allowed during panel draw.
        if splatting_props.is_rendering:
            from .splatting_data import get_state, _deferred_reset_props
            if not get_state().is_rendering:
                if not bpy.app.timers.is_registered(_deferred_reset_props):
                    bpy.app.timers.register(_deferred_reset_props, first_interval=0.0)

        # --- Splat instances (collapsible) ---
        meshes_header, meshes_body = layout.panel_prop(splatting_props, "ui_meshes_expanded")
        meshes_header.label(text="Splat Meshes", icon='OBJECT_DATA')
        if meshes_body:
            row = meshes_body.row()
            row.template_list(
                "SPLATTING_UL_instances", "",
                scene, "splatting_instances",
                splatting_props, "active_instance_index",
                rows=4,
            )
            col = row.column(align=True)
            if not splatting_props.is_rendering:
                col.operator("splatting.add_instance", text="", icon='ADD')
                col.operator("splatting.remove_instance", text="", icon='REMOVE')
                col.separator()
            col.operator("splatting.move_instance", text="", icon='TRIA_UP').direction = 'UP'
            col.operator("splatting.move_instance", text="", icon='TRIA_DOWN').direction = 'DOWN'

            # Light Probe baking
            idx = splatting_props.active_instance_index
            if 0 <= idx < len(scene.splatting_instances):
                item = scene.splatting_instances[idx]
                obj = bpy.data.objects.get(item.mesh_name)
                if obj and obj.type == 'MESH':
                    meshes_body.separator()
                    has_probes = len(obj.probe_points) > 0
                    row = meshes_body.row(align=True)
                    row.operator("splatting.bake_lightprobe", text="Bake Light Probe", icon='LIGHTPROBE_VOLUME')
                    if has_probes:
                        row.operator("splatting.remove_lightprobe", text="", icon='X')

        layout.separator()

        # --- Settings (collapsible, only when not rendering) ---
        settings_header, settings_body = layout.panel_prop(splatting_props, "ui_settings_expanded")
        settings_header.label(text="Settings", icon='SETTINGS')
        if settings_body:
            settings_body.prop(splatting_props, "show_block_grid", text="Display Grid")
            if splatting_props.show_block_grid:
                row = settings_body.row(align=True)
                row.prop(splatting_props, "grid_color", text="")
                row.prop(splatting_props, "grid_alpha", text="Alpha")
              
            if splatting_props.show_block_grid or not splatting_props.is_rendering: 
                row = settings_body.row(align=True)
                row.label(text="Block Offset")
                row.prop(splatting_props, "block_offset", index=0, text="X")
                row.prop(splatting_props, "block_offset", index=1, text="Y")
                row.prop(splatting_props, "block_offset", index=2, text="Z")

            if not splatting_props.is_rendering:
                settings_body.prop(splatting_props, "block_size", text="Block Size")
                settings_body.prop(splatting_props, "clip_alpha", text="Clip Alpha")
                settings_body.prop(splatting_props, "clip_size", text="Clip Size")

            if splatting_props.is_rendering:
                settings_body.prop(splatting_props, "block_sort_method", text="Sort Method")
                settings_body.separator()
                settings_body.prop(splatting_props, "debug_mix", text="Debug Mix", slider=True)

            # # Debug section (visible when not rendering)
            # if not splatting_props.is_rendering:
            #     settings_body.separator()
            #     dbg_row = settings_body.row(align=True)
            #     dbg_row.operator("splatting.debug_generate", text="Generate Debug", icon='MESH_GRID')

        # --- Render controls ---
        if not splatting_props.is_rendering:
            layout.operator("splatting.start_render", text="Start Render", icon='PLAY')
        else:
            layout.operator("splatting.stop_render", text="Stop Render", icon='CANCEL')

        layout.separator()

        # Color adjustments
        color_header, color_body = layout.panel_prop(splatting_props, "ui_color_expanded")
        color_header.label(text="Render Adjust", icon='COLOR')
        if color_body:
            instances = scene.splatting_instances
            idx = splatting_props.active_instance_index
            if 0 <= idx < len(instances):
                item = instances[idx]
                name = item.mesh_name
                obj = bpy.data.objects.get(name)
                if obj and obj.type == 'MESH':
                    color_body.label(text=obj.name, icon='OBJECT_DATA')
                else:
                    color_body.label(text=name or "(missing)", icon='ERROR')
                color_body.prop(item, "color_tint", text="Tint")
                color_body.prop(item, "color_brightness", text="Exposure", slider=True)
                color_body.prop(item, "color_gamma", text="Gamma", slider=True)
                color_body.prop(item, "color_hue", text="Hue", slider=True)
                color_body.prop(item, "color_saturation", text="Saturation", slider=True)
                color_body.split()
                color_body.prop(item, "quad_scale", text="Splat Scale", slider=True)
                row = color_body.split(factor=0.5)
                row.label(text="")
                row.operator("splatting.set_default_render", text="Set as Default")
            else:
                color_body.label(text="Select a mesh from Splat Meshes list", icon='INFO')

        layout.split()

        # Animation export
        export_header, export_body = layout.panel_prop(splatting_props, "ui_export_expanded")
        export_header.label(text="Export Animation", icon='RENDER_ANIMATION')
        if export_body:
            export_body.prop(splatting_props, "anim_start_frame", text="Start")
            export_body.prop(splatting_props, "anim_end_frame", text="End")
            export_body.prop(splatting_props, "anim_output_path", text="Output")
            export_body.prop(splatting_props, "anim_force_sort", text="Force Sorting Every Frame")
            if splatting_props.is_rendering:
                export_body.operator("splatting.export_animation", text="Export Frames", icon='OUTPUT')

        layout.split()

        # Statistics
        if splatting_props.is_rendering:
            from .splatting_data import get_state
            scene_state = get_state()
            stats_header, stats_body = layout.panel_prop(splatting_props, "ui_stats_expanded")
            stats_header.label(text="Statistics", icon='INFO')
            if stats_body:
                # Only count enabled instances (target_mesh may be freed)
                enabled_names = {item.mesh_name for item in scene.splatting_instances if item.enabled}
                total_points = total_blocks = 0
                for inst in scene_state.instances:
                    try:
                        if inst.target_mesh and inst.target_mesh.name in enabled_names:
                            total_points += inst.point_count
                            total_blocks += inst.block_count
                    except ReferenceError:
                        continue
                stats_body.label(
                    text=f"Total: {total_blocks:,} blocks, {total_points:,} splats")
                stats_body.label(
                    text=f"Displayed: {scene_state.displayed_block_count:,} blocks, {scene_state.displayed_splat_count:,} splats")


def register():
    bpy.utils.register_class(SPLATTING_UL_instances)
    bpy.utils.register_class(SPLATTING_PT_panel)


def unregister():
    bpy.utils.unregister_class(SPLATTING_PT_panel)
    bpy.utils.unregister_class(SPLATTING_UL_instances)
