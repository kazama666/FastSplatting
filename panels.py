import bpy
from bpy import types

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
        
        if not splatting_props.is_rendering:
            # Mesh selection
            layout.label(text="Target Mesh:")
            row = layout.row()
            row.prop(scene, "splatting_target_mesh", text="")
            row.operator("splatting.select_mesh", text="", icon='OBJECT_DATA')

        box = layout.box()
        box.label(text="Block Settings", icon='SETTINGS')
        if splatting_props.is_rendering:
            box.prop(splatting_props, "sort_blocks_per_frame", text="Sort Blocks per Frame")
        else:
            box.prop(splatting_props, "block_size", text="Block Size")
        layout.split()

        # Render controls
        if not splatting_props.is_rendering:
            layout.operator("splatting.start_render", text="Start Render", icon='PLAY')
        else:
            layout.operator("splatting.stop_render", text="Stop Render", icon='CANCEL')

        layout.separator()

        '''
        # LOD toggle
        row = layout.row()
        row.prop(splatting_props, "lod_enabled", text="Enable LOD")
        if splatting_props.lod_enabled:
            row = layout.row()
            row.prop(splatting_props, "lod_bias", text="LOD Bias", slider=True)

        layout.separator()
        '''

        # # Quad scale
        # layout.label(text="Splat Size:")
        # row = layout.row()
        # row.prop(splatting_props, "quad_scale", text="Scale", slider=True)

        layout.split()

        # Color adjustments
        col_box = layout.box()
        col_box.label(text="Render Adjust", icon='COLOR')
        col_box.prop(splatting_props, "color_tint", text="Tint")
        col_box.prop(splatting_props, "color_brightness", text="Exposure", slider=True)
        col_box.prop(splatting_props, "color_gamma", text="Gamma", slider=True)
        col_box.prop(splatting_props, "color_hue", text="Hue", slider=True)
        col_box.prop(splatting_props, "color_saturation", text="Saturation", slider=True)
        col_box.split()
        col_box.prop(splatting_props, "quad_scale", text="Splat Scale", slider=True)

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

        # Infomations
        if splatting_props.is_rendering:
            from .splatting_data import get_state
            state = get_state()
            stats_header, stats_body = layout.panel_prop(splatting_props, "ui_stats_expanded")
            stats_header.label(text="Statistics", icon='INFO')
            if stats_body:
                stats_body.label(text=f"Totle: {splatting_props.block_count:,} blocks, {splatting_props.point_count:,} splats")
                stats_body.label(text=f"Displayed: {state.displayed_block_count:,} blocks, {state.displayed_splat_count:,} splats")


class SPLATTING_OT_render_status(bpy.types.Operator):
    """Dummy operator for status updates"""
    bl_idname = "splatting.update_status"
    bl_label = "Update Status"

    def execute(self, context):
        return {'FINISHED'}


def register():
    # Register target mesh property
    bpy.types.Scene.splatting_target_mesh = bpy.props.StringProperty(
        name="Target Mesh",
        description="Mesh object containing splatting PLY data",
        default="",
    )

    bpy.utils.register_class(SPLATTING_OT_render_status)
    bpy.utils.register_class(SPLATTING_PT_panel)


def unregister():
    bpy.utils.unregister_class(SPLATTING_PT_panel)
    bpy.utils.unregister_class(SPLATTING_OT_render_status)

    del bpy.types.Scene.splatting_target_mesh
