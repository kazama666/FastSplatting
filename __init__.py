bl_info = {
    "name": "Fast Splatting",
    "author": "Kazama666",
    "description": "High-performance Gaussian Splatting viewer for Blender",
    "blender": (4, 2, 0),
    "version": (0, 7, 1),
    "location": "3D Viewport > Sidebar > FastSplatting",
    "warning": "",
    "category": "3D View",
}

# Support hot-reload for development
if "bpy" in locals():
    import importlib
    from . import operators, panels, splatting_data, gpu_renderer, load_spz
    importlib.reload(operators)
    importlib.reload(panels)
    importlib.reload(splatting_data)
    importlib.reload(gpu_renderer)
    importlib.reload(load_spz)
else:
    from . import operators
    from . import panels
    from . import splatting_data


def register():
    splatting_data.register()
    operators.register()
    panels.register()


def unregister():
    splatting_data.unregister()
    panels.unregister()
    operators.unregister()
