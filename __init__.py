bl_info = {
    "name": "Fast Splatting",
    "author": "Kazama666",
    "description": "High-performance Gaussian Splatting viewer for Blender",
    "blender": (4, 2, 0),
    "version": (0, 0, 1),
    "location": "3D Viewport > Sidebar > FastSplatting",
    "warning": "",
    "category": "3D View",
}

from . import operators
from . import panels
from . import splatting_data

def register():
    operators.register()
    panels.register()
    splatting_data.register()


def unregister():
    splatting_data.unregister()
    panels.unregister()
    operators.unregister()
