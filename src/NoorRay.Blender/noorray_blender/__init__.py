import bpy

from . import export, helpers, noorray, ui
from .engine import NoorRayRenderEngine
from .properties import NoorRaySettings


def register():
    noorray.set_helpers(helpers)
    bpy.utils.register_class(NoorRaySettings)
    bpy.types.Scene.noorray = bpy.props.PointerProperty(type=NoorRaySettings)
    bpy.utils.register_class(NoorRayRenderEngine)
    ui.register()
    export.register()
    #! A Vulkan device takes seconds to make, so the add-on makes its idle device (about 35 MB) on a thread before the viewport asks for it.
    if not bpy.app.background:
        bpy.app.timers.register(noorray.prepare, first_interval=0.0)


def unregister():
    export.unregister()
    ui.unregister()
    bpy.utils.unregister_class(NoorRayRenderEngine)
    del bpy.types.Scene.noorray
    bpy.utils.unregister_class(NoorRaySettings)
    noorray.set_helpers(None)
