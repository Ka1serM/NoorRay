# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import os

import bpy
from bpy.props import IntProperty, PointerProperty
from bpy.types import Panel, PropertyGroup
from bl_ui import properties_data_light, properties_world


def _plugin_directory() -> Path:
    override = os.environ.get("HDNOORRAY_PLUGIN_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "plugin"


class NoorRaySettings(PropertyGroup):
    samples: IntProperty(
        name="Samples",
        description="Progressive samples per pixel",
        default=64,
        min=1,
        max=1_000_000,
    )
    max_bounces: IntProperty(
        name="Maximum Bounces",
        default=8,
        min=1,
        max=65,
    )


class NoorRayHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = "NOORRAY_HYDRA"
    bl_label = "NoorRay"
    bl_info = "NoorRay CUDA/OptiX Hydra render delegate"

    bl_use_preview = True
    # Lets Blender bind its viewport framebuffer while Hydra executes. The
    # delegate copies NoorRay's CUDA image directly into that GL texture.
    bl_use_gpu_context = True
    bl_use_materialx = False
    bl_delegate_id = "HdNoorRayRendererPlugin"

    def get_render_settings(self, engine_type):
        settings = bpy.context.scene.hdnoorray
        result = {
            "samples": settings.samples,
            "maxBounces": settings.max_bounces,
        }
        if engine_type != "VIEWPORT":
            result |= {
                "aovToken:Combined": "color",
                "aovToken:Depth": "depth",
            }
        return result

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(
                scene, render_layer, "Combined", 4, "RGBA", "COLOR")
        if render_layer.use_pass_z:
            self.register_pass(
                scene, render_layer, "Depth", 1, "Z", "VALUE")


class NOORRAY_PT_render_settings(Panel):
    bl_label = "NoorRay"
    bl_idname = "NOORRAY_PT_render_settings"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hdnoorray
        layout.prop(settings, "samples")
        layout.prop(settings, "max_bounces")


_CLASSES = (
    NoorRaySettings,
    NoorRayHydraRenderEngine,
    NOORRAY_PT_render_settings,
)

_COMPATIBLE_PANELS = (
    properties_data_light.DATA_PT_context_light,
    properties_data_light.DATA_PT_preview,
    properties_data_light.DATA_PT_EEVEE_light,
    properties_data_light.DATA_PT_spot,
    properties_data_light.DATA_PT_light_animation,
    properties_data_light.DATA_PT_custom_props_light,
    properties_world.WORLD_PT_context_world,
    properties_world.EEVEE_WORLD_PT_surface,
    properties_world.WORLD_PT_animation,
    properties_world.WORLD_PT_custom_props,
)


def _set_panel_compatibility(enabled: bool):
    engine_id = NoorRayHydraRenderEngine.bl_idname
    for panel in _COMPATIBLE_PANELS:
        if enabled:
            panel.COMPAT_ENGINES.add(engine_id)
        else:
            panel.COMPAT_ENGINES.discard(engine_id)


def register():
    plugin_directory = _plugin_directory()
    plug_info = plugin_directory / "plugInfo.json"
    if not plug_info.is_file():
        raise RuntimeError(
            f"hdNoorRay plugin bundle is incomplete: {plug_info} is missing")

    bpy.utils.expose_bundled_modules()
    import pxr.Plug

    pxr.Plug.Registry().RegisterPlugins([str(plugin_directory)])
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hdnoorray = PointerProperty(type=NoorRaySettings)
    _set_panel_compatibility(True)


def unregister():
    _set_panel_compatibility(False)
    if hasattr(bpy.types.Scene, "hdnoorray"):
        del bpy.types.Scene.hdnoorray
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
