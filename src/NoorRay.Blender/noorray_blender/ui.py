import bpy

from . import noorray
from .engine import NoorRayRenderEngine
from .properties import uses_dlss_rr


class NoorRayPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {NoorRayRenderEngine.bl_idname}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def split_layout(self):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        return layout


#! A panel of Cycles' settings tells why it is empty when the Cycles add-on is off.
class CyclesPanel(NoorRayPanel):
    def draw(self, context):
        layout = self.split_layout()
        cycles = getattr(context.scene, "cycles", None)
        if cycles is None:
            layout.label(text="Enable the Cycles add-on for these settings", icon="ERROR")
            return
        self.draw_cycles(context, layout, cycles)


class NOORRAY_RENDER_PT_sampling(NoorRayPanel, bpy.types.Panel):
    bl_label = "Sampling"

    def draw(self, context):
        settings = context.scene.noorray
        layout = self.split_layout()
        layout.prop(settings, "denoiser")
        if settings.denoiser.lower() not in noorray.denoisers():
            layout.label(text="This build of NoorRay has no such denoiser", icon="ERROR")


class NOORRAY_RENDER_PT_sampling_viewport(NoorRayPanel, bpy.types.Panel):
    bl_label = "Viewport"
    bl_parent_id = "NOORRAY_RENDER_PT_sampling"

    def draw(self, context):
        self.split_layout().prop(context.scene.noorray, "viewport_samples", text="Max Frames")


class NOORRAY_RENDER_PT_sampling_render(CyclesPanel, bpy.types.Panel):
    bl_label = "Render"
    bl_parent_id = "NOORRAY_RENDER_PT_sampling"

    def draw_cycles(self, context, layout, cycles):
        layout.prop(context.scene.noorray, "samples", text="Max Samples")
        layout.prop(cycles, "time_limit")


class NOORRAY_RENDER_PT_sampling_lights(CyclesPanel, bpy.types.Panel):
    bl_label = "Lights"
    bl_parent_id = "NOORRAY_RENDER_PT_sampling"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_cycles(self, context, layout, cycles):
        layout.prop(cycles, "use_light_tree")
        layout.prop(context.scene.noorray, "visibility_cache")


class NOORRAY_RENDER_PT_sampling_advanced(CyclesPanel, bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "NOORRAY_RENDER_PT_sampling"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_cycles(self, context, layout, cycles):
        row = layout.row(align=True)
        row.prop(cycles, "seed")
        row.prop(cycles, "use_animated_seed", text="", icon="TIME")


class NOORRAY_RENDER_PT_realtime(NoorRayPanel, bpy.types.Panel):
    bl_label = "Realtime"

    def draw(self, context):
        settings = context.scene.noorray
        layout = self.split_layout()
        layout.prop(settings, "temporal")
        layout.prop(settings, "reuse")
        layout.prop(settings, "radiance_cache")


class NOORRAY_RENDER_PT_realtime_upscaling(NoorRayPanel, bpy.types.Panel):
    bl_label = "Upscaling"
    bl_parent_id = "NOORRAY_RENDER_PT_realtime"

    def draw_header(self, context):
        self.layout.prop(context.scene.noorray, "upscale", text="")

    def draw(self, context):
        settings = context.scene.noorray
        layout = self.split_layout()
        if uses_dlss_rr(settings.denoiser):
            layout.label(text="DLSS-RR upscales in place of FSR", icon="INFO")
            layout.prop(settings, "upscale_quality")
            return
        layout.active = settings.temporal and settings.upscale
        layout.prop(settings, "upscale_quality")
        layout.prop(settings, "sharpness")


class NOORRAY_RENDER_PT_light_paths(NoorRayPanel, bpy.types.Panel):
    bl_label = "Light Paths"

    def draw(self, context):
        pass


class NOORRAY_RENDER_PT_light_paths_max_bounces(CyclesPanel, bpy.types.Panel):
    bl_label = "Max Bounces"
    bl_parent_id = "NOORRAY_RENDER_PT_light_paths"

    def draw_cycles(self, context, layout, cycles):
        column = layout.column()
        column.prop(cycles, "max_bounces", text="Total")
        column = layout.column(align=True)
        column.prop(cycles, "diffuse_bounces", text="Diffuse")
        column.prop(cycles, "glossy_bounces", text="Glossy")
        column.prop(cycles, "transmission_bounces", text="Transmission")
        column.prop(cycles, "volume_bounces", text="Volume")
        layout.prop(cycles, "transparent_max_bounces", text="Transparent")


class NOORRAY_RENDER_PT_light_paths_clamping(CyclesPanel, bpy.types.Panel):
    bl_label = "Clamping"
    bl_parent_id = "NOORRAY_RENDER_PT_light_paths"

    def draw_cycles(self, context, layout, cycles):
        column = layout.column(align=True)
        column.prop(cycles, "sample_clamp_direct", text="Direct Light")
        column.prop(cycles, "sample_clamp_indirect", text="Indirect Light")


class NOORRAY_RENDER_PT_light_paths_caustics(CyclesPanel, bpy.types.Panel):
    bl_label = "Caustics"
    bl_parent_id = "NOORRAY_RENDER_PT_light_paths"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_cycles(self, context, layout, cycles):
        layout.prop(cycles, "blur_glossy")


class NOORRAY_RENDER_PT_motion_blur(NoorRayPanel, bpy.types.Panel):
    bl_label = "Motion Blur"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.render, "use_motion_blur", text="")

    def draw(self, context):
        render = context.scene.render
        layout = self.split_layout()
        layout.active = render.use_motion_blur
        layout.prop(render, "motion_blur_position", text="Position")
        layout.prop(render, "motion_blur_shutter")
        layout.prop(context.scene.noorray, "motion_steps")


class NOORRAY_RENDER_PT_motion_blur_curve(NoorRayPanel, bpy.types.Panel):
    bl_label = "Shutter Curve"
    bl_parent_id = "NOORRAY_RENDER_PT_motion_blur"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        render = context.scene.render
        self.layout.active = render.use_motion_blur
        self.layout.template_curve_mapping(render, "motion_blur_shutter_curve")


class NOORRAY_RENDER_PT_film(NoorRayPanel, bpy.types.Panel):
    bl_label = "Film"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.split_layout()
        cycles = getattr(context.scene, "cycles", None)
        if cycles is not None:
            layout.prop(cycles, "film_exposure")
        layout.prop(context.scene.render, "filter_size", text="Pixel Filter")
        layout.prop(context.scene.render, "film_transparent", text="Transparent")


class NOORRAY_VIEWLAYER_PT_passes(NoorRayPanel, bpy.types.Panel):
    bl_label = "Passes"
    bl_context = "view_layer"

    def draw(self, context):
        layout = self.split_layout()
        column = layout.column(heading="Data")
        column.prop(context.view_layer, "use_pass_combined")
        column.prop(context.view_layer, "use_pass_z")
        column.prop(context.view_layer, "use_pass_mist")
        column.prop(context.view_layer, "use_pass_normal")
        layout.column(heading="Light").prop(context.view_layer, "use_pass_diffuse_color")


classes = (NOORRAY_RENDER_PT_sampling, NOORRAY_RENDER_PT_sampling_viewport, NOORRAY_RENDER_PT_sampling_render, NOORRAY_RENDER_PT_sampling_lights,
           NOORRAY_RENDER_PT_sampling_advanced, NOORRAY_RENDER_PT_realtime, NOORRAY_RENDER_PT_realtime_upscaling, NOORRAY_RENDER_PT_light_paths,
           NOORRAY_RENDER_PT_light_paths_max_bounces, NOORRAY_RENDER_PT_light_paths_clamping, NOORRAY_RENDER_PT_light_paths_caustics,
           NOORRAY_RENDER_PT_motion_blur, NOORRAY_RENDER_PT_motion_blur_curve, NOORRAY_RENDER_PT_film, NOORRAY_VIEWLAYER_PT_passes)


#! The built-in panels that list BLENDER_RENDER work for every engine; the others hold settings of EEVEE or Workbench.
def compatible_panels():
    return [panel for panel in bpy.types.Panel.__subclasses__() if "BLENDER_RENDER" in getattr(panel, "COMPAT_ENGINES", ())]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    for panel in compatible_panels():
        panel.COMPAT_ENGINES.add(NoorRayRenderEngine.bl_idname)


def unregister():
    for panel in compatible_panels():
        panel.COMPAT_ENGINES.discard(NoorRayRenderEngine.bl_idname)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
