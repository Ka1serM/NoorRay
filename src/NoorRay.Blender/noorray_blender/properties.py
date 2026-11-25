import bpy

#! The identifiers are the names of noorray.Engine.configure; the numbers stay fixed, so saved files keep their choice.
DENOISERS = (("OFF", "Off", "Show the film without a denoiser", 0),
             ("NRD", "NRD", "Denoise with RELAX of NVIDIA NRD and upscale with FSR", 1),
             ("OPEN_DLSS_RR", "OpenDLSS-RR", "Denoise and upscale with OpenDLSS-RR, the Vulkan port of DLSS Ray Reconstruction", 3))
#! The quality modes of FSR 3 and DLSS and their render scales.
UPSCALE_QUALITIES = (("NATIVE", "Native AA", "Render at the display size; the upscaler only anti-aliases", 1.0),
                     ("QUALITY", "Quality", "Render at 1/1.5 of the display size", 1.0 / 1.5),
                     ("BALANCED", "Balanced", "Render at 1/1.7 of the display size", 1.0 / 1.7),
                     ("PERFORMANCE", "Performance", "Render at 1/2 of the display size", 0.5),
                     ("ULTRA_PERFORMANCE", "Ultra Performance", "Render at 1/3 of the display size", 1.0 / 3.0))
RENDER_SCALES = {key: scale for key, _, _, scale in UPSCALE_QUALITIES}
#! The identifiers are the reuse names of noorray.Engine.configure in upper case.
REUSES = (("OFF", "Off", "Trace each frame on its own", 0),
          ("DI", "ReSTIR DI", "Resample the direct light of neighbouring pixels and earlier frames; the bounces are path traced", 1),
          ("PT", "ReSTIR PT", "Resample whole paths of neighbouring pixels and earlier frames; slower, for scenes lit mostly by bounced light", 2))


def uses_dlss_rr(denoiser):
    return denoiser == "OPEN_DLSS_RR"


#! The light paths, the light tree and the seed are Cycles' settings of the scene, so that a scene made for Cycles renders the same.
def cycles_settings(scene):
    cycles = getattr(scene, "cycles", None)
    if cycles is None:
        raise RuntimeError("NoorRay reads the light paths of Cycles: enable the Cycles add-on")
    return cycles


#! Blender redraws on its own only for a change in the UI, not for one from Python.
def redraw(self, context):
    self.id_data.update_tag()


class NoorRaySettings(bpy.types.PropertyGroup):
    samples: bpy.props.IntProperty(name="Samples", description="The samples per pixel of a final render", default=1024, min=1, update=redraw)
    motion_steps: bpy.props.IntProperty(name="Steps", description="The times in the shutter interval that a final render with motion blur samples",
                                        default=16, min=2, max=256, update=redraw)
    viewport_samples: bpy.props.IntProperty(name="Viewport Samples",
                                            description="The realtime frames of the viewport before it stops; 0 renders without a limit", default=256, min=0, update=redraw)
    denoiser: bpy.props.EnumProperty(name="Denoiser", description="The denoiser of the film", items=DENOISERS, default="OPEN_DLSS_RR", update=redraw)
    temporal: bpy.props.BoolProperty(name="Temporal Reuse", description="Reuse the denoiser history and the ReSTIR reservoirs of earlier frames",
                                     default=True, update=redraw)
    visibility_cache: bpy.props.BoolProperty(name="Visibility Cache",
                                             description="Learn which lights are occluded in each region of the scene, so that fewer shadow rays "
                                                         "are wasted", default=True, update=redraw)
    reuse: bpy.props.EnumProperty(name="ReSTIR", description="The samples that each pixel reuses from neighbouring pixels and earlier frames",
                                  items=REUSES, default="OFF", update=redraw)
    radiance_cache: bpy.props.BoolProperty(name="Radiance Cache",
                                           description="End the realtime paths early in the SHaRC radiance cache; faster, but slightly biased",
                                           default=True, update=redraw)
    upscale: bpy.props.BoolProperty(name="FSR", description="Upscale and anti-alias the NRD frames with FSR 3; needs Temporal Reuse", default=True, update=redraw)
    upscale_quality: bpy.props.EnumProperty(name="Quality", description="The render resolution of FSR or DLSS-RR as a share of the display size",
                                            items=[item[:3] for item in UPSCALE_QUALITIES], default="QUALITY", update=redraw)
    sharpness: bpy.props.FloatProperty(name="Sharpness", description="The sharpening of the FSR output", default=0.2, min=0.0, max=1.0,
                                       subtype="FACTOR", update=redraw)

    def engine_settings(self, final):
        scene = self.id_data
        cycles = cycles_settings(scene)
        seed = cycles.seed + (scene.frame_current if cycles.use_animated_seed else 0)
        return {"samples": self.samples if final else self.viewport_samples, "denoiser": self.denoiser.lower(), "temporal": self.temporal,
                "final": final, "max_bounces": cycles.max_bounces, "volume_bounces": cycles.volume_bounces,
                "transparent_bounces": cycles.transparent_max_bounces, "diffuse_bounces": cycles.diffuse_bounces,
                "glossy_bounces": cycles.glossy_bounces, "transmission_bounces": cycles.transmission_bounces, "filter_glossy": cycles.blur_glossy,
                "clamp_direct": cycles.sample_clamp_direct, "clamp_indirect": cycles.sample_clamp_indirect, "filter_width": scene.render.filter_size,
                "light_tree": cycles.use_light_tree, "visibility_cache": self.visibility_cache, "seed": seed % (1 << 32), "reuse": self.reuse.lower(),
                "radiance_cache": self.radiance_cache, "upscale": self.upscale, "render_scale": RENDER_SCALES[self.upscale_quality],
                "sharpness": self.sharpness, "exposure": cycles.film_exposure}
