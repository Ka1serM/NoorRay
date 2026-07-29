# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import math
import os
import struct
import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Camera, Operator, Panel, PropertyGroup
from bl_ui import properties_data_light, properties_output, properties_render, properties_world

from .materialx_sync import (
    MaterialXRenderSettingsSync,
    register_handlers as _register_materialx_handlers,
    unregister_handlers as _unregister_materialx_handlers,
)

MAGIC = b"GSPLAT\x00"
PROXY_SIZE = 1.0e-5


def _encode_splat_path(mesh: bpy.types.Mesh, path: str) -> None:
    raw_path = bpy.path.abspath(path).encode("utf-8")
    payload = MAGIC + struct.pack("<I", len(raw_path)) + raw_path

    triangle_count = max(1, math.ceil(len(payload) / 6))
    payload += bytes(triangle_count * 6 - len(payload))

    vertices = []
    faces = []

    for triangle_index in range(triangle_count):
        base = len(vertices)
        offset = triangle_index * PROXY_SIZE * 0.01
        vertices.extend([
            (offset, 0.0, 0.0),
            (offset + PROXY_SIZE, 0.0, 0.0),
            (offset, PROXY_SIZE, 0.0),
        ])
        faces.append((base, base + 1, base + 2))

    mesh.clear_geometry()
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="GSplatData")
    for loop_index, uv_item in enumerate(uv_layer.data):
        byte_index = loop_index * 2
        byte_0 = payload[byte_index]
        byte_1 = payload[byte_index + 1]
        uv_item.uv = (byte_0 / 255.0, byte_1 / 255.0)

    mesh.update()


def _plugin_directory() -> Path:
    override = os.environ.get("HDNOORRAY_PLUGIN_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "plugin"


class NoorRaySettings(PropertyGroup):
    samples: IntProperty(
        name="Samples",
        description="Maximum progressive samples per pixel",
        default=64,
        min=1,
        max=1_000_000,
    )
    noise_limit_enabled: BoolProperty(
        name="Use Noise Limit",
        description="Stop rendering early when the noise estimate falls below the threshold",
        default=False,
    )
    noise_level: FloatProperty(
        name="Noise Threshold",
        description="Noise level at which rendering is considered converged",
        default=0.0001,
        min=0.0,
        max=1.0,
        precision=4,
    )
    max_bounces: IntProperty(
        name="Max Bounces",
        description="Maximum number of ray bounces (total depth)",
        default=8,
        min=1,
        max=65,
    )
    russian_roulette_start_bounce: IntProperty(
        name="Russian Roulette",
        description="Bounce at which Russian roulette ray termination begins (0 disables)",
        default=3,
        min=0,
        max=65,
    )
    optix_denoiser_enabled: BoolProperty(
        name="Use OptiX Denoiser",
        description="Apply the OptiX AI denoiser as a post-process",
        default=False,
    )
    optix_denoiser_min_samples: IntProperty(
        name="Denoiser Start Sample",
        description="First sample at which the denoiser runs",
        default=1,
        min=1,
        max=1_000_000,
    )
    transparent_background: BoolProperty(
        name="Transparent Background",
        description="Render the background as transparent instead of solid color",
        default=False,
    )
    gaussian_cutoff_sigma: FloatProperty(
        name="Gaussian Cutoff",
        description="Gaussian splat cutoff in standard deviations",
        default=3.0,
        min=0.1,
        max=10.0,
    )
    gaussian_proxy_type: EnumProperty(
        name="Gaussian Proxy",
        description="Gaussian proxy mesh type for OptiX intersection",
        items=[
            ("ICOSPHERE", "Icosphere", ""),
            ("OCTAHEDRON", "Octahedron", ""),
            ("ICOSAHEDRON", "Icosahedron", ""),
            ("ICOSPHERE_L2", "Icosphere L2", ""),
        ],
        default="ICOSPHERE_L2",
    )
    gaussian_shading_mode: EnumProperty(
        name="Gaussian Shading",
        description="Shading mode for Gaussian splats",
        items=[
            ("GI", "Global Illumination", "Full global illumination"),
            ("DIRECT", "Direct Color", "Direct color only"),
        ],
        default="DIRECT",
    )
    gaussian_proxy_overdraw: BoolProperty(
        name="Proxy Overdraw",
        description="Visualize Gaussian proxy overdraw count",
        default=False,
    )
    gaussian_sh_degree: EnumProperty(
        name="Gaussian SH Degree",
        description="Spherical harmonics degree for Gaussian splats",
        items=[
            ("0", "Degree 0", ""),
            ("1", "Degree 1", ""),
            ("2", "Degree 2", ""),
            ("3", "Degree 3", ""),
        ],
        default="3",
    )


class NoorRayHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = "NOORRAY_HYDRA"
    bl_label = "NoorRay"
    bl_info = "CUDA/OptiX Hydra render delegate"

    bl_use_preview = True
    bl_use_gpu_context = True
    # NoorRay translates Blender's evaluated node trees itself and transports
    # compact documents as render settings.  Disabling Blender's exporter
    # avoids building a second, slower and less compatible MaterialX graph.
    bl_use_materialx = False
    bl_delegate_id = "HdNoorRayRendererPlugin"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._materialx_sync = MaterialXRenderSettingsSync()
        self._settings_depsgraph = None

    def _engine_update(self, engine_type, depsgraph, context):
        import _bpy_hydra

        if not self.engine_ptr:
            self.engine_ptr = _bpy_hydra.engine_create(
                self, engine_type, self.bl_delegate_id)
        if not self.engine_ptr:
            return

        # Blender's base HydraRenderEngine applies settings after
        # engine_update().  Material documents must arrive first so the
        # material Sprim's initial Sync can consume them without forcing a
        # redundant second scene update.
        self._settings_depsgraph = depsgraph
        try:
            render_settings = self.get_render_settings(engine_type)
        finally:
            self._settings_depsgraph = None
        for key, value in render_settings.items():
            _bpy_hydra.engine_set_render_setting(self.engine_ptr, key, value)
        _bpy_hydra.engine_update(self.engine_ptr, depsgraph, context)

    def update(self, data, depsgraph):
        del data
        self._engine_update(
            "PREVIEW" if self.is_preview else "FINAL", depsgraph, None)

    def view_update(self, context, depsgraph):
        self._engine_update("VIEWPORT", depsgraph, context)

    def get_render_settings(self, engine_type):
        # The delegate hands Blender scene-linear data and leaves display
        # transform, exposure, look and curves to the scene's own colour
        # management, which the panels below expose unmodified.
        scene = bpy.context.scene
        settings = scene.hdnoorray
        gauss = settings.gaussian_proxy_type
        shading = settings.gaussian_shading_mode
        result = {
            "samples": settings.samples,
            "maxBounces": settings.max_bounces,
            "noiseLimitEnabled": int(settings.noise_limit_enabled),
            "noiseLevel": settings.noise_level,
            "optixDenoiserEnabled": int(settings.optix_denoiser_enabled),
            "optixDenoiserMinSamples": settings.optix_denoiser_min_samples,
            "russianRouletteStartBounce": settings.russian_roulette_start_bounce,
            "transparentBackground": int(settings.transparent_background),
            "gaussianCutoffSigma": settings.gaussian_cutoff_sigma,
            "gaussianProxyType": ["ICOSPHERE", "OCTAHEDRON", "ICOSAHEDRON", "ICOSPHERE_L2"].index(gauss),
            "gaussianShadingMode": ["GI", "DIRECT"].index(shading),
            "gaussianProxyOverdrawVisualization": int(settings.gaussian_proxy_overdraw),
            "gaussianRenderSphericalHarmonics": int(settings.gaussian_sh_degree),
        }
        depsgraph = self._settings_depsgraph
        if depsgraph is not None:
            scene_eval = getattr(depsgraph, "scene_eval", scene)
            result.update(self._materialx_sync.collect(depsgraph, scene_eval))

        # Per-camera settings
        cam = bpy.context.scene.camera
        if cam and cam.data:
            nrcam = cam.data.hdnoorray
            result["cameraProjection"] = [
                "PERSPECTIVE", "ORTHOGRAPHIC", "FISHEYE",
                "THINLENS", "REALISTIC", "HYBRIDPSF",
            ].index(nrcam.projection_type)
            if nrcam.projection_type in {"THINLENS", "FISHEYE", "REALISTIC", "HYBRIDPSF"}:
                result["cameraApertureDiameter"] = nrcam.aperture_diameter_mm
            if nrcam.projection_type in {"THINLENS", "FISHEYE"}:
                result["cameraBokehBias"] = nrcam.bokeh_bias
            if nrcam.projection_type in {"REALISTIC", "HYBRIDPSF"}:
                result["cameraLensPath"] = nrcam.lens_path
                result["cameraGlassCatalogs"] = nrcam.glass_catalogs
            if nrcam.projection_type == "HYBRIDPSF":
                result["cameraRayLutPath"] = nrcam.ray_lut_path
            result["cameraSensorType"] = ["RECTANGULAR", "SCATTERPSF", "GATHERPSF"].index(
                nrcam.sensor_type)
            if nrcam.sensor_type != "RECTANGULAR":
                result["cameraSensorPath"] = nrcam.sensor_path
                result["cameraPsfPath"] = nrcam.psf_path

        if engine_type != "VIEWPORT":
            result |= {
                "aovToken:Combined": "color",
            }
        return result

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(
                scene, render_layer, "Combined", 4, "RGBA", "COLOR")


class NOORRAY_PT_sampling(Panel):
    bl_label = "Sampling"
    bl_idname = "NOORRAY_PT_sampling"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.hdnoorray
        layout.prop(settings, "samples")
        layout.prop(settings, "noise_limit_enabled")
        row = layout.row()
        row.enabled = settings.noise_limit_enabled
        row.prop(settings, "noise_level")


class NOORRAY_PT_light_paths(Panel):
    bl_label = "Light Paths"
    bl_idname = "NOORRAY_PT_light_paths"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.hdnoorray
        layout.prop(settings, "max_bounces")
        layout.prop(settings, "russian_roulette_start_bounce")


class NOORRAY_PT_denoising(Panel):
    bl_label = "Denoising"
    bl_idname = "NOORRAY_PT_denoising"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.hdnoorray
        layout.prop(settings, "optix_denoiser_enabled")
        row = layout.row()
        row.enabled = settings.optix_denoiser_enabled
        row.prop(settings, "optix_denoiser_min_samples")


class NOORRAY_PT_output(Panel):
    bl_label = "Output"
    bl_idname = "NOORRAY_PT_output"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.hdnoorray
        layout.prop(settings, "transparent_background")


class NOORRAY_PT_gaussians(Panel):
    bl_label = "Gaussians"
    bl_idname = "NOORRAY_PT_gaussians"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.engine == NoorRayHydraRenderEngine.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.hdnoorray
        layout.prop(settings, "gaussian_cutoff_sigma")
        layout.prop(settings, "gaussian_proxy_type")
        layout.prop(settings, "gaussian_shading_mode")
        layout.prop(settings, "gaussian_proxy_overdraw")
        layout.prop(settings, "gaussian_sh_degree")


class NoorRayCameraSettings(PropertyGroup):
    projection_type: EnumProperty(
        name="Projection",
        description="camera projection model",
        items=[
            ("PERSPECTIVE", "Perspective", "Simple pinhole perspective"),
            ("THINLENS", "Thin Lens", "Perspective with depth of field"),
            ("ORTHOGRAPHIC", "Orthographic", "Orthographic projection"),
            ("FISHEYE", "Fisheye", "Fisheye with depth of field"),
            ("REALISTIC", "Realistic", "Physical lens model (ROSS)"),
            ("HYBRIDPSF", "Hybrid PSF", "Hybrid PSF lens model"),
        ],
        default="PERSPECTIVE",
    )
    aperture_diameter_mm: FloatProperty(
        name="Aperture",
        description="Aperture diameter in mm (0 = pinhole)",
        default=0.0,
        min=0.0,
        max=1000.0,
        precision=2,
    )
    bokeh_bias: FloatProperty(
        name="Bokeh Bias",
        description="Bokeh shape bias (1 = uniform)",
        default=1.0,
        min=0.01,
        max=10.0,
    )
    lens_path: StringProperty(
        name="Lens File",
        description="Path to a ROSS lens description file",
        default="",
        subtype="FILE_PATH",
    )
    glass_catalogs: StringProperty(
        name="Glass Catalogs",
        description="Comma-separated paths to glass catalog files",
        default="",
        subtype="FILE_PATH",
    )
    ray_lut_path: StringProperty(
        name="Ray LUT",
        description="Path to a precomputed ray lookup table",
        default="",
        subtype="FILE_PATH",
    )
    sensor_type: EnumProperty(
        name="Sensor Type",
        description="Sensor / PSF model",
        items=[
            ("RECTANGULAR", "Rectangular", "Standard rectangular sensor"),
            ("SCATTERPSF", "Scatter PSF", "Point spread function scattering"),
            ("GATHERPSF", "Gather PSF", "Point spread function gathering"),
        ],
        default="RECTANGULAR",
    )
    sensor_path: StringProperty(
        name="Sensor File",
        description="Path to an image sensor description file",
        default="",
        subtype="FILE_PATH",
    )
    psf_path: StringProperty(
        name="PSF Grid",
        description="Path to a PSF grid file",
        default="",
        subtype="FILE_PATH",
    )


class NOORRAY_OT_add_gaussian_splat(Operator):
    bl_idname = "hdnoorray.add_gaussian_splat"
    bl_label = "Gaussian Splat"
    bl_description = "Add a Gaussian splat object"
    bl_property = "filepath"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        path = self.filepath
        if not path:
            return {"CANCELLED"}
        raw_name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for ext in (".spz", ".ply", ".splat"):
            if raw_name.endswith(ext):
                raw_name = raw_name[: -len(ext)]
                break

        identifier = uuid.uuid4().hex

        mesh = bpy.data.meshes.new(f".GSplatMesh_{identifier}")
        _encode_splat_path(mesh, path)

        marker = bpy.data.objects.new(f".GSplatMarker_{identifier}", mesh)

        prototype_collection = bpy.data.collections.new(f".GSplatPrototype_{identifier}")
        prototype_collection.objects.link(marker)

        empty = bpy.data.objects.new(raw_name, None)
        context.collection.objects.link(empty)

        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 1.0
        empty.instance_type = "COLLECTION"
        empty.instance_collection = prototype_collection
        empty.rotation_euler = (math.radians(90), 0, 0)

        empty["gaussian_splat_path"] = path
        empty["gaussian_splat_marker_mesh"] = mesh.name

        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class NOORRAY_PT_gaussian_splat_object(Panel):
    bl_label = "Gaussian Splat"
    bl_idname = "NOORRAY_PT_gaussian_splat_object"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return (context.object
                and context.object.get("gaussian_splat_path")
                and context.engine == NoorRayHydraRenderEngine.bl_idname)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        obj = context.object
        layout.prop(obj, '["gaussian_splat_path"]', text="File Path")
        layout.prop(obj, "name")


def _add_menu_entry(self, context):
    self.layout.operator(
        "hdnoorray.add_gaussian_splat",
        text="Gaussian Splat",
        icon="FILE_FOLDER")


class NOORRAY_PT_camera(Panel):
    bl_label = "Camera"
    bl_idname = "NOORRAY_PT_camera"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return (context.camera and
                context.engine == NoorRayHydraRenderEngine.bl_idname)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        cam_data = context.camera
        nrcam = cam_data.hdnoorray

        layout.prop(nrcam, "projection_type")
        ptype = nrcam.projection_type
        if ptype != "ORTHOGRAPHIC":
            layout.prop(cam_data, "lens")
        layout.prop(cam_data, "sensor_width")
        layout.prop(cam_data, "sensor_height")
        if ptype in {"THINLENS", "FISHEYE", "REALISTIC", "HYBRIDPSF"}:
            layout.prop(cam_data.dof, "focus_distance")
            layout.prop(nrcam, "aperture_diameter_mm")
        if ptype in {"THINLENS", "FISHEYE"}:
            layout.prop(nrcam, "bokeh_bias")
        if ptype in {"REALISTIC", "HYBRIDPSF"}:
            col = layout.column(align=True)
            col.prop(nrcam, "lens_path")
            col.prop(nrcam, "glass_catalogs")
        if ptype == "HYBRIDPSF":
            layout.prop(nrcam, "ray_lut_path")

        layout.separator()
        layout.label(text="Sensor", icon="VIEW3D")
        layout.prop(nrcam, "sensor_type")
        stype = nrcam.sensor_type
        if stype != "RECTANGULAR":
            layout.prop(nrcam, "sensor_path")
            layout.prop(nrcam, "psf_path")


_CLASSES = (
    NoorRaySettings,
    NoorRayCameraSettings,
    NoorRayHydraRenderEngine,
    NOORRAY_OT_add_gaussian_splat,
    NOORRAY_PT_sampling,
    NOORRAY_PT_light_paths,
    NOORRAY_PT_denoising,
    NOORRAY_PT_output,
    NOORRAY_PT_gaussians,
    NOORRAY_PT_camera,
    NOORRAY_PT_gaussian_splat_object,
)

# Blender's own panels that work as-is with this engine. The color management
# ones are listed by name because the exact set of subpanels differs between
# Blender versions, and a missing one must not break registration.
_COMPATIBLE_PANEL_NAMES = (
    (properties_data_light, (
        "DATA_PT_context_light",
        "DATA_PT_preview",
        "DATA_PT_EEVEE_light",
        "DATA_PT_spot",
        "DATA_PT_light_animation",
        "DATA_PT_custom_props_light",
    )),
    (properties_world, (
        "WORLD_PT_context_world",
        "EEVEE_WORLD_PT_surface",
        "WORLD_PT_animation",
        "WORLD_PT_custom_props",
    )),
    # The engine outputs scene-linear data, so the full color management stack
    # applies to it unchanged.
    (properties_render, (
        "RENDER_PT_color_management",
        "RENDER_PT_color_management_working_space",
        "RENDER_PT_color_management_white_balance",
        "RENDER_PT_color_management_curves",
        "RENDER_PT_color_management_advanced",
    )),
    # Without these the Output tab is empty for this engine, leaving no way to
    # set resolution, frame range or the output path.
    (properties_output, (
        "RENDER_PT_format",
        "RENDER_PT_frame_range",
        "RENDER_PT_time_stretching",
        "RENDER_PT_stereoscopy",
        "RENDER_PT_output",
        "RENDER_PT_output_views",
        "RENDER_PT_output_color_management",
        "RENDER_PT_encoding",
        "RENDER_PT_encoding_video",
        "RENDER_PT_encoding_audio",
        "RENDER_PT_post_processing",
        "RENDER_PT_stamp",
        "RENDER_PT_stamp_note",
        "RENDER_PT_stamp_burn",
    )),
)

_COMPATIBLE_PANELS = [
    panel
    for module, names in _COMPATIBLE_PANEL_NAMES
    for panel in (getattr(module, name, None) for name in names)
    if panel is not None
]


@bpy.app.handlers.persistent
def _on_load_post(_dummy):
    _restore_forced_view_transform()


def _set_panel_compatibility(enabled: bool):
    engine_id = NoorRayHydraRenderEngine.bl_idname
    for panel in _COMPATIBLE_PANELS:
        if enabled:
            panel.COMPAT_ENGINES.add(engine_id)
        else:
            panel.COMPAT_ENGINES.discard(engine_id)


def _restore_forced_view_transform():
    # Earlier versions forced the view transform to Raw and stashed the previous
    # value on the scene. Hand it back, once, so upgrading does not leave a
    # scene stuck on Raw with a stray custom property.
    #
    # register() runs while Blender still restricts bpy.data (during startup and
    # in background mode), where there is nothing to restore yet — the load_post
    # handler covers those scenes once the file is open.
    if not hasattr(bpy.data, "scenes"):
        return
    for scene in bpy.data.scenes:
        previous = scene.get("hdnoorray_prev_vt")
        if not previous:
            continue
        try:
            scene.view_settings.view_transform = previous
        except (TypeError, AttributeError):
            pass
        del scene["hdnoorray_prev_vt"]


def register():
    plugin_directory = _plugin_directory()
    plug_info = plugin_directory / "plugInfo.json"
    if not plug_info.is_file():
        raise RuntimeError(
            f"hdplugin bundle is incomplete: {plug_info} is missing")

    bpy.utils.expose_bundled_modules()
    import pxr.Plug

    pxr.Plug.Registry().RegisterPlugins([str(plugin_directory)])
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hdnoorray = PointerProperty(type=NoorRaySettings)
    bpy.types.Camera.hdnoorray = PointerProperty(type=NoorRayCameraSettings)
    bpy.types.VIEW3D_MT_add.append(_add_menu_entry)
    _set_panel_compatibility(True)
    _register_materialx_handlers()
    bpy.app.handlers.load_post.append(_on_load_post)
    _restore_forced_view_transform()


def unregister():
    _set_panel_compatibility(False)
    _unregister_materialx_handlers()
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    bpy.types.VIEW3D_MT_add.remove(_add_menu_entry)
    if hasattr(bpy.types.Camera, "hdnoorray"):
        del bpy.types.Camera.hdnoorray
    if hasattr(bpy.types.Scene, "hdnoorray"):
        del bpy.types.Scene.hdnoorray

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
