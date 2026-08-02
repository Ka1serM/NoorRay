import os
from pathlib import Path
import importlib.util

import bpy

_EXPORTER_PATH = Path(__file__).resolve().parents[2] / \
    "src/hdnoorray/blender_addon/hdnoorray/materialx_export.py"
_EXPORTER_SPEC = importlib.util.spec_from_file_location(
    "noorray_source_materialx_export", _EXPORTER_PATH)
_EXPORTER_MODULE = importlib.util.module_from_spec(_EXPORTER_SPEC)
assert _EXPORTER_SPEC.loader is not None
_EXPORTER_SPEC.loader.exec_module(_EXPORTER_MODULE)
export_material = _EXPORTER_MODULE.export_material


OUTPUT = "/tmp/noorray_principled_diffuse_color.exr"
COLOR = (0.08, 0.52, 0.82, 1.0)


def make_material(name, node_type):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.get("Material Output")
    shader = nodes.get("Principled BSDF")
    if node_type == "ShaderNodeBsdfDiffuse":
        shader = nodes.new(node_type)
    if shader is None:
        shader = nodes.new(node_type)
    if node_type == "ShaderNodeBsdfPrincipled":
        shader.inputs["Base Color"].default_value = COLOR
        shader.inputs["Roughness"].default_value = 1.0
        shader.inputs["Metallic"].default_value = 0.0
        specular = shader.inputs.get("Specular IOR Level")
        if specular is not None:
            specular.default_value = 0.0
    else:
        shader.inputs["Color"].default_value = COLOR
        shader.inputs["Roughness"].default_value = 1.0
    links.new(shader.outputs[0], output.inputs["Surface"])
    return material


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(-1.1, 0.0, 0.0))
left = bpy.context.object
left.data.materials.append(make_material("PrincipledColor", "ShaderNodeBsdfPrincipled"))

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1.1, 0.0, 0.0))
right = bpy.context.object
right.data.materials.append(make_material("DiffuseColor", "ShaderNodeBsdfDiffuse"))

bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 3.0))
light = bpy.context.object
light.data.energy = 800.0
light.data.shape = "RECTANGLE"
light.data.size = 4.0

bpy.ops.object.camera_add(location=(0.0, 0.0, 5.0))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = 4.4
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "NOORRAY_HYDRA"
scene.render.resolution_x = 64
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = OUTPUT
scene.hdnoorray.samples = 8
scene.hdnoorray.max_bounces = 2
scene.world.color = (0.0, 0.0, 0.0)

bpy.ops.render.render(write_still=True)
if not os.path.exists(OUTPUT):
    raise AssertionError("Principled/Diffuse render was not written")

image = bpy.data.images.load(OUTPUT, check_existing=False)
try:
    width, height = image.size
    pixels = image.pixels[:]

    def region_mean(x0, x1):
        total = [0.0, 0.0, 0.0]
        count = 0
        for y in range(height // 4, height - height // 4):
            for x in range(x0, x1):
                index = 4 * (y * width + x)
                for channel in range(3):
                    total[channel] += pixels[index + channel]
                count += 1
        return tuple(value / count for value in total)

    principled = region_mean(width // 8, width * 3 // 8)
    diffuse = region_mean(width * 5 // 8, width * 7 // 8)
    print(f"Principled mean: {principled}")
    print(f"Diffuse mean: {diffuse}")
    for channel in range(3):
        if principled[channel] <= 0.0 or diffuse[channel] <= 0.0:
            raise AssertionError(f"Missing color channel: {principled} vs {diffuse}")
    principled_ratio = principled[0] / max(principled[2], 1.0e-8)
    diffuse_ratio = diffuse[0] / max(diffuse[2], 1.0e-8)
    if abs(principled_ratio - diffuse_ratio) > 0.08:
        raise AssertionError(
            f"Principled color is biased relative to Diffuse: {principled} vs {diffuse}")
finally:
    bpy.data.images.remove(image)
