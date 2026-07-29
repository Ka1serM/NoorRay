import os
from pathlib import Path

import bpy


BEFORE_PATH = "/tmp/noorray_texture_reload_before.exr"
AFTER_PATH = "/tmp/noorray_texture_reload_after.exr"


def fill_image(image, color):
    image.pixels = list(color) * (image.size[0] * image.size[1])
    image.update()


def center_mean(path):
    if not os.path.exists(path):
        raise AssertionError(f"Render output was not written: {path}")

    rendered = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = rendered.size
        pixels = rendered.pixels[:]
        x_begin = width // 4
        x_end = width - x_begin
        y_begin = height // 4
        y_end = height - y_begin

        total = [0.0, 0.0, 0.0]
        count = 0
        for y in range(y_begin, y_end):
            for x in range(x_begin, x_end):
                offset = 4 * (y * width + x)
                total[0] += pixels[offset]
                total[1] += pixels[offset + 1]
                total[2] += pixels[offset + 2]
                count += 1
        return tuple(channel / count for channel in total)
    finally:
        bpy.data.images.remove(rendered)


def exported_image_state():
    candidates = [
        path
        for root in Path("/tmp").glob("blender_*/usd/image_cache")
        for path in root.rglob("*")
        if path.is_file()
    ]
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    exported = bpy.data.images.load(str(path), check_existing=False)
    try:
        return str(path), tuple(exported.pixels[:4])
    finally:
        bpy.data.images.remove(exported)


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object

material = bpy.data.materials.new("TextureReload")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
principled = nodes.get("Principled BSDF")
if principled is None:
    raise AssertionError("Blender did not create a Principled BSDF node")

image = bpy.data.images.new("ReloadPattern", width=2, height=2)
image.colorspace_settings.name = "sRGB"
fill_image(image, (0.9, 0.02, 0.01, 1.0))
image.pack()

texture = nodes.new("ShaderNodeTexImage")
texture.image = image
texture.interpolation = "Closest"
links.new(texture.outputs["Color"], principled.inputs["Base Color"])

emission_color = principled.inputs.get("Emission Color")
if emission_color is None:
    emission_color = principled.inputs.get("Emission")
if emission_color is None:
    raise AssertionError("Principled BSDF has no emission color input")
links.new(texture.outputs["Color"], emission_color)

emission_strength = principled.inputs.get("Emission Strength")
if emission_strength is None:
    raise AssertionError("Principled BSDF has no emission strength input")
emission_strength.default_value = 1.0
principled.inputs["Roughness"].default_value = 1.0
plane.data.materials.append(material)

bpy.ops.object.camera_add(location=(0.0, 0.0, 3.0))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = 4.2
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "NOORRAY_HYDRA"
scene.render.resolution_x = 32
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.hdnoorray.samples = 2
scene.hdnoorray.max_bounces = 1

scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
background.inputs["Strength"].default_value = 0.0

scene.render.filepath = BEFORE_PATH
bpy.ops.render.render(write_still=True)
print(f"First exported texture: {exported_image_state()}")

# Keep the same generated, packed image datablock. Blender exports generated
# images through a stable pointer-derived cache path, so this specifically
# exercises reloading changed pixels from the same asset path.
fill_image(image, (0.01, 0.02, 0.9, 1.0))
# Refresh the bytes in the existing packed file; this does not replace the
# image datablock or Blender's pointer-derived export path.
image.pack()

scene.render.filepath = AFTER_PATH
bpy.ops.render.render(write_still=True)
print(f"Second exported texture: {exported_image_state()}")

before = center_mean(BEFORE_PATH)
after = center_mean(AFTER_PATH)
print(f"Texture reload center means: before={before}, after={after}")

minimum_shift = 0.01
if before[0] <= before[2] + minimum_shift:
    raise AssertionError(f"First render is not materially red-dominant: {before}")
if after[2] <= after[0] + minimum_shift:
    raise AssertionError(f"Second render is not materially blue-dominant: {after}")
if before[0] <= after[0] + minimum_shift:
    raise AssertionError(f"Red channel did not materially decrease: {before} -> {after}")
if after[2] <= before[2] + minimum_shift:
    raise AssertionError(f"Blue channel did not materially increase: {before} -> {after}")
