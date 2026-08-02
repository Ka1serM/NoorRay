import math
import os

import bpy


OUTPUT = "/tmp/noorray_mix_soft_light.exr"
A = (0.8, 0.1, 0.03)
B = (0.02, 0.2, 0.7)
FAC = 0.35


def soft_light(a, b):
    low = a - (1.0 - 2.0 * b) * a * (1.0 - a)
    d = ((16.0 * a - 12.0) * a + 4.0) * a if a < 0.25 else math.sqrt(a)
    high = a + (2.0 * b - 1.0) * (d - a)
    return low if b < 0.5 else high


def expected():
    return tuple((1.0 - FAC) * a + FAC * soft_light(a, b)
                 for a, b in zip(A, B))


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1.1, 0.0, 0.0))
plane = bpy.context.object
material = bpy.data.materials.new("MixSoftLight")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
output = nodes.get("Material Output")
emission = nodes.new("ShaderNodeEmission")
mix = nodes.new("ShaderNodeMixRGB")
mix.blend_type = "SOFT_LIGHT"
mix.inputs["Fac"].default_value = FAC
mix.inputs["Color1"].default_value = (*A, 1.0)
mix.inputs["Color2"].default_value = (*B, 1.0)
links.new(mix.outputs["Color"], emission.inputs["Color"])
emission.inputs["Strength"].default_value = 1.0
links.new(emission.outputs["Emission"], output.inputs["Surface"])
plane.data.materials.append(material)

bpy.ops.object.camera_add(location=(0.0, 0.0, 3.0))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = 2.2
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "NOORRAY_HYDRA"
scene.render.resolution_x = 32
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = OUTPUT
scene.world.color = (0.0, 0.0, 0.0)

scene.hdnoorray.samples = 2
scene.hdnoorray.max_bounces = 1

target = expected()
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(-1.1, 0.0, 0.0))
reference_plane = bpy.context.object
reference_material = bpy.data.materials.new("MixSoftLightReference")
reference_material.use_nodes = True
reference_nodes = reference_material.node_tree.nodes
reference_links = reference_material.node_tree.links
reference_output = reference_nodes.get("Material Output")
reference_emission = reference_nodes.new("ShaderNodeEmission")
reference_emission.inputs["Color"].default_value = (*target, 1.0)
reference_emission.inputs["Strength"].default_value = 1.0
reference_links.new(reference_emission.outputs["Emission"], reference_output.inputs["Surface"])
reference_plane.data.materials.append(reference_material)

bpy.ops.render.render(write_still=True)
if not os.path.exists(OUTPUT):
    raise AssertionError("NoorRay Soft Light render was not written")

image = bpy.data.images.load(OUTPUT, check_existing=False)
try:
    width, height = image.size
    pixels = image.pixels[:]
    def region_mean(x0, x1):
        values = [0.0, 0.0, 0.0]
        count = 0
        for y in range(height // 4, height - height // 4):
            for x in range(x0, x1):
                index = 4 * (y * width + x)
                for channel in range(3):
                    values[channel] += pixels[index + channel]
                count += 1
        return tuple(value / count for value in values)

    actual = region_mean(width * 5 // 8, width * 7 // 8)
    reference = region_mean(width // 8, width * 3 // 8)
    print(f"Soft Light result: actual={actual}, reference={reference}")
    if any(abs(a - b) > 0.03 for a, b in zip(actual, reference)):
        raise AssertionError(f"Soft Light mismatch: {actual} vs {reference}")
finally:
    bpy.data.images.remove(image)
