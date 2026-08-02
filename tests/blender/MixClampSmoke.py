import os

import bpy


OUTPUT = "/tmp/noorray_mix_clamp.exr"
REFERENCE_OUTPUT = "/tmp/noorray_mix_clamp_reference.exr"


def multiply_mix(a, b, factor, clamp_factor=False, clamp_result=False):
    if clamp_factor:
        factor = max(0.0, min(1.0, factor))
    value = tuple((1.0 - factor) * x + factor * x * y for x, y in zip(a, b))
    if clamp_result:
        value = tuple(max(0.0, min(1.0, component)) for component in value)
    return value


def add_emission_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.get("Material Output")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def add_mix_plane(name, location, a, b, factor, clamp_factor, clamp_result):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    plane = bpy.context.object
    plane.name = name
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.get("Material Output")
    emission = nodes.new("ShaderNodeEmission")
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.clamp_factor = clamp_factor
    mix.clamp_result = clamp_result
    inputs = {socket.identifier: socket for socket in mix.inputs}
    outputs = {socket.identifier: socket for socket in mix.outputs}
    inputs["Factor_Float"].default_value = factor
    inputs["A_Color"].default_value = (*a, 1.0)
    inputs["B_Color"].default_value = (*b, 1.0)
    links.new(outputs["Result_Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(material)


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

factor_case = {
    "a": (0.2, 0.4, 0.8),
    "b": (0.8, 0.2, 0.1),
    "factor": 1.4,
    "clamp_factor": True,
    "clamp_result": False,
}
unclamped_factor_case = {
    "a": (0.8, 0.6, 0.4),
    "b": (1.2, 1.3, 1.5),
    "factor": 1.4,
    "clamp_factor": False,
    "clamp_result": False,
}
result_case = {
    "a": (1.4, 0.7, 0.2),
    "b": (1.1, 0.8, 0.6),
    "factor": 0.5,
    "clamp_factor": False,
    "clamp_result": True,
}

add_mix_plane("MixClampFactor", (-2.2, 0.0, 0.0), **factor_case)
add_mix_plane("MixUnclampedFactor", (0.0, 0.0, 0.0), **unclamped_factor_case)
add_mix_plane("MixClampResult", (2.2, 0.0, 0.0), **result_case)

bpy.ops.object.camera_add(location=(0.0, 0.0, 3.0))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = 6.6
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "NOORRAY_HYDRA"
scene.render.resolution_x = 96
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = OUTPUT
scene.world.color = (0.0, 0.0, 0.0)
scene.hdnoorray.samples = 2
scene.hdnoorray.max_bounces = 1

bpy.ops.render.render(write_still=True)
if not os.path.exists(OUTPUT):
    raise AssertionError("NoorRay Mix clamp render was not written")

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

    factor_actual = region_mean(8, 24)
    unclamped_actual = region_mean(40, 56)
    result_actual = region_mean(72, 88)
    actuals = (factor_actual, unclamped_actual, result_actual)
finally:
    bpy.data.images.remove(image)

for object_name, case in (("MixClampFactor", factor_case),
                          ("MixUnclampedFactor", unclamped_factor_case),
                          ("MixClampResult", result_case)):
    color = multiply_mix(**case)
    bpy.data.objects[object_name].data.materials[0] = add_emission_material(
        object_name + "Reference", color)

scene.render.filepath = REFERENCE_OUTPUT
bpy.ops.render.render(write_still=True)
if not os.path.exists(REFERENCE_OUTPUT):
    raise AssertionError("NoorRay Mix clamp reference render was not written")

reference_image = bpy.data.images.load(REFERENCE_OUTPUT, check_existing=False)
try:
    width, height = reference_image.size
    pixels = reference_image.pixels[:]

    def reference_region_mean(x0, x1):
        values = [0.0, 0.0, 0.0]
        count = 0
        for y in range(height // 4, height - height // 4):
            for x in range(x0, x1):
                index = 4 * (y * width + x)
                for channel in range(3):
                    values[channel] += pixels[index + channel]
                count += 1
        return tuple(value / count for value in values)

    references = (reference_region_mean(8, 24),
                  reference_region_mean(40, 56),
                  reference_region_mean(72, 88))
finally:
    bpy.data.images.remove(reference_image)

print(f"Mix Clamp Factor: actual={actuals[0]}, reference={references[0]}")
print(f"Mix Unclamped Factor: actual={actuals[1]}, reference={references[1]}")
print(f"Mix Clamp Result: actual={actuals[2]}, reference={references[2]}")
for actual, reference in zip(actuals, references):
    if any(abs(a - b) > 0.03 for a, b in zip(actual, reference)):
        raise AssertionError(f"Mix clamp mismatch: {actual} vs {reference}")
