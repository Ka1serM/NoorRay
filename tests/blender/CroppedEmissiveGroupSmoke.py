import os

import bpy


NOORRAY_OUTPUT = "/tmp/noorray_cropped_emissive_group.exr"
BLENDER_OUTPUT = "/tmp/blender_cropped_emissive_group.exr"


def region_mean(image):
    width, height = image.size
    pixels = image.pixels[:]
    values = [0.0, 0.0, 0.0]
    count = 0
    for y in range(height // 4, height - height // 4):
        for x in range(width // 4, width - width // 4):
            index = 4 * (y * width + x)
            for channel in range(3):
                values[channel] += pixels[index + channel]
            count += 1
    return tuple(value / count for value in values)


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.object
material = bpy.data.materials.new("CroppedEmissiveGroupProbe")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
output = nodes.get("Material Output")
emission = nodes.new("ShaderNodeEmission")
group = nodes.new("ShaderNodeGroup")
group.node_tree = bpy.data.node_groups["FP Cropped Emissive"]
inputs = {socket.identifier: socket for socket in group.inputs}
inputs["Input_1"].default_value = 0.48
inputs["Input_3"].default_value = 0.0
inputs["Input_5"].default_value = 0.65
inputs["Input_7"].default_value = 0.17
links.new(group.outputs["OutVec"], emission.inputs["Color"])
emission.inputs["Strength"].default_value = 1.0
links.new(emission.outputs["Emission"], output.inputs["Surface"])
plane.data.materials.append(material)

bpy.ops.object.camera_add(location=(0.0, 0.0, 3.0))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = 2.0
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.world.color = (0.0, 0.0, 0.0)
scene.hdnoorray.samples = 2
scene.hdnoorray.max_bounces = 1

scene.render.engine = "NOORRAY_HYDRA"
scene.render.filepath = NOORRAY_OUTPUT
bpy.ops.render.render(write_still=True)
if not os.path.exists(NOORRAY_OUTPUT):
    raise AssertionError("NoorRay cropped emissive probe was not written")
noorray_image = bpy.data.images.load(NOORRAY_OUTPUT, check_existing=False)
try:
    noorray = region_mean(noorray_image)
finally:
    bpy.data.images.remove(noorray_image)

scene.render.engine = "BLENDER_EEVEE"
scene.render.filepath = BLENDER_OUTPUT
bpy.ops.render.render(write_still=True)
if not os.path.exists(BLENDER_OUTPUT):
    raise AssertionError("Blender cropped emissive reference was not written")
blender_image = bpy.data.images.load(BLENDER_OUTPUT, check_existing=False)
try:
    blender = region_mean(blender_image)
finally:
    bpy.data.images.remove(blender_image)

print(f"Cropped emissive group: NoorRay={noorray}, Blender={blender}")
if any(abs(a - b) > 0.03 for a, b in zip(noorray, blender)):
    raise AssertionError(f"Cropped emissive group mismatch: {noorray} vs {blender}")
