import bpy
import math

from mathutils import Vector


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
sphere = bpy.context.object
bpy.ops.object.shade_smooth()

material = bpy.data.materials.new("PrincipledNormal")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
principled = nodes.get("Principled BSDF")
principled.inputs["Base Color"].default_value = (0.8, 0.08, 0.02, 1.0)
principled.inputs["Roughness"].default_value = 0.85

image = bpy.data.images.new("NormalPattern", width=2, height=2)
image.colorspace_settings.name = "Non-Color"
image.pixels = [
    0.80, 0.50, 0.90, 1.0,
    0.20, 0.50, 0.90, 1.0,
    0.50, 0.80, 0.90, 1.0,
    0.50, 0.20, 0.90, 1.0,
]
image.pack()
texture = nodes.new("ShaderNodeTexImage")
texture.image = image
normal_map = nodes.new("ShaderNodeNormalMap")
normal_map.inputs["Strength"].default_value = 0.65
links.new(texture.outputs["Color"], normal_map.inputs["Color"])
links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
sphere.data.materials.append(material)

bpy.ops.object.light_add(type="AREA", location=(2.5, -3.0, 3.5))
light = bpy.context.object
light.data.energy = 900.0
light.data.shape = "DISK"
light.data.size = 2.0

bpy.ops.object.camera_add(location=(0.0, -4.0, 0.4))
camera = bpy.context.object
direction = Vector((0.0, 0.0, 0.0)) - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "NOORRAY_HYDRA"
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.filepath = "/tmp/noorray_principled_normal.exr"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.hdnoorray.samples = 8
scene.hdnoorray.max_bounces = 4
scene.world.color = (0.03, 0.03, 0.03)

bpy.ops.render.render(write_still=True)
