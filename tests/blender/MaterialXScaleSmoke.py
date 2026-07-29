"""Exercise NoorRay's complete Blender -> MaterialX -> OptiX scale path.

The default is deliberately small enough for a routine smoke test. Use
NOORRAY_MATERIALX_SCALE_COUNT=1000 for a stress run.
"""

import colorsys
import math
import os
import time

import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(index, count):
    material = bpy.data.materials.new(f"ScaleMaterial_{index}")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise AssertionError("Blender did not create a Principled BSDF node")

    color = colorsys.hsv_to_rgb(index / max(count, 1), 0.8, 0.8)
    rgba = (*color, 1.0)
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = (
        0.15 + 0.7 * index / max(count - 1, 1))
    shader.inputs["Emission Color"].default_value = rgba
    shader.inputs["Emission Strength"].default_value = 1.0
    return material


def make_grid(materials):
    side = math.ceil(math.sqrt(len(materials)))
    vertices = []
    faces = []
    for index in range(len(materials)):
        x = index % side
        y = index // side
        base = len(vertices)
        vertices.extend((
            (x, y, 0.0),
            (x + 0.94, y, 0.0),
            (x + 0.94, y + 0.94, 0.0),
            (x, y + 0.94, 0.0),
        ))
        faces.append((base, base + 1, base + 2, base + 3))

    mesh = bpy.data.meshes.new("MaterialXScaleGrid")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.clear()
    for material in materials:
        mesh.materials.append(material)
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index

    grid = bpy.data.objects.new("MaterialXScaleGrid", mesh)
    bpy.context.collection.objects.link(grid)
    return grid, side


def configure_scene(side):
    bpy.ops.object.camera_add(
        location=(side * 0.5, side * 0.5, 10.0),
        rotation=(0.0, 0.0, 0.0),
    )
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = side + 0.25

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "NOORRAY_HYDRA"
    resolution = min(max(side * 4, 32), 256)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.hdnoorray.samples = 1
    scene.hdnoorray.max_bounces = 1

    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    return scene


def render_seconds():
    start = time.perf_counter()
    bpy.ops.render.render()
    elapsed = time.perf_counter() - start
    result = bpy.data.images.get("Render Result")
    if result is None or not result.has_data:
        raise AssertionError("NoorRay did not produce a Render Result")
    return elapsed


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")
clear_scene()

count = int(os.environ.get("NOORRAY_MATERIALX_SCALE_COUNT", "32"))
if count < 1:
    raise ValueError("NOORRAY_MATERIALX_SCALE_COUNT must be positive")

materials = [make_material(index, count) for index in range(count)]
_grid, grid_side = make_grid(materials)
configure_scene(grid_side)

cold_seconds = render_seconds()

# Exercise one-material invalidation. This should not reconvert or recompile
# the other materials if Blender keeps the render engine alive between calls.
edited_shader = materials[0].node_tree.nodes.get("Principled BSDF")
edited_shader.inputs["Base Color"].default_value = (0.02, 0.9, 0.1, 1.0)
edited_shader.inputs["Emission Color"].default_value = (0.02, 0.9, 0.1, 1.0)
edited_seconds = render_seconds()

print(
    "NOORRAY_MATERIALX_SCALE_BENCHMARK "
    f"materials={count} cold_seconds={cold_seconds:.6f} "
    f"edited_seconds={edited_seconds:.6f}"
)
