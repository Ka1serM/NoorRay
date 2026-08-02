import argparse
import math
import os
import sys

import bpy


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material():
    material = bpy.data.materials.new("LightUnitDiffuse")
    material.diffuse_color = (0.5, 0.5, 0.5, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_scene(light_type, area_shape):
    reset_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = "/tmp/noorray_blender_light.exr"
    scene.view_settings.look = "None"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.0, 0.0, 0.0)

    plane_mesh = bpy.data.meshes.new("LightUnitPlaneMesh")
    plane_mesh.from_pydata(
        [(-5.0, -5.0, 0.0), (5.0, -5.0, 0.0),
         (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0)],
        [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("LightUnitPlane", plane_mesh)
    bpy.context.collection.objects.link(plane)
    plane.data.materials.append(make_material())

    camera_data = bpy.data.cameras.new("LightUnitCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.0
    camera = bpy.data.objects.new("LightUnitCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 6.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera

    light_data = bpy.data.lights.new("LightUnitLight", light_type)
    light_data.energy = 100.0
    light_data.color = (1.0, 1.0, 1.0)
    light = bpy.data.objects.new("LightUnitLight", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, 0.0, 3.0)
    if light_type == "AREA":
        light_data.shape = area_shape
        light_data.size = 2.0
        if area_shape == "RECTANGLE":
            light_data.size_y = 2.0
    elif light_type == "SPOT":
        light_data.spot_size = math.radians(45.0)
        light_data.spot_blend = 0.0
    elif light_type == "SUN":
        light_data.angle = math.radians(0.526)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", choices=("POINT", "SPOT", "AREA", "SUN"),
                        default="POINT")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--engine", choices=("BLENDER_EEVEE", "CYCLES", "NOORRAY_HYDRA"),
                        default="BLENDER_EEVEE")
    parser.add_argument("--output", default="/tmp/noorray_blender_light.exr")
    parser.add_argument("--area-shape", choices=("DISK", "RECTANGLE"),
                        default="DISK")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    build_scene(args.light, args.area_shape)
    bpy.context.scene.render.engine = args.engine
    bpy.context.scene.render.filepath = os.path.abspath(args.output)
    if args.render:
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.usd_export(
        filepath=os.path.abspath(args.usd),
        selected_objects_only=False,
        export_materials=True,
        generate_materialx_network=True,
        export_lights=True,
        export_cameras=True,
        export_meshes=True,
        convert_scene_units="METERS")


if __name__ == "__main__":
    main()
