"""Validate NoorRay's Blender-to-MaterialX exporter inside Blender 5.2.

Run after building/linking the extension:

  blender --background --factory-startup \
    --python tests/blender/MaterialXExportValidation.py

Set NOORRAY_MATERIALX_BENCHMARK_COUNT to increase the bulk-conversion sample.
The benchmark measures XML conversion only; OSL/OptiX compilation is exercised
by the render smoke tests.
"""

import os
import time

import bpy


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")

from bl_ext.user_default.hdnoorray.materialx_export import export_material

import MaterialX as mx


def new_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    return material


def principled(material):
    node = material.node_tree.nodes.get("Principled BSDF")
    if node is None:
        raise AssertionError("Blender did not create a Principled BSDF node")
    return node


def validate_export(material, image_path_resolver=None):
    exported = export_material(
        material, image_path_resolver=image_path_resolver)
    if not exported.document:
        raise AssertionError(f"{material.name}: exporter returned no document")

    document = mx.createDocument()
    mx.readFromXmlString(document, exported.document)
    libraries = mx.createDocument()
    loaded = mx.loadLibraries(
        mx.getDefaultDataLibraryFolders(),
        mx.getDefaultDataSearchPath(),
        libraries,
    )
    if not loaded:
        raise AssertionError("Blender's MaterialX libraries could not be loaded")
    document.setDataLibrary(libraries)
    valid, message = document.validate()
    if not valid:
        raise AssertionError(
            f"{material.name}: invalid MaterialX document:\n{message}\n"
            f"{exported.document}")
    return exported


def make_math_material():
    material = new_material("MaterialX_Math")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    value = nodes.new("ShaderNodeValue")
    value.outputs["Value"].default_value = 0.23
    operation = nodes.new("ShaderNodeMath")
    operation.operation = "ADD"
    operation.inputs[1].default_value = 0.17
    links.new(value.outputs["Value"], operation.inputs[0])
    links.new(operation.outputs["Value"], principled(material).inputs["Roughness"])
    return material


def make_color_material():
    material = new_material("MaterialX_Color")
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    foreground = nodes.new("ShaderNodeRGB")
    foreground.outputs["Color"].default_value = (0.8, 0.1, 0.03, 1.0)
    background = nodes.new("ShaderNodeRGB")
    background.outputs["Color"].default_value = (0.02, 0.2, 0.7, 1.0)
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = 0.35
    links.new(foreground.outputs["Color"], mix.inputs["Color1"])
    links.new(background.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], principled(material).inputs["Base Color"])
    return material


def make_vector_material():
    material = new_material("MaterialX_Vector")
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    vector_math = nodes.new("ShaderNodeVectorMath")
    vector_math.operation = "SCALE"
    vector_math.inputs["Scale"].default_value = 0.5
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    bump.inputs["Distance"].default_value = 0.1
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], vector_math.inputs[0])
    links.new(vector_math.outputs["Vector"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled(material).inputs["Normal"])
    return material


def make_image_normal_material():
    material = new_material("MaterialX_ImageNormal")
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    image = bpy.data.images.new("MaterialX_TestNormal", width=2, height=2)
    image.colorspace_settings.name = "Non-Color"
    image.pixels = [
        0.5, 0.5, 1.0, 1.0,
        0.6, 0.5, 1.0, 1.0,
        0.5, 0.6, 1.0, 1.0,
        0.4, 0.5, 1.0, 1.0,
    ]
    image.pack()

    texcoord = nodes.new("ShaderNodeTexCoord")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    normal = nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = 0.7
    links.new(texcoord.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], principled(material).inputs["Normal"])
    return material, image


for graph in (
    make_math_material(),
    make_color_material(),
    make_vector_material(),
):
    validate_export(graph)

image_material, test_image = make_image_normal_material()
image_document = validate_export(
    image_material,
    image_path_resolver=lambda _image, _node=None: "/tmp/noorray_test_normal.exr",
)
if test_image.as_pointer() not in image_document.image_pointers:
    raise AssertionError("Exporter did not report the image datablock dependency")


benchmark_count = int(
    os.environ.get("NOORRAY_MATERIALX_BENCHMARK_COUNT", "1000"))
benchmark_materials = []
for index in range(benchmark_count):
    material = new_material(f"MaterialX_Benchmark_{index}")
    shader = principled(material)
    hue = index / max(benchmark_count - 1, 1)
    shader.inputs["Base Color"].default_value = (
        0.1 + 0.8 * hue,
        0.7 - 0.5 * hue,
        0.2 + 0.3 * hue,
        1.0,
    )
    shader.inputs["Roughness"].default_value = 0.15 + 0.7 * hue
    benchmark_materials.append(material)

start = time.perf_counter()
documents = [export_material(material).document
             for material in benchmark_materials]
elapsed = time.perf_counter() - start
if any(not document for document in documents):
    raise AssertionError("Bulk conversion produced an empty document")

print(
    "NOORRAY_MATERIALX_EXPORT_BENCHMARK "
    f"materials={benchmark_count} seconds={elapsed:.6f} "
    f"materials_per_second={benchmark_count / max(elapsed, 1e-12):.1f}"
)
