"""Validate NoorRay's Blender-to-MaterialX exporter inside Blender 5.2.

Run after building/linking the extension:

  blender --background --factory-startup \
    --python tests/blender/MaterialXExportValidation.py

Set NOORRAY_MATERIALX_BENCHMARK_COUNT to increase the bulk-conversion sample.
The benchmark measures XML conversion; SVM compilation is exercised by the
render smoke tests.
"""

import os
from pathlib import Path
import importlib.util
import time

import bpy


bpy.ops.preferences.addon_enable(module="bl_ext.user_default.hdnoorray")

_EXPORTER_PATH = Path(__file__).resolve().parents[2] / \
    "src/hdnoorray/blender_addon/hdnoorray/materialx_export.py"
_EXPORTER_SPEC = importlib.util.spec_from_file_location(
    "noorray_source_materialx_export", _EXPORTER_PATH)
_EXPORTER_MODULE = importlib.util.module_from_spec(_EXPORTER_SPEC)
assert _EXPORTER_SPEC.loader is not None
_EXPORTER_SPEC.loader.exec_module(_EXPORTER_MODULE)
export_material = _EXPORTER_MODULE.export_material

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
    if exported.unsupported_nodes:
        raise AssertionError(
            f"{material.name}: exporter reached unsupported Blender nodes: "
            + ", ".join(sorted(exported.unsupported_nodes)))

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


def make_color_to_bw_material():
    material = new_material("MaterialX_ColorToBW")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    color = nodes.new("ShaderNodeRGB")
    color.outputs["Color"].default_value = (0.8, 0.2, 0.05, 1.0)
    to_bw = nodes.new("ShaderNodeRGBToBW")
    links.new(color.outputs["Color"], to_bw.inputs["Color"])
    links.new(to_bw.outputs["Val"], principled(material).inputs["Roughness"])
    return material


def make_attribute_bright_contrast_material():
    material = new_material("MaterialX_AttributeBrightContrast")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = principled(material)

    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "color"
    bright_contrast = nodes.new("ShaderNodeBrightContrast")
    bright_contrast.inputs["Bright"].default_value = 0.1
    bright_contrast.inputs["Contrast"].default_value = 0.25
    bump = nodes.new("ShaderNodeBump")
    links.new(attribute.outputs["Color"], bright_contrast.inputs["Color"])
    links.new(bright_contrast.outputs["Color"], shader.inputs["Base Color"])
    links.new(attribute.outputs["Vector"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(attribute.outputs["Fac"], shader.inputs["Roughness"])
    return material


BLENDER_MIX_MODES = (
    "MIX", "DARKEN", "MULTIPLY", "BURN", "LIGHTEN", "SCREEN",
    "DODGE", "ADD", "OVERLAY", "SOFT_LIGHT", "LINEAR_LIGHT",
    "DIFFERENCE", "EXCLUSION", "SUBTRACT", "DIVIDE", "HUE",
    "SATURATION", "COLOR", "VALUE",
)


def make_mix_mode_material(mode, node_type, data_type=None):
    suffix = data_type or "RGB"
    material = new_material(f"MaterialX_Mix_{node_type}_{suffix}_{mode}")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    foreground = nodes.new("ShaderNodeRGB")
    foreground.outputs["Color"].default_value = (0.8, 0.1, 0.03, 1.0)
    background = nodes.new("ShaderNodeRGB")
    background.outputs["Color"].default_value = (0.02, 0.2, 0.7, 1.0)
    mix = nodes.new(node_type)
    mix.blend_type = mode
    if node_type == "ShaderNodeMixRGB":
        mix.inputs["Fac"].default_value = 0.35
        links.new(foreground.outputs["Color"], mix.inputs["Color1"])
        links.new(background.outputs["Color"], mix.inputs["Color2"])
        result = mix.outputs["Color"]
    else:
        mix.data_type = data_type or "RGBA"
        sockets = {socket.identifier: socket for socket in mix.inputs}
        outputs = {socket.identifier: socket for socket in mix.outputs}
        factor_identifier = "Factor_Float"
        if data_type == "VECTOR":
            factor_identifier = "Factor_Vector"
            mix.factor_mode = "NON_UNIFORM"
            sockets[factor_identifier].default_value = (0.35, 0.5, 0.65)
            a_identifier, b_identifier, result_identifier = (
                "A_Vector", "B_Vector", "Result_Vector")
        elif data_type == "FLOAT":
            sockets[factor_identifier].default_value = 0.35
            a_identifier, b_identifier, result_identifier = (
                "A_Float", "B_Float", "Result_Float")
        else:
            sockets[factor_identifier].default_value = 0.35
            a_identifier, b_identifier, result_identifier = (
                "A_Color", "B_Color", "Result_Color")
        links.new(foreground.outputs["Color"], sockets[a_identifier])
        links.new(background.outputs["Color"], sockets[b_identifier])
        result = outputs[result_identifier]
    destination = "Roughness" if data_type == "FLOAT" else "Base Color"
    links.new(result, principled(material).inputs[destination])
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


def make_scalar_vector_multiply_material():
    material = new_material("MaterialX_ScalarVectorMultiply")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    color = nodes.new("ShaderNodeRGB")
    color.outputs["Color"].default_value = (0.2, 0.4, 0.8, 1.0)
    separate = nodes.new("ShaderNodeSeparateColor")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    links.new(color.outputs["Color"], separate.inputs["Color"])
    links.new(color.outputs["Color"], scale.inputs[0])
    links.new(separate.outputs["Green"], scale.inputs["Scale"])
    links.new(scale.outputs["Vector"], principled(material).inputs["Base Color"])
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


def make_context_material():
    material = new_material("MaterialX_StandardContextNodes")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = principled(material)

    object_info = nodes.new("ShaderNodeObjectInfo")
    camera_data = nodes.new("ShaderNodeCameraData")
    texcoord = nodes.new("ShaderNodeTexCoord")
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "VECTOR"
    transform.convert_from = "WORLD"
    transform.convert_to = "OBJECT"
    white_noise = nodes.new("ShaderNodeTexWhiteNoise")
    white_noise.noise_dimensions = "4D"
    white_noise.inputs["W"].default_value = 0.25
    geometry = nodes.new("ShaderNodeNewGeometry")
    light_path = nodes.new("ShaderNodeLightPath")
    vector_add = nodes.new("ShaderNodeVectorMath")
    vector_add.operation = "ADD"
    roughness_add = nodes.new("ShaderNodeMath")
    roughness_add.operation = "ADD"
    metallic_multiply = nodes.new("ShaderNodeMath")
    metallic_multiply.operation = "MULTIPLY"

    links.new(camera_data.outputs["View Vector"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], shader.inputs["Normal"])
    links.new(texcoord.outputs["Generated"], white_noise.inputs["Vector"])
    links.new(object_info.outputs["Color"], shader.inputs["Base Color"])
    links.new(texcoord.outputs["Generated"], vector_add.inputs[0])
    links.new(object_info.outputs["Location"], vector_add.inputs[1])
    links.new(vector_add.outputs["Vector"], white_noise.inputs["Vector"])
    links.new(white_noise.outputs["Color"], roughness_add.inputs[0])
    links.new(light_path.outputs["Ray Length"], roughness_add.inputs[1])
    links.new(roughness_add.outputs["Value"], shader.inputs["Roughness"])
    links.new(geometry.outputs["Random Per Island"], metallic_multiply.inputs[0])
    links.new(camera_data.outputs["View Distance"], metallic_multiply.inputs[1])
    links.new(metallic_multiply.outputs["Value"], shader.inputs["Metallic"])
    return material


def make_metallic_material():
    material = new_material("MaterialX_Metallic")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.new("ShaderNodeBsdfMetallic")
    shader.inputs["Roughness"].default_value = 0.25
    output = next(
        node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_physical_metallic_material():
    material = new_material("MaterialX_PhysicalMetallic")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.new("ShaderNodeBsdfMetallic")
    shader.fresnel_type = "PHYSICAL_CONDUCTOR"
    output = next(
        node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


for graph in (
    make_math_material(),
    make_color_material(),
    make_color_to_bw_material(),
    make_attribute_bright_contrast_material(),
    make_vector_material(),
    make_scalar_vector_multiply_material(),
):
    validate_export(graph)

scalar_vector_document = validate_export(make_scalar_vector_multiply_material())
if scalar_vector_document.document.count("<combine3") < 1:
    raise AssertionError(
        "Scalar/vector multiplication was not lowered to an explicit component broadcast")

attribute_document = validate_export(make_attribute_bright_contrast_material())
for required in ("<geompropvalue", "<contrast", "<max"):
    if required not in attribute_document.document:
        raise AssertionError(
            f"Attribute/BrightContrast did not lower to standard MaterialX: {required}")

for mix_node_type in ("ShaderNodeMixRGB", "ShaderNodeMix"):
    for mode in BLENDER_MIX_MODES:
        data_types = (None,) if mix_node_type == "ShaderNodeMixRGB" else (
            "FLOAT", "VECTOR", "RGBA")
        for data_type in data_types:
            mix_document = validate_export(
                make_mix_mode_material(mode, mix_node_type, data_type))
        if mode in {"HUE", "SATURATION", "COLOR", "VALUE"}:
            if "<rgbtohsv" not in mix_document.document or "<hsvtorgb" not in mix_document.document:
                raise AssertionError(
                    f"{mix_node_type} {mode} did not expand through MaterialX HSV nodes")
        if mode == "SOFT_LIGHT" and "<ifgreater" not in mix_document.document:
            raise AssertionError(
                f"{mix_node_type} SOFT_LIGHT did not expand through MaterialX conditionals")
        if "<ifless" in mix_document.document:
            raise AssertionError(
                f"{mix_node_type} {mode} emitted non-standard MaterialX ifless")

image_material, test_image = make_image_normal_material()
image_document = validate_export(
    image_material,
    image_path_resolver=lambda _image, _node=None: "/tmp/noorray_test_normal.exr",
)
if test_image.as_pointer() not in image_document.image_pointers:
    raise AssertionError("Exporter did not report the image datablock dependency")

context_document = validate_export(make_context_material())
for forbidden in ("<objectinfo", "<cameradata", "<white_noise",
                  'geomprop="object_', 'geomprop="camera_'):
    if forbidden in context_document.document:
        raise AssertionError(
            f"Exporter emitted a non-standard MaterialX node/property: {forbidden}")
for required in ("<position", "<geomcolor", "<viewdirection", "<transformvector",
                 "<magnitude", "<noise3d", "<combine4", "<randomfloat"):
    if required not in context_document.document:
        raise AssertionError(
            f"Exporter did not emit the standard MaterialX node: {required}")

metallic_document = validate_export(make_metallic_material())
if "<generalized_schlick_bsdf" not in metallic_document.document:
    raise AssertionError(
        "Metallic BSDF did not export to generalized_schlick_bsdf")

physical_metallic_document = validate_export(make_physical_metallic_material())
if "<conductor_bsdf" not in physical_metallic_document.document:
    raise AssertionError("Physical Metallic BSDF did not export to conductor_bsdf")


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
