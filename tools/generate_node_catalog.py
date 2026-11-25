import itertools
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import bpy
from bl_ui import node_add_menu_shader

ROOT = Path(__file__).resolve().parent.parent
NODES = ROOT / "src" / "NoorRay" / "Materials" / "Nodes"
OUTPUT = NODES / "NodeCatalogData.cpp"
HIDDEN = {"mapping.curve_count", "mapping.clip", "socket_idname", "active_index", "object", "ies", "sun_direction", "sample_attribute_items", "world_distance"}
POINTERS = {"image", "object", "ies"}
SKIPPED = {"ShaderNodeScript", "ShaderNodeGroup", "NodeGroupInput", "NodeGroupOutput"}
KINDS = ("Material", "Light", "World")
MAXIMUM_COMBINATIONS = 4096
SOCKET_TYPES = {
    "VALUE": "Float", "INT": "Int", "BOOLEAN": "Bool", "VECTOR": "Vector", "RGBA": "Color", "SHADER": "Shader", "STRING": "String",
    "ROTATION": "Rotation", "MATRIX": "Matrix", "MENU": "Menu", "CLOSURE": "Closure",
}
VALUE_TYPES = {"VALUE", "INT", "BOOLEAN", "VECTOR", "RGBA", "ROTATION"}
INTERFACE_TYPES = {
    "VALUE": "NodeSocketFloat", "INT": "NodeSocketInt", "VECTOR": "NodeSocketVector",
}
MENUS = (
    (node_add_menu_shader.NODE_MT_shader_node_input_base, "Input"),
    (node_add_menu_shader.NODE_MT_shader_node_input_constant_base, "Input"),
    (node_add_menu_shader.NODE_MT_shader_node_output_base, "Output"),
    (node_add_menu_shader.NODE_MT_shader_node_shader_base, "Shader"),
    (node_add_menu_shader.NODE_MT_shader_node_displacement_base, "Displacement"),
    (node_add_menu_shader.NODE_MT_shader_node_color_base, "Color"),
    (node_add_menu_shader.NODE_MT_shader_node_texture_base, "Texture"),
    (node_add_menu_shader.NODE_MT_shader_node_math_base, "Math"),
    (node_add_menu_shader.NODE_MT_shader_node_vector_base, "Vector"),
    (node_add_menu_shader.NODE_MT_shader_node_utilities_base, "Utilities"),
)


class MenuRecorder:
    bl_label = ""
    menu_path = ""

    def __init__(self):
        self.types = []
        self.layout = SimpleNamespace(separator=lambda: None)

    def node_operator(self, _layout, node_type, *, poll=None, **_options):
        if poll is None or poll is True:
            self.types.append(node_type)
        return SimpleNamespace(settings=SimpleNamespace(add=SimpleNamespace))

    def node_operator_with_outputs(self, _context, layout, node_type, _names, **options):
        return self.node_operator(layout, node_type, poll=options.get("poll"))

    def node_operator_with_searchable_enum(self, _context, layout, node_type, _property, **_options):
        self.node_operator(layout, node_type)

    def color_mix_node(self, _context, layout):
        self.node_operator(layout, "ShaderNodeMix")

    def draw_menu(self, *_arguments, **_options):
        pass

    draw_assets_for_catalog = repeat_zone = closure_zone = draw_menu


def plain_type(menu_type):
    tree = bpy.data.node_groups.new("menu", "ShaderNodeTree")
    try:
        result = tree.nodes.new(menu_type).bl_idname
    except RuntimeError:
        result = menu_type
    bpy.data.node_groups.remove(tree)
    return result


def read_menus():
    categories = {}
    kinds = {}
    for kind in KINDS:
        owner = SimpleNamespace(type={"Material": "MESH", "Light": "LIGHT", "World": "WORLD"}[kind])
        space = SimpleNamespace(type="NODE_EDITOR", tree_type="ShaderNodeTree", shader_type="WORLD" if kind == "World" else "OBJECT", id_from=owner)
        context = SimpleNamespace(space_data=space, engine="CYCLES", is_menu_search=False)
        for menu, category in MENUS:
            recorder = MenuRecorder()
            menu.draw(recorder, context)
            for node_type in map(plain_type, recorder.types):
                categories.setdefault(node_type, category)
                kinds.setdefault(node_type, set()).add(kind)
        for node_type in ("NodeFrame", "NodeReroute"):
            categories[node_type] = "Layout"
            kinds.setdefault(node_type, set()).add(kind)
    return categories, kinds


def compiled_types():
    table = re.findall(r'\{"(\w+)", "(\w+)"\}', (NODES / "NodeAccess.cpp").read_text())
    quoted = set()
    for source in sorted(NODES.glob("*.cpp")):
        if source.name not in ("NodeAccess.cpp", OUTPUT.name):
            quoted.update(re.findall(r'"(\w+)"', source.read_text()))
    quoted.update(("REROUTE", "FRAME"))
    return [idname for idname, kind in table if (kind in quoted or idname in quoted) and idname not in SKIPPED]


def base_properties():
    return {item.identifier for item in bpy.types.Node.bl_rna.properties}


def values_of(value):
    return [float(item) for item in value] if hasattr(value, "__len__") else [float(value)]


def describe_socket(node, socket, group):
    result = SimpleNamespace(identifier=socket.identifier, name=socket.name, kind=socket.type, enabled=socket.enabled, hidden=socket.hide, hide_value=socket.hide_value,
                             components=1, value=[0.0] * 4, text="", minimum=1.0, maximum=0.0, subtype="")
    value = getattr(socket, "default_value", None)
    if value is None:
        return result
    if socket.type in ("STRING", "MENU"):
        result.text = str(value)
        return result
    if socket.type not in VALUE_TYPES:
        return result
    values = values_of(value)
    result.components = len(values)
    result.value = (values + [0.0] * 4)[:4]
    if socket.type in INTERFACE_TYPES:
        item = group.interface.new_socket("socket", in_out="INPUT", socket_type=INTERFACE_TYPES[socket.type])
        item.from_socket(node, socket)
        result.minimum, result.maximum, result.subtype = float(item.min_value), float(item.max_value), item.subtype
        group.interface.remove(item)
    return result


def enum_items(node, identifier, description):
    items = []
    before = getattr(node, identifier)
    for item in description.enum_items_static:
        if not item.identifier:
            continue
        try:
            setattr(node, identifier, item.identifier)
        except TypeError:
            continue
        if getattr(node, identifier) == item.identifier:
            items.append((item.identifier, item.name))
    setattr(node, identifier, before)
    return items


def describe_property(node, description):
    name = description.identifier
    if name in ("node_tree", "texture_mapping", "color_mapping", "image_user"):
        return None
    if description.type == "COLLECTION":
        return plain_property(name, description.name, "Integer", integer=len(getattr(node, name)))
    result = SimpleNamespace(name=name, label=description.name, kind="Text", number=0.0, integer=0, text="", items=[], numbers=[], minimum=1.0, maximum=0.0)
    if description.type == "POINTER":
        return result if name in POINTERS else None
    value = getattr(node, name)
    if description.type == "STRING":
        result.text = value
    elif description.type == "ENUM":
        if description.is_enum_flag:
            return None
        result.text = value
        result.items = enum_items(node, name, description)
    elif description.is_array:
        result.kind = "Numbers"
        result.numbers = values_of(value)
    elif description.type == "BOOLEAN":
        result.kind, result.integer = "Boolean", int(value)
    else:
        result.kind = "Integer" if description.type == "INT" else "Number"
        result.minimum, result.maximum = float(description.soft_min), float(description.soft_max)
        if result.kind == "Integer":
            result.integer = int(value)
        else:
            result.number = float(value)
    return result


def plain_property(name, label, kind, **values):
    return SimpleNamespace(**({"name": name, "label": label, "kind": kind, "number": 0.0, "integer": 0, "text": "", "items": [], "numbers": [], "minimum": 1.0,
                               "maximum": 0.0} | values))


def static_items(owner, identifier):
    return [(item.identifier, item.name) for item in owner.bl_rna.properties[identifier].enum_items_static]


def ramp_properties(ramp):
    elements = []
    for element in ramp.elements:
        elements += [element.position, *element.color]
    return [plain_property(f"color_ramp.{name}", ramp.bl_rna.properties[name].name, "Text", text=getattr(ramp, name), items=static_items(ramp, name))
            for name in ("interpolation", "color_mode", "hue_interpolation")] + [plain_property("color_ramp.elements", "Elements", "Numbers", numbers=elements)]


def curve_properties(mapping):
    mapping.initialize()
    count = len(mapping.curves)
    clip = [float(mapping.use_clip), mapping.clip_min_x, mapping.clip_min_y, mapping.clip_max_x, mapping.clip_max_y]
    handles = [item.identifier for item in bpy.types.CurveMapPoint.bl_rna.properties["handle_type"].enum_items_static]
    result = [plain_property("mapping.curve_count", "Curves", "Integer", integer=count),
              plain_property("mapping.extend", mapping.bl_rna.properties["extend"].name, "Text", text=mapping.extend, items=static_items(mapping, "extend")),
              plain_property("mapping.clip", mapping.bl_rna.properties["use_clip"].name, "Numbers", numbers=clip)]
    names = {1: ("Value",), 3: ("X", "Y", "Z"), 4: ("R", "G", "B", "C")}[count]
    for index, curve in enumerate(mapping.curves):
        points = [value for point in curve.points for value in point.location]
        result.append(plain_property(f"mapping.curve{index}", names[index], "Numbers", numbers=list(points)))
        result.append(plain_property(f"mapping.handles{index}", f"{names[index]} Handles", "Numbers",
                                     numbers=[float(handles.index(point.handle_type)) for point in curve.points]))
    return result


def socket_state(node):
    return (tuple(socket.enabled for socket in node.inputs), tuple(socket.enabled for socket in node.outputs),
            tuple(socket.hide_value for socket in node.inputs), tuple(socket.name for socket in node.inputs), tuple(socket.name for socket in node.outputs))


def read_rule(node, properties):
    options = [(item.name, [value for value, _ in item.items] if item.items else [False, True]) for item in properties
               if (item.items and hasattr(node, item.name)) or item.kind == "Boolean" and item.name != "is_active_output"]
    defaults = {name: getattr(node, name) for name, _ in options}
    size = 1
    for _, values in options:
        size *= len(values)
    if not options:
        return [], []
    if size > MAXIMUM_COMBINATIONS:
        raise SystemExit(f"{node.bl_idname} has {size} combinations of settings")
    states = {}
    for combination in itertools.product(*(range(len(values)) for _, values in options)):
        for (name, values), index in zip(options, combination):
            if getattr(node, name) != values[index]:
                setattr(node, name, values[index])
        states[combination] = socket_state(node)
    for name, value in defaults.items():
        setattr(node, name, value)

    def matters(position):
        return any(states[key] != states[key[:position] + (0,) + key[position + 1:]] for key in states)

    used = [position for position in range(len(options)) if matters(position)]
    if not used:
        return [], []
    rows = []
    for combination in itertools.product(*(range(len(options[position][1])) for position in used)):
        key = [0] * len(options)
        for position, index in zip(used, combination):
            key[position] = index
        rows.append(states[tuple(key)])
    return [options[position][0] for position in used], rows


def read_muting(tree, node):
    source = tree.nodes.new("ShaderNodeRGB")
    shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    target = tree.nodes.new("ShaderNodeMixShader")
    sink = tree.nodes.new("ShaderNodeMath")
    node.mute = True
    masks = []
    for output in node.outputs:
        mask = 0
        default = -1
        if output.enabled:
            tree.links.new(output, target.inputs[1] if output.type == "SHADER" else sink.inputs[0])
            plain = [index for index, socket in enumerate(node.inputs) if any(item.from_socket == socket and item.to_socket == output for item in node.internal_links)]
            default = plain[0] if plain else -1
            for index, input_socket in enumerate(node.inputs):
                if not input_socket.enabled:
                    continue
                link = tree.links.new(shader.outputs[0] if input_socket.type == "SHADER" else source.outputs[0], input_socket)
                if any(item.from_socket == input_socket and item.to_socket == output for item in node.internal_links):
                    mask |= 1 << index
                tree.links.remove(link)
        masks.append((mask, default))
    node.mute = False
    for extra in (source, shader, target, sink):
        tree.nodes.remove(extra)
    return masks


def all_muting(tree, node, properties, settings):
    items = [item for item in properties if item.name in settings and item.items]
    masks = read_muting(tree, node)
    for combination in itertools.product(*(item.items for item in items)):
        for item, (value, _) in zip(items, combination):
            setattr(node, item.name, value)
        masks = [(first[0] | second[0], max(first[1], second[1])) for first, second in zip(masks, read_muting(tree, node))]
    for item in items:
        setattr(node, item.name, item.text)
    return masks


def read_type(tree, group, idname, categories, kinds, base):
    node = tree.nodes.new(idname)
    properties = []
    for description in node.bl_rna.properties:
        if description.identifier in base:
            continue
        if description.identifier == "color_ramp":
            properties += ramp_properties(node.color_ramp)
        elif description.identifier == "mapping":
            properties += curve_properties(node.mapping)
        elif (item := describe_property(node, description)) is not None:
            properties.append(item)
    if idname == "ShaderNodeAmbientOcclusion":
        world = bpy.data.worlds.new("distance")
        properties.append(plain_property("world_distance", "", "Number", number=float(world.light_settings.distance)))
    result = SimpleNamespace(idname=idname, name=node.bl_rna.name, node_name=node.name, category=categories.get(idname, ""), width=node.width,
                             kinds=[kind for kind in KINDS if kind in kinds.get(idname, ())], properties=properties,
                             inputs=[describe_socket(node, socket, group) for socket in node.inputs],
                             outputs=[describe_socket(node, socket, group) for socket in node.outputs])
    result.rule_settings, result.rule_rows = read_rule(node, properties)
    result.muting = all_muting(tree, node, properties, result.rule_settings)
    tree.nodes.remove(node)
    return result


def read_owner_defaults():
    material = bpy.data.materials.new("material")
    light = bpy.data.lights.new("light", "POINT")
    world = bpy.data.worlds.new("world")
    visibility = world.cycles_visibility
    boolean = lambda name, value: plain_property(name, "", "Boolean", integer=int(value))
    number = lambda name, value: plain_property(name, "", "Number", number=float(value))
    text = lambda name, value: plain_property(name, "", "Text", text=value)
    numbers = lambda name, value: plain_property(name, "", "Numbers", numbers=values_of(value))
    owners = [
        ("Material", material.node_tree, [
            boolean("use_nodes", True), plain_property("pass_index", "", "Integer", integer=material.pass_index),
            text("displacement_method", material.displacement_method), text("emission_sampling", material.cycles.emission_sampling),
            numbers("diffuse_color", material.diffuse_color), number("metallic", material.metallic), number("roughness", material.roughness)]),
        ("Light", light.node_tree, [
            boolean("use_nodes", True), number("energy", light.energy), number("exposure", light.exposure), numbers("color", light.color),
            boolean("use_temperature", light.use_temperature), boolean("normalize", light.normalize)]),
        ("World", world.node_tree, [
            text("name", "World"), boolean("use_nodes", True), boolean("film_transparent", False),
            *(boolean(f"cycles_visibility.{ray}", getattr(visibility, ray)) for ray in ("camera", "diffuse", "glossy", "transmission", "scatter")),
            text("sampling_method", world.cycles.sampling_method)]),
    ]
    result = []
    for kind, tree, properties in owners:
        nodes = [(node.bl_idname, node.name, tuple(node.location), node.width) for node in tree.nodes]
        names = [node.name for node in tree.nodes]
        links = [(names.index(link.from_node.name), list(link.from_node.outputs).index(link.from_socket), names.index(link.to_node.name),
                  list(link.to_node.inputs).index(link.to_socket)) for link in tree.links]
        values = []
        for index, node in enumerate(tree.nodes):
            fresh = tree.nodes.new(node.bl_idname)
            for socket_index, (socket, plain) in enumerate(zip(node.inputs, fresh.inputs)):
                if socket.type in VALUE_TYPES and values_of(socket.default_value) != values_of(plain.default_value):
                    values.append((index, socket_index, (values_of(socket.default_value) + [0.0] * 4)[:4]))
            tree.nodes.remove(fresh)
        result.append((kind, tree.name, nodes, links, values, properties))
    return result


def quote(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def number(value):
    text = f"{value:.9g}"
    return text + ("f" if "." in text or "e" in text else ".0f")


def boolean(value):
    return "true" if value else "false"


def braces(items):
    return "{" + ", ".join(items) + "}"


def mask(flags):
    return f"0x{sum(1 << index for index, flag in enumerate(flags) if flag):x}"


class Writer:
    def __init__(self):
        self.lines = []
        self.arrays = {}

    def array(self, kind, items):
        if not items:
            return "{}"
        body = ", ".join(items)
        key = (kind, body)
        if key not in self.arrays:
            self.arrays[key] = f"table{len(self.arrays)}"
            self.lines.append(f"constexpr {kind} {self.arrays[key]}[] = {{{body}}};")
        return self.arrays[key]

    def socket(self, item):
        values = braces([number(value) for value in item.value])
        kind = SOCKET_TYPES.get(item.kind, "Custom")
        return braces([quote(item.identifier), quote(item.name), f"NodeSocketType::{kind}", str(item.components), boolean(item.enabled), boolean(item.hidden),
                       boolean(item.hide_value), values, quote(item.text), number(item.minimum), number(item.maximum), quote(item.subtype)])

    def property(self, item):
        items = self.array("NodeEnumData", [braces([quote(identifier), quote(name)]) for identifier, name in item.items])
        numbers = self.array("float", [number(value) for value in item.numbers])
        return braces([quote(item.name), quote(item.label), f"NodePropertyKind::{item.kind}", number(item.number), str(item.integer), quote(item.text), numbers, items,
                       number(item.minimum), number(item.maximum), boolean(item.name not in HIDDEN and ".handles" not in item.name),
                       boolean(item.name not in POINTERS)])

    def rule(self, item):
        states = []
        rows = []
        default = (tuple(socket.name for socket in item.inputs), tuple(socket.name for socket in item.outputs))
        for inputs, outputs, hidden, input_names, output_names in item.rule_rows:
            changes = [braces([boolean(side == 1), str(index), quote(name)]) for side, (names, plain) in enumerate(zip((input_names, output_names), default))
                       for index, name in enumerate(names) if name != plain[index]]
            state = braces([mask(inputs), mask(outputs), mask(hidden), self.array("NodeSocketName", changes)])
            if state not in states:
                states.append(state)
            rows.append(str(states.index(state)))
        settings = self.array("std::string_view", [quote(name) for name in item.rule_settings])
        return settings, self.array("std::uint8_t", rows), self.array("NodeSocketState", states)

    def type(self, item):
        inputs = self.array("NodeSocketData", [self.socket(socket) for socket in item.inputs])
        outputs = self.array("NodeSocketData", [self.socket(socket) for socket in item.outputs])
        properties = self.array("NodePropertyData", [self.property(setting) for setting in item.properties])
        kinds = self.array("NodeTreeKind", [f"NodeTreeKind::{kind}" for kind in item.kinds])
        muting = self.array("NodeMuting", [braces([f"0x{inputs:x}", str(default)]) for inputs, default in item.muting])
        settings, rows, states = self.rule(item)
        return braces([quote(item.idname), quote(item.name), quote(item.category), number(item.width), inputs, outputs, properties, kinds, settings, rows, states,
                       muting])

    def owner(self, item):
        kind, name, nodes, links, values, properties = item
        node_table = self.array("NodeDefault", [braces([quote(idname), braces([number(value) for value in location]), number(width)])
                                                for idname, _, location, width in nodes])
        link_table = self.array("NodeLink", [braces([str(value) for value in link]) for link in links])
        value_table = self.array("NodeDefaultValue", [braces([str(node), str(socket), braces([number(part) for part in value])])
                                                      for node, socket, value in values])
        property_table = self.array("NodePropertyData", [self.property(setting) for setting in properties])
        return braces([f"NodeTreeKind::{kind}", quote(name), node_table, link_table, value_table, property_table])


def main():
    categories, kinds = read_menus()
    tree = bpy.data.materials.new("catalog").node_tree
    tree.nodes.clear()
    group = bpy.data.node_groups.new("ranges", "ShaderNodeTree")
    base = base_properties()
    types = [read_type(tree, group, idname, categories, kinds, base) for idname in compiled_types()]
    types.sort(key=lambda item: item.idname)
    owners = read_owner_defaults()
    writer = Writer()
    type_rows = ",\n".join(writer.type(item) for item in types)
    owner_rows = ",\n".join(writer.owner(item) for item in owners)
    source = "\n".join([
        "//! tools/generate_node_catalog.py made this file from Blender " + bpy.app.version_string + ".",
        '#include "NoorRay/Materials/Nodes/NodeCatalogData.h"', "", "namespace NoorRay::Materials {", "", "namespace {", "", *writer.lines, "",
        f"constexpr NodeTypeData types[] = {{\n{type_rows}}};", "", f"constexpr NodeTreeDefault defaults[] = {{\n{owner_rows}}};", "", "}", "",
        "std::span<const NodeTypeData> nodeTypeData() {", "    return types;", "}", "",
        "std::span<const NodeTreeDefault> nodeTreeDefaults() {", "    return defaults;", "}", "", "}", ""])
    OUTPUT.write_text(subprocess.run(["clang-format", f"--assume-filename={OUTPUT}"], input=source, capture_output=True, text=True, check=True).stdout)
    for item in types:
        if item.rule_settings:
            print(f"{item.idname}: {', '.join(item.rule_settings)}")
    print(f"wrote {len(types)} node types, {sum(1 for item in types if item.rule_settings)} with availability rules, to {OUTPUT}")
    for item in types:
        if item.name != item.node_name:
            print(f"{item.idname} makes nodes named {item.node_name}, not {item.name}")
    print("not in the add menu:", ", ".join(item.idname for item in types if not item.kinds))
    print("menu types without a compiler:", ", ".join(sorted(set(categories) - {item.idname for item in types})))


main()
