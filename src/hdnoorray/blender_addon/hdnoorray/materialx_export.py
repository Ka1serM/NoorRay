# SPDX-License-Identifier: GPL-3.0-or-later
"""Dependency-free Blender shader graph to MaterialX translator."""

from dataclasses import dataclass
import math
from typing import Callable, Optional
from xml.sax.saxutils import escape as _xml_escape


_FLOAT = "float"
_INTEGER = "integer"
_BOOLEAN = "boolean"
_COLOR3 = "color3"
_COLOR4 = "color4"
_VECTOR2 = "vector2"
_VECTOR3 = "vector3"
_VECTOR4 = "vector4"
_SURFACE = "surfaceshader"
_BSDF = "BSDF"
_EDF = "EDF"
# Blender Shader sockets may carry BSDF, EDF, or surfaceshader values.
_SHADER = "shader"
_SHADER_FAMILY = {_SHADER, _BSDF, _EDF, _SURFACE}

@dataclass(frozen=True, slots=True)
class ExportedMaterial:
    document: str
    image_pointers: frozenset[int]
    # Reached Blender node types without a semantic exporter handler.
    unsupported_nodes: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Value:
    type: str
    value: Optional[str] = None
    node: Optional[str] = None
    # Set when `node` has more than one declared output (e.g. artistic_ior's
    # "ior"/"extinction") and this value refers to one specific one, rather
    # than a single-output node's implicit sole output.
    output: Optional[str] = None


def _float_text(value) -> str:
    value = float(value)
    if not math.isfinite(value):
        value = 0.0
    if value == 0.0:
        return "0"
    return format(value, ".9g")


def _literal(type_name: str, value) -> _Value:
    if type_name == _BOOLEAN:
        text = "true" if bool(value) else "false"
    elif type_name == _INTEGER:
        text = str(int(value))
    elif type_name in {_FLOAT}:
        text = _float_text(value)
    elif type_name in {_COLOR3, _VECTOR3}:
        text = ",".join(_float_text(value[i]) for i in range(3))
    elif type_name in {_COLOR4, _VECTOR4}:
        text = ",".join(_float_text(value[i]) for i in range(4))
    elif type_name == _VECTOR2:
        text = ",".join(_float_text(value[i]) for i in range(2))
    else:
        text = str(value)
    return _Value(type_name, value=text)


def _default(type_name: str) -> _Value:
    if type_name == _BOOLEAN:
        return _literal(type_name, False)
    if type_name == _INTEGER:
        return _literal(type_name, 0)
    if type_name == _FLOAT:
        return _literal(type_name, 0.0)
    if type_name == _VECTOR2:
        return _literal(type_name, (0.0, 0.0))
    if type_name in {_COLOR3, _VECTOR3}:
        return _literal(type_name, (0.0, 0.0, 0.0))
    if type_name in {_COLOR4, _VECTOR4}:
        return _literal(type_name, (0.0, 0.0, 0.0, 0.0))
    return _Value(type_name, value="")


def _attr(value) -> str:
    return _xml_escape(str(value), {'"': "&quot;"})


class _Document:
    __slots__ = ("_nodes", "_next_name", "_conversions", "transport_nonce")

    def __init__(self, transport_nonce: int = 0):
        self._nodes: list[str] = []
        self._next_name = 0
        self._conversions: dict[tuple[_Value, str], _Value] = {}
        self.transport_nonce = int(transport_nonce)

    def node(
        self,
        category: str,
        type_name: str,
        inputs=(),
        *,
        attributes=(),
    ) -> _Value:
        name = f"n{self._next_name}"
        self._next_name += 1
        pieces = [
            "<",
            category,
            ' name="',
            name,
            '" type="',
            type_name,
            '"',
        ]
        for key, value in attributes:
            if value is not None and value != "":
                pieces.extend((' ', key, '="', _attr(value), '"'))
        inputs = tuple(inputs)
        if not inputs:
            pieces.append("/>")
        else:
            pieces.append(">")
            for input_spec in inputs:
                input_name, input_type, input_value = input_spec[:3]
                input_attributes = input_spec[3] if len(input_spec) > 3 else ()
                value = self.convert(input_value, input_type)
                pieces.extend(
                    (
                        '<input name="',
                        input_name,
                        '" type="',
                        input_type,
                        '"',
                    )
                )
                pieces.append(' ')
                if value.node is not None:
                    pieces.extend(('nodename="', value.node, '"'))
                    if value.output:
                        pieces.extend((' output="', value.output, '"'))
                else:
                    pieces.extend(('value="', _attr(value.value or ""), '"'))
                for key, attribute_value in input_attributes:
                    if attribute_value is not None and attribute_value != "":
                        pieces.extend((' ', key, '="', _attr(attribute_value), '"'))
                pieces.append("/>")
            pieces.extend(("</", category, ">"))
        self._nodes.append("".join(pieces))
        return _Value(type_name, node=name)

    def convert(self, value: _Value, target: str) -> _Value:
        if value.type == target:
            return value
        if value.type in _SHADER_FAMILY or target in _SHADER_FAMILY:
            # Closure-family values need context-specific handling; do not emit
            # generic scalar conversion nodes for them.
            # Closure-family values pass through until a concrete consumer
            # requests a compatible MaterialX type.
            return value
        key = (value, target)
        cached = self._conversions.get(key)
        if cached is not None:
            return cached

        source = value.type
        if source == _COLOR3 and target == _FLOAT:
            # Luminance preserves its input width; extract a channel for a
            # scalar result.
            luma = self.node("luminance", _COLOR3, (("in", _COLOR3, value),))
            result = self.extract(luma, 0)
        elif source == _COLOR4 and target == _FLOAT:
            luma = self.node("luminance", _COLOR4, (("in", _COLOR4, value),))
            result = self.extract(luma, 0)
        elif source in {_VECTOR2, _VECTOR3, _VECTOR4} and target == _FLOAT:
            result = self.node("magnitude", _FLOAT, (("in", source, value),))
        elif source == _BOOLEAN and target == _FLOAT:
            result = self.node("convert", _FLOAT, (("in", _BOOLEAN, value),))
        elif source == _INTEGER and target == _FLOAT:
            result = self.node("convert", _FLOAT, (("in", _INTEGER, value),))
        elif source == _FLOAT and target == _BOOLEAN:
            result = self.node(
                "ifgreater",
                _BOOLEAN,
                (
                    ("value1", _FLOAT, value),
                    ("value2", _FLOAT, _literal(_FLOAT, 0.0)),
                    ("in1", _BOOLEAN, _literal(_BOOLEAN, True)),
                    ("in2", _BOOLEAN, _literal(_BOOLEAN, False)),
                ),
            )
        elif source == _FLOAT and target == _INTEGER and value.node is None:
            result = _literal(_INTEGER, int(float(value.value or 0)))
        else:
            # MaterialX defines the remaining scalar/color/vector conversions.
            result = self.node("convert", target, (("in", source, value),))
        self._conversions[key] = result
        return result

    def named_output(self, value: _Value, output_name: str, type_name: str) -> _Value:
        """A reference to one specific output of a multi-output node (e.g.
        artistic_ior's "ior"/"extinction"), instantiated with type="multioutput"."""
        return _Value(type_name, node=value.node, output=output_name)

    def extract(self, value: _Value, index: int) -> _Value:
        if value.node is None:
            fields = (value.value or "").split(",")
            field = fields[index] if index < len(fields) else "0"
            return _Value(_FLOAT, value=field)
        return self.node(
            "extract",
            _FLOAT,
            (
                ("in", value.type, value),
                ("index", _INTEGER, _literal(_INTEGER, index)),
            ),
        )

    def serialize(self, surface: _Value) -> str:
        material = self.node(
            "surfacematerial",
            "material",
            (("surfaceshader", _SURFACE, surface),),
        )
        del material
        nonce = (
            f"<!--noorray-image-revision:{self.transport_nonce}-->"
            if self.transport_nonce
            else ""
        )
        return (
            '<?xml version="1.0"?><materialx version="1.39">'
            + nonce
            + "".join(self._nodes)
            + "</materialx>"
        )


class _Converter:
    __slots__ = (
        "doc",
        "material",
        "resolve_image_path",
        "cache",
        "visiting",
        "image_pointers",
        "unsupported_nodes",
        "group_stack",
    )

    def __init__(self, material, resolve_image_path, transport_nonce: int):
        self.doc = _Document(transport_nonce)
        self.material = material
        self.resolve_image_path = resolve_image_path
        self.cache: dict[tuple, _Value] = {}
        self.visiting: set[tuple] = set()
        self.image_pointers: set[int] = set()
        # Track Blender nodes without an _export_ handler.
        self.unsupported_nodes: set[str] = set()
        # Isolate nested group cache contexts and resolve group inputs through
        # the caller's context.
        self.group_stack: list[tuple[int, Callable]] = []

    @staticmethod
    def _socket(collection, name_or_index):
        if isinstance(name_or_index, int):
            try:
                return collection[name_or_index]
            except (IndexError, TypeError):
                return None
        socket = collection.get(name_or_index)
        if socket is not None:
            return socket
        for item in collection:
            if item.identifier == name_or_index or item.name == name_or_index:
                return item
        return None

    def _input_socket(self, node, name_or_index):
        return self._socket(node.inputs, name_or_index)

    def _input(self, node, name_or_index, type_name: str) -> _Value:
        socket = self._input_socket(node, name_or_index)
        if socket is None:
            return _default(type_name)
        if socket.is_linked and socket.links:
            link = socket.links[0]
            return self.output(link.from_node, link.from_socket, type_name)
        return self._socket_default(socket, type_name)

    def _linked_input(self, node, name_or_index, type_name: str) -> Optional[_Value]:
        socket = self._input_socket(node, name_or_index)
        if socket is None or not socket.is_linked or not socket.links:
            return None
        link = socket.links[0]
        return self.output(link.from_node, link.from_socket, type_name)

    @staticmethod
    def _socket_default(socket, type_name: str) -> _Value:
        try:
            value = socket.default_value
        except AttributeError:
            return _default(type_name)
        if type_name == _BOOLEAN:
            return _literal(type_name, bool(value))
        if type_name == _INTEGER:
            return _literal(type_name, int(value))
        if type_name == _FLOAT:
            if isinstance(value, (int, float, bool)):
                return _literal(type_name, value)
            try:
                return _literal(type_name, sum(value[:3]) / 3.0)
            except (TypeError, IndexError):
                return _default(type_name)
        if type_name == _VECTOR2:
            return _literal(type_name, value[:2])
        if type_name in {_COLOR3, _VECTOR3}:
            return _literal(type_name, value[:3])
        if type_name in {_COLOR4, _VECTOR4}:
            try:
                return _literal(type_name, value[:4])
            except (TypeError, IndexError):
                return _literal(type_name, (value, value, value, 1.0))
        return _default(type_name)

    def _group_context(self) -> tuple:
        return tuple(pointer for pointer, _ in self.group_stack)

    def output(self, node, socket, expected: str) -> _Value:
        key = (self._group_context(), node.as_pointer(), socket.identifier, expected)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if key in self.visiting:
            return _default(expected)
        self.visiting.add(key)
        try:
            value = self._node_output(node, socket)
            value = self.doc.convert(value, expected)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            value = _default(expected)
        self.visiting.remove(key)
        self.cache[key] = value
        return value

    def _node_output(self, node, socket) -> _Value:
        node_type = node.bl_idname
        method = getattr(self, f"_export_{node_type}", None)
        if method is None:
            # Preserve the default fallback, but expose the node type so
            # callers can reject the incomplete export.
            self.unsupported_nodes.add(node_type)
            return self._socket_default(socket, self._socket_type(socket))
        return method(node, socket)

    @staticmethod
    def _socket_type(socket) -> str:
        return {
            "VALUE": _FLOAT,
            "INT": _INTEGER,
            "BOOLEAN": _BOOLEAN,
            "RGBA": _COLOR4,
            "VECTOR": _VECTOR3,
            "SHADER": _SHADER,
        }.get(socket.type, _FLOAT)

    def _op(self, category: str, type_name: str, *values: _Value) -> _Value:
        names = ("in",) if len(values) == 1 else tuple(
            f"in{i + 1}" for i in range(len(values))
        )
        return self.doc.node(
            category,
            type_name,
            tuple((name, type_name, value) for name, value in zip(names, values)),
        )

    def _binary(
        self, category: str, a: _Value, b: _Value, type_name: Optional[str] = None
    ) -> _Value:
        type_name = type_name or a.type
        # MaterialX has scalar-factor arithmetic nodedefs, but make scalar
        # broadcasting explicit when a Blender vector/color operation receives
        # a float (for example Vector Math SCALE driven by Separate Color).
        # This keeps the graph component-wise even for consumers that do not
        # select the *FA nodedef variant during node resolution.
        a = self._broadcast_scalar(a, type_name)
        b = self._broadcast_scalar(b, type_name)
        return self.doc.node(
            category,
            type_name,
            (("in1", type_name, a), ("in2", type_name, b)),
        )

    def _broadcast_scalar(self, value: _Value, target: str) -> _Value:
        if value.type != _FLOAT:
            return value
        if target == _VECTOR2:
            return self.doc.node(
                "combine2",
                target,
                (("in1", _FLOAT, value), ("in2", _FLOAT, value)),
            )
        if target in {_COLOR3, _VECTOR3}:
            return self.doc.node(
                "combine3",
                target,
                (("in1", _FLOAT, value), ("in2", _FLOAT, value),
                 ("in3", _FLOAT, value)),
            )
        if target in {_COLOR4, _VECTOR4}:
            return self.doc.node(
                "combine4",
                target,
                (("in1", _FLOAT, value), ("in2", _FLOAT, value),
                 ("in3", _FLOAT, value), ("in4", _FLOAT, value)),
            )
        return value

    def _mix(self, factor: _Value, background: _Value, foreground: _Value) -> _Value:
        return self.doc.node(
            "mix",
            background.type,
            (
                ("fg", background.type, foreground),
                ("bg", background.type, background),
                ("mix", factor.type, factor),
            ),
        )

    def _clamp(
        self,
        value: _Value,
        low: _Value | float = 0.0,
        high: _Value | float = 1.0,
    ) -> _Value:
        if not isinstance(low, _Value):
            low = _literal(_FLOAT, low)
        if not isinstance(high, _Value):
            high = _literal(_FLOAT, high)
        return self.doc.node(
            "clamp",
            value.type,
            (
                ("in", value.type, value),
                ("low", low.type, low),
                ("high", high.type, high),
            ),
        )

    def _compare(
        self,
        category: str,
        a: _Value,
        b: _Value,
        true_value: _Value,
        false_value: _Value,
    ) -> _Value:
        return self.doc.node(
            category,
            true_value.type,
            (
                ("value1", a.type, a),
                ("value2", b.type, b),
                ("in1", true_value.type, true_value),
                ("in2", false_value.type, false_value),
            ),
        )

    def _combine_channels(self, type_name: str, channels: list[_Value]) -> _Value:
        if type_name in {_COLOR3, _VECTOR3}:
            return self.doc.node(
                "combine3",
                type_name,
                tuple((f"in{i + 1}", _FLOAT, channel) for i, channel in enumerate(channels)),
            )
        if type_name in {_COLOR4, _VECTOR4}:
            if len(channels) == 3:
                channels = [*channels, _literal(_FLOAT, 1.0)]
            return self.doc.node(
                "combine4",
                type_name,
                tuple((f"in{i + 1}", _FLOAT, channel) for i, channel in enumerate(channels)),
            )
        return channels[0]

    def _soft_light_scalar(self, a: _Value, b: _Value) -> _Value:
        """MaterialX-only expansion of Blender's per-channel soft-light blend."""
        one = _literal(_FLOAT, 1.0)
        two = _literal(_FLOAT, 2.0)
        quarter = _literal(_FLOAT, 0.25)
        half = _literal(_FLOAT, 0.5)
        one_minus_a = self._binary("subtract", one, a, _FLOAT)
        low = self._binary(
            "subtract",
            a,
            self._binary(
                "multiply",
                self._binary(
                    "multiply",
                    self._binary("subtract", one, self._binary("multiply", two, b, _FLOAT), _FLOAT),
                    a,
                    _FLOAT,
                ),
                one_minus_a,
                _FLOAT,
            ),
            _FLOAT,
        )
        cubic = self._binary(
            "multiply",
            self._binary(
                "add",
                self._binary(
                    "multiply",
                    self._binary(
                        "add",
                        self._binary("multiply", _literal(_FLOAT, 16.0), a, _FLOAT),
                        _literal(_FLOAT, -12.0),
                        _FLOAT,
                    ),
                    a,
                    _FLOAT,
                ),
                _literal(_FLOAT, 4.0),
                _FLOAT,
            ),
            a,
            _FLOAT,
        )
        # MaterialX stdlib exposes the greater-than comparisons, but not
        # `ifless`. Swap the operands to keep this graph standard-compliant.
        d = self._compare("ifgreater", quarter, a, cubic, self._op("sqrt", _FLOAT, a))
        high = self._binary(
            "add",
            a,
            self._binary(
                "multiply",
                self._binary("subtract", self._binary("multiply", two, b, _FLOAT), one, _FLOAT),
                self._binary("subtract", d, a, _FLOAT),
                _FLOAT,
            ),
            _FLOAT,
        )
        return self._compare("ifgreater", half, b, low, high)

    def _hsv_blend(self, blend_type: str, a: _Value, b: _Value) -> _Value:
        """Replace Blender's HSV blend modes with standard MaterialX nodes."""
        if a.type not in {_COLOR3, _COLOR4, _VECTOR3, _VECTOR4}:
            return b
        working_type = a.type
        color_type = {
            _VECTOR3: _COLOR3,
            _VECTOR4: _COLOR4,
        }.get(working_type, working_type)
        a_color = self.doc.convert(a, color_type)
        b_color = self.doc.convert(b, color_type)
        hsv_a = self.doc.node("rgbtohsv", color_type, (("in", color_type, a_color),))
        hsv_b = self.doc.node("rgbtohsv", color_type, (("in", color_type, b_color),))
        channel_count = 4 if color_type == _COLOR4 else 3
        a_channels = [self.doc.extract(hsv_a, i) for i in range(channel_count)]
        b_channels = [self.doc.extract(hsv_b, i) for i in range(channel_count)]
        if blend_type == "HUE":
            channels = [b_channels[0], a_channels[1], a_channels[2]]
        elif blend_type == "SATURATION":
            channels = [a_channels[0], b_channels[1], a_channels[2]]
        elif blend_type == "COLOR":
            channels = [b_channels[0], b_channels[1], a_channels[2]]
        else:  # VALUE
            channels = [a_channels[0], a_channels[1], b_channels[2]]
        if color_type == _COLOR4:
            channels.append(b_channels[3])
        hsv = self._combine_channels(color_type, channels)
        result = self.doc.node("hsvtorgb", color_type, (("in", color_type, hsv),))
        return self.doc.convert(result, working_type)

    def _blend_scalar_mode(self, blend_type: str, factor: _Value, a: _Value, b: _Value) -> _Value:
        categories = {
            "ADD": "plus",
            "SUBTRACT": "minus",
            "DIFFERENCE": "difference",
            "BURN": "burn",
            "DODGE": "dodge",
            "SCREEN": "screen",
            "OVERLAY": "overlay",
        }
        if blend_type == "MIX":
            return self._mix(factor, a, b)
        if blend_type in categories:
            return self.doc.node(
                categories[blend_type],
                a.type,
                (("fg", a.type, b), ("bg", a.type, a), ("mix", _FLOAT, factor)),
            )
        if blend_type == "MULTIPLY":
            blended = self._binary("multiply", a, b, a.type)
        elif blend_type == "DIVIDE":
            blended = self._binary("divide", a, b, a.type)
        elif blend_type == "DARKEN":
            blended = self._binary("min", a, b, a.type)
        elif blend_type == "LIGHTEN":
            blended = self._binary("max", a, b, a.type)
        elif blend_type == "LINEAR_LIGHT":
            blended = self._binary(
                "subtract",
                self._binary("add", a, self._binary("multiply", b, _literal(_FLOAT, 2.0), a.type), a.type),
                _literal(_FLOAT, 1.0),
                a.type,
            )
        elif blend_type == "EXCLUSION":
            blended = self._binary(
                "subtract",
                self._binary("add", a, b, a.type),
                self._binary("multiply", self._literal_like(a.type, 2.0), self._binary("multiply", a, b, a.type), a.type),
                a.type,
            )
        elif blend_type == "SOFT_LIGHT":
            blended = self._soft_light_scalar(a, b)
        else:
            blended = b
        return self._mix(factor, a, blended)

    @staticmethod
    def _literal_like(type_name: str, value: float) -> _Value:
        if type_name == _FLOAT:
            return _literal(_FLOAT, value)
        if type_name in {_COLOR3, _VECTOR3}:
            return _literal(type_name, (value, value, value))
        if type_name in {_COLOR4, _VECTOR4}:
            return _literal(type_name, (value, value, value, value))
        return _literal(_FLOAT, value)

    def _negate(self, value: _Value) -> _Value:
        return self._binary(
            "subtract", _default(value.type), value, value.type
        )

    def _default_position(self) -> _Value:
        return self.doc.node(
            "position",
            _VECTOR3,
            (("space", "string", _literal("string", "object")),),
        )

    def _default_texcoord(self) -> _Value:
        return self.doc.node(
            "texcoord",
            _VECTOR2,
            (("index", _INTEGER, _literal(_INTEGER, 0)),),
        )

    # -- Terminals and constants -----------------------------------------

    def surface(self) -> _Value:
        tree = getattr(self.material, "node_tree", None)
        if tree is not None:
            output_nodes = [
                node
                for node in tree.nodes
                if node.bl_idname == "ShaderNodeOutputMaterial"
                and getattr(node, "is_active_output", True)
            ]
            if not output_nodes:
                output_nodes = [
                    node
                    for node in tree.nodes
                    if node.bl_idname == "ShaderNodeOutputMaterial"
                ]
            if output_nodes:
                surface = self._input_socket(output_nodes[0], "Surface")
                if surface and surface.is_linked and surface.links:
                    link = surface.links[0]
                    shader = self.output(link.from_node, link.from_socket, _SHADER)
                    return self._as_surfaceshader(shader)
        return self._fallback_surface()

    def _as_surfaceshader(self, value: _Value) -> _Value:
        """Wrap a closure-family value in a MaterialX surfaceshader."""
        if value.type == _SURFACE:
            return value
        inputs = []
        if value.node is not None:
            if value.type == _BSDF:
                inputs.append(("bsdf", _BSDF, value))
            elif value.type == _EDF:
                inputs.append(("edf", _EDF, value))
        return self.doc.node("surface", _SURFACE, inputs)

    def _add_closure(
        self, closure_type: str, a: Optional[_Value], b: Optional[_Value]
    ) -> Optional[_Value]:
        """Add two closure values when both are present."""
        a = a if a is not None and a.node is not None else None
        b = b if b is not None and b.node is not None else None
        if a is None:
            return b
        if b is None:
            return a
        return self.doc.node(
            "add", closure_type,
            (("in1", closure_type, a), ("in2", closure_type, b)),
        )

    def _mix_shader(self, fac: _Value, a: _Value, b: _Value) -> _Value:
        # Mix the complete surfaceshader values, including opacity and EDF.
        return self.doc.node(
            "mix",
            _SURFACE,
            (
                ("fg", _SURFACE, self._as_surfaceshader(b)),
                ("bg", _SURFACE, self._as_surfaceshader(a)),
                ("mix", _FLOAT, fac),
            ),
        )

    def _add_shader(self, a: _Value, b: _Value) -> _Value:
        # Add decomposed BSDF and EDF leaves directly. A complete surfaceshader
        # cannot be decomposed here, so use the representable mixed form.
        if a.type == _SURFACE or b.type == _SURFACE:
            return self._mix_shader(_literal(_FLOAT, 0.5), a, b)
        bsdf = self._add_closure(_BSDF, a if a.type == _BSDF else None,
            b if b.type == _BSDF else None)
        edf = self._add_closure(_EDF, a if a.type == _EDF else None,
            b if b.type == _EDF else None)
        if bsdf is not None and edf is None:
            return bsdf
        if edf is not None and bsdf is None:
            return edf
        inputs = []
        if bsdf is not None:
            inputs.append(("bsdf", _BSDF, bsdf))
        if edf is not None:
            inputs.append(("edf", _EDF, edf))
        return self.doc.node("surface", _SURFACE, inputs)

    def _fallback_surface(self) -> _Value:
        color = getattr(self.material, "diffuse_color", (0.8, 0.8, 0.8, 1.0))
        metallic = getattr(self.material, "metallic", 0.0)
        roughness = getattr(self.material, "roughness", 0.5)
        return self.doc.node(
            "open_pbr_surface",
            _SURFACE,
            (
                ("base_color", _COLOR3, _literal(_COLOR3, color[:3])),
                ("base_metalness", _FLOAT, _literal(_FLOAT, metallic)),
                ("specular_roughness", _FLOAT, _literal(_FLOAT, roughness)),
                (
                    "geometry_opacity",
                    _FLOAT,
                    _literal(_FLOAT, color[3] if len(color) > 3 else 1.0),
                ),
            ),
        )

    def _export_ShaderNodeValue(self, node, socket):
        return self._socket_default(socket, _FLOAT)

    def _export_ShaderNodeRGB(self, node, socket):
        return self._socket_default(socket, _COLOR3)

    def _export_ShaderNodeRGBToBW(self, node, socket):
        del socket
        # Blender's RGB to BW node is a luminance reduction. MaterialX's
        # luminance node preserves the color width, so _Document.convert adds
        # the standard extract step needed for the scalar output.
        return self.doc.convert(
            self._input(node, "Color", _COLOR3), _FLOAT)

    def _export_ShaderNodeBlackbody(self, node, socket):
        del socket
        return self.doc.node(
            "blackbody",
            _COLOR3,
            (("temperature", _FLOAT, self._input(node, "Temperature", _FLOAT)),),
        )

    # -- Principled ------------------------------------------------------

    def _export_ShaderNodeBsdfPrincipled(self, node, socket):
        del socket
        value = lambda name, type_name: self._input(node, name, type_name)
        base_color = value("Base Color", _COLOR3)
        specular_level = value("Specular IOR Level", _FLOAT)
        specular_weight = self._binary(
            "multiply", specular_level, _literal(_FLOAT, 2.0), _FLOAT
        )
        anisotropy = self._binary(
            "multiply",
            value("Anisotropic", _FLOAT),
            _literal(_FLOAT, 0.7),
            _FLOAT,
        )
        thin_film_thickness = self._binary(
            "multiply",
            value("Thin Film Thickness", _FLOAT),
            _literal(_FLOAT, 0.001),
            _FLOAT,
        )
        thin_film_weight = self._compare(
            "ifgreater",
            thin_film_thickness,
            _literal(_FLOAT, 0.0),
            _literal(_FLOAT, 1.0),
            _literal(_FLOAT, 0.0),
        )
        inputs = [
            ("base_color", _COLOR3, base_color),
            ("base_diffuse_roughness", _FLOAT, value("Diffuse Roughness", _FLOAT)),
            ("base_metalness", _FLOAT, value("Metallic", _FLOAT)),
            ("specular_weight", _FLOAT, specular_weight),
            ("specular_color", _COLOR3, value("Specular Tint", _COLOR3)),
            ("specular_roughness", _FLOAT, value("Roughness", _FLOAT)),
            ("specular_ior", _FLOAT, value("IOR", _FLOAT)),
            ("specular_roughness_anisotropy", _FLOAT, anisotropy),
            ("transmission_weight", _FLOAT, value("Transmission Weight", _FLOAT)),
            ("transmission_color", _COLOR3, base_color),
            ("subsurface_weight", _FLOAT, value("Subsurface Weight", _FLOAT)),
            ("subsurface_color", _COLOR3, base_color),
            ("subsurface_radius_scale", _COLOR3, value("Subsurface Radius", _COLOR3)),
            ("subsurface_radius", _FLOAT, value("Subsurface Scale", _FLOAT)),
            (
                "subsurface_scatter_anisotropy",
                _FLOAT,
                value("Subsurface Anisotropy", _FLOAT),
            ),
            ("fuzz_weight", _FLOAT, value("Sheen Weight", _FLOAT)),
            ("fuzz_color", _COLOR3, value("Sheen Tint", _COLOR3)),
            ("fuzz_roughness", _FLOAT, value("Sheen Roughness", _FLOAT)),
            ("coat_weight", _FLOAT, value("Coat Weight", _FLOAT)),
            ("coat_color", _COLOR3, value("Coat Tint", _COLOR3)),
            ("coat_roughness", _FLOAT, value("Coat Roughness", _FLOAT)),
            ("coat_ior", _FLOAT, value("Coat IOR", _FLOAT)),
            ("emission_luminance", _FLOAT, value("Emission Strength", _FLOAT)),
            ("emission_color", _COLOR3, value("Emission Color", _COLOR3)),
            ("thin_film_weight", _FLOAT, thin_film_weight),
            ("thin_film_thickness", _FLOAT, thin_film_thickness),
            ("thin_film_ior", _FLOAT, value("Thin Film IOR", _FLOAT)),
            ("geometry_opacity", _FLOAT, value("Alpha", _FLOAT)),
            ("geometry_thin_walled", _BOOLEAN, value("Thin Wall", _BOOLEAN)),
        ]
        for blender_name, mx_name in (
            ("Normal", "geometry_normal"),
            ("Coat Normal", "geometry_coat_normal"),
            ("Tangent", "geometry_tangent"),
        ):
            linked = self._linked_input(node, blender_name, _VECTOR3)
            if linked is not None:
                inputs.append((mx_name, _VECTOR3, linked))
        return self.doc.node("open_pbr_surface", _SURFACE, inputs)

    # -- Raw BSDF/EDF closures --------------------------------------------
    #
    # Keep leaf closures decomposed until the material output.

    @staticmethod
    def _mx_distribution(node) -> str:
        # The MaterialX dielectric and conductor nodes use GGX here. Blender
        # distributions without a direct MaterialX equivalent use this mapping.
        return "ggx"

    def _export_ShaderNodeBsdfDiffuse(self, node, socket):
        del socket
        return self.doc.node(
            "oren_nayar_diffuse_bsdf",
            _BSDF,
            (
                ("color", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("roughness", _FLOAT, self._input(node, "Roughness", _FLOAT)),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
            ),
        )

    def _export_ShaderNodeBsdfTranslucent(self, node, socket):
        del socket
        return self.doc.node(
            "translucent_bsdf",
            _BSDF,
            (
                ("color", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
            ),
        )

    def _export_ShaderNodeBsdfTransparent(self, node, socket):
        del socket
        return self.doc.node(
            "dielectric_bsdf",
            _BSDF,
            (
                ("tint", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("ior", _FLOAT, _literal(_FLOAT, 1.0)),
                ("scatter_mode", "string", _literal("string", "T")),
            ),
        )

    def _export_ShaderNodeBsdfGlass(self, node, socket):
        del socket
        roughness = self._input(node, "Roughness", _FLOAT)
        return self.doc.node(
            "dielectric_bsdf",
            _BSDF,
            (
                ("tint", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("ior", _FLOAT, self._input(node, "IOR", _FLOAT)),
                ("roughness", _VECTOR2, self.doc.node(
                    "combine2", _VECTOR2,
                    (("in1", _FLOAT, roughness), ("in2", _FLOAT, roughness)))),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
                ("distribution", "string", _literal("string", self._mx_distribution(node))),
                ("scatter_mode", "string", _literal("string", "RT")),
            ),
        )

    def _export_ShaderNodeBsdfRefraction(self, node, socket):
        del socket
        roughness = self._input(node, "Roughness", _FLOAT)
        return self.doc.node(
            "dielectric_bsdf",
            _BSDF,
            (
                ("tint", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("ior", _FLOAT, self._input(node, "IOR", _FLOAT)),
                ("roughness", _VECTOR2, self.doc.node(
                    "combine2", _VECTOR2,
                    (("in1", _FLOAT, roughness), ("in2", _FLOAT, roughness)))),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
                ("distribution", "string", _literal("string", self._mx_distribution(node))),
                ("scatter_mode", "string", _literal("string", "T")),
            ),
        )

    def _export_ShaderNodeBsdfGlossy(self, node, socket):
        del socket
        roughness = self._input(node, "Roughness", _FLOAT)
        # Convert Blender's colored mirror input to MaterialX's conductor
        # IOR/extinction representation.
        artistic = self.doc.node(
            "artistic_ior",
            "multioutput",
            (
                ("reflectivity", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("edge_color", _COLOR3, _literal(_COLOR3, (1.0, 1.0, 1.0))),
            ),
        )
        return self.doc.node(
            "conductor_bsdf",
            _BSDF,
            (
                ("ior", _COLOR3, self.doc.named_output(artistic, "ior", _COLOR3)),
                ("extinction", _COLOR3, self.doc.named_output(artistic, "extinction", _COLOR3)),
                ("roughness", _VECTOR2, self.doc.node(
                    "combine2", _VECTOR2,
                    (("in1", _FLOAT, roughness), ("in2", _FLOAT, roughness)))),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
                ("distribution", "string", _literal("string", self._mx_distribution(node))),
            ),
        )

    _export_ShaderNodeBsdfAnisotropic = _export_ShaderNodeBsdfGlossy

    def _export_ShaderNodeBsdfMetallic(self, node, socket):
        del socket
        roughness = self._input(node, "Roughness", _FLOAT)
        if getattr(node, "fresnel_type", "F82") == "PHYSICAL_CONDUCTOR":
            return self.doc.node(
                "conductor_bsdf",
                _BSDF,
                (
                    ("ior", _COLOR3, self._input(node, "IOR", _COLOR3)),
                    ("extinction", _COLOR3, self._input(node, "Extinction", _COLOR3)),
                    ("roughness", _VECTOR2, self.doc.node(
                        "combine2", _VECTOR2,
                        (("in1", _FLOAT, roughness), ("in2", _FLOAT, roughness)))),
                    ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
                    ("tangent", _VECTOR3, self._input(node, "Tangent", _VECTOR3)),
                ),
            )

        # Blender's default F82 mode uses Base Color as normal-incidence
        # reflectivity and Edge Tint as grazing-angle reflectivity. MaterialX's
        # standard generalized Schlick node expresses that same model.
        return self.doc.node(
            "generalized_schlick_bsdf",
            _BSDF,
            (
                ("color0", _COLOR3, self._input(node, "Base Color", _COLOR3)),
                ("color90", _COLOR3, self._input(node, "Edge Tint", _COLOR3)),
                ("roughness", _VECTOR2, self.doc.node(
                    "combine2", _VECTOR2,
                    (("in1", _FLOAT, roughness), ("in2", _FLOAT, roughness)))),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
                ("tangent", _VECTOR3, self._input(node, "Tangent", _VECTOR3)),
                ("scatter_mode", "string", _literal("string", "R")),
            ),
        )

    def _export_ShaderNodeBsdfSheen(self, node, socket):
        del socket
        roughness_name = "Roughness" if self._input_socket(node, "Roughness") is not None else "Sigma"
        return self.doc.node(
            "sheen_bsdf",
            _BSDF,
            (
                ("color", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("roughness", _FLOAT, self._input(node, roughness_name, _FLOAT)),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
            ),
        )

    _export_ShaderNodeBsdfVelvet = _export_ShaderNodeBsdfSheen

    def _export_ShaderNodeSubsurfaceScattering(self, node, socket):
        del socket
        return self.doc.node(
            "subsurface_bsdf",
            _BSDF,
            (
                ("color", _COLOR3, self._input(node, "Color", _COLOR3)),
                ("radius", _COLOR3, self._input(node, "Radius", _COLOR3)),
                ("anisotropy", _FLOAT, self._input(node, "Anisotropy", _FLOAT)),
                ("normal", _VECTOR3, self._input(node, "Normal", _VECTOR3)),
            ),
        )

    def _export_ShaderNodeEmission(self, node, socket):
        del socket
        color = self._input(node, "Color", _COLOR3)
        strength = self._input(node, "Strength", _FLOAT)
        weighted = self._binary("multiply", color, strength, _COLOR3)
        return self.doc.node("uniform_edf", _EDF, (("color", _COLOR3, weighted),))

    def _export_ShaderNodeMixShader(self, node, socket):
        del socket
        fac = self._input(node, "Fac", _FLOAT)
        a = self._input(node, 1, _SHADER)
        b = self._input(node, 2, _SHADER)
        return self._mix_shader(fac, a, b)

    def _export_ShaderNodeAddShader(self, node, socket):
        del socket
        a = self._input(node, 0, _SHADER)
        b = self._input(node, 1, _SHADER)
        return self._add_shader(a, b)

    # -- Node groups (including nested groups) ---------------------------

    def _export_NodeGroupInput(self, node, socket):
        del node
        if not self.group_stack:
            return self._socket_default(socket, self._socket_type(socket))
        _, resolver = self.group_stack[-1]
        return resolver(socket.identifier, self._socket_type(socket))

    def _export_ShaderNodeGroup(self, node, socket):
        tree = node.node_tree
        if tree is None:
            return self._socket_default(socket, self._socket_type(socket))
        output_nodes = [
            inner for inner in tree.nodes
            if inner.bl_idname == "NodeGroupOutput"
            and getattr(inner, "is_active_output", True)
        ]
        if not output_nodes:
            output_nodes = [
                inner for inner in tree.nodes if inner.bl_idname == "NodeGroupOutput"
            ]
        if not output_nodes:
            return self._socket_default(socket, self._socket_type(socket))
        inner_socket = self._input_socket(output_nodes[0], socket.identifier)
        if inner_socket is None:
            return self._socket_default(socket, self._socket_type(socket))

        # Resolve inner group inputs against the caller's context.
        outer_stack = list(self.group_stack)

        def resolve_outer_input(identifier: str, type_name: str) -> _Value:
            saved = self.group_stack
            self.group_stack = outer_stack
            try:
                return self._input(node, identifier, type_name)
            finally:
                self.group_stack = saved

        self.group_stack.append((node.as_pointer(), resolve_outer_input))
        try:
            return self._input(
                output_nodes[0], inner_socket.identifier, self._socket_type(socket))
        finally:
            self.group_stack.pop()

    # -- Reroute -----------------------------------------------------------
    #
    # Forward the linked value instead of using the reroute socket default.
    def _export_NodeReroute(self, node, socket):
        return self._input(node, 0, self._socket_type(socket))

    # -- Images and coordinates -----------------------------------------

    def _export_ShaderNodeUVMap(self, node, socket):
        del node, socket
        return self._default_texcoord()

    def _export_ShaderNodeTexCoord(self, node, socket):
        del node
        if socket.identifier == "UV":
            return self._default_texcoord()
        if socket.identifier in {"Normal", "Reflection"}:
            return self.doc.node(
                "normal",
                _VECTOR3,
                (("space", "string", _literal("string", "world")),),
            )
        space = "object" if socket.identifier in {"Generated", "Object"} else "world"
        return self.doc.node(
            "position",
            _VECTOR3,
            (("space", "string", _literal("string", space)),),
        )

    def _export_ShaderNodeTexImage(self, node, socket):
        image = getattr(node, "image", None)
        if image is None:
            return (
                _literal(_FLOAT, 1.0)
                if socket.identifier == "Alpha"
                else _literal(_COLOR3, (1.0, 0.0, 1.0))
            )
        image_pointer = image.as_pointer()
        self.image_pointers.add(image_pointer)
        path = self.resolve_image_path(image, node) if self.resolve_image_path else ""
        if not path:
            path = getattr(image, "filepath", "") or getattr(image, "filepath_raw", "")

        vector = self._linked_input(node, "Vector", _VECTOR2) or self._default_texcoord()
        filter_type = {
            "Closest": "closest",
            "Linear": "linear",
            "Cubic": "cubic",
            "Smart": "cubic",
        }.get(getattr(node, "interpolation", "Linear"), "linear")
        address_mode = {
            "REPEAT": "periodic",
            "EXTEND": "clamp",
            "CLIP": "constant",
            "MIRROR": "mirror",
        }.get(getattr(node, "extension", "REPEAT"), "periodic")

        settings = getattr(image, "colorspace_settings", None)
        is_data = bool(getattr(settings, "is_data", False))
        color_space_name = getattr(settings, "name", "") if settings else ""
        image_type = _VECTOR4 if is_data else _COLOR4
        color_space = None
        if not is_data:
            lowered = color_space_name.casefold()
            if "non-color" in lowered or "data" in lowered:
                image_type = _VECTOR4
            elif "srgb" in lowered:
                color_space = "srgb_texture"
            elif "linear" in lowered:
                color_space = "lin_rec709"
        image_value = self.doc.node(
            "image",
            image_type,
            (
                ("file", "filename", _literal("filename", path),
                    (("colorspace", color_space),)),
                ("texcoord", _VECTOR2, vector),
                ("uaddressmode", "string", _literal("string", address_mode)),
                ("vaddressmode", "string", _literal("string", address_mode)),
                ("filtertype", "string", _literal("string", filter_type)),
            ),
        )
        if socket.identifier == "Alpha":
            return self.doc.extract(image_value, 3)
        return image_value

    def _rotate_euler(
        self, vector: _Value, rotation: _Value, inverse: bool = False
    ) -> _Value:
        order = (2, 1, 0) if inverse else (0, 1, 2)
        result = vector
        axes = (
            _literal(_VECTOR3, (1.0, 0.0, 0.0)),
            _literal(_VECTOR3, (0.0, 1.0, 0.0)),
            _literal(_VECTOR3, (0.0, 0.0, 1.0)),
        )
        for index in order:
            angle = self.doc.extract(rotation, index)
            if inverse:
                angle = self._negate(angle)
            angle = self._binary(
                "multiply", angle, _literal(_FLOAT, 180.0 / math.pi), _FLOAT
            )
            result = self.doc.node(
                "rotate3d",
                _VECTOR3,
                (
                    ("in", _VECTOR3, result),
                    ("amount", _FLOAT, angle),
                    ("axis", _VECTOR3, axes[index]),
                ),
            )
        return result

    def _export_ShaderNodeMapping(self, node, socket):
        del socket
        vector = self._input(node, "Vector", _VECTOR3)
        location = self._input(node, "Location", _VECTOR3)
        rotation = self._input(node, "Rotation", _VECTOR3)
        scale = self._input(node, "Scale", _VECTOR3)
        mapping_type = getattr(node, "vector_type", None) or getattr(
            node, "mapping_type", "POINT"
        )
        if mapping_type == "TEXTURE":
            result = self._binary("subtract", vector, location, _VECTOR3)
            result = self._rotate_euler(result, rotation, inverse=True)
            return self._binary("divide", result, scale, _VECTOR3)
        if mapping_type == "VECTOR":
            result = self._binary("multiply", vector, scale, _VECTOR3)
            signed_rotation = self._binary(
                "multiply",
                rotation,
                _literal(_VECTOR3, (1.0, 1.0, -1.0)),
                _VECTOR3,
            )
            return self._rotate_euler(result, signed_rotation)
        if mapping_type == "NORMAL":
            result = self._binary("divide", vector, scale, _VECTOR3)
            result = self._rotate_euler(result, rotation)
            return self._op("normalize", _VECTOR3, result)
        result = self._binary("multiply", vector, scale, _VECTOR3)
        result = self._rotate_euler(result, rotation)
        return self._binary("add", result, location, _VECTOR3)

    # -- Scalar math -----------------------------------------------------

    def _export_ShaderNodeMath(self, node, socket):
        del socket
        x = self._input(node, 0, _FLOAT)
        operation = node.operation
        unary = {
            "SINE": "sin",
            "COSINE": "cos",
            "TANGENT": "tan",
            "ARCSINE": "asin",
            "ARCCOSINE": "acos",
            "ABSOLUTE": "absval",
            "FLOOR": "floor",
            "CEIL": "ceil",
            "SQRT": "sqrt",
            "SIGN": "sign",
            "EXPONENT": "exp",
            "ROUND": "round",
        }
        if operation in unary:
            result = self._op(unary[operation], _FLOAT, x)
        elif operation == "ARCTANGENT":
            result = self._binary("atan2", x, _literal(_FLOAT, 1.0), _FLOAT)
        elif operation == "FRACT":
            result = self._binary(
                "subtract", x, self._op("floor", _FLOAT, x), _FLOAT
            )
        elif operation == "INVERSE_SQRT":
            result = self._binary(
                "divide",
                _literal(_FLOAT, 1.0),
                self._op("sqrt", _FLOAT, x),
                _FLOAT,
            )
        elif operation in {"RADIANS", "DEGREES"}:
            scale = math.pi / 180.0 if operation == "RADIANS" else 180.0 / math.pi
            result = self._binary("multiply", x, _literal(_FLOAT, scale), _FLOAT)
        elif operation in {"SINH", "COSH", "TANH"}:
            pos = self._op("exp", _FLOAT, x)
            neg = self._op("exp", _FLOAT, self._negate(x))
            numerator = self._binary(
                "subtract" if operation in {"SINH", "TANH"} else "add",
                pos,
                neg,
                _FLOAT,
            )
            denominator = (
                self._binary("add", pos, neg, _FLOAT)
                if operation == "TANH"
                else _literal(_FLOAT, 2.0)
            )
            result = self._binary("divide", numerator, denominator, _FLOAT)
        elif operation == "TRUNC":
            result = self._binary(
                "multiply",
                self._op("sign", _FLOAT, x),
                self._op("floor", _FLOAT, self._op("absval", _FLOAT, x)),
                _FLOAT,
            )
        else:
            y = self._input(node, 1, _FLOAT)
            binary = {
                "ADD": "add",
                "SUBTRACT": "subtract",
                "MULTIPLY": "multiply",
                "DIVIDE": "divide",
                "POWER": "power",
                "MINIMUM": "min",
                "MAXIMUM": "max",
                "MODULO": "modulo",
                "ARCTAN2": "atan2",
            }
            if operation in binary:
                result = self._binary(binary[operation], x, y, _FLOAT)
            elif operation == "LOGARITHM":
                result = self._binary(
                    "divide",
                    self._op("ln", _FLOAT, x),
                    self._op("ln", _FLOAT, y),
                    _FLOAT,
                )
            elif operation in {"LESS_THAN", "GREATER_THAN"}:
                result = self._compare(
                    "ifgreater",
                    y if operation == "LESS_THAN" else x,
                    x if operation == "LESS_THAN" else y,
                    _literal(_FLOAT, 1.0),
                    _literal(_FLOAT, 0.0),
                )
            elif operation in {"SNAP", "FLOORED_MODULO"}:
                divided = self._binary("divide", x, y, _FLOAT)
                floored = self._op("floor", _FLOAT, divided)
                if operation == "SNAP":
                    result = self._binary("multiply", floored, y, _FLOAT)
                else:
                    result = self._binary(
                        "subtract",
                        x,
                        self._binary("multiply", floored, y, _FLOAT),
                        _FLOAT,
                    )
            elif operation == "PINGPONG":
                period = self._binary(
                    "multiply", y, _literal(_FLOAT, 2.0), _FLOAT
                )
                wrapped = self._binary("modulo", x, period, _FLOAT)
                result = self._op(
                    "absval",
                    _FLOAT,
                    self._binary("subtract", wrapped, y, _FLOAT),
                )
            else:
                z = self._input(node, 2, _FLOAT)
                if operation == "MULTIPLY_ADD":
                    result = self._binary(
                        "add",
                        self._binary("multiply", x, y, _FLOAT),
                        z,
                        _FLOAT,
                    )
                elif operation == "COMPARE":
                    result = self._compare(
                        "ifgreater",
                        z,
                        self._op(
                            "absval",
                            _FLOAT,
                            self._binary("subtract", x, y, _FLOAT),
                        ),
                        _literal(_FLOAT, 1.0),
                        _literal(_FLOAT, 0.0),
                    )
                elif operation == "WRAP":
                    span = self._binary("subtract", y, z, _FLOAT)
                    normalized = self._binary(
                        "divide",
                        self._binary("subtract", x, z, _FLOAT),
                        span,
                        _FLOAT,
                    )
                    result = self._binary(
                        "subtract",
                        x,
                        self._binary(
                            "multiply",
                            span,
                            self._op("floor", _FLOAT, normalized),
                            _FLOAT,
                        ),
                        _FLOAT,
                    )
                elif operation in {"SMOOTH_MIN", "SMOOTH_MAX"}:
                    a, b, smooth = x, y, z
                    if operation == "SMOOTH_MAX":
                        a, b, smooth = self._negate(a), self._negate(b), self._negate(smooth)
                    h = self._binary(
                        "divide",
                        self._op(
                            "max",
                            _FLOAT,
                            self._binary(
                                "subtract",
                                smooth,
                                self._op(
                                    "absval",
                                    _FLOAT,
                                    self._binary("subtract", a, b, _FLOAT),
                                ),
                                _FLOAT,
                            ),
                            _literal(_FLOAT, 0.0),
                        ),
                        smooth,
                        _FLOAT,
                    )
                    h3 = self._binary(
                        "multiply",
                        self._binary("multiply", h, h, _FLOAT),
                        h,
                        _FLOAT,
                    )
                    correction = self._binary(
                        "multiply",
                        self._binary("multiply", h3, smooth, _FLOAT),
                        _literal(_FLOAT, 1.0 / 6.0),
                        _FLOAT,
                    )
                    result = self._binary(
                        "subtract",
                        self._binary("min", a, b, _FLOAT),
                        correction,
                        _FLOAT,
                    )
                    if operation == "SMOOTH_MAX":
                        result = self._negate(result)
                else:
                    result = x
        if getattr(node, "use_clamp", False):
            result = self._clamp(result)
        return result

    # -- Vector math -----------------------------------------------------

    def _export_ShaderNodeVectorMath(self, node, socket):
        operation = node.operation
        x = self._input(node, 0, _VECTOR3)
        unary = {
            "SINE": "sin",
            "COSINE": "cos",
            "TANGENT": "tan",
            "ABSOLUTE": "absval",
            "FLOOR": "floor",
            "CEIL": "ceil",
            "NORMALIZE": "normalize",
            "SIGN": "sign",
            "ROUND": "round",
        }
        if operation in unary:
            result = self._op(unary[operation], _VECTOR3, x)
        elif operation == "FRACTION":
            result = self._binary(
                "subtract", x, self._op("floor", _VECTOR3, x), _VECTOR3
            )
        elif operation == "LENGTH":
            result = self.doc.node(
                "magnitude", _FLOAT, (("in", _VECTOR3, x),)
            )
        else:
            y = self._input(node, 1, _VECTOR3)
            binary = {
                "ADD": "add",
                "SUBTRACT": "subtract",
                "MULTIPLY": "multiply",
                "DIVIDE": "divide",
                "MINIMUM": "min",
                "MAXIMUM": "max",
                "MODULO": "modulo",
                "POWER": "power",
                "CROSS_PRODUCT": "crossproduct",
                "DOT_PRODUCT": "dotproduct",
            }
            if operation in binary:
                output_type = (
                    _FLOAT if operation == "DOT_PRODUCT" else _VECTOR3
                )
                result = self.doc.node(
                    binary[operation],
                    output_type,
                    (("in1", _VECTOR3, x), ("in2", _VECTOR3, y)),
                )
            elif operation == "DISTANCE":
                result = self._op(
                    "magnitude",
                    _VECTOR3,
                    self._binary("subtract", y, x, _VECTOR3),
                )
            elif operation == "SCALE":
                result = self._binary(
                    "multiply", x, self._input(node, "Scale", _FLOAT), _VECTOR3
                )
            elif operation == "SNAP":
                result = self._binary(
                    "multiply",
                    self._op(
                        "floor",
                        _VECTOR3,
                        self._binary("divide", x, y, _VECTOR3),
                    ),
                    y,
                    _VECTOR3,
                )
            elif operation == "PROJECT":
                dot_yy = self.doc.node(
                    "dotproduct",
                    _FLOAT,
                    (("in1", _VECTOR3, y), ("in2", _VECTOR3, y)),
                )
                dot_xy = self.doc.node(
                    "dotproduct",
                    _FLOAT,
                    (("in1", _VECTOR3, x), ("in2", _VECTOR3, y)),
                )
                result = self._binary(
                    "multiply",
                    y,
                    self._binary("divide", dot_xy, dot_yy, _FLOAT),
                    _VECTOR3,
                )
            elif operation == "REFLECT":
                result = self.doc.node(
                    "reflect",
                    _VECTOR3,
                    (("in", _VECTOR3, x), ("normal", _VECTOR3, y)),
                )
            elif operation == "REFRACT":
                result = self.doc.node(
                    "refract",
                    _VECTOR3,
                    (
                        ("in", _VECTOR3, x),
                        ("normal", _VECTOR3, y),
                        ("ior", _FLOAT, self._input(node, "Scale", _FLOAT)),
                    ),
                )
            else:
                z = self._input(node, 2, _VECTOR3)
                if operation == "MULTIPLY_ADD":
                    result = self._binary(
                        "add",
                        self._binary("multiply", x, y, _VECTOR3),
                        z,
                        _VECTOR3,
                    )
                elif operation == "FACEFORWARD":
                    dot = self.doc.node(
                        "dotproduct",
                        _FLOAT,
                        (("in1", _VECTOR3, z), ("in2", _VECTOR3, y)),
                    )
                    result = self._compare(
                        "ifgreatereq",
                        dot,
                        _literal(_FLOAT, 0.0),
                        self._negate(x),
                        x,
                    )
                elif operation == "WRAP":
                    span = self._binary("subtract", y, z, _VECTOR3)
                    result = self._binary(
                        "subtract",
                        x,
                        self._binary(
                            "multiply",
                            span,
                            self._op(
                                "floor",
                                _VECTOR3,
                                self._binary(
                                    "divide",
                                    self._binary("subtract", x, z, _VECTOR3),
                                    span,
                                    _VECTOR3,
                                ),
                            ),
                            _VECTOR3,
                        ),
                        _VECTOR3,
                    )
                else:
                    result = x
        wanted = _FLOAT if socket.type == "VALUE" else _VECTOR3
        return self.doc.convert(result, wanted)

    # -- Mix and color adjustment ---------------------------------------

    def _blend(
        self,
        blend_type: str,
        factor: _Value,
        a: _Value,
        b: _Value,
        clamp_factor: bool,
        clamp_result: bool,
    ) -> _Value:
        if clamp_factor:
            factor = self._clamp(factor)
        if blend_type in {"HUE", "SATURATION", "COLOR", "VALUE"}:
            blended = self._hsv_blend(blend_type, a, b)
            result = self._mix(factor, a, blended)
        elif (
            a.type in {_COLOR3, _COLOR4, _VECTOR3, _VECTOR4}
            and (
                factor.type != _FLOAT
                or (
                    blend_type == "SOFT_LIGHT"
                    or (
                        a.type in {_VECTOR3, _VECTOR4}
                        and blend_type in {
                            "ADD", "SUBTRACT", "DIFFERENCE", "BURN", "DODGE",
                            "SCREEN", "OVERLAY",
                        }
                    )
                )
            )
        ):
            channels = []
            for index in range(3 if a.type in {_COLOR3, _VECTOR3} else 4):
                channels.append(
                    self._blend_scalar_mode(
                        blend_type,
                        factor if factor.type == _FLOAT else self.doc.extract(factor, index),
                        self.doc.extract(a, index),
                        self.doc.extract(b, index),
                    )
                )
            result = self._combine_channels(a.type, channels)
        elif blend_type in {"ADD", "SUBTRACT", "DIFFERENCE", "BURN", "DODGE", "SCREEN", "OVERLAY"} and a.type in {_FLOAT, _COLOR3, _COLOR4}:
            result = self._blend_scalar_mode(blend_type, factor, a, b)
        else:
            result = self._blend_scalar_mode(blend_type, factor, a, b)
        return self._clamp(result) if clamp_result else result

    def _export_ShaderNodeMix(self, node, socket):
        data_type = node.data_type
        if data_type == "FLOAT":
            type_name, factor_name, a_name, b_name = (
                _FLOAT,
                "Factor_Float",
                "A_Float",
                "B_Float",
            )
        elif data_type == "VECTOR":
            type_name, factor_name, a_name, b_name = (
                _VECTOR3,
                "Factor_Vector" if getattr(node, "factor_mode", "UNIFORM") != "UNIFORM" else "Factor_Float",
                "A_Vector",
                "B_Vector",
            )
        else:
            type_name, factor_name, a_name, b_name = (
                _COLOR4,
                "Factor_Float",
                "A_Color",
                "B_Color",
            )
        factor_type = _VECTOR3 if factor_name == "Factor_Vector" else _FLOAT
        result = self._blend(
            getattr(node, "blend_type", "MIX"),
            self._input(node, factor_name, factor_type),
            self._input(node, a_name, type_name),
            self._input(node, b_name, type_name),
            bool(getattr(node, "clamp_factor", False)),
            bool(getattr(node, "clamp_result", False)),
        )
        return self.doc.convert(result, self._socket_type(socket))

    def _export_ShaderNodeMixRGB(self, node, socket):
        del socket
        return self._blend(
            getattr(node, "blend_type", "MIX"),
            self._input(node, "Fac", _FLOAT),
            self._input(node, "Color1", _COLOR3),
            self._input(node, "Color2", _COLOR3),
            True,
            bool(getattr(node, "use_clamp", False)),
        )

    def _export_ShaderNodeInvert(self, node, socket):
        del socket
        color = self._input(node, "Color", _COLOR3)
        inverted = self._binary(
            "subtract", _literal(_COLOR3, (1.0, 1.0, 1.0)), color, _COLOR3
        )
        return self._mix(self._input(node, "Fac", _FLOAT), color, inverted)

    def _export_ShaderNodeHueSaturation(self, node, socket):
        del socket
        color = self._input(node, "Color", _COLOR3)
        amount = self.doc.node(
            "combine3",
            _VECTOR3,
            (
                (
                    "in1",
                    _FLOAT,
                    self._binary(
                        "subtract",
                        self._input(node, "Hue", _FLOAT),
                        _literal(_FLOAT, 0.5),
                        _FLOAT,
                    ),
                ),
                ("in2", _FLOAT, self._input(node, "Saturation", _FLOAT)),
                ("in3", _FLOAT, self._input(node, "Value", _FLOAT)),
            ),
        )
        adjusted = self.doc.node(
            "hsvadjust",
            _COLOR3,
            (("in", _COLOR3, color), ("amount", _VECTOR3, amount)),
        )
        return self._mix(self._input(node, "Fac", _FLOAT), color, adjusted)

    def _export_ShaderNodeGamma(self, node, socket):
        del socket
        return self._binary(
            "power",
            self._input(node, "Color", _COLOR3),
            self._input(node, "Gamma", _FLOAT),
            _COLOR3,
        )

    def _export_ShaderNodeClamp(self, node, socket):
        del socket
        value = self._input(node, "Value", _FLOAT)
        low = self._input(node, "Min", _FLOAT)
        high = self._input(node, "Max", _FLOAT)
        if getattr(node, "clamp_type", "MINMAX") == "RANGE":
            normal = self._clamp(value, low, high)
            reverse = self._clamp(value, high, low)
            return self._compare("ifgreater", low, high, reverse, normal)
        return self._clamp(value, low, high)

    # -- Color ramp ------------------------------------------------------

    def _export_ShaderNodeValToRGB(self, node, socket):
        factor = self._input(node, "Fac", _FLOAT)
        ramp = node.color_ramp
        elements = sorted(ramp.elements, key=lambda element: element.position)
        if not elements:
            result = _literal(_COLOR4, (0.0, 0.0, 0.0, 1.0))
        elif len(elements) == 1:
            result = _literal(_COLOR4, elements[0].color)
        else:
            result = _literal(_COLOR4, elements[-1].color)
            interpolation = ramp.interpolation
            for left, right in reversed(tuple(zip(elements, elements[1:]))):
                left_color = _literal(_COLOR4, left.color)
                right_color = _literal(_COLOR4, right.color)
                if interpolation == "CONSTANT":
                    segment = left_color
                else:
                    if interpolation == "EASE":
                        t = self.doc.node(
                            "smoothstep",
                            _FLOAT,
                            (
                                ("in", _FLOAT, factor),
                                ("low", _FLOAT, _literal(_FLOAT, left.position)),
                                ("high", _FLOAT, _literal(_FLOAT, right.position)),
                            ),
                        )
                    else:
                        t = self.doc.node(
                            "remap",
                            _FLOAT,
                            (
                                ("in", _FLOAT, factor),
                                ("inlow", _FLOAT, _literal(_FLOAT, left.position)),
                                ("inhigh", _FLOAT, _literal(_FLOAT, right.position)),
                            ),
                        )
                        t = self._clamp(t)
                    segment = self._mix(t, left_color, right_color)
                result = self._compare(
                    "ifgreatereq",
                    factor,
                    _literal(_FLOAT, right.position),
                    result,
                    segment,
                )
        return self.doc.extract(result, 3) if socket.identifier == "Alpha" else result

    # -- RGB curves ------------------------------------------------------

    # MaterialX has no curve node, so sampled curves become piecewise nodes.
    _CURVE_SAMPLES = 16

    @staticmethod
    def _curve_samples(mapping, curve, count: int) -> Optional[list]:
        # Support both Blender curve-evaluation APIs.
        divisor = float(count - 1)
        samples = []
        for index in range(count):
            position = index / divisor
            try:
                samples.append(float(mapping.evaluate(curve, position)))
            except (AttributeError, TypeError, ValueError):
                try:
                    samples.append(float(curve.evaluate(position)))
                except (AttributeError, TypeError, ValueError):
                    return None
        return samples

    @staticmethod
    def _is_identity_curve(samples: list, tolerance: float = 1.0e-4) -> bool:
        divisor = float(len(samples) - 1)
        return all(
            abs(sample - index / divisor) <= tolerance
            for index, sample in enumerate(samples)
        )

    def _piecewise_curve(self, value: _Value, samples: list) -> _Value:
        # Build the conditional chain from the last segment toward the first.
        divisor = float(len(samples) - 1)
        result = _literal(_FLOAT, samples[-1])
        for index in range(len(samples) - 2, -1, -1):
            low = index / divisor
            high = (index + 1) / divisor
            t = self._clamp(
                self.doc.node(
                    "remap",
                    _FLOAT,
                    (
                        ("in", _FLOAT, value),
                        ("inlow", _FLOAT, _literal(_FLOAT, low)),
                        ("inhigh", _FLOAT, _literal(_FLOAT, high)),
                    ),
                )
            )
            segment = self._mix(
                t, _literal(_FLOAT, samples[index]), _literal(_FLOAT, samples[index + 1])
            )
            result = self._compare(
                "ifgreatereq", value, _literal(_FLOAT, high), result, segment
            )
        return result

    def _export_ShaderNodeRGBCurve(self, node, socket):
        del socket
        color = self._input(node, "Color", _COLOR3)
        factor = self._input(node, "Fac", _FLOAT)
        mapping = getattr(node, "mapping", None)
        if mapping is None:
            return color
        try:
            mapping.update()
            curves = tuple(mapping.curves)
        except (AttributeError, RuntimeError):
            return color
        if len(curves) < 4:
            return color

        # Apply per-channel curves, followed by the combined curve.
        channel_samples = [
            self._curve_samples(mapping, curves[i], self._CURVE_SAMPLES)
            for i in range(3)
        ]
        combined_samples = self._curve_samples(
            mapping, curves[3], self._CURVE_SAMPLES)
        if combined_samples is None or any(s is None for s in channel_samples):
            self.unsupported_nodes.add("ShaderNodeRGBCurve")
            return color
        combined_identity = self._is_identity_curve(combined_samples)
        if combined_identity and all(
            self._is_identity_curve(samples) for samples in channel_samples
        ):
            return color

        components = []
        for index in range(3):
            component = self.doc.extract(color, index)
            if not self._is_identity_curve(channel_samples[index]):
                component = self._piecewise_curve(component, channel_samples[index])
            if not combined_identity:
                component = self._piecewise_curve(component, combined_samples)
            components.append(component)
        mapped = self.doc.node(
            "combine3",
            _COLOR3,
            tuple((f"in{i + 1}", _FLOAT, components[i]) for i in range(3)),
        )
        # Fac blends between the untouched input and the mapped result.
        return self._mix(factor, color, mapped)

    # -- Geometry --------------------------------------------------------

    @staticmethod
    def _materialx_geomprop_name(attribute_name: str) -> str:
        """Map Blender's built-in attribute spellings to MaterialX names.

        Custom names are deliberately preserved.  They are valid MaterialX
        geomprop names even when the current Hydra geometry source does not
        provide that primvar; the node's declared default then supplies the
        standard fallback instead of dropping the whole graph.
        """
        name = (attribute_name or "").strip()
        aliases = {
            "generated": "Pobject",
            "position": "Pworld",
            "p": "Pworld",
            "pworld": "Pworld",
            "object": "Pobject",
            "pobject": "Pobject",
            "normal": "Nworld",
            "n": "Nworld",
            "nworld": "Nworld",
            "tangent": "Tworld",
            "tworld": "Tworld",
            "bitangent": "Bworld",
            "bworld": "Bworld",
            "uv": "UV0",
            "uv0": "UV0",
            "uv_map": "UV0",
            "st": "st",
            "color": "color",
            "col": "color",
            "cd": "Cd",
            "vertex_color": "vertex_color",
        }
        return aliases.get(name.casefold(), name or "color")

    def _attribute_value(self, node, type_name: str) -> _Value:
        attribute_name = self._materialx_geomprop_name(
            getattr(node, "attribute_name", ""))
        defaults = {
            _COLOR4: _literal(_COLOR4, (0.0, 0.0, 0.0, 1.0)),
            _VECTOR3: _default(_VECTOR3),
            _FLOAT: _default(_FLOAT),
        }
        return self.doc.node(
            "geompropvalue",
            type_name,
            (
                ("geomprop", "string", _literal("string", attribute_name)),
                ("default", type_name, defaults[type_name]),
            ),
        )

    def _export_ShaderNodeAttribute(self, node, socket):
        """Lower Blender's legacy Attribute node to standard MaterialX.

        MaterialX's geompropvalue is the portable equivalent of Blender's
        named geometry attribute query.  NoorRay currently has native streams
        for the standard position/normal/tangent/UV/display-color names and
        uses the authored default for custom primvars not present on a mesh.
        """
        if socket.identifier == "Color":
            return self._attribute_value(node, _COLOR4)
        if socket.identifier == "Vector":
            return self._attribute_value(node, _VECTOR3)
        if socket.identifier == "Alpha":
            return self.doc.extract(
                self._attribute_value(node, _COLOR4), 3)
        if socket.identifier in {"Fac", "Factor"}:
            return self._attribute_value(node, _FLOAT)
        return self._socket_default(socket, self._socket_type(socket))

    def _export_ShaderNodeBrightContrast(self, node, socket):
        """Expand Cycles' brightness/contrast equation into stdlib nodes.

        Blender evaluates this as max((1 + contrast) * color + brightness -
        0.5 * contrast, 0).  MaterialX's contrast node expresses the first
        affine part around a pivot, so the complete lowering is contrast,
        add, then max, with no renderer-specific node involved.
        """
        del socket
        color = self._input(node, "Color", _COLOR4)
        brightness = self._input(node, "Bright", _FLOAT)
        contrast = self._input(node, "Contrast", _FLOAT)
        amount = self._binary(
            "add", _literal(_FLOAT, 1.0), contrast, _FLOAT)
        adjusted = self.doc.node(
            "contrast",
            color.type,
            (
                ("in", color.type, color),
                ("amount", _FLOAT, amount),
                ("pivot", _FLOAT, _literal(_FLOAT, 0.5)),
            ),
        )
        brightened = self._binary(
            "add", adjusted, brightness, color.type)
        return self._binary(
            "max", brightened, _literal(color.type, (0.0,) * 4), color.type)

    def _export_ShaderNodeVertexColor(self, node, socket):
        del node
        # Blender's layer_name selects a colour attribute; the renderer carries
        # exactly one per vertex (Surface::color), which is what geomcolor
        # reads, so the name is not forwarded.
        if socket.identifier == "Alpha":
            return self.doc.extract(
                self.doc.node("geomcolor", _COLOR4), 3
            )
        return self.doc.node("geomcolor", _COLOR3)

    def _shading_normal(self, node) -> _Value:
        # Blender's Normal input overrides the shading normal when linked.
        linked = self._linked_input(node, "Normal", _VECTOR3)
        if linked is not None:
            return linked
        return self.doc.node(
            "normal", _VECTOR3,
            (("space", "string", _literal("string", "world")),))

    def _facing_cosine(self, node) -> _Value:
        # |N.V|, the quantity both Fresnel and Layer Weight are built on.
        return self._op(
            "absval",
            _FLOAT,
            self.doc.node(
                "dotproduct",
                _FLOAT,
                (
                    ("in1", _VECTOR3, self._shading_normal(node)),
                    ("in2", _VECTOR3, self.doc.node("viewdirection", _VECTOR3)),
                ),
            ),
        )

    def _schlick(self, cosine: _Value, f0: _Value) -> _Value:
        # Schlick approximation used for the exported Fresnel value.
        one_minus = self._binary(
            "subtract", _literal(_FLOAT, 1.0), cosine, _FLOAT)
        power = self._binary(
            "power", one_minus, _literal(_FLOAT, 5.0), _FLOAT)
        weight = self._binary(
            "subtract", _literal(_FLOAT, 1.0), f0, _FLOAT)
        return self._binary(
            "add", f0, self._binary("multiply", weight, power, _FLOAT), _FLOAT)

    def _export_ShaderNodeFresnel(self, node, socket):
        del socket
        ior = self._input(node, "IOR", _FLOAT)
        # f0 = ((ior - 1) / (ior + 1))^2
        ratio = self._binary(
            "divide",
            self._binary("subtract", ior, _literal(_FLOAT, 1.0), _FLOAT),
            self._binary("add", ior, _literal(_FLOAT, 1.0), _FLOAT),
            _FLOAT,
        )
        f0 = self._binary("multiply", ratio, ratio, _FLOAT)
        return self._schlick(self._facing_cosine(node), f0)

    def _export_ShaderNodeLayerWeight(self, node, socket):
        blend = self._clamp(
            self._input(node, "Blend", _FLOAT),
            _literal(_FLOAT, 0.0),
            _literal(_FLOAT, 0.99999),
        )
        cosine = self._facing_cosine(node)
        if socket.identifier == "Facing":
            # Cycles remaps Blend into an exponent: below 0.5 it scales to
            # [0, 1], above it to [1, inf).
            low = self._binary(
                "multiply", blend, _literal(_FLOAT, 2.0), _FLOAT)
            high = self._binary(
                "divide",
                _literal(_FLOAT, 0.5),
                self._binary(
                    "subtract", _literal(_FLOAT, 1.0), blend, _FLOAT),
                _FLOAT,
            )
            exponent = self._compare(
                "ifgreatereq", blend, _literal(_FLOAT, 0.5), high, low)
            return self._binary(
                "subtract",
                _literal(_FLOAT, 1.0),
                self._binary("power", cosine, exponent, _FLOAT),
                _FLOAT,
            )
        # Fresnel output: eta = 1 / (1 - blend), as a normal-incidence
        # reflectance through the same Schlick form used above.
        eta = self._binary(
            "divide",
            _literal(_FLOAT, 1.0),
            self._binary("subtract", _literal(_FLOAT, 1.0), blend, _FLOAT),
            _FLOAT,
        )
        ratio = self._binary(
            "divide",
            self._binary("subtract", eta, _literal(_FLOAT, 1.0), _FLOAT),
            self._binary("add", eta, _literal(_FLOAT, 1.0), _FLOAT),
            _FLOAT,
        )
        return self._schlick(cosine, self._binary("multiply", ratio, ratio, _FLOAT))

    def _export_ShaderNodeNewGeometry(self, node, socket):
        identifier = socket.identifier
        if identifier == "Position":
            return self.doc.node(
                "position", _VECTOR3,
                (("space", "string", _literal("string", "world")),))
        if identifier == "Normal":
            return self.doc.node(
                "normal", _VECTOR3,
                (("space", "string", _literal("string", "world")),))
        if identifier in {"True Normal", "TrueNormal"}:
            return self.doc.node(
                "geometry_normal", _VECTOR3,
                (("space", "string", _literal("string", "world")),))
        if identifier == "Tangent":
            return self.doc.node(
                "tangent", _VECTOR3,
                (("space", "string", _literal("string", "world")),))
        if identifier == "Incoming":
            # Blender's Incoming points from the surface toward the viewer,
            # which is what the renderer stores as viewdirection.
            return self.doc.node("viewdirection", _VECTOR3)
        if identifier == "Backfacing":
            # MaterialX has no direct backface input; derive it from normal/view.
            facing = self.doc.node(
                "dotproduct",
                _FLOAT,
                (
                    ("in1", _VECTOR3, self.doc.node(
                        "geometry_normal", _VECTOR3,
                        (("space", "string", _literal("string", "world")),))),
                    ("in2", _VECTOR3, self.doc.node("viewdirection", _VECTOR3)),
                ),
            )
            return self._compare(
                "ifgreater",
                facing,
                _literal(_FLOAT, 0.0),
                _literal(_FLOAT, 0.0),
                _literal(_FLOAT, 1.0),
            )
        if identifier in {"Random", "Random Per Island"}:
            # MaterialX has no mesh-island identifier stream.  A standard
            # randomfloat driven by position is deterministic and keeps the
            # graph portable, but is necessarily a spatial approximation of
            # Blender's per-object/per-island value.
            position = self.doc.node(
                "position",
                _VECTOR3,
                (('space', "string", _literal("string", "world")),),
            )
            return self.doc.node(
                "randomfloat",
                _FLOAT,
                (
                    ("in", _FLOAT, self.doc.node(
                        "magnitude", _FLOAT, (("in", _VECTOR3, position),))),
                    ("min", _FLOAT, _literal(_FLOAT, 0.0)),
                    ("max", _FLOAT, _literal(_FLOAT, 1.0)),
                    ("seed", _INTEGER, _literal(_INTEGER, 0)),
                ),
            )
        # Pointiness needs mesh-wide preprocessing that default MaterialX does
        # not expose.  Keep the fallback visible to callers as unsupported.
        self.unsupported_nodes.add(f"{node.bl_idname}.{identifier}")
        return self._socket_default(socket, self._socket_type(socket))

    def _export_ShaderNodeLightPath(self, node, socket):
        """Approximate Blender Light Path with portable MaterialX values.

        Default MaterialX intentionally has no ray-type, bounce-depth, or
        shadow-ray query.  The exporter therefore models the common primary
        camera path and derives ray length/depth-like values from standard
        geometric inputs.  This keeps graphs valid and predictable without
        inventing a renderer-specific node or geomprop.
        """
        identifier = socket.identifier
        primary_camera = {
            "Is Camera Ray": True,
            "Is Diffuse Ray": True,
        }
        if identifier in primary_camera:
            return _literal(_BOOLEAN, primary_camera[identifier])
        if identifier == "Ray Length":
            position = self.doc.node(
                "position",
                _VECTOR3,
                (('space', "string", _literal("string", "world")),),
            )
            return self.doc.node(
                "magnitude", _FLOAT, (("in", _VECTOR3, position),))
        if identifier in {
            "Ray Depth",
            "Transparent Depth",
            "Transmission Depth",
        }:
            return _literal(_FLOAT, 0.0)
        if identifier in {
            "Is Shadow Ray",
            "Is Glossy Ray",
            "Is Singular Ray",
            "Is Reflection Ray",
            "Is Transmission Ray",
        }:
            return _literal(_BOOLEAN, False)
        # Blender can add outputs in future versions.  Keep their socket
        # defaults rather than emitting a non-standard MaterialX construct.
        self.unsupported_nodes.add(f"{node.bl_idname}.{identifier}")
        return self._socket_default(socket, self._socket_type(socket))

    def _export_ShaderNodeObjectInfo(self, node, socket):
        del node
        position = self.doc.node(
            "position",
            _VECTOR3,
            (("space", "string", _literal("string", "world")),),
        )
        if socket.identifier == "Location":
            # Default MaterialX has no object-origin stream; world position is
            # the closest standard geometric signal.
            return position
        if socket.identifier in {"Color", "Alpha"}:
            color = self.doc.node(
                "geomcolor",
                _COLOR4,
                (("index", _INTEGER, _literal(_INTEGER, 0)),),
            )
            if socket.identifier == "Color":
                return color
            return self.doc.node(
                "extract",
                _FLOAT,
                (
                    ("in", _COLOR4, color),
                    ("index", _INTEGER, _literal(_INTEGER, 3)),
                ),
            )

        random_input = self.doc.node(
            "magnitude", _FLOAT, (("in", _VECTOR3, position),))
        random_value = self.doc.node(
            "randomfloat",
            _FLOAT,
            (
                ("in", _FLOAT, random_input),
                ("min", _FLOAT, _literal(_FLOAT, 0.0)),
                ("max", _FLOAT, _literal(_FLOAT, 1.0)),
                ("seed", _INTEGER, _literal(_INTEGER, 0)),
            ),
        )
        if socket.identifier in {"Object Index", "Random"}:
            return random_value
        # MaterialX has no material-slot stream. A stable zero is preferable
        # to a private geomprop or a fabricated MaterialX node.
        return _literal(_FLOAT, 0.0)

    def _export_ShaderNodeCameraData(self, node, socket):
        if socket.identifier == "View Vector":
            return self.doc.node("viewdirection", _VECTOR3)
        position = self.doc.node(
            "position",
            _VECTOR3,
            (("space", "string", _literal("string", "world")),),
        )
        if socket.identifier == "View Distance":
            return self.doc.node("magnitude", _FLOAT, (("in", _VECTOR3, position),))
        if socket.identifier == "View Z Depth":
            return self.doc.node(
                "dotproduct",
                _FLOAT,
                (
                    ("in1", _VECTOR3, position),
                    ("in2", _VECTOR3, self.doc.node("viewdirection", _VECTOR3)),
                ),
            )
        self.unsupported_nodes.add(f"{node.bl_idname}.{socket.identifier}")
        return _default(self._socket_type(socket))

    def _export_ShaderNodeVectorTransform(self, node, socket):
        del socket
        vector_type = {
            "POINT": "transformpoint",
            "NORMAL": "transformnormal",
            "VECTOR": "transformvector",
        }.get(getattr(node, "vector_type", "VECTOR"), "transformvector")
        space = lambda name: {
            "OBJECT": "object",
            "WORLD": "world",
            # Default MaterialX has no camera space. Treat Blender camera
            # space as world space so the graph remains valid and useful,
            # while documenting the approximation.
            "CAMERA": "world",
        }.get(name, "world")
        return self.doc.node(
            vector_type,
            _VECTOR3,
            (
                ("in", _VECTOR3, self._input(node, "Vector", _VECTOR3)),
                ("fromspace", "string", _literal(
                    "string", space(getattr(node, "convert_from", "WORLD")))),
                ("tospace", "string", _literal(
                    "string", space(getattr(node, "convert_to", "OBJECT")))),
            ),
        )

    # -- Normal and bump -------------------------------------------------

    def _export_ShaderNodeNormalMap(self, node, socket):
        del socket
        color = self._input(node, "Color", _VECTOR3)
        if getattr(node, "space", "TANGENT") == "TANGENT":
            return self.doc.node(
                "normalmap",
                _VECTOR3,
                (
                    ("in", _VECTOR3, color),
                    ("scale", _FLOAT, self._input(node, "Strength", _FLOAT)),
                ),
            )
        return self._op("normalize", _VECTOR3, color)

    def _export_ShaderNodeBump(self, node, socket):
        del socket
        height = self._input(node, "Height", _FLOAT)
        strength = self._input(node, "Strength", _FLOAT)
        distance = self._input(node, "Distance", _FLOAT)
        scale = self._binary("multiply", strength, distance, _FLOAT)
        if getattr(node, "invert", False):
            scale = self._negate(scale)
        inputs = [("height", _FLOAT, height), ("scale", _FLOAT, scale)]
        normal = self._linked_input(node, "Normal", _VECTOR3)
        if normal is not None:
            inputs.append(("normal", _VECTOR3, normal))
        return self.doc.node("bump", _VECTOR3, inputs)

    # -- Procedural textures --------------------------------------------

    def _texture_vector(self, node, type_name=_VECTOR3) -> _Value:
        return self._linked_input(node, "Vector", type_name) or (
            self._default_texcoord() if type_name == _VECTOR2 else self._default_position()
        )

    def _export_ShaderNodeTexWhiteNoise(self, node, socket):
        position = self._texture_vector(node, _VECTOR3)
        w = self._input(node, "W", _FLOAT)
        w_vector = self.doc.node(
            "combine3",
            _VECTOR3,
            (("in1", _FLOAT, w), ("in2", _FLOAT, w), ("in3", _FLOAT, w)),
        )
        position = self._binary("add", position, w_vector, _VECTOR3)
        if socket.identifier == "Value":
            return self.doc.node(
                "noise3d",
                _FLOAT,
                (
                    ("amplitude", _FLOAT, _literal(_FLOAT, 1.0)),
                    ("pivot", _FLOAT, _literal(_FLOAT, 0.5)),
                    ("position", _VECTOR3, position),
                ),
            )
        color = self.doc.node(
            "noise3d",
            _COLOR3,
            (
                ("amplitude", _VECTOR3, _literal(_VECTOR3, (1.0, 1.0, 1.0))),
                ("pivot", _FLOAT, _literal(_FLOAT, 0.5)),
                ("position", _VECTOR3, position),
            ),
        )
        return self.doc.node(
            "combine4",
            _COLOR4,
            (
                ("in1", _FLOAT, self.doc.extract(color, 0)),
                ("in2", _FLOAT, self.doc.extract(color, 1)),
                ("in3", _FLOAT, self.doc.extract(color, 2)),
                ("in4", _FLOAT, _literal(_FLOAT, 1.0)),
            ),
        )

    def _export_ShaderNodeTexChecker(self, node, socket):
        vector = self._texture_vector(node, _VECTOR3)
        scaled = self._binary(
            "multiply", vector, self._input(node, "Scale", _FLOAT), _VECTOR3
        )
        floors = [
            self._op("floor", _FLOAT, self.doc.extract(scaled, index))
            for index in range(3)
        ]
        parity = self._binary(
            "modulo",
            self._binary(
                "add",
                self._binary("add", floors[0], floors[1], _FLOAT),
                floors[2],
                _FLOAT,
            ),
            _literal(_FLOAT, 2.0),
            _FLOAT,
        )
        if socket.type == "VALUE":
            first, second = _literal(_FLOAT, 1.0), _literal(_FLOAT, 0.0)
        else:
            first = self._input(node, "Color1", _COLOR3)
            second = self._input(node, "Color2", _COLOR3)
        return self._compare(
            "ifequal", parity, _literal(_FLOAT, 1.0), first, second
        )

    def _export_ShaderNodeTexGradient(self, node, socket):
        vector = self._texture_vector(node, _VECTOR3)
        x, y = self.doc.extract(vector, 0), self.doc.extract(vector, 1)
        gradient = node.gradient_type
        if gradient == "QUADRATIC":
            result = self._binary("multiply", x, x, _FLOAT)
        elif gradient == "EASING":
            x = self._clamp(x)
            result = self._binary(
                "multiply",
                self._binary("multiply", x, x, _FLOAT),
                self._binary(
                    "subtract",
                    _literal(_FLOAT, 3.0),
                    self._binary("multiply", _literal(_FLOAT, 2.0), x, _FLOAT),
                    _FLOAT,
                ),
                _FLOAT,
            )
        elif gradient == "DIAGONAL":
            result = self._binary(
                "multiply",
                self._binary("add", x, y, _FLOAT),
                _literal(_FLOAT, 0.5),
                _FLOAT,
            )
        elif gradient == "RADIAL":
            result = self._binary(
                "add",
                self._binary(
                    "divide",
                    self._binary("atan2", y, x, _FLOAT),
                    _literal(_FLOAT, math.tau),
                    _FLOAT,
                ),
                _literal(_FLOAT, 0.5),
                _FLOAT,
            )
        elif gradient in {"SPHERICAL", "QUADRATIC_SPHERE"}:
            length = self.doc.node(
                "magnitude", _FLOAT, (("in", _VECTOR3, vector),)
            )
            result = self._binary(
                "max",
                self._binary(
                    "subtract", _literal(_FLOAT, 1.0), length, _FLOAT
                ),
                _literal(_FLOAT, 0.0),
                _FLOAT,
            )
            if gradient == "QUADRATIC_SPHERE":
                result = self._binary("multiply", result, result, _FLOAT)
        else:
            result = x
        return (
            result
            if socket.type == "VALUE"
            else self.doc.convert(result, _COLOR3)
        )

    def _export_ShaderNodeTexWave(self, node, socket):
        vector = self._texture_vector(node, _VECTOR3)
        position = self._binary(
            "multiply", vector, self._input(node, "Scale", _FLOAT), _VECTOR3
        )
        if node.wave_type == "RINGS":
            mask = {
                "X": (0.0, 1.0, 1.0),
                "Y": (1.0, 0.0, 1.0),
                "Z": (1.0, 1.0, 0.0),
            }.get(node.rings_direction, (1.0, 1.0, 1.0))
            ring_position = self._binary(
                "multiply", position, _literal(_VECTOR3, mask), _VECTOR3
            )
            wave = self._binary(
                "multiply",
                self.doc.node(
                    "magnitude", _FLOAT, (("in", _VECTOR3, ring_position),)
                ),
                _literal(_FLOAT, 20.0),
                _FLOAT,
            )
        elif node.bands_direction == "DIAGONAL":
            wave = self._binary(
                "multiply",
                self._binary(
                    "add",
                    self._binary(
                        "add",
                        self.doc.extract(position, 0),
                        self.doc.extract(position, 1),
                        _FLOAT,
                    ),
                    self.doc.extract(position, 2),
                    _FLOAT,
                ),
                _literal(_FLOAT, 10.0),
                _FLOAT,
            )
        else:
            index = {"X": 0, "Y": 1, "Z": 2}.get(node.bands_direction, 0)
            wave = self._binary(
                "multiply",
                self.doc.extract(position, index),
                _literal(_FLOAT, 20.0),
                _FLOAT,
            )
        detail_socket = self._input_socket(node, "Detail")
        octaves = max(1, min(13, int(getattr(detail_socket, "default_value", 2.0))))
        fractal = self.doc.node(
            "fractal3d",
            _FLOAT,
            (
                ("position", _VECTOR3, position),
                ("octaves", _INTEGER, _literal(_INTEGER, octaves)),
                ("lacunarity", _FLOAT, _literal(_FLOAT, 2.0)),
                ("diminish", _FLOAT, self._input(node, "Detail Roughness", _FLOAT)),
            ),
        )
        distortion = self._binary(
            "multiply",
            self._input(node, "Distortion", _FLOAT),
            _literal(_FLOAT, 10.0),
            _FLOAT,
        )
        detail_scale = self._binary(
            "multiply",
            self._input(node, "Detail Scale", _FLOAT),
            _literal(_FLOAT, 10.0),
            _FLOAT,
        )
        wave = self._binary(
            "add",
            self._binary("add", wave, self._input(node, "Phase Offset", _FLOAT), _FLOAT),
            self._binary(
                "multiply",
                self._binary("multiply", distortion, detail_scale, _FLOAT),
                fractal,
                _FLOAT,
            ),
            _FLOAT,
        )
        if node.wave_profile == "SAW":
            normalized = self._binary(
                "divide", wave, _literal(_FLOAT, math.tau), _FLOAT
            )
            result = self._binary(
                "subtract", normalized, self._op("floor", _FLOAT, normalized), _FLOAT
            )
        elif node.wave_profile == "TRI":
            normalized = self._binary(
                "divide", wave, _literal(_FLOAT, math.tau), _FLOAT
            )
            result = self._binary(
                "multiply",
                self._op(
                    "absval",
                    _FLOAT,
                    self._binary(
                        "subtract",
                        normalized,
                        self._op(
                            "floor",
                            _FLOAT,
                            self._binary(
                                "add", normalized, _literal(_FLOAT, 0.5), _FLOAT
                            ),
                        ),
                        _FLOAT,
                    ),
                ),
                _literal(_FLOAT, 2.0),
                _FLOAT,
            )
        else:
            result = self._binary(
                "add",
                _literal(_FLOAT, 0.5),
                self._binary(
                    "multiply",
                    _literal(_FLOAT, 0.5),
                    self._op(
                        "sin",
                        _FLOAT,
                        self._binary(
                            "subtract",
                            wave,
                            _literal(_FLOAT, math.pi / 2.0),
                            _FLOAT,
                        ),
                    ),
                    _FLOAT,
                ),
                _FLOAT,
            )
        return result if socket.type == "VALUE" else self.doc.convert(result, _COLOR3)

    def _export_ShaderNodeTexNoise(self, node, socket):
        position = self._texture_vector(node, _VECTOR3)
        position = self._binary(
            "multiply", position, self._input(node, "Scale", _FLOAT), _VECTOR3
        )
        detail_socket = self._input_socket(node, "Detail")
        octaves = max(1, min(13, int(getattr(detail_socket, "default_value", 2.0))))
        type_name = _FLOAT if socket.type == "VALUE" else _COLOR3
        return self.doc.node(
            "fractal3d",
            type_name,
            (
                ("position", _VECTOR3, position),
                ("octaves", _INTEGER, _literal(_INTEGER, octaves)),
                ("lacunarity", _FLOAT, self._input(node, "Lacunarity", _FLOAT)),
                ("diminish", _FLOAT, self._input(node, "Roughness", _FLOAT)),
            ),
        )

    def _export_ShaderNodeTexVoronoi(self, node, socket):
        dimensions = getattr(node, "voronoi_dimensions", "3D")
        vector_type = _VECTOR2 if dimensions == "2D" else _VECTOR3
        coordinate = self._texture_vector(node, vector_type)
        coordinate = self._binary(
            "multiply", coordinate, self._input(node, "Scale", _FLOAT), vector_type
        )
        category = "worleynoise2d" if dimensions == "2D" else "worleynoise3d"
        coordinate_name = "texcoord" if dimensions == "2D" else "position"
        if socket.identifier in {"Color", "Position"}:
            return self.doc.node(
                category,
                _VECTOR3,
                (
                    (coordinate_name, vector_type, coordinate),
                    ("jitter", _FLOAT, self._input(node, "Randomness", _FLOAT)),
                    ("style", _INTEGER, _literal(_INTEGER, 1)),
                ),
            )
        return self.doc.node(
            category,
            _FLOAT,
            (
                (coordinate_name, vector_type, coordinate),
                ("jitter", _FLOAT, self._input(node, "Randomness", _FLOAT)),
                ("style", _INTEGER, _literal(_INTEGER, 0)),
            ),
        )

    # -- Separate/combine and map range ---------------------------------

    def _export_ShaderNodeSeparateXYZ(self, node, socket):
        index = {"X": 0, "Y": 1, "Z": 2}.get(socket.identifier, 0)
        return self.doc.extract(self._input(node, "Vector", _VECTOR3), index)

    def _export_ShaderNodeCombineXYZ(self, node, socket):
        del socket
        return self.doc.node(
            "combine3",
            _VECTOR3,
            tuple(
                (f"in{index + 1}", _FLOAT, self._input(node, name, _FLOAT))
                for index, name in enumerate(("X", "Y", "Z"))
            ),
        )

    def _export_ShaderNodeSeparateColor(self, node, socket):
        color = self._input(node, "Color", _COLOR3)
        mode = getattr(node, "mode", "RGB")
        if mode in {"HSV", "HSL"}:
            color = self.doc.node("rgbtohsv", _COLOR3, (("in", _COLOR3, color),))
        index = {"Red": 0, "Green": 1, "Blue": 2}.get(socket.identifier, 0)
        return self.doc.extract(color, index)

    def _export_ShaderNodeCombineColor(self, node, socket):
        del socket
        color = self.doc.node(
            "combine3",
            _COLOR3,
            tuple(
                (f"in{index + 1}", _FLOAT, self._input(node, name, _FLOAT))
                for index, name in enumerate(("Red", "Green", "Blue"))
            ),
        )
        if getattr(node, "mode", "RGB") in {"HSV", "HSL"}:
            color = self.doc.node("hsvtorgb", _COLOR3, (("in", _COLOR3, color),))
        return color

    def _export_ShaderNodeMapRange(self, node, socket):
        vector_mode = node.data_type == "FLOAT_VECTOR" or socket.type == "VECTOR"
        type_name = _VECTOR3 if vector_mode else _FLOAT
        names = (
            ("Vector", "From_Min_FLOAT3", "From_Max_FLOAT3", "To_Min_FLOAT3", "To_Max_FLOAT3", "Steps_FLOAT3")
            if vector_mode
            else ("Value", "From Min", "From Max", "To Min", "To Max", "Steps")
        )
        value, low, high, out_low, out_high, steps = (
            self._input(node, name, type_name) for name in names
        )
        interpolation = node.interpolation_type
        if interpolation == "LINEAR":
            result = self.doc.node(
                "remap",
                type_name,
                (
                    ("in", type_name, value),
                    ("inlow", type_name, low),
                    ("inhigh", type_name, high),
                    ("outlow", type_name, out_low),
                    ("outhigh", type_name, out_high),
                ),
            )
        elif interpolation == "STEPPED":
            factor = self.doc.node(
                "remap",
                type_name,
                (
                    ("in", type_name, value),
                    ("inlow", type_name, low),
                    ("inhigh", type_name, high),
                ),
            )
            quantized = self._binary(
                "divide",
                self._op(
                    "floor",
                    type_name,
                    self._binary(
                        "multiply",
                        factor,
                        self._binary(
                            "add", steps, _literal(_FLOAT, 1.0), type_name
                        ),
                        type_name,
                    ),
                ),
                steps,
                type_name,
            )
            result = self.doc.node(
                "remap",
                type_name,
                (
                    ("in", type_name, quantized),
                    ("outlow", type_name, out_low),
                    ("outhigh", type_name, out_high),
                ),
            )
        else:
            factor = self.doc.node(
                "smoothstep",
                type_name,
                (
                    ("in", type_name, value),
                    ("low", type_name, low),
                    ("high", type_name, high),
                ),
            )
            result = self.doc.node(
                "remap",
                type_name,
                (
                    ("in", type_name, factor),
                    ("outlow", type_name, out_low),
                    ("outhigh", type_name, out_high),
                ),
            )
        if getattr(node, "clamp", False):
            result = self._clamp(result, out_low, out_high)
        return result


def export_material(
    material,
    image_path_resolver: Optional[Callable] = None,
    *,
    transport_nonce: int = 0,
) -> ExportedMaterial:
    """Translate one evaluated Blender material into compact MaterialX XML."""
    converter = _Converter(material, image_path_resolver, transport_nonce)
    document = converter.doc.serialize(converter.surface())
    return ExportedMaterial(
        document,
        frozenset(converter.image_pointers),
        frozenset(converter.unsupported_nodes),
    )
