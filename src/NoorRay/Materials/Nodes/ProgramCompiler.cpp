#include "NoorRay/Materials/Nodes/ProgramCompiler.h"

#include <algorithm>
#include <array>
#include <format>
#include <string_view>

#include "NoorRay/Shaders/Materials/Program.h"

namespace NoorRay::Materials {

namespace {

constexpr std::array<std::string_view, 41> mathOperations = {
    "ADD",       "SUBTRACT",   "MULTIPLY",  "DIVIDE",         "MULTIPLY_ADD", "POWER",   "LOGARITHM",  "SQRT",       "INVERSE_SQRT", "ABSOLUTE", "EXPONENT",
    "MINIMUM",   "MAXIMUM",    "LESS_THAN", "GREATER_THAN",   "SIGN",         "COMPARE", "SMOOTH_MIN", "SMOOTH_MAX", "ROUND",        "FLOOR",    "CEIL",
    "TRUNC",     "FRACT",      "MODULO",    "FLOORED_MODULO", "WRAP",         "SNAP",    "PINGPONG",   "SINE",       "COSINE",       "TANGENT",  "ARCSINE",
    "ARCCOSINE", "ARCTANGENT", "ARCTAN2",   "SINH",           "COSH",         "TANH",    "RADIANS",    "DEGREES"};

constexpr std::array<std::string_view, 29> vectorMathOperations = {
    "ADD",         "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD", "CROSS_PRODUCT", "PROJECT", "REFLECT", "REFRACT", "FACEFORWARD",
    "DOT_PRODUCT", "DISTANCE", "LENGTH",   "SCALE",  "NORMALIZE",    "ABSOLUTE",      "POWER",   "SIGN",    "MINIMUM", "MAXIMUM",
    "FLOOR",       "CEIL",     "FRACTION", "MODULO", "WRAP",         "SNAP",          "SINE",    "COSINE",  "TANGENT"};

static_assert(mathOperations.size() == Shaders::mathDegrees + 1);
static_assert(vectorMathOperations.size() == Shaders::vectorMathTangent + 1);

template <std::size_t Size>
std::optional<std::uint32_t> indexOf(const std::array<std::string_view, Size>& names, const std::string_view name) {
    const auto found = std::ranges::find(names, name);
    if (found == names.end())
        return std::nullopt;
    return static_cast<std::uint32_t>(found - names.begin());
}

std::optional<std::uint32_t> direction(const std::string_view name, const std::string_view last) {
    constexpr std::array<std::string_view, 3> axes = {"X", "Y", "Z"};
    if (name == last)
        return 3;
    return indexOf(axes, name);
}

std::optional<std::uint32_t> waveProfile(const std::string_view name) {
    constexpr std::array<std::string_view, 3> profiles = {"SIN", "SAW", "TRI"};
    return indexOf(profiles, name);
}

}

void ProgramCompiler::fail(std::string message) {
    if (!failure)
        failure = std::move(message);
}

ProgramCompiler::Register ProgramCompiler::allocate(const Core::float4 value) {
    if (program.registers.size() >= Shaders::maxProgramRegisters) {
        fail(std::format("the displacement needs more than {} registers", Shaders::maxProgramRegisters));
        return 0;
    }
    program.registers.push_back(value);
    return static_cast<Register>(program.registers.size() - 1);
}

ProgramCompiler::Register ProgramCompiler::constant(const Core::float4 value) {
    const std::array<float, 4> key{value.x, value.y, value.z, value.w};
    if (const auto found = constants.find(key); found != constants.end())
        return found->second;
    const Register result = allocate(value);
    constants.emplace(key, result);
    return result;
}

ProgramCompiler::Register ProgramCompiler::scalar(const float value) {
    return constant({value, value, value, value});
}

ProgramCompiler::Register ProgramCompiler::emit(const std::uint32_t op, const std::uint32_t mode, const std::initializer_list<Register> inputs) {
    std::array<std::uint32_t, 4> operands{};
    std::ranges::copy(inputs, operands.begin());
    const Register result = allocate({});
    program.instructions.push_back({op | mode << 8 | std::uint32_t{result} << 16 | operands[0] << 24, operands[1] | operands[2] << 8 | operands[3] << 16});
    return result;
}

ProgramCompiler::Register ProgramCompiler::input(const SocketRef socket, const std::function<Register()>& unlinked) {
    const NodeLinkEnd link = links.linkedNode(socket);
    if (!link.node) {
        if (unlinked && !links.isLinked(socket))
            return unlinked();
        const NodeValues value = socketValues(socket);
        if (value.count == 1)
            return scalar(value.values[0]);
        return constant({value.values[0], value.values[1], value.values[2], value.values[3]});
    }
    const Register result = output(link.node, link.socket);
    const std::string_view to = socketTypeName(socket->type);
    const std::string_view from = socketTypeName(link.socket->type);
    if (to == "VALUE" && from == "VECTOR")
        return emit(Shaders::programOpAverage, 0, {result});
    if (to == "VALUE" && from == "RGBA")
        return emit(Shaders::programOpGrayscale, 0, {result});
    return result;
}

ProgramCompiler::Register ProgramCompiler::input(const NodeRef node, const char* name, const std::function<Register()>& unlinked) {
    return input(inputNamed(node, name), unlinked);
}

ProgramCompiler::Register ProgramCompiler::inputAt(const NodeRef node, const std::size_t index) {
    return input(Materials::inputAt(node, index));
}

ProgramCompiler::Register ProgramCompiler::output(const NodeRef node, const SocketRef socket) {
    const std::pair<std::uintptr_t, std::string> key{reinterpret_cast<std::uintptr_t>(&node.get()), socket->identifier};
    if (const auto found = outputs.find(key); found != outputs.end())
        return found->second;
    const Register result = compileOutput(node, socket);
    outputs.emplace(key, result);
    return result;
}

ProgramCompiler::Register ProgramCompiler::compileOutput(const NodeRef node, const SocketRef socket) {
    const std::string_view kind = nodeType(node);
    const std::string name = node->name;
    if (kind == "VALUE")
        return scalar(socketNumber(outputAt(node, 0)));
    if (kind == "MATH") {
        const std::optional<std::uint32_t> operation = indexOf(mathOperations, text(node, "operation"));
        if (!operation) {
            fail(std::format("the displacement uses the math operation {}", text(node, "operation")));
            return 0;
        }
        const std::uint32_t mode = *operation | (flag(node, "use_clamp") ? Shaders::mathClamp : 0);
        return emit(Shaders::programOpMath, mode, {inputAt(node, 0), inputAt(node, 1), inputAt(node, 2)});
    }
    if (kind == "VECT_MATH") {
        const std::optional<std::uint32_t> operation = indexOf(vectorMathOperations, text(node, "operation"));
        if (!operation) {
            fail(std::format("the displacement uses the vector math operation {}", text(node, "operation")));
            return 0;
        }
        return emit(Shaders::programOpVectorMath, *operation, {inputAt(node, 0), inputAt(node, 1), inputAt(node, 2), inputAt(node, 3)});
    }
    if (kind == "TEX_WAVE")
        return wave(node);
    if (kind == "COMBXYZ")
        return emit(Shaders::programOpCombine, 0, {input(node, "X"), input(node, "Y"), input(node, "Z"), scalar(0.0f)});
    if (kind == "SEPXYZ") {
        constexpr std::array<std::string_view, 3> axes = {"X", "Y", "Z"};
        return emit(Shaders::programOpSeparate, indexOf(axes, socket->name).value_or(0), {input(node, "Vector")});
    }
    if (kind == "ATTRIBUTE")
        return attribute(node, socket);
    const std::string output = socket->name;
    if (kind == "TEX_COORD") {
        if (output == "Generated")
            return emit(Shaders::programOpGenerated);
        if (output == "Normal")
            return emit(Shaders::programOpNormal);
        if (output == "Object" && text(node, "object").empty())
            return emit(Shaders::programOpPosition);
        fail(std::format("the displacement reads the texture coordinate '{}' of '{}'", output, name));
        return 0;
    }
    if (kind == "NEW_GEOMETRY") {
        if (output == "Position")
            return emit(Shaders::programOpPosition);
        if (output == "Normal")
            return emit(Shaders::programOpNormal);
        fail(std::format("the displacement reads the geometry output '{}' of '{}'", output, name));
        return 0;
    }
    fail(std::format("the displacement uses the node '{}' ({}), which NoorRay programs do not support", name, kind));
    return 0;
}

ProgramCompiler::Register ProgramCompiler::wave(const NodeRef node) {
    const bool bands = text(node, "wave_type") == "BANDS";
    const std::optional<std::uint32_t> axis =
        bands ? direction(text(node, "bands_direction"), "DIAGONAL") : direction(text(node, "rings_direction"), "SPHERICAL");
    const std::optional<std::uint32_t> profile = waveProfile(text(node, "wave_profile"));
    if (!axis || !profile) {
        fail(std::format("the wave texture '{}' has an unknown direction or profile", node->name));
        return 0;
    }
    const std::uint32_t mode = (bands ? 0u : 1u) + 2 * *axis + 8 * *profile;
    const Register vector = input(node, "Vector", [this] { return emit(Shaders::programOpGenerated); });
    const Register settings = pack(node, {"Scale", "Distortion", "Detail", "Detail Scale"});
    const Register shape = pack(node, {"Detail Roughness", "Phase Offset"});
    return emit(Shaders::programOpWave, mode, {vector, settings, shape});
}

ProgramCompiler::Register ProgramCompiler::pack(const NodeRef node, const std::initializer_list<const char*> names) {
    bool linked = false;
    std::array<float, 4> values{};
    std::size_t index = 0;
    for (const char* name : names) {
        const SocketRef socket = inputNamed(node, name);
        linked = linked || links.isLinked(socket);
        values[index++] = socketNumber(socket);
    }
    if (!linked)
        return constant({values[0], values[1], values[2], values[3]});
    std::array<Register, 4> registers{};
    index = 0;
    for (const char* name : names)
        registers[index++] = input(node, name);
    for (; index < registers.size(); ++index)
        registers[index] = scalar(0.0f);
    return emit(Shaders::programOpCombine, 0, {registers[0], registers[1], registers[2], registers[3]});
}

ProgramCompiler::Register ProgramCompiler::attribute(const NodeRef node, const SocketRef socket) {
    const std::string name = text(node, "attribute_name");
    if (text(node, "attribute_type") != "GEOMETRY" || name.empty()) {
        fail(std::format("the displacement reads the attribute node '{}', which is not a named geometry attribute", node->name));
        return 0;
    }
    auto found = std::ranges::find(program.attributes, name);
    if (found == program.attributes.end())
        found = program.attributes.insert(program.attributes.end(), name);
    const std::pair<std::uintptr_t, std::string> key{reinterpret_cast<std::uintptr_t>(&node.get()), "Vector"};
    auto vector = outputs.find(key);
    if (vector == outputs.end())
        vector = outputs.emplace(key, emit(Shaders::programOpAttribute, static_cast<std::uint32_t>(found - program.attributes.begin()))).first;
    const std::string output = socket->name;
    if (output == "Vector" || output == "Color")
        return vector->second;
    if (output == "Fac")
        return emit(Shaders::programOpAverage, 0, {vector->second});
    fail(std::format("the displacement reads the output '{}' of the attribute node '{}'", output, node->name));
    return 0;
}

ProgramCompiler::Register ProgramCompiler::displacement(const NodeRef node) {
    const std::string_view kind = nodeType(node);
    const std::string name = node->name;
    if (kind != "DISPLACEMENT" && kind != "VECTOR_DISPLACEMENT") {
        fail(std::format("the displacement comes from '{}' ({}), not a Displacement or Vector Displacement node", name, kind));
        return 0;
    }
    const std::string space = text(node, "space");
    if (space != "OBJECT") {
        fail(std::format("the displacement node '{}' is in {} space; NoorRay displaces in object space", name, space));
        return 0;
    }
    if (kind == "VECTOR_DISPLACEMENT")
        return emit(Shaders::programOpVectorDisplacement, 0, {input(node, "Vector"), input(node, "Midlevel"), input(node, "Scale")});
    const Register normal = input(node, "Normal", [this] { return emit(Shaders::programOpNormal); });
    return emit(Shaders::programOpDisplacement, 0, {input(node, "Height"), input(node, "Midlevel"), input(node, "Scale"), normal});
}

std::expected<std::optional<Materials::Program::Source>, std::string> ProgramCompiler::compile(const NodeRef output) {
    const NodeLinkEnd link = links.linkedNode(inputNamed(output, "Displacement"));
    if (!link.node)
        return std::nullopt;
    program.displacement = displacement(link.node);
    if (failure)
        return std::unexpected(std::move(*failure));
    return std::move(program);
}

}
