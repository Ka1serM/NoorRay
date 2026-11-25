#include "NoorRay/Materials/GraphShader.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRHI/Shaders/WgslModule.h"
#include "NoorRay/Core/Hash.h"

namespace NoorRay::Materials {

namespace {

//! The donor names the context member n as @member<n>; the splice puts in the member names of the host, because slangc adds a suffix to each.
constexpr std::string_view memberMarker = "@member";

std::string floatType(const std::uint8_t width) {
    return width == 1 ? "f32" : std::format("vec{}<f32>", width);
}

//! A literal for any float; a value that is not finite is a bit cast, because WGSL has no literal for it.
std::string floatLiteral(const float value) {
    if (!std::isfinite(value))
        return std::format("bitcast<f32>({}u)", std::bit_cast<std::uint32_t>(value));
    std::string text = std::format("{:.9g}", value);
    if (text.find_first_of(".e") == std::string::npos)
        text += ".0";
    return text;
}

//! Writes the body of one graph; each value is a let, so that WGSL never evaluates an operation on constants when it creates the shader.
class Body {
public:
    explicit Body(const Graph& graph) : graph(graph) {}

    std::string write() {
        const std::span<const GraphInstruction> instructions = graph.getInstructions();
        std::vector<bool> live(instructions.size(), false);
        for (std::size_t output = 0; output < graphOutputCount; ++output)
            live[graph.getOutput(static_cast<GraphOutput>(output))] = true;
        for (std::size_t at = instructions.size(); at-- > 0;)
            if (live[at])
                for (const std::uint32_t operand : graph.operands(instructions[at]))
                    live[operand] = true;
        values.assign(instructions.size(), {});
        for (std::size_t at = 0; at < instructions.size(); ++at)
            if (live[at])
                values[at] = let(instructions[at].width, value(instructions[at]));
        std::string outputs;
        for (std::size_t output = 0; output + 1 < graphOutputCount; ++output)
            outputs += values[graph.getOutput(static_cast<GraphOutput>(output))] + ", ";
        outputs += narrow(values[graph.getOutput(GraphOutput::Transparency)], 4, 1);
        text += std::format("    return MaterialGraphOutput({});\n", outputs);
        return std::move(text);
    }

private:
    std::string let(const std::uint8_t width, const std::string& expression) {
        const std::string name = std::format("v{}", count++);
        text += std::format("    let {} : {} = {};\n", name, floatType(width), expression);
        return name;
    }

    static std::string splat(const std::string& value, const std::uint8_t from, const std::uint8_t to) {
        if (from == to)
            return value;
        return std::format("{}({})", floatType(to), value);
    }

    static std::string narrow(const std::string& value, const std::uint8_t from, const std::uint8_t to, const std::uint32_t first = 0) {
        if (from == to)
            return value;
        return std::format("{}.{}", value, std::string_view("xyzw").substr(first, to));
    }

    static std::string zero(const std::uint8_t width) { return splat("0.0", 1, width); }

    static std::string call(const std::string_view function, const std::initializer_list<std::string> arguments) {
        std::string list;
        for (const std::string& argument : arguments)
            list += (list.empty() ? "" : ", ") + argument;
        return std::format("{}({})", function, list);
    }

    //! The result where the test of a against zero holds, else zero.
    std::string guarded(const std::string_view test, const std::string& a, const std::string& result, const std::uint8_t width) {
        return std::format("select({}, {}, {} {} {})", zero(width), result, a, test, zero(width));
    }

    static std::string input(const GraphInput input, const std::uint8_t width) {
        const GraphInputLayout layout = graphInputLayout(input);
        return narrow(std::format("shadingContext.{}{}", memberMarker, layout.member), 4, width, layout.component);
    }

    std::string value(const GraphInstruction& instruction) {
        const std::span<const std::uint32_t> operands = graph.operands(instruction);
        const std::uint8_t width = instruction.width;
        const auto widthOf = [&](const std::size_t at) { return graph.getInstructions()[operands[at]].width; };
        const auto raw = [&](const std::size_t at) { return values[operands[at]]; };
        const auto operand = [&](const std::size_t at) { return let(width, splat(values[operands[at]], widthOf(at), width)); };
        const auto unary = [&](const std::string_view function) { return call(function, {operand(0)}); };
        const auto binary = [&](const std::string_view function) { return call(function, {operand(0), operand(1)}); };
        const auto ternary = [&](const std::string_view function) { return call(function, {operand(0), operand(1), operand(2)}); };
        switch (instruction.op) {
        case GraphOp::Constant: {
            const std::span<const float> constants = graph.getConstants().subspan(instruction.index, width);
            if (width == 1)
                return floatLiteral(constants[0]);
            std::string list;
            for (const float constant : constants)
                list += (list.empty() ? "" : ", ") + floatLiteral(constant);
            return std::format("{}({})", floatType(width), list);
        }
        case GraphOp::Input:
            return input(static_cast<GraphInput>(instruction.index), width);
        case GraphOp::Parameter:
            return narrow(call("graphParameter", {"material", floatLiteral(static_cast<float>(instruction.index))}), 4, width);
        case GraphOp::Texture:
            return call("graphTexture", {"material", raw(1), floatLiteral(static_cast<float>(instruction.index)), raw(0)});
        case GraphOp::Call: {
            std::string list;
            for (const std::uint32_t argument : operands)
                list += (list.empty() ? "" : ", ") + values[argument];
            return std::format("{}({})", graphFunctionInfo(static_cast<GraphFunction>(instruction.index)).name, list);
        }
        case GraphOp::Extract:
            return narrow(raw(0), widthOf(0), 1, instruction.index);
        case GraphOp::Combine: {
            std::string list;
            for (const std::uint32_t part : operands)
                list += (list.empty() ? "" : ", ") + values[part];
            return std::format("{}({})", floatType(width), list);
        }
        case GraphOp::Add:
            return std::format("{} + {}", operand(0), operand(1));
        case GraphOp::Sub:
            return std::format("{} - {}", operand(0), operand(1));
        case GraphOp::Mul:
            return std::format("{} * {}", operand(0), operand(1));
        case GraphOp::Div: {
            const std::string divisor = operand(1);
            return guarded("!=", divisor, std::format("{} / {}", operand(0), divisor), width);
        }
        case GraphOp::Min:
            return binary("min");
        case GraphOp::Max:
            return binary("max");
        case GraphOp::Pow: {
            const std::string base = operand(0);
            const std::string exponent = operand(1);
            const std::string magnitude = let(width, std::format("pow(abs({}), {})", base, exponent));
            const std::string half = let(width, std::format("{} * 0.5", exponent));
            const std::string negated = std::format("select({0}, -{0}, trunc({1}) != {1})", magnitude, half);
            const std::string integer = std::format("select({0}, {1}, trunc({2}) == {2})", zero(width), negated, exponent);
            return std::format("select({}, {}, {} < {})", magnitude, integer, base, zero(width));
        }
        case GraphOp::Atan2:
            return binary("atan2");
        case GraphOp::Modulo: {
            const std::string divisor = operand(1);
            return guarded("!=", divisor, std::format("{} % {}", operand(0), divisor), width);
        }
        case GraphOp::FlooredModulo: {
            const std::string dividend = operand(0);
            const std::string divisor = operand(1);
            return guarded("!=", divisor, std::format("{0} - {1} * floor({0} / {1})", dividend, divisor), width);
        }
        case GraphOp::Abs:
            return unary("abs");
        case GraphOp::Floor:
            return unary("floor");
        case GraphOp::Ceil:
            return unary("ceil");
        case GraphOp::Fract:
            return unary("fract");
        case GraphOp::Round: {
            const std::string x = operand(0);
            return std::format("sign({0}) * floor(abs({0}) + 0.5)", x);
        }
        case GraphOp::Trunc:
            return unary("trunc");
        case GraphOp::Sign:
            return unary("sign");
        case GraphOp::Sqrt: {
            const std::string x = operand(0);
            return guarded(">", x, std::format("sqrt(max({}, {}))", x, zero(width)), width);
        }
        case GraphOp::InverseSqrt: {
            const std::string x = operand(0);
            return guarded(">", x, std::format("inverseSqrt(max({}, {}))", x, splat("1e-30", 1, width)), width);
        }
        case GraphOp::Exp:
            return unary("exp");
        case GraphOp::Log: {
            const std::string x = operand(0);
            return guarded(">", x, std::format("log(max({}, {}))", x, splat("1e-30", 1, width)), width);
        }
        case GraphOp::Sin:
            return unary("sin");
        case GraphOp::Cos:
            return unary("cos");
        case GraphOp::Tan:
            return unary("tan");
        case GraphOp::Asin:
        case GraphOp::Acos:
            return std::format("{}(clamp({}, {}, {}))", instruction.op == GraphOp::Asin ? "asin" : "acos", operand(0), splat("-1.0", 1, width),
                               splat("1.0", 1, width));
        case GraphOp::Atan:
            return unary("atan");
        case GraphOp::Sinh:
            return unary("sinh");
        case GraphOp::Cosh:
            return unary("cosh");
        case GraphOp::Tanh:
            return unary("tanh");
        case GraphOp::Less:
        case GraphOp::Greater:
            return std::format("select({}, {}, {} {} {})", zero(width), splat("1.0", 1, width), operand(0), instruction.op == GraphOp::Less ? "<" : ">",
                               operand(1));
        case GraphOp::Select: {
            const std::string condition = operand(0);
            return std::format("select({}, {}, {} > {})", operand(2), operand(1), condition, splat("0.5", 1, width));
        }
        case GraphOp::Dot:
            if (widthOf(0) == 1)
                return std::format("{} * {}", raw(0), raw(1));
            return call("dot", {raw(0), raw(1)});
        case GraphOp::Cross:
            return call("cross", {raw(0), raw(1)});
        case GraphOp::Length:
            return call("length", {raw(0)});
        case GraphOp::Distance:
            return call("distance", {raw(0), raw(1)});
        case GraphOp::Normalize: {
            const std::string size = let(width, splat(std::format("length({})", raw(0)), 1, width));
            return guarded(">", size, std::format("{} / max({}, {})", raw(0), size, splat("1e-30", 1, width)), width);
        }
        case GraphOp::Reflect:
            return call("reflect", {raw(0), raw(1)});
        case GraphOp::Refract:
            return call("refract", {raw(0), raw(1), raw(2)});
        case GraphOp::FaceForward:
            return call("faceForward", {raw(0), raw(1), raw(2)});
        case GraphOp::Clamp: {
            const std::string low = operand(1);
            const std::string high = operand(2);
            return std::format("clamp({}, min({}, {}), max({}, {}))", operand(0), low, high, low, high);
        }
        case GraphOp::Mix:
            return ternary("mix");
        case GraphOp::Smoothstep:
            return ternary("smoothstep");
        case GraphOp::MulAdd:
            return std::format("{} * {} + {}", operand(0), operand(1), operand(2));
        }
        return zero(width);
    }

    const Graph& graph;
    std::vector<std::string> values;
    std::string text;
    std::size_t count = 0;
};

//! The member names of the struct whose name is base and a slangc suffix, in order.
std::optional<std::vector<std::string_view>> structMembers(const std::string_view text, const std::string_view base) {
    for (std::size_t at = text.find("struct "); at != std::string_view::npos; at = text.find("struct ", at + 1)) {
        if (at != 0 && text[at - 1] != '\n')
            continue;
        const std::size_t name = at + 7;
        const std::size_t nameEnd = text.find_first_of(" \n{", name);
        const std::string_view found = text.substr(name, nameEnd - name);
        const std::size_t underscore = found.rfind('_');
        if (underscore == std::string_view::npos || found.substr(0, underscore) != base)
            continue;
        const std::size_t open = text.find('{', nameEnd);
        const std::size_t close = text.find('}', open);
        if (open == std::string_view::npos || close == std::string_view::npos)
            return std::nullopt;
        std::vector<std::string_view> members;
        for (std::size_t begin = open + 1; begin < close;) {
            const std::size_t colon = text.find(':', begin);
            if (colon == std::string_view::npos || colon > close)
                break;
            std::size_t end = colon;
            while (end > begin && text[end - 1] == ' ')
                --end;
            std::size_t start = end;
            while (start > begin && text[start - 1] != ' ' && text[start - 1] != '\n' && text[start - 1] != ')')
                --start;
            members.push_back(text.substr(start, end - start));
            const std::size_t comma = text.find(',', colon);
            begin = comma == std::string_view::npos ? close : comma + 1;
        }
        return members;
    }
    return std::nullopt;
}

}

std::expected<GraphCode, Core::Error> writeGraphCode(const Graph& graph, const std::string_view name) {
    const std::string body = Body(graph).write();
    const std::string text = std::format("fn {}(shadingContext : ShadingContext, material : vec4<u32>) -> MaterialGraphOutput\n{{\n{}}}\n", name, body);
    GraphCode code;
    code.code.resize(text.size());
    std::memcpy(code.code.data(), text.data(), text.size());
    code.key = Core::hashBytes(code.code);
    return code;
}

std::expected<bool, Core::Error> hasGraphFunction(const std::span<const std::byte> code, const std::string_view name) {
    const NoorRHI::WgslModule module(std::string(reinterpret_cast<const char*>(code.data()), code.size()));
    return module.hasFunction(name);
}

std::expected<std::vector<std::byte>, Core::Error> spliceGraphCode(const std::span<const std::byte> code, const std::span<const std::byte> graphCode,
                                                                   const std::string_view name) {
    std::string host(reinterpret_cast<const char*>(code.data()), code.size());
    std::string donor(reinterpret_cast<const char*>(graphCode.data()), graphCode.size());
    const std::optional<std::vector<std::string_view>> members = structMembers(host, "ShadingContext");
    if (!members || members->size() != graphInputMembers)
        return Core::fail(Core::ErrorCode::InvalidState, std::format("the shader has no ShadingContext struct with {} members", graphInputMembers));
    std::string linked;
    linked.reserve(donor.size());
    for (std::size_t at = 0; at < donor.size();) {
        const std::size_t marker = donor.find(memberMarker, at);
        if (marker == std::string::npos) {
            linked.append(donor, at);
            break;
        }
        linked.append(donor, at, marker - at);
        std::size_t end = marker + memberMarker.size();
        std::size_t member = 0;
        while (end < donor.size() && donor[end] >= '0' && donor[end] <= '9')
            member = member * 10 + static_cast<std::size_t>(donor[end++] - '0');
        if (member >= members->size())
            return Core::fail(Core::ErrorCode::InvalidState, std::format("the graph code reads the context member {}, which does not exist", member));
        linked += (*members)[member];
        at = end;
    }
    NoorRHI::WgslModule module(std::move(host));
    if (auto replaced = Core::check(module.replaceFunction(name, linked)); !replaced)
        return std::unexpected(std::move(replaced.error()));
    const std::string& text = module.getText();
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

std::expected<std::vector<std::byte>, Core::Error> linkAlphaGraphCode(const std::span<const std::byte> code,
                                                                      const std::span<const std::span<const std::byte>> donors) {
    NoorRHI::WgslModule module(std::string(reinterpret_cast<const char*>(code.data()), code.size()));
    std::vector<std::string> names;
    for (std::size_t index = 0; index < donors.size(); ++index) {
        const std::string& name = names.emplace_back(std::format("alphaGraph{}", index));
        const std::string_view donor(reinterpret_cast<const char*>(donors[index].data()), donors[index].size());
        if (auto added = Core::check(module.addFunction(name, donor, alphaGraphDonorName)); !added)
            return std::unexpected(std::move(added.error()));
    }
    const std::vector<std::string_view> callees(names.begin(), names.end());
    if (auto written = Core::check(module.writeSwitch(alphaGraphSwitchName, callees)); !written)
        return std::unexpected(std::move(written.error()));
    const std::string& text = module.getText();
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

}
