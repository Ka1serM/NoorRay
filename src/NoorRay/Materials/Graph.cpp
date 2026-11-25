#include "NoorRay/Materials/Graph.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

namespace NoorRay::Materials {

namespace {

using Values = std::array<float, 4>;

constexpr std::array<GraphFunctionInfo, graphFunctionCount> functionInfos{{
    {"graphParameter", {1}, 4},
    {"graphTexture", {1, 2}, 4},
    {"graphNoise", {4, 4, 4, 4}, 4},
    {"graphVoronoi", {4, 4, 4, 4}, 4},
    {"graphVoronoiPosition", {4, 4, 4, 4}, 4},
    {"graphWave", {3, 4, 4, 1}, 1},
    {"graphMagic", {3, 4}, 4},
    {"graphChecker", {3, 1}, 1},
    {"graphBrick", {3, 4, 4, 2}, 2},
    {"graphGradient", {3, 1}, 1},
    {"graphWhiteNoise", {4, 1}, 4},
    {"graphHueSaturationValue", {3, 4}, 3},
    {"graphRgbToHsv", {3}, 3},
    {"graphHsvToRgb", {3}, 3},
    {"graphBlackbody", {1}, 3},
    {"graphWavelength", {1}, 3},
    {"graphFresnel", {1, 3, 3, 1}, 1},
    {"graphLayerWeight", {1, 3, 3, 1}, 2},
    {"graphAo", {3, 3, 4}, 1},
    {"graphBevel", {3, 3, 3, 4}, 3},
    {"graphRaycast", {3, 3, 4}, 3},
    {"graphRaycastNormal", {3, 3, 4}, 3},
    {"graphGabor", {3, 4, 3, 1}, 3},
    {"graphRadialTiling", {3, 4}, 4},
    {"graphRadialTilingSegment", {3, 1}, 1},
    {"graphBoxWeights", {3, 1}, 3},
    {"graphInstanceAttribute", {1}, 4},
    {"graphParametric", {3}, 3},
    {"graphWireframe", {3, 1}, 1},
}};

constexpr std::array<GraphInputLayout, graphInputCount> inputLayouts{{
    {0, 0, 3},  {1, 0, 3},  {2, 0, 3},  {3, 0, 4},  {4, 0, 3},  {5, 0, 2},  {5, 2, 2},  {6, 0, 2},  {6, 2, 2},  {7, 0, 4},  {11, 0, 3}, {12, 0, 3}, {1, 3, 1},
    {4, 3, 1},  {11, 3, 1}, {12, 3, 1}, {9, 0, 3},  {10, 0, 3}, {8, 0, 4},  {13, 0, 3}, {13, 3, 1}, {14, 0, 3}, {15, 0, 3}, {16, 0, 3}, {17, 0, 4}, {18, 0, 4},
    {19, 0, 4}, {20, 0, 4}, {21, 0, 4}, {22, 0, 4}, {23, 0, 4}, {24, 0, 4}, {25, 0, 4}, {26, 0, 4}, {27, 0, 1}, {27, 1, 1}, {27, 2, 1}, {27, 3, 1}, {28, 0, 1},
    {28, 1, 1}, {28, 2, 1}, {28, 3, 1}, {29, 0, 1}, {29, 1, 1}, {29, 2, 1}, {29, 3, 1}, {30, 0, 4}, {31, 0, 1}, {31, 1, 1}, {31, 2, 1},
}};

bool isElementWise(const GraphOp op) {
    return (op >= GraphOp::Add && op <= GraphOp::Select) || op == GraphOp::Clamp || op == GraphOp::Mix || op == GraphOp::Smoothstep || op == GraphOp::MulAdd;
}

std::size_t operandCount(const GraphOp op) {
    switch (op) {
    case GraphOp::Constant:
    case GraphOp::Input:
    case GraphOp::Parameter:
        return 0;
    case GraphOp::Extract:
    case GraphOp::Abs:
    case GraphOp::Floor:
    case GraphOp::Ceil:
    case GraphOp::Fract:
    case GraphOp::Round:
    case GraphOp::Trunc:
    case GraphOp::Sign:
    case GraphOp::Sqrt:
    case GraphOp::InverseSqrt:
    case GraphOp::Exp:
    case GraphOp::Log:
    case GraphOp::Sin:
    case GraphOp::Cos:
    case GraphOp::Tan:
    case GraphOp::Asin:
    case GraphOp::Acos:
    case GraphOp::Atan:
    case GraphOp::Sinh:
    case GraphOp::Cosh:
    case GraphOp::Tanh:
    case GraphOp::Length:
    case GraphOp::Normalize:
        return 1;
    case GraphOp::Select:
    case GraphOp::Refract:
    case GraphOp::FaceForward:
    case GraphOp::Clamp:
    case GraphOp::Mix:
    case GraphOp::Smoothstep:
    case GraphOp::MulAdd:
        return 3;
    case GraphOp::Call:
    case GraphOp::Combine:
        return 0;
    default:
        return 2;
    }
}

float evaluateUnary(const GraphOp op, const float x) {
    switch (op) {
    case GraphOp::Abs:
        return std::abs(x);
    case GraphOp::Floor:
        return std::floor(x);
    case GraphOp::Ceil:
        return std::ceil(x);
    case GraphOp::Fract:
        return x - std::floor(x);
    case GraphOp::Round:
        return std::round(x);
    case GraphOp::Trunc:
        return std::trunc(x);
    case GraphOp::Sign:
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    case GraphOp::Sqrt:
        return x > 0.0f ? std::sqrt(x) : 0.0f;
    case GraphOp::InverseSqrt:
        return x > 0.0f ? 1.0f / std::sqrt(x) : 0.0f;
    case GraphOp::Exp:
        return std::exp(x);
    case GraphOp::Log:
        return x > 0.0f ? std::log(x) : 0.0f;
    case GraphOp::Sin:
        return std::sin(x);
    case GraphOp::Cos:
        return std::cos(x);
    case GraphOp::Tan:
        return std::tan(x);
    case GraphOp::Asin:
        return std::asin(std::clamp(x, -1.0f, 1.0f));
    case GraphOp::Acos:
        return std::acos(std::clamp(x, -1.0f, 1.0f));
    case GraphOp::Atan:
        return std::atan(x);
    case GraphOp::Sinh:
        return std::sinh(x);
    case GraphOp::Cosh:
        return std::cosh(x);
    default:
        return std::tanh(x);
    }
}

//! The result of an element-wise operation as GraphSpirv computes it; Div, Pow, Sqrt and Log give zero where they are undefined, as Blender does.
float evaluateElement(const GraphOp op, const float a, const float b, const float c) {
    switch (op) {
    case GraphOp::Add:
        return a + b;
    case GraphOp::Sub:
        return a - b;
    case GraphOp::Mul:
        return a * b;
    case GraphOp::Div:
        return b != 0.0f ? a / b : 0.0f;
    case GraphOp::Min:
        return std::min(a, b);
    case GraphOp::Max:
        return std::max(a, b);
    case GraphOp::Pow: {
        if (a >= 0.0f)
            return std::pow(a, b);
        if (std::trunc(b) != b)
            return 0.0f;
        const float magnitude = std::pow(-a, b);
        return std::trunc(b * 0.5f) != b * 0.5f ? -magnitude : magnitude;
    }
    case GraphOp::Atan2:
        return std::atan2(a, b);
    case GraphOp::Modulo:
        return b != 0.0f ? std::fmod(a, b) : 0.0f;
    case GraphOp::FlooredModulo:
        return b != 0.0f ? a - std::floor(a / b) * b : 0.0f;
    case GraphOp::Less:
        return a < b ? 1.0f : 0.0f;
    case GraphOp::Greater:
        return a > b ? 1.0f : 0.0f;
    case GraphOp::Select:
        return a > 0.5f ? b : c;
    case GraphOp::Clamp:
        return std::clamp(a, std::min(b, c), std::max(b, c));
    case GraphOp::Mix:
        return a + (b - a) * c;
    case GraphOp::Smoothstep: {
        const float t = b != a ? std::clamp((c - a) / (b - a), 0.0f, 1.0f) : (c < a ? 0.0f : 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }
    case GraphOp::MulAdd:
        return a * b + c;
    default:
        return evaluateUnary(op, a);
    }
}

//! component(operand, at) is one component of an operand, where an operand of width 1 broadcasts.
template <class Component>
Values evaluateOperation(const GraphOp op, const std::size_t resultSize, const std::size_t operandCount, const std::size_t firstWidth,
                         const Component& component) {
    Values result{};
    switch (op) {
    case GraphOp::Dot:
    case GraphOp::Length:
    case GraphOp::Distance: {
        float sum = 0.0f;
        for (std::size_t at = 0; at < firstWidth; ++at) {
            const float a = component(0, at);
            const float b = op == GraphOp::Dot ? component(1, at) : (op == GraphOp::Distance ? a - component(1, at) : a);
            sum += op == GraphOp::Distance ? b * b : a * b;
        }
        result[0] = op == GraphOp::Dot ? sum : std::sqrt(sum);
        return result;
    }
    case GraphOp::Cross: {
        const Core::float3 c = Core::cross({component(0, 0), component(0, 1), component(0, 2)}, {component(1, 0), component(1, 1), component(1, 2)});
        return {c.x, c.y, c.z, 0.0f};
    }
    case GraphOp::Normalize: {
        float sum = 0.0f;
        for (std::size_t at = 0; at < resultSize; ++at)
            sum += component(0, at) * component(0, at);
        const float scale = sum > 0.0f ? 1.0f / std::sqrt(sum) : 0.0f;
        for (std::size_t at = 0; at < resultSize; ++at)
            result[at] = component(0, at) * scale;
        return result;
    }
    default:
        for (std::size_t at = 0; at < resultSize; ++at)
            result[at] = evaluateElement(op, component(0, at), operandCount > 1 ? component(1, at) : 0.0f, operandCount > 2 ? component(2, at) : 0.0f);
        return result;
    }
}

void hashBytes(std::uint64_t& hash, const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i)
        hash = (hash ^ bytes[i]) * 0x100000001b3ull;
}

}

GraphInputLayout graphInputLayout(const GraphInput input) {
    return inputLayouts[static_cast<std::size_t>(input)];
}

std::size_t GraphFunctionInfo::parameterCount() const {
    return static_cast<std::size_t>(std::ranges::find(parameters, std::uint8_t(0)) - parameters.begin());
}

const GraphFunctionInfo& graphFunctionInfo(const GraphFunction function) {
    return functionInfos[static_cast<std::size_t>(function)];
}

std::expected<void, Core::Error> Graph::open(Source opened, const std::string_view materialName) {
    const auto invalid = [&](const std::size_t at, const std::string_view problem) {
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the graph of material '{}' {} at instruction {}", materialName, problem, at));
    };
    const std::span<const GraphInstruction> instructions = opened.instructions;
    for (std::size_t at = 0; at < instructions.size(); ++at) {
        const GraphInstruction& instruction = instructions[at];
        if (static_cast<std::size_t>(instruction.op) >= graphOpCount)
            return invalid(at, "has an unknown operation");
        if (instruction.width == 0 || instruction.width > 4)
            return invalid(at, "has a width outside 1 to 4");
        if (std::size_t(instruction.first) + instruction.count > opened.operands.size())
            return invalid(at, "has operands outside the operand list");
        const std::span<const std::uint32_t> operands = std::span(opened.operands).subspan(instruction.first, instruction.count);
        for (const std::uint32_t operand : operands)
            if (operand >= at)
                return invalid(at, "reads a value that comes after it");
        const auto widthOf = [&](const std::size_t operand) { return instructions[operands[operand]].width; };
        const std::size_t expected = operandCount(instruction.op);
        if (instruction.op != GraphOp::Call && instruction.op != GraphOp::Combine && operands.size() != expected)
            return invalid(at, "has the wrong number of operands");
        switch (instruction.op) {
        case GraphOp::Constant:
            if (std::size_t(instruction.index) + instruction.width > opened.constants.size())
                return invalid(at, "reads a constant that does not exist");
            break;
        case GraphOp::Input: {
            if (instruction.index >= graphInputCount)
                return invalid(at, "reads an unknown input");
            if (instruction.width != graphInputLayout(static_cast<GraphInput>(instruction.index)).width)
                return invalid(at, "reads an input with the wrong width");
            break;
        }
        case GraphOp::Parameter:
            if (instruction.index >= opened.parameterSlots)
                return invalid(at, "reads a parameter slot that does not exist");
            break;
        case GraphOp::Texture:
            if (instruction.index >= opened.textureCount || widthOf(0) != 2 || widthOf(1) != 4 || instruction.width != 4)
                return invalid(at, "samples a texture that does not exist, with a UV that is not a float2 or gradients that are not a float4");
            break;
        case GraphOp::Call: {
            if (instruction.index >= graphFunctionCount)
                return invalid(at, "calls an unknown function");
            const GraphFunctionInfo& info = graphFunctionInfo(static_cast<GraphFunction>(instruction.index));
            if (operands.size() != info.parameterCount() || instruction.width != info.result)
                return invalid(at, std::format("calls {} with the wrong signature", info.name));
            for (std::size_t operand = 0; operand < operands.size(); ++operand)
                if (widthOf(operand) != info.parameters[operand])
                    return invalid(at, std::format("calls {} with a parameter of the wrong width", info.name));
            break;
        }
        case GraphOp::Extract:
            if (instruction.width != 1 || instruction.index >= widthOf(0))
                return invalid(at, "extracts a component that does not exist");
            break;
        case GraphOp::Combine: {
            std::size_t width = 0;
            for (std::size_t operand = 0; operand < operands.size(); ++operand)
                width += widthOf(operand);
            if (width != instruction.width)
                return invalid(at, "combines values whose widths do not add up");
            break;
        }
        case GraphOp::Dot:
        case GraphOp::Distance:
            if (instruction.width != 1 || widthOf(0) != widthOf(1))
                return invalid(at, "takes the dot product or distance of different widths");
            break;
        case GraphOp::Length:
            if (instruction.width != 1)
                return invalid(at, "has a length that is not a scalar");
            break;
        case GraphOp::Cross:
            if (instruction.width != 3 || widthOf(0) != 3 || widthOf(1) != 3)
                return invalid(at, "takes the cross product of values that are not float3");
            break;
        case GraphOp::Normalize:
        case GraphOp::Reflect:
        case GraphOp::Refract:
        case GraphOp::FaceForward:
            for (std::size_t operand = 0; operand < operands.size(); ++operand) {
                const bool scalar = instruction.op == GraphOp::Refract && operand == 2;
                if (widthOf(operand) != (scalar ? 1 : instruction.width))
                    return invalid(at, "has operands of the wrong width");
            }
            break;
        default:
            if (!isElementWise(instruction.op))
                return invalid(at, "has an unknown operation");
            for (std::size_t operand = 0; operand < operands.size(); ++operand)
                if (widthOf(operand) != 1 && widthOf(operand) != instruction.width)
                    return invalid(at, "has an operand that neither matches the width nor is a scalar");
            break;
        }
    }
    for (const std::uint32_t output : opened.outputs)
        if (output >= instructions.size() || instructions[output].width != 4)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("the graph of material '{}' has an output that is not a float4 value", materialName));
    for (const GraphParameter& parameter : opened.parameters)
        if (parameter.slot >= opened.parameterSlots || parameter.width == 0 || parameter.width > 4)
            return Core::fail(Core::ErrorCode::InvalidFile,
                              std::format("the graph of material '{}' has the parameter '{}' outside its slots", materialName, parameter.name));
    source = std::move(opened);
    hash = 0xcbf29ce484222325ull;
    hashBytes(hash, source.instructions.data(), source.instructions.size() * sizeof(GraphInstruction));
    hashBytes(hash, source.operands.data(), source.operands.size() * sizeof(std::uint32_t));
    hashBytes(hash, source.constants.data(), source.constants.size() * sizeof(float));
    hashBytes(hash, source.outputs.data(), source.outputs.size() * sizeof(std::uint32_t));
    hashBytes(hash, &source.textureCount, sizeof(source.textureCount));
    return {};
}

const float* Graph::constantOf(const std::uint32_t value) const {
    const GraphInstruction& instruction = source.instructions[value];
    return instruction.op == GraphOp::Constant ? &source.constants[instruction.index] : nullptr;
}

std::optional<Core::float4> Graph::evaluateUniform(const std::uint32_t value, const std::span<const Core::float4> parameters) const {
    std::vector<std::optional<Values>> values(value + 1);
    for (std::uint32_t index = 0; index <= value; ++index) {
        const GraphInstruction& instruction = source.instructions[index];
        const std::span<const std::uint32_t> inputs = operands(instruction);
        if (!std::ranges::all_of(inputs, [&](const std::uint32_t input) { return values[input].has_value(); }))
            continue;
        const auto component = [&](const std::size_t operand, const std::size_t at) {
            const Values& input = *values[inputs[operand]];
            return source.instructions[inputs[operand]].width == 1 ? input[0] : input[at];
        };
        switch (instruction.op) {
        case GraphOp::Constant:
            values[index] = Values{};
            std::copy_n(source.constants.begin() + instruction.index, instruction.width, values[index]->begin());
            break;
        case GraphOp::Parameter:
            if (instruction.index < parameters.size())
                values[index] =
                    Values{parameters[instruction.index].x, parameters[instruction.index].y, parameters[instruction.index].z, parameters[instruction.index].w};
            break;
        case GraphOp::Extract:
            values[index] = Values{(*values[inputs[0]])[instruction.index]};
            break;
        case GraphOp::Combine: {
            Values combined{};
            std::size_t at = 0;
            for (const std::uint32_t input : inputs)
                for (std::size_t part = 0; part < source.instructions[input].width; ++part)
                    combined[at++] = (*values[input])[part];
            values[index] = combined;
            break;
        }
        case GraphOp::Input:
        case GraphOp::Texture:
        case GraphOp::Call:
        case GraphOp::Reflect:
        case GraphOp::Refract:
        case GraphOp::FaceForward:
            break;
        default:
            values[index] = evaluateOperation(instruction.op, instruction.width, inputs.size(), source.instructions[inputs[0]].width, component);
            break;
        }
    }
    if (!values[value])
        return std::nullopt;
    const Values& result = *values[value];
    return Core::float4{result[0], result[1], result[2], result[3]};
}

bool Graph::reads(const GraphOutput output, const GraphInput input) const {
    std::vector<bool> used(source.instructions.size());
    used[getOutput(output)] = true;
    for (std::size_t index = source.instructions.size(); index-- > 0;) {
        if (!used[index])
            continue;
        const GraphInstruction& instruction = source.instructions[index];
        if (instruction.op == GraphOp::Input && static_cast<GraphInput>(instruction.index) == input)
            return true;
        for (const std::uint32_t operand : operands(instruction))
            used[operand] = true;
    }
    return false;
}

bool Graph::hasSameProgram(const Graph& other) const {
    return hash == other.hash && source.instructions == other.source.instructions && source.operands == other.source.operands
           && std::ranges::equal(source.constants, other.source.constants,
                                 [](const float a, const float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); })
           && source.outputs == other.source.outputs && source.textureCount == other.source.textureCount
           && source.parameterSlots == other.source.parameterSlots;
}

std::uint32_t GraphBuilder::emit(const GraphInstruction instruction, const std::span<const std::uint32_t> operands) {
    std::u32string key;
    key.reserve(2 + operands.size());
    key.push_back(static_cast<char32_t>(std::to_underlying(instruction.op) | instruction.width << 8u));
    key.push_back(instruction.index);
    key.append(operands.begin(), operands.end());
    const auto [found, added] = known.try_emplace(std::move(key), static_cast<std::uint32_t>(source.instructions.size()));
    if (!added)
        return found->second;
    GraphInstruction stored = instruction;
    stored.first = static_cast<std::uint32_t>(source.operands.size());
    stored.count = static_cast<std::uint32_t>(operands.size());
    source.operands.insert(source.operands.end(), operands.begin(), operands.end());
    source.instructions.push_back(stored);
    return found->second;
}

std::uint32_t GraphBuilder::constantValues(const std::span<const float> values) {
    std::u32string key;
    key.push_back(static_cast<char32_t>(0xffffffffu));
    for (const float value : values)
        key.push_back(std::bit_cast<char32_t>(value));
    const auto [found, added] = known.try_emplace(std::move(key), static_cast<std::uint32_t>(source.instructions.size()));
    if (!added)
        return found->second;
    const GraphInstruction stored{.op = GraphOp::Constant,
                                  .width = static_cast<std::uint8_t>(values.size()),
                                  .index = static_cast<std::uint16_t>(source.constants.size()),
                                  .first = static_cast<std::uint32_t>(source.operands.size())};
    source.constants.insert(source.constants.end(), values.begin(), values.end());
    source.instructions.push_back(stored);
    return found->second;
}

std::uint32_t GraphBuilder::constant(const Core::float4 value) {
    return constantValues(std::array{value.x, value.y, value.z, value.w});
}

std::uint32_t GraphBuilder::constant(const float value) {
    return constantValues(std::array{value});
}

std::uint32_t GraphBuilder::constant(const Core::float2 value) {
    return combine(std::array{constant(value.x), constant(value.y)});
}

std::uint32_t GraphBuilder::constant(const Core::float3 value) {
    return combine(std::array{constant(value.x), constant(value.y), constant(value.z)});
}

const float* GraphBuilder::constantOf(const std::uint32_t value) const {
    const GraphInstruction& instruction = source.instructions[value];
    return instruction.op == GraphOp::Constant ? &source.constants[instruction.index] : nullptr;
}

std::uint32_t GraphBuilder::input(const GraphInput input) {
    return emit({.op = GraphOp::Input, .width = graphInputLayout(input).width, .index = static_cast<std::uint16_t>(input)}, {});
}

std::uint32_t GraphBuilder::parameter(std::string name, const std::uint8_t width) {
    const auto slot = source.parameterSlots++;
    values.resize(source.parameterSlots);
    source.parameters.push_back({std::move(name), slot, width});
    GraphInstruction instruction{.op = GraphOp::Parameter, .width = width, .index = static_cast<std::uint16_t>(slot)};
    instruction.first = static_cast<std::uint32_t>(source.operands.size());
    source.instructions.push_back(instruction);
    return static_cast<std::uint32_t>(source.instructions.size() - 1);
}

std::uint32_t GraphBuilder::texture(const std::uint32_t texture, const std::uint32_t uv, const std::uint32_t gradients) {
    return emit({.op = GraphOp::Texture, .width = 4, .index = static_cast<std::uint16_t>(texture)}, std::array{uv, gradients});
}

std::uint32_t GraphBuilder::call(const GraphFunction function, const std::span<const std::uint32_t> operands) {
    return emit({.op = GraphOp::Call, .width = graphFunctionInfo(function).result, .index = static_cast<std::uint16_t>(function)}, operands);
}

std::uint32_t GraphBuilder::extract(const std::uint32_t value, const std::uint32_t component) {
    const GraphInstruction& instruction = source.instructions[value];
    if (instruction.width == 1)
        return value;
    if (const float* values = constantOf(value))
        return constant(values[component]);
    if (instruction.op == GraphOp::Combine) {
        std::uint32_t offset = 0;
        for (const std::uint32_t part : std::span(source.operands).subspan(instruction.first, instruction.count)) {
            const std::uint8_t partWidth = width(part);
            if (component < offset + partWidth)
                return extract(part, component - offset);
            offset += partWidth;
        }
    }
    return emit({.op = GraphOp::Extract, .width = 1, .index = static_cast<std::uint16_t>(component)}, std::array{value});
}

std::uint32_t GraphBuilder::combine(const std::span<const std::uint32_t> values) {
    if (values.size() == 1)
        return values[0];
    std::uint8_t total = 0;
    bool constants = true;
    for (const std::uint32_t value : values) {
        total = static_cast<std::uint8_t>(total + width(value));
        constants = constants && constantOf(value);
    }
    if (const std::optional<std::uint32_t> whole = wholeOfExtracts(values))
        return *whole;
    if (constants) {
        std::array<float, 4> folded{};
        std::size_t at = 0;
        for (const std::uint32_t value : values)
            for (std::size_t component = 0; component < width(value); ++component)
                folded[at++] = constantOf(value)[component];
        return constantValues(std::span(folded).first(total));
    }
    return emit({.op = GraphOp::Combine, .width = total}, values);
}

//! The value whose components values extracts in order, so that a vector taken apart and put back together stays one value.
std::optional<std::uint32_t> GraphBuilder::wholeOfExtracts(const std::span<const std::uint32_t> values) const {
    const GraphInstruction& first = source.instructions[values[0]];
    if (first.op != GraphOp::Extract)
        return std::nullopt;
    const std::uint32_t whole = source.operands[first.first];
    if (width(whole) != values.size())
        return std::nullopt;
    for (std::size_t component = 0; component < values.size(); ++component) {
        const GraphInstruction& part = source.instructions[values[component]];
        if (part.op != GraphOp::Extract || part.index != component || source.operands[part.first] != whole)
            return std::nullopt;
    }
    return whole;
}

std::uint32_t GraphBuilder::resize(const std::uint32_t value, const std::uint8_t target, const float fill) {
    const std::uint8_t current = width(value);
    if (current == target)
        return value;
    std::vector<std::uint32_t> parts;
    for (std::uint8_t component = 0; component < target; ++component)
        parts.push_back(current == 1 ? value : (component < current ? extract(value, component) : constant(fill)));
    return combine(parts);
}

std::uint8_t GraphBuilder::resultWidth(const GraphOp op, const std::span<const std::uint32_t> operands) const {
    switch (op) {
    case GraphOp::Dot:
    case GraphOp::Length:
    case GraphOp::Distance:
        return 1;
    case GraphOp::Cross:
        return 3;
    case GraphOp::Normalize:
    case GraphOp::Reflect:
    case GraphOp::Refract:
    case GraphOp::FaceForward:
        return width(operands[0]);
    default: {
        std::uint8_t result = 1;
        for (const std::uint32_t operand : operands)
            result = std::max(result, width(operand));
        return result;
    }
    }
}

std::uint32_t GraphBuilder::fold(const GraphOp op, const std::uint8_t resultSize, const std::span<const std::uint32_t> operands) {
    const auto component = [&](const std::size_t operand, const std::size_t at) {
        const float* values = constantOf(operands[operand]);
        return width(operands[operand]) == 1 ? values[0] : values[at];
    };
    const Values result = evaluateOperation(op, resultSize, operands.size(), width(operands[0]), component);
    if (resultSize == 4)
        return constant(Core::float4{result[0], result[1], result[2], result[3]});
    std::vector<std::uint32_t> parts;
    for (std::size_t at = 0; at < resultSize; ++at)
        parts.push_back(constant(result[at]));
    return combine(parts);
}

std::uint32_t GraphBuilder::op(const GraphOp op, const std::span<const std::uint32_t> operands) {
    const std::uint8_t resultSize = resultWidth(op, operands);
    const bool constants = std::ranges::all_of(operands, [&](const std::uint32_t operand) { return constantOf(operand) != nullptr; });
    const bool foldable = op != GraphOp::Reflect && op != GraphOp::Refract && op != GraphOp::FaceForward;
    if (constants && foldable)
        return fold(op, resultSize, operands);
    const auto isConstant = [&](const std::uint32_t value, const float wanted) {
        const float* values = constantOf(value);
        if (!values)
            return false;
        for (std::size_t at = 0; at < width(value); ++at)
            if (values[at] != wanted)
                return false;
        return true;
    };
    const auto widened = [&](const std::uint32_t value) { return width(value) == resultSize ? value : resize(value, resultSize); };
    switch (op) {
    case GraphOp::Add:
        if (isConstant(operands[1], 0.0f))
            return widened(operands[0]);
        if (isConstant(operands[0], 0.0f))
            return widened(operands[1]);
        break;
    case GraphOp::Sub: {
        if (isConstant(operands[1], 0.0f))
            return widened(operands[0]);
        const GraphInstruction& negated = source.instructions[operands[1]];
        if (isConstant(operands[0], 0.0f) && negated.op == GraphOp::Sub && isConstant(source.operands[negated.first], 0.0f))
            return widened(source.operands[negated.first + 1]);
        break;
    }
    case GraphOp::Mul:
        if (isConstant(operands[1], 1.0f))
            return widened(operands[0]);
        if (isConstant(operands[0], 1.0f))
            return widened(operands[1]);
        if (isConstant(operands[0], 0.0f) || isConstant(operands[1], 0.0f))
            return resize(constant(0.0f), resultSize);
        break;
    case GraphOp::Div:
        if (isConstant(operands[1], 1.0f))
            return widened(operands[0]);
        break;
    case GraphOp::Mix:
        if (isConstant(operands[2], 0.0f))
            return widened(operands[0]);
        if (isConstant(operands[2], 1.0f))
            return widened(operands[1]);
        break;
    case GraphOp::Select:
        if (constantOf(operands[0]) && width(operands[0]) == 1)
            return widened(constantOf(operands[0])[0] > 0.5f ? operands[1] : operands[2]);
        break;
    default:
        break;
    }
    return emit({.op = op, .width = resultSize}, operands);
}

void GraphBuilder::setOutput(const GraphOutput output, const std::uint32_t value) {
    source.outputs[static_cast<std::size_t>(output)] = resize(value, 4);
}

}
