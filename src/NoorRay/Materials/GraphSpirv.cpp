#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <string_view>

#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp11>

#include "NoorRHI/Shaders/SpirvModule.h"
#include "NoorRay/Core/Hash.h"
#include "NoorRay/Materials/GraphShader.h"

namespace NoorRay::Materials {

namespace {

using Words = std::vector<std::uint32_t>;

Words toWords(const std::span<const std::byte> bytes) {
    Words words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), words.size() * sizeof(std::uint32_t));
    return words;
}

void append(Words& stream, const spv::Op op, const std::span<const std::uint32_t> operands) {
    stream.push_back(static_cast<std::uint32_t>((operands.size() + 1) << 16) | static_cast<std::uint32_t>(op));
    stream.insert(stream.end(), operands.begin(), operands.end());
}

void append(Words& stream, const spv::Op op, const std::initializer_list<std::uint32_t> operands) {
    append(stream, op, std::span(operands.begin(), operands.size()));
}

Words literal(const std::string_view text) {
    Words words((text.size() + 4) / 4, 0);
    std::memcpy(words.data(), text.data(), text.size());
    return words;
}

//! The module sections and the ids; types and constants are unique by their operands.
class Writer {
public:
    std::uint32_t newId() { return bound++; }

    std::uint32_t global(const spv::Op op, const Words& operands, const bool typed) {
        Words key = operands;
        key.insert(key.begin(), static_cast<std::uint32_t>(op));
        if (const auto found = cache.find(key); found != cache.end())
            return found->second;
        const std::uint32_t id = newId();
        Words full = operands;
        full.insert(full.begin() + (typed ? 1 : 0), id);
        append(globals, op, full);
        cache.emplace(std::move(key), id);
        return id;
    }

    std::uint32_t floatType(const std::uint8_t width) {
        const std::uint32_t scalar = global(spv::Op::OpTypeFloat, {32}, false);
        return width == 1 ? scalar : global(spv::Op::OpTypeVector, {scalar, width}, false);
    }

    std::uint32_t boolType(const std::uint8_t width) {
        const std::uint32_t scalar = global(spv::Op::OpTypeBool, {}, false);
        return width == 1 ? scalar : global(spv::Op::OpTypeVector, {scalar, width}, false);
    }

    std::uint32_t uintType() { return global(spv::Op::OpTypeInt, {32, 0}, false); }

    std::uint32_t materialType() { return global(spv::Op::OpTypeVector, {uintType(), 4}, false); }

    std::uint32_t contextType() { return global(spv::Op::OpTypeStruct, Words(graphInputMembers, floatType(4)), false); }

    //! Shaders::MaterialGraphOutput: a float4 per output but the last, which is a float.
    std::uint32_t outputType() {
        Words members(graphOutputCount - 1, floatType(4));
        members.push_back(floatType(1));
        return global(spv::Op::OpTypeStruct, members, false);
    }

    std::uint32_t pointerType(const std::uint32_t pointee) {
        return global(spv::Op::OpTypePointer, {static_cast<std::uint32_t>(spv::StorageClass::Function), pointee}, false);
    }

    std::uint32_t functionType(const std::uint32_t result, const std::span<const std::uint32_t> parameters) {
        Words operands{result};
        operands.insert(operands.end(), parameters.begin(), parameters.end());
        return global(spv::Op::OpTypeFunction, operands, false);
    }

    std::uint32_t uintConstant(const std::uint32_t value) { return global(spv::Op::OpConstant, {uintType(), value}, true); }

    std::uint32_t floatConstant(const float value) { return global(spv::Op::OpConstant, {floatType(1), std::bit_cast<std::uint32_t>(value)}, true); }

    std::uint32_t vectorConstant(const std::span<const float> values) {
        if (values.size() == 1)
            return floatConstant(values[0]);
        Words operands{floatType(static_cast<std::uint8_t>(values.size()))};
        for (const float value : values)
            operands.push_back(floatConstant(value));
        return global(spv::Op::OpConstantComposite, operands, true);
    }

    std::uint32_t splatConstant(const float value, const std::uint8_t width) {
        const float values[] = {value, value, value, value};
        return vectorConstant(std::span(values, width));
    }

    void name(const std::uint32_t id, const std::string_view text) {
        Words operands{id};
        const Words words = literal(text);
        operands.insert(operands.end(), words.begin(), words.end());
        append(names, spv::Op::OpName, operands);
    }

    Words finish(const std::uint32_t glsl) const {
        Words module{spv::MagicNumber, 0x00010500u, 0, bound, 0};
        append(module, spv::Op::OpCapability, {static_cast<std::uint32_t>(spv::Capability::Shader)});
        Words import{glsl};
        const Words words = literal("GLSL.std.450");
        import.insert(import.end(), words.begin(), words.end());
        append(module, spv::Op::OpExtInstImport, import);
        append(module, spv::Op::OpMemoryModel,
               {static_cast<std::uint32_t>(spv::AddressingModel::Logical), static_cast<std::uint32_t>(spv::MemoryModel::GLSL450)});
        module.insert(module.end(), names.begin(), names.end());
        module.insert(module.end(), globals.begin(), globals.end());
        module.insert(module.end(), declarations.begin(), declarations.end());
        module.insert(module.end(), functions.begin(), functions.end());
        return module;
    }

    Words names;
    Words globals;
    Words declarations;
    Words functions;

private:
    std::map<Words, std::uint32_t> cache;
    std::uint32_t bound = 1;
};

//! Writes the instructions of one graph as the body of evaluateMaterialGraph.
class Body {
public:
    Body(Writer& writer, const Graph& graph, const std::uint32_t glsl, const std::uint32_t context, const std::uint32_t material,
         const std::map<GraphFunction, std::uint32_t>& helpers)
        : writer(writer), graph(graph), glsl(glsl), context(context), material(material), helpers(helpers) {}

    void write() {
        const std::span<const GraphInstruction> instructions = graph.getInstructions();
        std::vector<bool> live(instructions.size(), false);
        for (std::size_t output = 0; output < graphOutputCount; ++output)
            live[graph.getOutput(static_cast<GraphOutput>(output))] = true;
        for (std::size_t at = instructions.size(); at-- > 0;)
            if (live[at])
                for (const std::uint32_t operand : graph.operands(instructions[at]))
                    live[operand] = true;
        values.assign(instructions.size(), 0);
        for (std::size_t at = 0; at < instructions.size(); ++at)
            if (live[at])
                values[at] = value(instructions[at]);
        Words operands{writer.outputType(), writer.newId()};
        for (std::size_t output = 0; output + 1 < graphOutputCount; ++output)
            operands.push_back(values[graph.getOutput(static_cast<GraphOutput>(output))]);
        operands.push_back(narrow(values[graph.getOutput(GraphOutput::Transparency)], 4, 1));
        append(writer.functions, spv::Op::OpCompositeConstruct, operands);
        append(writer.functions, spv::Op::OpReturnValue, {operands[1]});
    }

private:
    std::uint32_t emit(const spv::Op op, const std::uint32_t type, const std::span<const std::uint32_t> operands) {
        const std::uint32_t id = writer.newId();
        Words full{type, id};
        full.insert(full.end(), operands.begin(), operands.end());
        append(writer.functions, op, full);
        return id;
    }

    std::uint32_t emit(const spv::Op op, const std::uint32_t type, const std::initializer_list<std::uint32_t> operands) {
        return emit(op, type, std::span(operands.begin(), operands.size()));
    }

    std::uint32_t extended(const GLSLstd450 instruction, const std::uint8_t width, const std::initializer_list<std::uint32_t> operands) {
        Words full{glsl, static_cast<std::uint32_t>(instruction)};
        full.insert(full.end(), operands.begin(), operands.end());
        return emit(spv::Op::OpExtInst, writer.floatType(width), full);
    }

    std::uint32_t splat(const std::uint32_t id, const std::uint8_t from, const std::uint8_t to) {
        if (from == to)
            return id;
        return emit(spv::Op::OpCompositeConstruct, writer.floatType(to), Words(to, id));
    }

    std::uint32_t narrow(const std::uint32_t id, const std::uint8_t from, const std::uint8_t to, const std::uint32_t first = 0) {
        if (from == to)
            return id;
        if (to == 1)
            return emit(spv::Op::OpCompositeExtract, writer.floatType(1), {id, first});
        Words operands{id, id};
        for (std::uint32_t component = 0; component < to; ++component)
            operands.push_back(first + component);
        return emit(spv::Op::OpVectorShuffle, writer.floatType(to), operands);
    }

    std::uint32_t select(const std::uint32_t condition, const std::uint32_t a, const std::uint32_t b, const std::uint8_t width) {
        return emit(spv::Op::OpSelect, writer.floatType(width), {condition, a, b});
    }

    std::uint32_t compare(const spv::Op op, const std::uint32_t a, const std::uint32_t b, const std::uint8_t width) {
        return emit(op, writer.boolType(width), {a, b});
    }

    //! The result where test(a) holds, else zero.
    std::uint32_t guarded(const spv::Op test, const std::uint32_t a, const std::uint32_t result, const std::uint8_t width) {
        const std::uint32_t zero = writer.splatConstant(0.0f, width);
        return select(compare(test, a, zero, width), result, zero, width);
    }

    std::uint32_t call(const GraphFunction function, const std::span<const std::uint32_t> arguments) {
        Words operands{helpers.at(function)};
        operands.insert(operands.end(), arguments.begin(), arguments.end());
        return emit(spv::Op::OpFunctionCall, writer.floatType(graphFunctionInfo(function).result), operands);
    }

    std::uint32_t input(const GraphInput input, const std::uint8_t width) {
        const GraphInputLayout layout = graphInputLayout(input);
        const std::uint32_t pointer = emit(spv::Op::OpAccessChain, writer.pointerType(writer.floatType(4)), {context, writer.uintConstant(layout.member)});
        return narrow(emit(spv::Op::OpLoad, writer.floatType(4), {pointer}), 4, width, layout.component);
    }

    std::uint32_t value(const GraphInstruction& instruction) {
        const std::span<const std::uint32_t> operands = graph.operands(instruction);
        const std::uint8_t width = instruction.width;
        const auto widthOf = [&](const std::size_t operand) { return graph.getInstructions()[operands[operand]].width; };
        const auto operand = [&](const std::size_t at) { return splat(values[operands[at]], widthOf(at), width); };
        const std::uint32_t type = writer.floatType(width);
        switch (instruction.op) {
        case GraphOp::Constant:
            return writer.vectorConstant(graph.getConstants().subspan(instruction.index, width));
        case GraphOp::Input:
            return input(static_cast<GraphInput>(instruction.index), width);
        case GraphOp::Parameter: {
            const std::uint32_t slot = writer.floatConstant(static_cast<float>(instruction.index));
            return narrow(call(GraphFunction::Parameter, std::array{material, slot}), 4, width);
        }
        case GraphOp::Texture:
            return call(GraphFunction::Texture,
                        std::array{material, values[operands[1]], writer.floatConstant(static_cast<float>(instruction.index)), values[operands[0]]});
        case GraphOp::Call: {
            Words arguments;
            for (const std::uint32_t argument : operands)
                arguments.push_back(values[argument]);
            return call(static_cast<GraphFunction>(instruction.index), arguments);
        }
        case GraphOp::Extract:
            return emit(spv::Op::OpCompositeExtract, type, {values[operands[0]], instruction.index});
        case GraphOp::Combine: {
            Words parts;
            for (const std::uint32_t part : operands)
                parts.push_back(values[part]);
            return emit(spv::Op::OpCompositeConstruct, type, parts);
        }
        case GraphOp::Add:
            return emit(spv::Op::OpFAdd, type, {operand(0), operand(1)});
        case GraphOp::Sub:
            return emit(spv::Op::OpFSub, type, {operand(0), operand(1)});
        case GraphOp::Mul:
            return emit(spv::Op::OpFMul, type, {operand(0), operand(1)});
        case GraphOp::Div: {
            const std::uint32_t divisor = operand(1);
            return guarded(spv::Op::OpFOrdNotEqual, divisor, emit(spv::Op::OpFDiv, type, {operand(0), divisor}), width);
        }
        case GraphOp::Min:
            return extended(GLSLstd450FMin, width, {operand(0), operand(1)});
        case GraphOp::Max:
            return extended(GLSLstd450FMax, width, {operand(0), operand(1)});
        case GraphOp::Pow: {
            const std::uint32_t base = operand(0);
            const std::uint32_t exponent = operand(1);
            const std::uint32_t zero = writer.splatConstant(0.0f, width);
            const std::uint32_t magnitude = extended(GLSLstd450Pow, width, {extended(GLSLstd450FAbs, width, {base}), exponent});
            const std::uint32_t integer = compare(spv::Op::OpFOrdEqual, extended(GLSLstd450Trunc, width, {exponent}), exponent, width);
            const std::uint32_t half = emit(spv::Op::OpFMul, type, {exponent, writer.splatConstant(0.5f, width)});
            const std::uint32_t odd = compare(spv::Op::OpFOrdNotEqual, extended(GLSLstd450Trunc, width, {half}), half, width);
            const std::uint32_t negated = select(odd, emit(spv::Op::OpFNegate, type, {magnitude}), magnitude, width);
            const std::uint32_t negative = compare(spv::Op::OpFOrdLessThan, base, zero, width);
            return select(negative, select(integer, negated, zero, width), magnitude, width);
        }
        case GraphOp::Atan2:
            return extended(GLSLstd450Atan2, width, {operand(0), operand(1)});
        case GraphOp::Modulo: {
            const std::uint32_t divisor = operand(1);
            return guarded(spv::Op::OpFOrdNotEqual, divisor, emit(spv::Op::OpFRem, type, {operand(0), divisor}), width);
        }
        case GraphOp::FlooredModulo: {
            const std::uint32_t divisor = operand(1);
            return guarded(spv::Op::OpFOrdNotEqual, divisor, emit(spv::Op::OpFMod, type, {operand(0), divisor}), width);
        }
        case GraphOp::Abs:
            return extended(GLSLstd450FAbs, width, {operand(0)});
        case GraphOp::Floor:
            return extended(GLSLstd450Floor, width, {operand(0)});
        case GraphOp::Ceil:
            return extended(GLSLstd450Ceil, width, {operand(0)});
        case GraphOp::Fract:
            return extended(GLSLstd450Fract, width, {operand(0)});
        case GraphOp::Round: {
            const std::uint32_t x = operand(0);
            const std::uint32_t rounded =
                extended(GLSLstd450Floor, width, {emit(spv::Op::OpFAdd, type, {extended(GLSLstd450FAbs, width, {x}), writer.splatConstant(0.5f, width)})});
            return emit(spv::Op::OpFMul, type, {extended(GLSLstd450FSign, width, {x}), rounded});
        }
        case GraphOp::Trunc:
            return extended(GLSLstd450Trunc, width, {operand(0)});
        case GraphOp::Sign:
            return extended(GLSLstd450FSign, width, {operand(0)});
        case GraphOp::Sqrt: {
            const std::uint32_t x = operand(0);
            return guarded(spv::Op::OpFOrdGreaterThan, x, extended(GLSLstd450Sqrt, width, {x}), width);
        }
        case GraphOp::InverseSqrt: {
            const std::uint32_t x = operand(0);
            return guarded(spv::Op::OpFOrdGreaterThan, x, extended(GLSLstd450InverseSqrt, width, {x}), width);
        }
        case GraphOp::Exp:
            return extended(GLSLstd450Exp, width, {operand(0)});
        case GraphOp::Log: {
            const std::uint32_t x = operand(0);
            return guarded(spv::Op::OpFOrdGreaterThan, x, extended(GLSLstd450Log, width, {x}), width);
        }
        case GraphOp::Sin:
            return extended(GLSLstd450Sin, width, {operand(0)});
        case GraphOp::Cos:
            return extended(GLSLstd450Cos, width, {operand(0)});
        case GraphOp::Tan:
            return extended(GLSLstd450Tan, width, {operand(0)});
        case GraphOp::Asin:
        case GraphOp::Acos: {
            const std::uint32_t clamped =
                extended(GLSLstd450FClamp, width, {operand(0), writer.splatConstant(-1.0f, width), writer.splatConstant(1.0f, width)});
            return extended(instruction.op == GraphOp::Asin ? GLSLstd450Asin : GLSLstd450Acos, width, {clamped});
        }
        case GraphOp::Atan:
            return extended(GLSLstd450Atan, width, {operand(0)});
        case GraphOp::Sinh:
            return extended(GLSLstd450Sinh, width, {operand(0)});
        case GraphOp::Cosh:
            return extended(GLSLstd450Cosh, width, {operand(0)});
        case GraphOp::Tanh:
            return extended(GLSLstd450Tanh, width, {operand(0)});
        case GraphOp::Less:
        case GraphOp::Greater: {
            const spv::Op test = instruction.op == GraphOp::Less ? spv::Op::OpFOrdLessThan : spv::Op::OpFOrdGreaterThan;
            return select(compare(test, operand(0), operand(1), width), writer.splatConstant(1.0f, width), writer.splatConstant(0.0f, width), width);
        }
        case GraphOp::Select: {
            const std::uint32_t condition = compare(spv::Op::OpFOrdGreaterThan, operand(0), writer.splatConstant(0.5f, width), width);
            return select(condition, operand(1), operand(2), width);
        }
        case GraphOp::Dot:
            if (widthOf(0) == 1)
                return emit(spv::Op::OpFMul, type, {values[operands[0]], values[operands[1]]});
            return emit(spv::Op::OpDot, type, {values[operands[0]], values[operands[1]]});
        case GraphOp::Cross:
            return extended(GLSLstd450Cross, width, {values[operands[0]], values[operands[1]]});
        case GraphOp::Length:
            return extended(GLSLstd450Length, width, {values[operands[0]]});
        case GraphOp::Distance:
            return extended(GLSLstd450Distance, width, {values[operands[0]], values[operands[1]]});
        case GraphOp::Normalize: {
            const std::uint32_t vector = values[operands[0]];
            const std::uint32_t size = splat(extended(GLSLstd450Length, 1, {vector}), 1, width);
            return guarded(spv::Op::OpFOrdGreaterThan, size, emit(spv::Op::OpFDiv, type, {vector, size}), width);
        }
        case GraphOp::Reflect:
            return extended(GLSLstd450Reflect, width, {values[operands[0]], values[operands[1]]});
        case GraphOp::Refract:
            return extended(GLSLstd450Refract, width, {values[operands[0]], values[operands[1]], values[operands[2]]});
        case GraphOp::FaceForward:
            return extended(GLSLstd450FaceForward, width, {values[operands[0]], values[operands[1]], values[operands[2]]});
        case GraphOp::Clamp: {
            const std::uint32_t low = extended(GLSLstd450FMin, width, {operand(1), operand(2)});
            const std::uint32_t high = extended(GLSLstd450FMax, width, {operand(1), operand(2)});
            return extended(GLSLstd450FClamp, width, {operand(0), low, high});
        }
        case GraphOp::Mix:
            return extended(GLSLstd450FMix, width, {operand(0), operand(1), operand(2)});
        case GraphOp::Smoothstep:
            return extended(GLSLstd450SmoothStep, width, {operand(0), operand(1), operand(2)});
        case GraphOp::MulAdd:
            return emit(spv::Op::OpFAdd, type, {emit(spv::Op::OpFMul, type, {operand(0), operand(1)}), operand(2)});
        }
        return 0;
    }

    Writer& writer;
    const Graph& graph;
    std::uint32_t glsl;
    std::uint32_t context;
    std::uint32_t material;
    const std::map<GraphFunction, std::uint32_t>& helpers;
    std::vector<std::uint32_t> values;
};
}

std::expected<GraphCode, Core::Error> writeGraphCode(const Graph& graph, const std::string_view name) {
    Writer writer;
    const std::uint32_t glsl = writer.newId();
    std::map<GraphFunction, std::uint32_t> helpers;
    for (const GraphInstruction& instruction : graph.getInstructions()) {
        if (instruction.op == GraphOp::Parameter)
            helpers.emplace(GraphFunction::Parameter, 0);
        if (instruction.op == GraphOp::Texture)
            helpers.emplace(GraphFunction::Texture, 0);
        if (instruction.op == GraphOp::Call)
            helpers.emplace(static_cast<GraphFunction>(instruction.index), 0);
    }
    for (auto& [function, id] : helpers) {
        const GraphFunctionInfo& info = graphFunctionInfo(function);
        Words parameters;
        if (function == GraphFunction::Parameter || function == GraphFunction::Texture)
            parameters.push_back(writer.materialType());
        if (function == GraphFunction::Texture)
            parameters.push_back(writer.floatType(4));
        for (std::size_t parameter = 0; parameter < info.parameterCount(); ++parameter)
            parameters.push_back(writer.floatType(info.parameters[parameter]));
        const std::uint32_t result = writer.floatType(info.result);
        const std::uint32_t type = writer.functionType(result, parameters);
        id = writer.newId();
        writer.name(id, info.name);
        append(writer.declarations, spv::Op::OpFunction, {result, id, static_cast<std::uint32_t>(spv::FunctionControlMask::MaskNone), type});
        for (const std::uint32_t parameter : parameters)
            append(writer.declarations, spv::Op::OpFunctionParameter, {parameter, writer.newId()});
        append(writer.declarations, spv::Op::OpFunctionEnd, {});
    }

    const std::uint32_t output = writer.outputType();
    const std::uint32_t contextPointer = writer.pointerType(writer.contextType());
    const std::uint32_t function = writer.newId();
    writer.name(function, name);
    const std::array parameterTypes{contextPointer, writer.materialType()};
    append(writer.functions, spv::Op::OpFunction,
           {output, function, static_cast<std::uint32_t>(spv::FunctionControlMask::DontInline), writer.functionType(output, parameterTypes)});
    const std::uint32_t context = writer.newId();
    append(writer.functions, spv::Op::OpFunctionParameter, {contextPointer, context});
    const std::uint32_t material = writer.newId();
    append(writer.functions, spv::Op::OpFunctionParameter, {writer.materialType(), material});
    append(writer.functions, spv::Op::OpLabel, {writer.newId()});
    Body(writer, graph, glsl, context, material, helpers).write();
    append(writer.functions, spv::Op::OpFunctionEnd, {});
    const Words words = writer.finish(glsl);
    GraphCode code;
    code.code.resize(words.size() * sizeof(std::uint32_t));
    std::memcpy(code.code.data(), words.data(), code.code.size());
    code.key = Core::hashBytes(code.code);
    return code;
}

std::expected<bool, Core::Error> hasGraphFunction(const std::span<const std::byte> code, const std::string_view name) {
    NoorRHI::SpirvModule module;
    if (auto read = Core::check(module.read(toWords(code))); !read)
        return std::unexpected(std::move(read.error()));
    return module.hasFunction(name);
}

std::expected<std::vector<std::byte>, Core::Error> spliceGraphCode(const std::span<const std::byte> code, const std::span<const std::byte> graphCode,
                                                                   const std::string_view name) {
    NoorRHI::SpirvModule module;
    if (auto read = Core::check(module.read(toWords(code))); !read)
        return std::unexpected(std::move(read.error()));
    if (auto replaced = Core::check(module.replaceFunction(name, toWords(graphCode))); !replaced)
        return std::unexpected(std::move(replaced.error()));
    std::vector<std::uint32_t> written = module.write();
    //! The base keeps graph functions apart for the splice; after it the driver may inline them, as NVIDIA faults on some calls to the large noises.
    for (std::size_t at = 5; at < written.size(); at += written[at] >> 16) {
        if (written[at] >> 16 == 0)
            return Core::fail(Core::ErrorCode::InvalidState, "the spliced shader has an instruction of no words");
        if ((written[at] & 0xffffu) == static_cast<std::uint32_t>(spv::Op::OpFunction))
            written[at + 3] &= ~static_cast<std::uint32_t>(spv::FunctionControlMask::DontInline);
    }
    std::vector<std::byte> bytes(written.size() * sizeof(std::uint32_t));
    std::memcpy(bytes.data(), written.data(), bytes.size());
    return bytes;
}

std::expected<std::vector<std::byte>, Core::Error> linkAlphaGraphCode(const std::span<const std::byte> code,
                                                                      const std::span<const std::span<const std::byte>> donors) {
    NoorRHI::SpirvModule module;
    if (auto read = Core::check(module.read(toWords(code))); !read)
        return std::unexpected(std::move(read.error()));
    std::vector<std::string> names;
    for (std::size_t index = 0; index < donors.size(); ++index) {
        const std::string& name = names.emplace_back(std::format("alphaGraph{}", index));
        if (auto added = Core::check(module.addFunction(name, toWords(donors[index]), alphaGraphDonorName)); !added)
            return std::unexpected(std::move(added.error()));
    }
    const std::vector<std::string_view> callees(names.begin(), names.end());
    if (auto written = Core::check(module.writeSwitch(alphaGraphSwitchName, callees)); !written)
        return std::unexpected(std::move(written.error()));
    Words words = module.write();
    //! The switch and the alpha graphs stay calls: inlined at each visibility test of the kernel, they take the driver minutes to compile.
    std::set<std::uint32_t> calls;
    for (std::size_t at = 5; at < words.size(); at += words[at] >> 16) {
        if (words[at] >> 16 == 0)
            return Core::fail(Core::ErrorCode::InvalidState, "the linked shader has an instruction of no words");
        if ((words[at] & 0xffffu) != static_cast<std::uint32_t>(spv::Op::OpName))
            continue;
        const std::string_view name(reinterpret_cast<const char*>(words.data() + at + 2));
        if (name.starts_with(alphaGraphSwitchName) || name.starts_with("alphaGraph"))
            calls.insert(words[at + 1]);
    }
    for (std::size_t at = 5; at < words.size(); at += words[at] >> 16)
        if ((words[at] & 0xffffu) == static_cast<std::uint32_t>(spv::Op::OpFunction) && calls.contains(words[at + 2]))
            words[at + 3] = (words[at + 3] & ~static_cast<std::uint32_t>(spv::FunctionControlMask::Inline))
                            | static_cast<std::uint32_t>(spv::FunctionControlMask::DontInline);
    std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

}
