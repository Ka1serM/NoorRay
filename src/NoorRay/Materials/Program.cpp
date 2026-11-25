#include "NoorRay/Materials/Program.h"

#include <format>

#include "NoorRay/Shaders/Materials/Program.h"

namespace NoorRay::Materials {

namespace {

bool isValidMode(const std::uint32_t op, const std::uint32_t mode, const std::size_t attributeCount) {
    switch (op) {
    case Shaders::programOpMath:
        return (mode & ~Shaders::mathClamp) <= Shaders::mathDegrees;
    case Shaders::programOpVectorMath:
        return mode <= Shaders::vectorMathTangent;
    case Shaders::programOpWave:
        return (mode >> 3u) <= Shaders::waveProfileTriangle;
    case Shaders::programOpSeparate:
        return mode < 4;
    case Shaders::programOpAttribute:
        return mode < attributeCount;
    default:
        return op <= Shaders::programOpVectorDisplacement && mode == 0;
    }
}

}

std::expected<void, Core::Error> Program::open(Source source, const std::string_view materialName) {
    const auto invalid = [&](const std::string_view problem) {
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the program of material '{}' {}", materialName, problem));
    };
    const std::size_t registerCount = source.registers.size();
    if (registerCount > Shaders::maxProgramRegisters)
        return invalid(std::format("has more than {} registers", Shaders::maxProgramRegisters));
    for (const Core::uint2 instruction : source.instructions) {
        if (!isValidMode(Shaders::getProgramOp(instruction), Shaders::getProgramMode(instruction), source.attributes.size()))
            return invalid("has an unknown operation");
        for (std::uint32_t input = 0; input < 4; ++input)
            if (Shaders::getProgramInput(instruction, input) >= registerCount)
                return invalid("reads a register that does not exist");
        if (Shaders::getProgramOutput(instruction) >= registerCount)
            return invalid("writes a register that does not exist");
    }
    for (const ProgramParameter& parameter : source.parameters)
        if (parameter.reg >= registerCount || parameter.components == 0 || parameter.components > 4)
            return invalid(std::format("has the parameter '{}' outside its registers", parameter.name));
    if (source.displacement && *source.displacement >= registerCount)
        return invalid("has its displacement outside its registers");
    registers = std::move(source.registers);
    instructions = std::move(source.instructions);
    parameters = std::move(source.parameters);
    attributes = std::move(source.attributes);
    displacement = source.displacement;
    return {};
}

}
