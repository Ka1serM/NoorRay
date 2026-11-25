#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Materials {

struct ProgramParameter {
    std::string name;
    std::uint8_t reg = 0;
    std::uint8_t components = 1;
};

//! A flat register program of a node tree, run by Shaders/Materials/Program.h; each instruction is a schema ProgramInstruction as two words.
class Program {
public:
    struct Source {
        std::vector<Core::float4> registers;
        std::vector<Core::uint2> instructions;
        std::vector<ProgramParameter> parameters;
        std::vector<std::string> attributes;
        std::optional<std::uint8_t> displacement;
    };

    [[nodiscard]] std::expected<void, Core::Error> open(Source source, std::string_view materialName);

    std::span<const Core::float4> getRegisters() const { return registers; }
    std::span<const Core::uint2> getInstructions() const { return instructions; }
    std::span<const ProgramParameter> getParameters() const { return parameters; }
    std::span<const std::string> getAttributes() const { return attributes; }
    std::optional<std::uint8_t> getDisplacement() const { return displacement; }

private:
    std::vector<Core::float4> registers;
    std::vector<Core::uint2> instructions;
    std::vector<ProgramParameter> parameters;
    std::vector<std::string> attributes;
    std::optional<std::uint8_t> displacement;
};

}
