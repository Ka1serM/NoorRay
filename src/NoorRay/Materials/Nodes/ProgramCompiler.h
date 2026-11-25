#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Program.h"

namespace NoorRay::Materials {

//! Compiles the displacement of a Blender material into a NoorRay material program (Shaders/Materials/Program.h).
class ProgramCompiler {
public:
    //! The program of the displacement input of the output node, or nothing when it is not linked; the error names a node that programs cannot run.
    std::expected<std::optional<Materials::Program::Source>, std::string> compile(const NodeRef output);

private:
    using Register = std::uint8_t;

    Register allocate(Core::float4 value);
    Register constant(Core::float4 value);
    Register scalar(float value);
    Register emit(std::uint32_t op, std::uint32_t mode = 0, std::initializer_list<Register> inputs = {});
    Register input(const SocketRef socket, const std::function<Register()>& unlinked = nullptr);
    Register input(const NodeRef node, const char* name, const std::function<Register()>& unlinked = nullptr);
    Register inputAt(const NodeRef node, std::size_t index);
    Register output(const NodeRef node, const SocketRef socket);
    Register compileOutput(const NodeRef node, const SocketRef socket);
    Register wave(const NodeRef node);
    Register pack(const NodeRef node, std::initializer_list<const char*> names);
    Register attribute(const NodeRef node, const SocketRef socket);
    Register displacement(const NodeRef node);
    void fail(std::string message);

    NodeLinks links;
    Materials::Program::Source program;
    std::map<std::array<float, 4>, Register> constants;
    std::map<std::pair<std::uintptr_t, std::string>, Register> outputs;
    std::optional<std::string> failure;
};

}
