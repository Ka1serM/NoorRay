#pragma once

#include <cstdint>
#include <functional>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Material.h"
#include "Materials/SVM/SvmCompiler.h"

// Host-side material resource. The shader record from Shared/Material.h is
// reached through the inherited `data` member; this object owns the published
// buffers and the compiled SVM program that feed it.
struct Material : noorrhi::Shared<nr::graphics::Material>
{
    nr::svm::CompiledSvmProgram program;
    std::uint32_t shadowOpaque{};
    std::uint32_t mayEmit{};

    bool hasProgram() const { return !program.bytecode.empty(); }

    void upload(noorrhi::Device& device,
        const std::function<std::uint32_t(std::uint32_t)>& resolveTexture);
    void releaseGpu() { bytecode = {}; textureHandles = {}; release(); }

    noorrhi::Buffer<std::uint32_t> bytecode;
    noorrhi::Buffer<std::uint32_t> textureHandles;
};
