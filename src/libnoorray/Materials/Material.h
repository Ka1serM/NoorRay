#pragma once

#include <cstdint>
#include <functional>

#include <gpu/gpu.hpp>

#include "Shared/Material.h"
#include "Materials/SVM/SvmCompiler.h"

// Host-side material resource. The shader record from Shared/Material.h is
// reached through the inherited `data` member; this object owns the published
// buffers and the compiled SVM program that feed it.
struct Material : gpu::Shared<nr::graphics::Material>
{
    nr::svm::CompiledSvmProgram program;
    std::uint32_t shadowOpaque{};
    std::uint32_t mayEmit{};

    bool hasProgram() const { return !program.bytecode.empty(); }

    void upload(gpu::Device& device,
        const std::function<std::uint32_t(std::uint32_t)>& resolveTexture);
    void releaseGpu() { bytecode = {}; textureHandles = {}; release(); }

    gpu::Buffer<std::uint32_t> bytecode;
    gpu::Buffer<std::uint32_t> textureHandles;
};
