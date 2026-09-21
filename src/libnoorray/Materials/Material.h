#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Material.h"
#include "Materials/SVM/SvmCompiler.h"
#include "Materials/MaterialX/SlangMaterialCompiler.h"

// A parameter-block word that holds a scene texture's descriptor handle.
struct MaterialTextureWord
{
    std::uint32_t word{};
    // Index into the scene's textures; one that does not resolve samples white.
    std::uint32_t texture{~0u};
};

// A material's MaterialX surface compiled for the realtime renderer.
struct MaterialShaderProgram
{
    std::shared_ptr<const nr::materialx::MaterialShader> shader;
    std::vector<std::uint32_t> parameters;
    std::vector<MaterialTextureWord> textures;
};

// Host-side material resource. The shader record from Shared/Material.h is
// reached through the inherited `data` member; this object owns the published
// buffers and the compiled SVM and realtime programs that feed it.
struct Material : noorrhi::Shared<nr::graphics::Material>
{
    nr::svm::CompiledSvmProgram program;
    MaterialShaderProgram shaderProgram;
    std::uint32_t shadowOpaque{};
    std::uint32_t mayEmit{};
    // Set once the material runtime published its programs.
    bool compiled{};

    bool hasProgram() const { return !program.bytecode.empty(); }

    // `shaderIndex` is the callable index of shaderProgram.shader, ~0u without one.
    void upload(noorrhi::Device& device,
        const std::function<std::uint32_t(std::uint32_t)>& resolveTexture,
        std::uint32_t shaderIndex);
    void releaseGpu() { bytecode = {}; textureHandles = {}; shaderParameters = {}; release(); }

    noorrhi::Buffer<std::uint32_t> bytecode;
    noorrhi::Buffer<std::uint32_t> textureHandles;
    noorrhi::Buffer<std::uint32_t> shaderParameters;
};
