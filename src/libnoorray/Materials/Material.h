#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Material.h"
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
    // The opacity may be below one somewhere, with this material's values:
    // its geometry is traced with the shader's any-hit stages.
    bool transparent{};
};

// Host-side material resource. The shader record from Shared/Material.h is
// reached through the inherited `data` member; this object owns the published
// buffer and the compiled realtime program that feeds it.
struct Material : noorrhi::Shared<nr::graphics::Material>
{
    MaterialShaderProgram shaderProgram;
    // Set once the material runtime published its program.
    bool compiled{};

    void upload(noorrhi::Device& device,
        const std::function<std::uint32_t(std::uint32_t)>& resolveTexture);
    void releaseGpu() { shaderParameters = {}; release(); }

    noorrhi::Buffer<std::uint32_t> shaderParameters;
};
