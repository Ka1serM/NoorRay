#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "DiffRenderer/Methods/Mesh/Shaders/EnvironmentLight.h"
#include "DiffRenderer/Training/GpuAdam.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! The learned environment light of nvdiffrec: a 6 x 512^2 cube map, prefiltered for the split sum after every step, with the white balance term.
class TrainableLight {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::uint32_t baseSize);
    void close();
    //! Sets every texel of the base to the value and clears the moments.
    [[nodiscard]] std::expected<void, NoorRHI::Error> reset(float value);
    //! Clears the moments, as nvdiffrec starts a new optimizer for its second pass.
    [[nodiscard]] std::expected<void, NoorRHI::Error> restartMoments();
    void recordBuild(NoorRHI::CommandList& list) const;
    void recordClearGradients(NoorRHI::CommandList& list) const;
    void recordBackward(NoorRHI::CommandList& list) const;
    void recordStep(NoorRHI::CommandList& list, const Training::GpuAdam& adam, float learningRate);
    [[nodiscard]] std::expected<void, NoorRHI::Error> downloadBase(std::vector<float>& values) const;

    const EnvironmentLight::Light& getLight() const { return arguments.light; }
    std::uint32_t getBaseSize() const { return arguments.light.baseSize; }
    std::size_t getByteCount() const;

private:
    void dispatchLevels(NoorRHI::CommandList& list, const NoorRHI::ComputePipeline& pipeline, std::uint32_t first, std::uint32_t last, bool down) const;
    std::uint32_t getTexelCount(std::uint32_t level) const;
    std::uint32_t getTotalTexelCount() const;

    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline downsample;
    NoorRHI::ComputePipeline filterSpecular;
    NoorRHI::ComputePipeline filterDiffuse;
    NoorRHI::ComputePipeline clearGradients;
    NoorRHI::ComputePipeline filterSpecularBackward;
    NoorRHI::ComputePipeline filterDiffuseBackward;
    NoorRHI::ComputePipeline upsampleGradient;
    NoorRHI::ComputePipeline finishBaseGradient;
    NoorRHI::ComputePipeline clampBase;
    NoorRHI::Buffer<float> levels;
    NoorRHI::Buffer<NoorRHI::float4> specular;
    NoorRHI::Buffer<float> diffuse;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> specularGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> diffuseGradient;
    NoorRHI::Buffer<float> levelGradient;
    NoorRHI::Buffer<float> first;
    NoorRHI::Buffer<float> second;
    std::uint32_t step = 0;
    EnvironmentLight::Arguments arguments{};
};

}
