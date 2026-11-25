#pragma once

#include <array>
#include <cstdint>
#include <expected>

#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

struct LargeStepSettings {
    float lambda = 20.0f;
    std::uint32_t solverIterations = 16;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

//! Nicolet et al. 2021: the vertices move in u = (I + lambda L) x, and the uniform Adam of the paper steps u with the gradient (I + lambda L)^-1 dLoss/dx.
class LargeStepOptimizer {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const LargeStepSettings& settings);
    void close();
    //! Takes the neighbours of the mesh and restarts the moments.
    [[nodiscard]] std::expected<void, NoorRHI::Error> setMesh(const SurfaceMesh& mesh);
    void record(NoorRHI::CommandList& list, NoorRHI::Ptr<float> positions, NoorRHI::Ptr<float> gradients, float stepSize);

private:
    std::uint32_t getGroupCount() const;
    void recordSolve(NoorRHI::CommandList& list, NoorRHI::Ptr<float> right) const;

    NoorRHI::Device* device = nullptr;
    LargeStepSettings settings;
    NoorRHI::ComputePipeline startSolve;
    NoorRHI::ComputePipeline multiply;
    NoorRHI::ComputePipeline update;
    NoorRHI::ComputePipeline advance;
    NoorRHI::ComputePipeline uniformMoments;
    NoorRHI::ComputePipeline uniformStep;
    NoorRHI::ComputePipeline resetLargest;
    NoorRHI::ComputePipeline applyOffsets;
    NoorRHI::Buffer<NoorRHI::uint> offsets;
    NoorRHI::Buffer<NoorRHI::uint> neighbors;
    NoorRHI::Buffer<float> solution;
    NoorRHI::Buffer<float> residual;
    NoorRHI::Buffer<float> direction;
    NoorRHI::Buffer<float> product;
    NoorRHI::Buffer<float> first;
    NoorRHI::Buffer<float> second;
    NoorRHI::Buffer<float> steps;
    NoorRHI::Buffer<float> products;
    std::array<NoorRHI::Buffer<float>, 2> residuals;
    NoorRHI::Buffer<NoorRHI::Atomic<NoorRHI::uint>> largestSecond;
    std::uint32_t vertexCount = 0;
    std::uint32_t step = 0;
};

}
