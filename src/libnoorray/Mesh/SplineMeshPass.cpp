#include "Mesh/SplineMeshPass.h"

#include "Realtime/ShaderLoading.h"

namespace
{
// Matches SplineMesh.slang.
constexpr uint32_t GroupSize = 64u;
}

SplineMeshPass::SplineMeshPass(noorrhi::Device& device)
    : device_(device), pipeline_(device.compute(loadShader(device, "Mesh/SplineMesh.spv")))
{
}

void SplineMeshPass::record(noorrhi::Buffer<glm::vec3>& positions, noorrhi::Buffer<TangentFrame>& tangents,
    const nr::graphics::SplineMesh& spline) const
{
    const auto vertexCount = static_cast<uint32_t>(positions.size());
    pipeline_.launch({divideRoundingUp(vertexCount, GroupSize), 1, 1},
        nr::graphics::SplineMeshArguments{positions.ptr().address, tangents.ptr().address, vertexCount, spline});
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
}
