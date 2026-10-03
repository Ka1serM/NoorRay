#include "Mesh/LandscapePass.h"
#include "Realtime/ShaderLoading.h"

LandscapePass::LandscapePass(noorrhi::Device& device)
    : device_(device), pipeline_(device.compute(loadShader(device, "Mesh/LandscapeMesh.spv"))) {}

void LandscapePass::record(noorrhi::Buffer<glm::vec3>& positions) const
{
    const auto count = static_cast<uint32_t>(positions.size());
    pipeline_.launch({(count + 63u) / 64u, 1, 1},
        nr::graphics::LandscapeMeshArguments{positions.ptr().address, count});
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
}
