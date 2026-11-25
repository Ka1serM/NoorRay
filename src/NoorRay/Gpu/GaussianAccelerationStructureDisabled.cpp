#include "NoorRay/Gpu/GaussianAccelerationStructure.h"

namespace NoorRay::Gpu {

bool GaussianAccelerationStructure::isAvailable() {
    return false;
}

std::expected<void, Core::Error> GaussianAccelerationStructure::build(NoorRHI::Device&, const Scene::Scene&, const bool traced) {
    if (!traced)
        return {};
    return Core::fail(Core::ErrorCode::Unsupported, "this build has no Gaussian ray tracing");
}

}
