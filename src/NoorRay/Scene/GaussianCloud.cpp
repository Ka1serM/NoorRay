#include "NoorRay/Scene/GaussianCloud.h"

namespace NoorRay::Scene {

Core::Bounds GaussianCloud::computeBounds() const {
    Core::Bounds bounds;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const float radius = 3.0f * Core::maxComponent(scales[i]);
        bounds.grow(positions[i] - Core::float3{radius, radius, radius});
        bounds.grow(positions[i] + Core::float3{radius, radius, radius});
    }
    return bounds;
}

}
