#include "NoorRay/Scene/Camera.h"

#include <algorithm>

namespace NoorRay::Scene {

namespace {

constexpr float emptyNear = 0.01f;
constexpr float emptyFar = 1000.0f;
//! The rasterizer keeps a plain depth buffer, so near stays at least this part of far.
constexpr float depthRange = 1e-4f;

}

void fitClipRange(Camera& camera, const Core::Bounds& bounds) {
    if (bounds.min.x > bounds.max.x) {
        camera.near = emptyNear;
        camera.far = emptyFar;
        return;
    }
    const float radius = std::max(Core::length(bounds.max - bounds.min) * 0.5f, 1e-3f);
    const float distance = Core::length((bounds.min + bounds.max) * 0.5f - camera.transform.getTranslation());
    camera.far = (distance + radius) * 1.5f;
    camera.near = std::max({(distance - radius) * 0.5f, camera.far * depthRange, 1e-5f});
}

}
