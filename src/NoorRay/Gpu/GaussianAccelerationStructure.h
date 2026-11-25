#pragma once

#include <expected>

#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Gpu {

//! The TLAS of one proxy instance per Gaussian, which Extend and Shadow trace; NOORRAY_GAUSSIAN_TRACING OFF builds a version without it.
class GaussianAccelerationStructure {
public:
    static bool isAvailable();

    void markChanged() { stale = true; }
    bool isStale() const { return stale; }
    //! traced false releases the TLAS, so that a scene whose Gaussians are not traced holds none.
    bool needsBuild(const bool traced) const { return traced ? stale : static_cast<bool>(tlas); }
    [[nodiscard]] std::expected<void, Core::Error> build(NoorRHI::Device& device, const Scene::Scene& scene, bool traced);

    NoorRHI::AccelerationStructureHandle getTlas() const { return tlas ? tlas.getHandle() : NoorRHI::AccelerationStructureHandle{}; }

private:
    [[nodiscard]] std::expected<void, Core::Error> buildProxy(NoorRHI::Device& device);

    NoorRHI::AccelerationStructure proxyBlas;
    NoorRHI::AccelerationStructure tlas;
    bool stale = true;
};

}
