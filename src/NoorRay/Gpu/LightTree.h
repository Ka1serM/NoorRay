#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! What the tree knows of a light: the bounds and cones that the build splits by, and the SG light that the importance of its nodes takes.
struct LightBounds {
    Core::Bounds bounds;
    //! A distant light has its direction to the light and its half angle as theta_o.
    Core::float3 axis{0.0f, 0.0f, 1.0f};
    float cosThetaO = 1.0f;
    float cosThetaE = 1.0f;
    //! The largest intensity in W/sr times the luminance of the colour, or the irradiance of a distant light; zero leaves the light out.
    float power = 0.0f;
    bool twoSided = false;
    //! A one-sided light in one plane, dot(axis, p) = planeOffset, lights nothing behind it.
    bool planar = false;
    float planeOffset = 0.0f;
    bool distant = false;
    //! The mean position and the spatial variance (the trace of the covariance) of the emitting points.
    Core::float3 mean{};
    float variance = 0.0f;
    //! The mean emission direction weighted by intensity; its length sets the vMF sharpness, as Banerjee et al. 2005.
    Core::float3 emission{};
    //! The emitted flux times the luminance of the colour; the SG nodes weigh their children by it.
    float flux = 0.0f;
    //! The light sets that the light reaches, as Cycles' light linking.
    std::uint64_t sets = 0;
};

struct LightTrees {
    std::vector<Shaders::LightTreeNode> nodes;
    std::vector<Shaders::LightTreeSet> sets;
    //! The trail of light l in set s is at s * light count + l; bit d is set when the leaf is under the second child at depth d.
    std::vector<Core::uint2> trails;
};

//! One tree per light set over the local lights that reach it, split by the surface area orientation heuristic, with the distant lights of the set after it.
[[nodiscard]] std::expected<LightTrees, Core::Error> buildLightTrees(std::span<const LightBounds> lights, std::uint32_t setCount);

}
