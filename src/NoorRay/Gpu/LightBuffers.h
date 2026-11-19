#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/MaterialTable.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! Analytic lights first, then one triangle light per triangle of each emissive instance. The alias table samples by estimated power.
class LightBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, std::span<const MaterialInfo> materials,
                                                          float sceneRadius);

    static Shaders::LightRecord makeRecord(const Scene::Light& light, float& weight, float sceneRadius);

    NoorRHI::Ptr<Shaders::LightRecord> getRecords() const { return records.getPtr(); }
    NoorRHI::Ptr<Shaders::AliasEntry> getAlias() const { return alias.getPtr(); }
    std::uint32_t getCount() const { return count; }
    float getTotalPower() const { return totalPower; }
    //! The first light of each instance in dense order, or invalidIndex when the instance does not emit.
    std::span<const std::uint32_t> getInstanceLights() const { return instanceLights; }

private:
    NoorRHI::Buffer<Shaders::LightRecord> records;
    NoorRHI::Buffer<Shaders::AliasEntry> alias;
    std::vector<std::uint32_t> instanceLights;
    std::uint32_t count = 0;
    float totalPower = 0.0f;
};

}
