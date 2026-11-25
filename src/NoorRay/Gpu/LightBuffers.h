#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GraphBakes.h"
#include "NoorRay/Gpu/LightTree.h"
#include "NoorRay/Gpu/MaterialTable.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! Analytic lights first, then one triangle light per triangle of each emissive instance. A light tree per light set samples them by importance.
class LightBuffers {
public:
    //! bakes holds the light shader of each light in dense order.
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, std::span<const MaterialInfo> materials,
                                                          std::span<const LightBake> bakes);

    static Shaders::LightRecord makeRecord(const Scene::Light& light, const LightBake& bake, LightBounds& bounds);

    NoorRHI::Ptr<Shaders::LightRecord> getRecords() const { return records.getPtr(); }
    NoorRHI::Ptr<Shaders::LightTreeNode> getNodes() const { return nodes.getPtr(); }
    NoorRHI::Ptr<Shaders::LightTreeSet> getTreeSets() const { return treeSets.getPtr(); }
    NoorRHI::Ptr<NoorRHI::uint2> getTrails() const { return trails.getPtr(); }
    std::uint32_t getCount() const { return count; }
    bool areVisibleToCamera() const { return visibleToCamera; }
    //! The first light of each instance in dense order, or invalidIndex when the instance does not emit.
    std::span<const std::uint32_t> getInstanceLights() const { return instanceLights; }
    NoorRHI::Ptr<std::uint32_t> getEmitterLights() const { return emitterLights.getPtr(); }
    NoorRHI::Ptr<NoorRHI::uint2> getEmitterRanges() const { return emitterRanges.getPtr(); }
    //! The graph programs with triangle lights to bake, and their light counts.
    std::span<const std::pair<std::uint32_t, std::uint32_t>> getEmitterPrograms() const { return emitterPrograms; }
    //! Changes when a light index can name another light than before (an emissive instance hidden); older light indices hold only while it stays.
    std::uint64_t getLayout() const { return layout; }
    //! Changes when the light trees are built again; a node index holds only while it stays.
    std::uint64_t getTrees() const { return trees; }

private:
    NoorRHI::Buffer<Shaders::LightRecord> records;
    NoorRHI::Buffer<Shaders::LightTreeNode> nodes;
    NoorRHI::Buffer<Shaders::LightTreeSet> treeSets;
    NoorRHI::Buffer<NoorRHI::uint2> trails;
    NoorRHI::Buffer<std::uint32_t> emitterLights;
    bool visibleToCamera = false;
    NoorRHI::Buffer<NoorRHI::uint2> emitterRanges;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> emitterPrograms;
    std::vector<std::uint32_t> instanceLights;
    std::uint32_t count = 0;
    std::uint64_t layout = 0;
    std::uint64_t trees = 0;
};

}
