#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Gpu {

class MeshBuffers;

class AccelerationStructures {
public:
    void markInstancesMoved() { instancesMoved = true; }
    void markInstanceSetChanged() { instanceSetChanged = true; }
    [[nodiscard]] std::expected<void, Core::Error> build(NoorRHI::Device& device, const Scene::Scene& scene, MeshBuffers& meshes);

    static std::uint8_t getMask(Scene::InstanceFlags flags);
    NoorRHI::AccelerationStructureHandle getTlas() const { return tlas ? tlas.getHandle() : NoorRHI::AccelerationStructureHandle{}; }
    bool isBuilt() const { return built; }
    bool isStale() const { return !built || instancesMoved || instanceSetChanged; }

private:
    NoorRHI::AccelerationStructure tlas;
    std::vector<NoorRHI::Instance> instances;
    bool built = false;
    bool instancesMoved = true;
    bool instanceSetChanged = true;
};

}
