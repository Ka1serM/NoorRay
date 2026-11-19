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
    void markGaussiansChanged() { gaussiansStale = true; }
    [[nodiscard]] std::expected<void, Core::Error> build(NoorRHI::Device& device, const Scene::Scene& scene, MeshBuffers& meshes);

    static std::uint8_t getMask(Scene::InstanceFlags flags);
    NoorRHI::AccelerationStructureHandle getTlas() const { return tlas ? tlas.getHandle() : NoorRHI::AccelerationStructureHandle{}; }
    NoorRHI::AccelerationStructureHandle getGaussianTlas() const { return gaussianTlas ? gaussianTlas.getHandle() : NoorRHI::AccelerationStructureHandle{}; }
    bool isBuilt() const { return built; }
    bool isStale() const { return !built || instancesMoved || instanceSetChanged || gaussiansStale; }

private:
    [[nodiscard]] std::expected<void, Core::Error> buildProxy(NoorRHI::Device& device);
    [[nodiscard]] std::expected<void, Core::Error> buildGaussians(NoorRHI::Device& device, const Scene::Scene& scene);

    NoorRHI::AccelerationStructure tlas;
    NoorRHI::AccelerationStructure proxyBlas;
    NoorRHI::AccelerationStructure gaussianTlas;
    std::vector<NoorRHI::Instance> instances;
    bool built = false;
    bool instancesMoved = true;
    bool instanceSetChanged = true;
    bool gaussiansStale = true;
};

}
