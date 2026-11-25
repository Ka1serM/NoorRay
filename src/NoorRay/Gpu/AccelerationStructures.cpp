#include "NoorRay/Gpu/AccelerationStructures.h"

#include <span>

#include "NoorRHI/Device/Device.h"
#include "NoorRay/Gpu/MeshBuffers.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

std::uint8_t AccelerationStructures::getMask(const Scene::InstanceFlags flags) {
    if (Scene::hasFlag(flags, Scene::InstanceFlags::Hidden))
        return 0;
    std::uint32_t mask = 0;
    const auto show = [&](const Scene::InstanceFlags hidden, const std::uint32_t bit) { mask |= Scene::hasFlag(flags, hidden) ? 0u : bit; };
    show(Scene::InstanceFlags::NoShadow, Shaders::instanceMaskShadow);
    show(Scene::InstanceFlags::NoCameraRays, Shaders::instanceMaskCamera);
    show(Scene::InstanceFlags::NoDiffuseRays, Shaders::instanceMaskDiffuse);
    show(Scene::InstanceFlags::NoGlossyRays, Shaders::instanceMaskGlossy);
    show(Scene::InstanceFlags::NoTransmissionRays, Shaders::instanceMaskTransmission);
    show(Scene::InstanceFlags::NoVolumeScatterRays, Shaders::instanceMaskVolumeScatter);
    return static_cast<std::uint8_t>(mask);
}

std::expected<void, Core::Error> AccelerationStructures::build(NoorRHI::Device& device, const Scene::Scene& scene, MeshBuffers& meshes) {
    bool blasChanged = false;
    for (std::uint32_t index = 0; index < meshes.getCount(); ++index) {
        MeshGpu& mesh = meshes.getMesh(index);
        const Scene::Mesh& source = scene.getMeshes().getValues()[index];
        if (source.getTriangleCount() == 0) {
            blasChanged = blasChanged || mesh.blas;
            mesh.blas = {};
            continue;
        }
        if (mesh.blasStale || !mesh.blas) {
            const NoorRHI::TriangleGeometry geometry{mesh.positions.getPtr(), mesh.indices.getPtr(), source.getTriangleCount(), 3 * sizeof(float), mesh.opaque};
            if (auto result = mesh.blas.build(device, std::span(&geometry, 1), NoorRHI::BuildMode::Dynamic); !result)
                return Core::fail(Core::ErrorCode::Gpu, "BLAS of mesh '" + source.name + "': " + result.error().message);
            blasChanged = true;
        } else if (mesh.blasRefit) {
            if (auto result = mesh.blas.update(); !result)
                return Core::fail(std::move(result.error()));
            blasChanged = true;
        }
        mesh.blasStale = false;
        mesh.blasRefit = false;
    }
    if (built && !instancesMoved && !instanceSetChanged && !blasChanged)
        return {};
    const bool sameSet = built && !instanceSetChanged && !blasChanged;
    instances.clear();
    const std::span<const Scene::Instance> sceneInstances = scene.getInstances().getValues();
    for (std::uint32_t index = 0; index < sceneInstances.size(); ++index) {
        const Scene::Instance& instance = sceneInstances[index];
        const std::uint32_t mesh = scene.getMeshes().denseIndex(instance.mesh);
        if (mesh < meshes.getCount() && meshes.getMesh(mesh).blas)
            instances.push_back({&meshes.getMesh(mesh).blas, instance.transform.toMatrix(), index, 0, getMask(instance.flags)});
    }
    if (instances.empty()) {
        tlas = {};
        built = true;
        instancesMoved = false;
        instanceSetChanged = false;
        return {};
    }
    std::expected<void, NoorRHI::Error> result = sameSet && tlas ? tlas.update(instances) : tlas.build(device, instances);
    if (!result)
        return Core::fail(Core::ErrorCode::Gpu, "TLAS: " + result.error().message);
    built = true;
    instancesMoved = false;
    instanceSetChanged = false;
    return {};
}

}
