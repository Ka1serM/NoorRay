#include "NoorRay/Gpu/AccelerationStructures.h"

#include <algorithm>

#include "NoorRHI/Device/Device.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/GaussianBuffers.h"
#include "NoorRay/Gpu/MeshBuffers.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

std::uint8_t AccelerationStructures::getMask(const Scene::InstanceFlags flags) {
    if (Scene::hasFlag(flags, Scene::InstanceFlags::Hidden))
        return 0;
    std::uint32_t mask = Shaders::instanceMaskIndirect;
    if (!Scene::hasFlag(flags, Scene::InstanceFlags::NoShadow))
        mask |= Shaders::instanceMaskShadow;
    if (!Scene::hasFlag(flags, Scene::InstanceFlags::NoCameraRays))
        mask |= Shaders::instanceMaskCamera;
    return static_cast<std::uint8_t>(mask);
}

std::expected<void, Core::Error> AccelerationStructures::buildProxy(NoorRHI::Device& device) {
    const GaussianProxy& proxy = GaussianBuffers::getProxy(device);
    NoorRHI::Buffer<NoorRHI::float3> positions;
    NoorRHI::Buffer<std::uint32_t> indices;
    if (auto uploaded = uploadValues<NoorRHI::float3>(device, positions, proxy.vertices, false); !uploaded)
        return uploaded;
    if (auto uploaded = uploadValues<std::uint32_t>(device, indices, proxy.indices, false); !uploaded)
        return uploaded;
    const NoorRHI::TriangleGeometry geometry{positions.getPtr(), indices.getPtr(), proxy.getTriangleCount(), 3 * sizeof(float), false};
    if (auto result = proxyBlas.build(device, std::span(&geometry, 1), NoorRHI::BuildMode::Static); !result)
        return Core::fail(Core::ErrorCode::Gpu, "the BLAS of the Gaussian proxy: " + result.error().message);
    return {};
}

std::expected<void, Core::Error> AccelerationStructures::buildGaussians(NoorRHI::Device& device, const Scene::Scene& scene) {
    gaussiansStale = false;
    const std::span<const Scene::GaussianCloud> clouds = scene.getGaussianClouds().getValues();
    std::uint64_t total = 0;
    for (const Scene::GaussianCloud& cloud : clouds)
        total += cloud.getCount();
    gaussianTlas = {};
    if (total == 0)
        return {};
    if (total > UINT32_MAX)
        return Core::fail(Core::ErrorCode::Gpu, "the Gaussian TLAS holds at most 2^32 - 1 Gaussians");
    if (!proxyBlas)
        if (auto built = buildProxy(device); !built)
            return built;
    NoorRHI::Buffer<NoorRHI::InstanceRecord> records;
    if (auto opened = Core::check(records.open(device, total)); !opened)
        return opened;
    //! The records go up in batches, so the CPU never holds the instances of a whole cloud.
    std::vector<NoorRHI::InstanceRecord> batch(std::min<std::uint64_t>(total, std::uint64_t{1} << 16));
    std::size_t uploaded = 0;
    for (std::uint32_t index = 0; index < clouds.size(); ++index)
        for (std::uint32_t first = 0; first < clouds[index].getCount(); first += static_cast<std::uint32_t>(batch.size())) {
            const std::span<NoorRHI::InstanceRecord> part = std::span(batch).first(std::min<std::size_t>(batch.size(), clouds[index].getCount() - first));
            GaussianBuffers::writeInstances(clouds[index], index, first, part, proxyBlas.getAddress());
            if (auto result = Core::check(records.upload(part, uploaded)); !result)
                return result;
            uploaded += part.size();
        }
    if (auto result = gaussianTlas.build(device, records.getPtr(), static_cast<std::uint32_t>(total)); !result)
        return Core::fail(Core::ErrorCode::Gpu, "the Gaussian TLAS: " + result.error().message);
    return {};
}

std::expected<void, Core::Error> AccelerationStructures::build(NoorRHI::Device& device, const Scene::Scene& scene, MeshBuffers& meshes) {
    bool blasChanged = false;
    for (std::uint32_t index = 0; index < meshes.getCount(); ++index) {
        MeshGpu& mesh = meshes.getMesh(index);
        const Scene::Mesh& source = scene.getMeshes().getValues()[index];
        if (source.getTriangleCount() == 0)
            continue;
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
    if (gaussiansStale)
        if (auto result = buildGaussians(device, scene); !result)
            return result;
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
