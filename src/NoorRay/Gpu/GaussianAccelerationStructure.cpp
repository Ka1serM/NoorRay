#include "NoorRay/Gpu/GaussianAccelerationStructure.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

//! Encloses the unit sphere; outward faces wind counterclockwise.
struct GaussianProxy {
    std::vector<Core::float3> vertices;
    std::vector<std::uint32_t> indices;
    std::uint32_t getTriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
};

constexpr float goldenRatio = 1.6180339887498949f;

GaussianProxy makeIcosahedron() {
    GaussianProxy proxy;
    proxy.vertices = {{-1.0f, goldenRatio, 0.0f}, {1.0f, goldenRatio, 0.0f}, {-1.0f, -goldenRatio, 0.0f}, {1.0f, -goldenRatio, 0.0f},
                      {0.0f, -1.0f, goldenRatio}, {0.0f, 1.0f, goldenRatio}, {0.0f, -1.0f, -goldenRatio}, {0.0f, 1.0f, -goldenRatio},
                      {goldenRatio, 0.0f, -1.0f}, {goldenRatio, 0.0f, 1.0f}, {-goldenRatio, 0.0f, -1.0f}, {-goldenRatio, 0.0f, 1.0f}};
    proxy.indices = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
                     3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};
    return proxy;
}

GaussianProxy makeOctahedron() {
    GaussianProxy proxy;
    proxy.vertices = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}};
    proxy.indices = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    return proxy;
}

//! Splits each triangle into four, with the new vertices on the sphere through the old ones.
GaussianProxy subdivide(const GaussianProxy& coarse) {
    GaussianProxy fine;
    fine.vertices = coarse.vertices;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> midpoints;
    const auto midpoint = [&](const std::uint32_t a, const std::uint32_t b) {
        const auto [found, added] = midpoints.try_emplace({std::min(a, b), std::max(a, b)}, static_cast<std::uint32_t>(fine.vertices.size()));
        if (added)
            fine.vertices.push_back(Core::normalize(fine.vertices[a] + fine.vertices[b]) * Core::length(fine.vertices[a]));
        return found->second;
    };
    for (std::size_t first = 0; first < coarse.indices.size(); first += 3) {
        const std::uint32_t a = coarse.indices[first];
        const std::uint32_t b = coarse.indices[first + 1];
        const std::uint32_t c = coarse.indices[first + 2];
        const std::uint32_t ab = midpoint(a, b);
        const std::uint32_t bc = midpoint(b, c);
        const std::uint32_t ca = midpoint(c, a);
        fine.indices.insert(fine.indices.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    return fine;
}

//! Scales the polyhedron so that its nearest face plane touches the unit sphere, which it then encloses.
GaussianProxy encloseUnitSphere(GaussianProxy proxy) {
    float inradius = INFINITY;
    for (std::size_t first = 0; first < proxy.indices.size(); first += 3) {
        const Core::float3 a = proxy.vertices[proxy.indices[first]];
        const Core::float3 normal = Core::normalize(Core::cross(proxy.vertices[proxy.indices[first + 1]] - a, proxy.vertices[proxy.indices[first + 2]] - a));
        inradius = std::min(inradius, std::abs(Core::dot(normal, a)));
    }
    for (Core::float3& vertex : proxy.vertices)
        vertex = vertex * (1.0f / inradius);
    return proxy;
}

GaussianProxy makeProxy(const std::uint32_t triangles) {
    if (triangles == 8)
        return encloseUnitSphere(makeOctahedron());
    GaussianProxy proxy = makeIcosahedron();
    for (std::uint32_t count = 20; count < triangles; count *= 4)
        proxy = subdivide(proxy);
    return encloseUnitSphere(std::move(proxy));
}

const GaussianProxy& getProxy(const NoorRHI::Device& device) {
    static const GaussianProxy hardware = makeProxy(80);
    static const GaussianProxy software = makeProxy(20);
    return device.getFeatures().hardwareRayTracing ? hardware : software;
}

Core::float3 rotate(const Core::float4 q, const Core::float3 v) {
    const Core::float3 axis{q.x, q.y, q.z};
    const Core::float3 t = Core::cross(axis, v) * 2.0f;
    return v + t * q.w + Core::cross(axis, t);
}

//! The opacity cutoff 1/255 ends a Gaussian at sqrt(2 ln(255 opacity)) standard deviations.
void writeInstances(const Scene::GaussianCloud& cloud, const std::uint32_t cloudIndex, const std::uint32_t first,
                    const std::span<NoorRHI::InstanceRecord> records, const std::uint64_t proxyAddress) {
    for (std::uint32_t offset = 0; offset < records.size(); ++offset) {
        const std::uint32_t i = first + offset;
        const float opacity = cloud.opacities[i];
        const float cutoff = opacity * 255.0f > 1.0f ? std::sqrt(2.0f * std::log(opacity * 255.0f)) : 0.0f;
        const Core::float3 scale = cloud.scales[i] * cutoff;
        const Core::float4 rotation = cloud.rotations[i];
        const Core::float3 x = rotate(rotation, {scale.x, 0.0f, 0.0f});
        const Core::float3 y = rotate(rotation, {0.0f, scale.y, 0.0f});
        const Core::float3 z = rotate(rotation, {0.0f, 0.0f, scale.z});
        const Core::float3 mean = cloud.positions[i];
        const Core::Transform shape{{{x.x, y.x, z.x, mean.x}, {x.y, y.y, z.y, mean.y}, {x.z, y.z, z.z, mean.z}}};
        const Core::Transform world = cloud.transform * shape;
        NoorRHI::InstanceRecord& record = records[offset];
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 4; ++column)
                record.transform[row * 4 + column] = world.rows[row][column];
        record.customIndexAndMask = cloudIndex | (cutoff > 0.0f ? 0xffu : 0u) << 24;
        record.shaderBindingTableOffsetAndFlags = 0;
        record.blasAddress = proxyAddress;
    }
}

}

bool GaussianAccelerationStructure::isAvailable() {
    return true;
}

std::expected<void, Core::Error> GaussianAccelerationStructure::buildProxy(NoorRHI::Device& device) {
    const GaussianProxy& proxy = getProxy(device);
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

std::expected<void, Core::Error> GaussianAccelerationStructure::build(NoorRHI::Device& device, const Scene::Scene& scene, const bool traced) {
    if (!traced) {
        tlas = {};
        stale = true;
        return {};
    }
    if (!stale)
        return {};
    const std::span<const Scene::GaussianCloud> clouds = scene.getGaussianClouds().getValues();
    std::uint64_t total = 0;
    for (const Scene::GaussianCloud& cloud : clouds)
        total += cloud.getCount();
    tlas = {};
    if (total == 0) {
        stale = false;
        return {};
    }
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
            writeInstances(clouds[index], index, first, part, proxyBlas.getAddress());
            if (auto result = Core::check(records.upload(part, uploaded)); !result)
                return result;
            uploaded += part.size();
        }
    if (auto result = tlas.build(device, records.getPtr(), static_cast<std::uint32_t>(total)); !result)
        return Core::fail(Core::ErrorCode::Gpu, "the Gaussian TLAS: " + result.error().message);
    stale = false;
    return {};
}

}
