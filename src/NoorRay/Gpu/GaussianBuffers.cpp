#include "NoorRay/Gpu/GaussianBuffers.h"

#include <cmath>
#include <map>
#include <utility>

#include "NoorRHI/Device/Device.h"
#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

constexpr float goldenRatio = 1.6180339887498949f;

GaussianProxy makeIcosahedron() {
    GaussianProxy proxy;
    proxy.vertices = {{-1.0f, goldenRatio, 0.0f}, {1.0f, goldenRatio, 0.0f},  {-1.0f, -goldenRatio, 0.0f}, {1.0f, -goldenRatio, 0.0f},
                      {0.0f, -1.0f, goldenRatio}, {0.0f, 1.0f, goldenRatio},  {0.0f, -1.0f, -goldenRatio}, {0.0f, 1.0f, -goldenRatio},
                      {goldenRatio, 0.0f, -1.0f}, {goldenRatio, 0.0f, 1.0f},  {-goldenRatio, 0.0f, -1.0f}, {-goldenRatio, 0.0f, 1.0f}};
    proxy.indices = {0, 11, 5, 0, 5,  1,  0,  1,  7,  0,  7,  10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                     3, 9,  4, 3, 4,  2,  3,  2,  6,  3,  6,  8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6, 2,  10, 8, 6, 7, 9, 8, 1};
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

Core::float3 rotate(const Core::float4 q, const Core::float3 v) {
    const Core::float3 axis{q.x, q.y, q.z};
    const Core::float3 t = Core::cross(axis, v) * 2.0f;
    return v + t * q.w + Core::cross(axis, t);
}

}

const GaussianProxy& GaussianBuffers::getProxy(const NoorRHI::Device& device) {
    static const GaussianProxy hardware = makeProxy(80);
    static const GaussianProxy software = makeProxy(20);
    return device.getFeatures().hardwareRayTracing ? hardware : software;
}

void GaussianBuffers::writeInstances(const Scene::GaussianCloud& cloud, const std::uint32_t cloudIndex, const std::uint32_t first,
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

std::expected<void, Core::Error> GaussianBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene) {
    const std::span<const Scene::GaussianCloud> sceneClouds = scene.getGaussianClouds().getValues();
    clouds.resize(sceneClouds.size());
    std::vector<Shaders::GaussianCloudRecord> values(sceneClouds.size());
    gaussianCount = 0;
    std::vector<NoorRHI::float4> packed;
    for (std::size_t c = 0; c < sceneClouds.size(); ++c) {
        const Scene::GaussianCloud& cloud = sceneClouds[c];
        if (!clouds[c])
            clouds[c] = std::make_unique<GaussianCloudGpu>();
        GaussianCloudGpu& gpu = *clouds[c];
        const std::uint32_t count = cloud.getCount();
        gaussianCount += count;
        packed.resize(count);
        for (std::uint32_t i = 0; i < count; ++i)
            packed[i] = {cloud.positions[i].x, cloud.positions[i].y, cloud.positions[i].z, cloud.opacities[i]};
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.positionOpacity, packed, false); !result)
            return result;
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.rotations, cloud.rotations, false); !result)
            return result;
        for (std::uint32_t i = 0; i < count; ++i)
            packed[i] = {cloud.scales[i].x, cloud.scales[i].y, cloud.scales[i].z, 0.0f};
        if (auto result = uploadValues<NoorRHI::float4>(device, gpu.scales, packed, false); !result)
            return result;
        //! Two little-endian halves are one word, so the coefficients upload as they are, without a packed copy.
        const std::size_t shWords = (cloud.sh.size() + 1) / 2;
        if (!gpu.sh || gpu.sh.getSize() != std::max<std::size_t>(shWords, 1))
            if (auto opened = gpu.sh.open(device, std::max<std::size_t>(shWords, 1)); !opened)
                return Core::fail(std::move(opened.error()));
        const std::span<const std::uint32_t> pairs(reinterpret_cast<const std::uint32_t*>(cloud.sh.data()), cloud.sh.size() / 2);
        if (auto uploaded = Core::check(gpu.sh.upload(pairs)); !uploaded)
            return uploaded;
        if (cloud.sh.size() % 2 != 0) {
            const std::uint32_t last = cloud.sh.back();
            if (auto uploaded = Core::check(gpu.sh.upload(std::span(&last, 1), pairs.size())); !uploaded)
                return uploaded;
        }
        Shaders::GaussianCloudRecord& record = values[c];
        record.positionOpacity = gpu.positionOpacity.getPtr();
        record.rotations = gpu.rotations.getPtr();
        record.scales = gpu.scales.getPtr();
        record.sh = gpu.sh.getPtr();
        const Core::Transform inverse = cloud.transform.inverse();
        for (int row = 0; row < 3; ++row) {
            record.objectToWorld[row] = {cloud.transform.rows[row][0], cloud.transform.rows[row][1], cloud.transform.rows[row][2],
                                         cloud.transform.rows[row][3]};
            record.worldToObject[row] = {inverse.rows[row][0], inverse.rows[row][1], inverse.rows[row][2], inverse.rows[row][3]};
        }
        record.count = count;
        record.shDegree = cloud.shDegree;
        record.shStride = 3 * Scene::getShCoefficientCount(cloud.shDegree);
        record.firstInstance = static_cast<std::uint32_t>(gaussianCount - count);
    }
    return uploadValues<Shaders::GaussianCloudRecord>(device, records, values);
}

}
