#include "DiffRenderer/Core/GaussianProxies.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "GaussianInstances.slang.h"

namespace DiffRenderer::Core {

using namespace NoorRHI;

namespace {

constexpr float goldenRatio = 1.6180339887498949f;
//! A dispatch has at most 65535 groups in x; larger arrays loop with a stride of the thread count.
constexpr std::uint32_t maxThreads = 65535u * 256u;

float3 subtract(const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 scale(const float3 value, const float factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

float length(const float3 value) {
    return std::sqrt(dot(value, value));
}

float3 rotate(const std::span<const float, 4> xyzw, const float3 value) {
    const float3 axis{xyzw[0], xyzw[1], xyzw[2]};
    const float3 t = scale(cross(axis, value), 2.0f);
    const float3 turned = cross(axis, t);
    return {value.x + t.x * xyzw[3] + turned.x, value.y + t.y * xyzw[3] + turned.y, value.z + t.z * xyzw[3] + turned.z};
}

InstanceRecord getRecord(const std::span<const float, 3> mean, const std::span<const float, 4> rotation, const std::span<const float, 3> scales,
                         const float opacity, const std::uint64_t blasAddress) {
    const bool visible = opacity * 255.0f > 1.0f;
    //! Vulkan needs an invertible transform, so a Gaussian below the cutoff keeps a size and its mask hides it.
    const float cutoff = visible ? std::sqrt(2.0f * std::log(opacity * 255.0f)) : 1.0f;
    const std::array<float3, 3> columns{rotate(rotation, {scales[0] * cutoff, 0.0f, 0.0f}), rotate(rotation, {0.0f, scales[1] * cutoff, 0.0f}),
                                        rotate(rotation, {0.0f, 0.0f, scales[2] * cutoff})};
    return {.transform = {columns[0].x, columns[1].x, columns[2].x, mean[0], columns[0].y, columns[1].y, columns[2].y, mean[1], columns[0].z, columns[1].z,
                          columns[2].z, mean[2]},
            .customIndexAndMask = visible ? 0xff000000u : 0u,
            .shaderBindingTableOffsetAndFlags = 0,
            .blasAddress = blasAddress};
}

}

GaussianProxyMesh::GaussianProxyMesh() {
    vertices = {{-1.0f, goldenRatio, 0.0f}, {1.0f, goldenRatio, 0.0f}, {-1.0f, -goldenRatio, 0.0f}, {1.0f, -goldenRatio, 0.0f},
                {0.0f, -1.0f, goldenRatio}, {0.0f, 1.0f, goldenRatio}, {0.0f, -1.0f, -goldenRatio}, {0.0f, 1.0f, -goldenRatio},
                {goldenRatio, 0.0f, -1.0f}, {goldenRatio, 0.0f, 1.0f}, {-goldenRatio, 0.0f, -1.0f}, {-goldenRatio, 0.0f, 1.0f}};
    indices = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
               3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};
    subdivide();
    const float factor = 1.0f / getInradius();
    for (float3& vertex : vertices)
        vertex = scale(vertex, factor);
}

void GaussianProxyMesh::subdivide() {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> midpoints;
    const auto getMidpoint = [&](const std::uint32_t a, const std::uint32_t b) {
        const auto [found, added] = midpoints.try_emplace({std::min(a, b), std::max(a, b)}, static_cast<std::uint32_t>(vertices.size()));
        if (added) {
            const float3 sum{vertices[a].x + vertices[b].x, vertices[a].y + vertices[b].y, vertices[a].z + vertices[b].z};
            vertices.push_back(scale(sum, length(vertices[a]) / length(sum)));
        }
        return found->second;
    };
    std::vector<std::uint32_t> fine;
    fine.reserve(indices.size() * 4);
    for (std::size_t first = 0; first < indices.size(); first += 3) {
        const std::uint32_t a = indices[first];
        const std::uint32_t b = indices[first + 1];
        const std::uint32_t c = indices[first + 2];
        const std::uint32_t ab = getMidpoint(a, b);
        const std::uint32_t bc = getMidpoint(b, c);
        const std::uint32_t ca = getMidpoint(c, a);
        fine.insert(fine.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    indices = std::move(fine);
}

float GaussianProxyMesh::getInradius() const {
    float inradius = std::numeric_limits<float>::infinity();
    for (std::size_t first = 0; first < indices.size(); first += 3) {
        const float3 a = vertices[indices[first]];
        const float3 normal = cross(subtract(vertices[indices[first + 1]], a), subtract(vertices[indices[first + 2]], a));
        inradius = std::min(inradius, std::abs(dot(normal, a)) / length(normal));
    }
    return inradius;
}

std::expected<void, Error> GaussianProxies::open(Device& target, const std::uint32_t instanceCapacity) {
    device = &target;
    capacity = instanceCapacity;
    built = 0;
    tlas = {};
    if (!target.getFeatures().hardwareRayTracing)
        return fail(ErrorCode::Unsupported, "the device has no hardware ray tracing");
    const GaussianProxyMesh mesh;
    for (std::expected<void, Error> step : {writeInstances.open(target, GaussianInstances::writeInstances), vertices.open(target, mesh.getVertices().size()),
                                            indices.open(target, mesh.getIndices().size()), records.open(target, capacity), vertices.upload(mesh.getVertices()),
                                            indices.upload(mesh.getIndices())})
        if (!step)
            return step;
    const TriangleGeometry geometry{.positions = vertices.getPtr(),
                                    .indices = indices.getPtr(),
                                    .triangleCount = static_cast<std::uint32_t>(mesh.getIndices().size() / 3),
                                    .opaque = false};
    return blas.build(target, {&geometry, 1}, BuildMode::Static);
}

std::expected<void, Error> GaussianProxies::update(Device& target, const std::span<const float> means, const std::span<const float> quaternionsXyzw,
                                                   const std::span<const float> scales, const std::span<const float> opacities, const std::uint32_t count) {
    if (&target != device)
        return fail(ErrorCode::InvalidState, "the Gaussian proxies are open on another device");
    if (count > capacity)
        return fail(ErrorCode::InvalidArgument, "the Gaussians exceed the capacity of the proxies");
    hostRecords.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        hostRecords[index] = getRecord(means.subspan(3 * index).first<3>(), quaternionsXyzw.subspan(4 * index).first<4>(), scales.subspan(3 * index).first<3>(),
                                       opacities[index], blas.getAddress());
    if (std::expected<void, Error> uploaded = records.upload(hostRecords); !uploaded)
        return uploaded;
    return rebuild(count);
}

void GaussianProxies::recordInstances(CommandList& list, const Instances::Arguments& parameters) const {
    Instances::Arguments arguments = parameters;
    arguments.records = records.getPtr().as<uint>();
    arguments.proxy = blas.getHandle();
    arguments.threads = std::clamp(arguments.count, 1u, maxThreads);
    list.dispatch(writeInstances, {arguments.threads}, arguments);
}

std::expected<void, Error> GaussianProxies::rebuild(const std::uint32_t count) {
    if (count > capacity)
        return fail(ErrorCode::InvalidArgument, "the Gaussians exceed the capacity of the proxies");
    if (std::expected<void, Error> result = tlas.build(*device, records.getPtr(), count); !result)
        return result;
    built = count;
    return {};
}

std::expected<void, Error> GaussianProxies::refit() {
    return tlas.update(records.getPtr(), built);
}

}
