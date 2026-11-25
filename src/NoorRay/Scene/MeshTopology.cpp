#include "NoorRay/Scene/MeshTopology.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <ranges>

#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Scene/Mesh.h"

namespace NoorRay::Scene {

namespace {

Core::float3 normalizeOrZero(const Core::float3 value) {
    const float size = Core::length(value);
    return size > 0.0f ? value * (1.0f / size) : Core::float3{};
}

}

std::expected<void, Core::Error> MeshTopology::open(const NormalSource source, std::vector<std::uint32_t> corners, std::vector<std::uint32_t> offsets,
                                                    const Mesh& mesh) {
    const auto invalid = [&](const std::string_view problem) {
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the topology of mesh '{}' {}", mesh.name, problem));
    };
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != corners.size() || !std::ranges::is_sorted(offsets))
        return invalid("has face offsets that do not match its corners");
    if (std::ranges::any_of(corners, [&](const std::uint32_t vertex) { return vertex >= mesh.positions.size(); }))
        return invalid("has a corner past the vertices");
    const bool normals = source == NormalSource::Face
                             ? mesh.normals.empty()
                             : !mesh.normals.empty() && mesh.normalDomain == (source == NormalSource::Stored ? MeshDomain::Corner : MeshDomain::Vertex);
    if (!normals)
        return invalid("has normals that do not match its normal source");
    if (!mesh.tangents.empty() && mesh.uvs[0].empty())
        return invalid("has tangents but no UV set");
    normalSource = source;
    cornerVertices = std::move(corners);
    faceOffsets = std::move(offsets);
    tangentUvs.clear();
    if (mesh.tangents.empty())
        return {};
    tangentUvs.resize(mesh.uvs[0].size());
    for (std::size_t index = 0; index < tangentUvs.size(); ++index) {
        const std::uint32_t packed = mesh.uvs[0][index];
        tangentUvs[index] = {Core::fromHalf(Core::half{static_cast<std::uint16_t>(packed & 0xFFFFu)}),
                             Core::fromHalf(Core::half{static_cast<std::uint16_t>(packed >> 16)})};
    }
    return {};
}

std::vector<Core::float3> MeshTopology::computeFaceNormals(const std::span<const Core::float3> positions) const {
    const std::size_t faceCount = faceOffsets.size() - 1;
    std::vector<Core::float3> normals(faceCount);
    for (std::size_t face = 0; face < faceCount; ++face) {
        const std::uint32_t first = faceOffsets[face];
        const std::uint32_t count = faceOffsets[face + 1] - first;
        const auto at = [&](const std::uint32_t corner) { return positions[cornerVertices[first + corner]]; };
        Core::float3 normal{};
        if (count == 3)
            normal = Core::cross(at(0) - at(1), at(1) - at(2));
        else if (count == 4)
            normal = Core::cross(at(0) - at(2), at(1) - at(3));
        else if (count > 4)
            for (std::uint32_t corner = 0; corner < count; ++corner) {
                const Core::float3 current = at(corner);
                const Core::float3 next = at((corner + 1) % count);
                normal += Core::float3{(current.y - next.y) * (current.z + next.z), (current.z - next.z) * (current.x + next.x),
                                       (current.x - next.x) * (current.y + next.y)};
            }
        normal = normalizeOrZero(normal);
        //! Blender gives a face without area the normal +Z.
        normals[face] = Core::dot(normal, normal) > 0.0f ? normal : Core::float3{0.0f, 0.0f, 1.0f};
    }
    return normals;
}

std::vector<Core::float3> MeshTopology::computeVertexNormals(const std::span<const Core::float3> positions,
                                                             const std::span<const Core::float3> faceNormals) const {
    std::vector<Core::float3> normals(positions.size());
    for (std::size_t face = 0; face + 1 < faceOffsets.size(); ++face) {
        const std::uint32_t first = faceOffsets[face];
        const std::uint32_t count = faceOffsets[face + 1] - first;
        for (std::uint32_t corner = 0; corner < count; ++corner) {
            const std::uint32_t vertex = cornerVertices[first + corner];
            const Core::float3 position = positions[vertex];
            const Core::float3 toPrevious = normalizeOrZero(positions[cornerVertices[first + (corner + count - 1) % count]] - position);
            const Core::float3 toNext = normalizeOrZero(positions[cornerVertices[first + (corner + 1) % count]] - position);
            const float angle = std::acos(std::clamp(Core::dot(toPrevious, toNext), -1.0f, 1.0f));
            normals[vertex] += faceNormals[face] * angle;
        }
    }
    for (std::size_t vertex = 0; vertex < normals.size(); ++vertex) {
        Core::float3 normal = normalizeOrZero(normals[vertex]);
        //! Blender uses the direction of the position for a vertex without a normal.
        if (Core::dot(normal, normal) == 0.0f)
            normal = normalizeOrZero(positions[vertex]);
        normals[vertex] = Core::dot(normal, normal) > 0.0f ? normal : Core::float3{0.0f, 0.0f, 1.0f};
    }
    return normals;
}

std::expected<MeshTopology::Streams, Core::Error> MeshTopology::rebuild(const std::span<const Core::float3> positions,
                                                                        const std::span<const std::uint32_t> storedNormals, const Mesh& mesh) const {
    if (positions.size() != mesh.positions.size())
        return Core::fail(Core::ErrorCode::InvalidFile, "a positions frame does not have a position for each vertex");
    if (normalSource == NormalSource::Stored && storedNormals.size() != mesh.indices.size())
        return Core::fail(Core::ErrorCode::InvalidFile, "a positions frame does not have a normal for each triangle corner");
    Streams streams;
    std::vector<Core::float3> normals;
    if (normalSource == NormalSource::Stored) {
        streams.normals.assign(storedNormals.begin(), storedNormals.end());
        if (!tangentUvs.empty())
            normals = storedNormals | std::views::transform(Core::unpackOctahedral) | std::ranges::to<std::vector>();
    } else if (normalSource == NormalSource::Vertex) {
        normals = computeVertexNormals(positions, computeFaceNormals(positions));
        streams.normals = normals | std::views::transform(Core::packOctahedral) | std::ranges::to<std::vector>();
    }
    if (tangentUvs.empty())
        return streams;
    const std::vector<Core::float4> tangents =
        Import::MeshBuilder::computeTangents(positions, mesh.indices, normals, mesh.normalDomain, tangentUvs, mesh.uvDomain, mesh.tangentDomain);
    streams.tangents.resize(tangents.size());
    for (std::size_t index = 0; index < tangents.size(); ++index)
        streams.tangents[index] = Core::packTangent(Core::float3{tangents[index].x, tangents[index].y, tangents[index].z}, tangents[index].w);
    return streams;
}

}
