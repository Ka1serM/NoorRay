#include "NoorRay/Import/MeshBuilder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <utility>

namespace NoorRay::Import {

void MeshData::appendSubset(const Core::Handle<Scene::Material> material, const std::uint32_t firstIndex, const std::uint32_t indexCount) {
    if (indexCount == 0)
        return;
    if (!subsets.empty() && subsets.back().material == material && subsets.back().firstIndex + subsets.back().indexCount == firstIndex) {
        subsets.back().indexCount += indexCount;
        return;
    }
    subsets.push_back({firstIndex, indexCount, material});
}

std::vector<Core::float3> MeshBuilder::computeNormals(const std::vector<Core::float3>& positions, const std::vector<std::uint32_t>& indices) {
    std::vector<Core::float3> normals(positions.size());
    for (std::size_t triangle = 0; triangle + 2 < indices.size(); triangle += 3) {
        const std::uint32_t a = indices[triangle];
        const std::uint32_t b = indices[triangle + 1];
        const std::uint32_t c = indices[triangle + 2];
        const Core::float3 normal = Core::cross(positions[b] - positions[a], positions[c] - positions[a]);
        normals[a] += normal;
        normals[b] += normal;
        normals[c] += normal;
    }
    for (Core::float3& normal : normals)
        normal = Core::normalize(normal);
    return normals;
}

std::vector<Core::float4> MeshBuilder::computeTangents(const std::span<const Core::float3> positions, const std::span<const std::uint32_t> indices,
                                                       const std::span<const Core::float3> normals, const Scene::MeshDomain normalDomain,
                                                       const std::span<const Core::float2> uvs, const Scene::MeshDomain uvDomain,
                                                       const Scene::MeshDomain tangentDomain) {
    const std::size_t corners = indices.size() - indices.size() % 3;
    const auto at = [&](const Scene::MeshDomain domain, const std::size_t corner) { return domain == Scene::MeshDomain::Vertex ? indices[corner] : corner; };
    const auto normalOf = [&](const std::size_t corner) {
        if (!normals.empty())
            return normals[at(normalDomain, corner)];
        const std::size_t first = corner - corner % 3;
        return Core::normalize(
            Core::cross(positions[indices[first + 1]] - positions[indices[first]], positions[indices[first + 2]] - positions[indices[first]]));
    };
    //! The group of each corner: its vertex in the vertex domain, else the first corner with its vertex, normal and UV.
    std::vector<std::uint32_t> groups(corners);
    if (tangentDomain == Scene::MeshDomain::Vertex)
        for (std::size_t corner = 0; corner < corners; ++corner)
            groups[corner] = indices[corner];
    else {
        using Key = std::array<std::uint32_t, 4>;
        const auto keyOf = [&](const std::size_t corner) {
            const Core::float2 uv = uvs[at(uvDomain, corner)];
            return Key{indices[corner], Core::packOctahedral(normalOf(corner)), std::bit_cast<std::uint32_t>(uv.x), std::bit_cast<std::uint32_t>(uv.y)};
        };
        std::vector<std::pair<Key, std::uint32_t>> keys(corners);
        for (std::size_t corner = 0; corner < corners; ++corner)
            keys[corner] = {keyOf(corner), static_cast<std::uint32_t>(corner)};
        std::ranges::sort(keys);
        for (std::size_t index = 0; index < corners; ++index)
            groups[keys[index].second] = index > 0 && keys[index].first == keys[index - 1].first ? groups[keys[index - 1].second] : keys[index].second;
    }
    const std::size_t count = tangentDomain == Scene::MeshDomain::Vertex ? positions.size() : corners;
    std::vector<Core::float3> tangents(count);
    std::vector<Core::float3> bitangents(count);
    for (std::size_t triangle = 0; triangle < corners; triangle += 3) {
        const Core::float3 edge1 = positions[indices[triangle + 1]] - positions[indices[triangle]];
        const Core::float3 edge2 = positions[indices[triangle + 2]] - positions[indices[triangle]];
        const Core::float2 uv = uvs[at(uvDomain, triangle)];
        const Core::float2 delta1 = uvs[at(uvDomain, triangle + 1)] - uv;
        const Core::float2 delta2 = uvs[at(uvDomain, triangle + 2)] - uv;
        const float determinant = delta1.x * delta2.y - delta2.x * delta1.y;
        if (std::abs(determinant) < 1e-12f)
            continue;
        const float inverse = 1.0f / determinant;
        const Core::float3 tangent = (edge1 * delta2.y - edge2 * delta1.y) * inverse;
        const Core::float3 bitangent = (edge2 * delta1.x - edge1 * delta2.x) * inverse;
        for (std::size_t corner = triangle; corner < triangle + 3; ++corner) {
            tangents[groups[corner]] += tangent;
            bitangents[groups[corner]] += bitangent;
        }
    }
    std::vector<Core::float4> result(count);
    for (std::size_t corner = 0; corner < corners; ++corner) {
        const std::uint32_t group = groups[corner];
        const std::size_t target = tangentDomain == Scene::MeshDomain::Vertex ? group : corner;
        const Core::float3 normal = normalOf(corner);
        Core::float3 tangent = tangents[group] - normal * Core::dot(normal, tangents[group]);
        if (Core::dot(tangent, tangent) < 1e-20f) {
            const Core::float3 axis = std::abs(normal.x) < 0.9f ? Core::float3{1.0f, 0.0f, 0.0f} : Core::float3{0.0f, 1.0f, 0.0f};
            tangent = Core::cross(axis, normal);
        }
        tangent = Core::normalize(tangent);
        const float sign = Core::dot(Core::cross(normal, tangent), bitangents[group]) < 0.0f ? -1.0f : 1.0f;
        result[target] = {tangent.x, tangent.y, tangent.z, sign};
    }
    return result;
}

Scene::Mesh MeshBuilder::build(MeshData data) {
    const std::size_t count = data.positions.size();
    if (data.normals.size() != count)
        data.normals = computeNormals(data.positions, data.indices);
    if (data.tangents.size() != count && data.uvs[0].size() == count)
        data.tangents = computeTangents(data.positions, data.indices, data.normals, Scene::MeshDomain::Vertex, data.uvs[0], Scene::MeshDomain::Vertex,
                                        Scene::MeshDomain::Vertex);

    Scene::Mesh mesh;
    mesh.name = std::move(data.name);
    mesh.normals.resize(count);
    for (std::size_t vertex = 0; vertex < count; ++vertex)
        mesh.normals[vertex] = Core::packOctahedral(Core::normalize(data.normals[vertex]));
    if (data.tangents.size() == count) {
        mesh.tangents.resize(count);
        for (std::size_t vertex = 0; vertex < count; ++vertex) {
            const Core::float4 tangent = data.tangents[vertex];
            mesh.tangents[vertex] = Core::packTangent(Core::normalize(Core::float3{tangent.x, tangent.y, tangent.z}), tangent.w);
        }
    }
    for (std::size_t set = 0; set < Scene::maxUvSets; ++set) {
        if (data.uvs[set].size() != count)
            break;
        mesh.uvs[set].resize(count);
        for (std::size_t vertex = 0; vertex < count; ++vertex)
            mesh.uvs[set][vertex] = Core::packHalf2(data.uvs[set][vertex].x, data.uvs[set][vertex].y);
    }
    if (data.colors.size() == count) {
        mesh.colors.resize(count);
        for (std::size_t vertex = 0; vertex < count; ++vertex)
            mesh.colors[vertex] = Core::packUnorm4x8(data.colors[vertex]);
    }
    mesh.positions = std::move(data.positions);
    mesh.indices = std::move(data.indices);
    mesh.subsets = std::move(data.subsets);
    mesh.computeBounds();
    return mesh;
}

}
