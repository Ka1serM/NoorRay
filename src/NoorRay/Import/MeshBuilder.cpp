#include "NoorRay/Import/MeshBuilder.h"

#include <cmath>

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

std::vector<Core::float4> MeshBuilder::computeTangents(const MeshData& data) {
    const std::size_t count = data.positions.size();
    std::vector<Core::float3> tangents(count);
    std::vector<Core::float3> bitangents(count);
    const std::vector<Core::float2>& uvs = data.uvs[0];
    for (std::size_t triangle = 0; triangle + 2 < data.indices.size(); triangle += 3) {
        const std::uint32_t a = data.indices[triangle];
        const std::uint32_t b = data.indices[triangle + 1];
        const std::uint32_t c = data.indices[triangle + 2];
        const Core::float3 edge1 = data.positions[b] - data.positions[a];
        const Core::float3 edge2 = data.positions[c] - data.positions[a];
        const Core::float2 delta1 = uvs[b] - uvs[a];
        const Core::float2 delta2 = uvs[c] - uvs[a];
        const float determinant = delta1.x * delta2.y - delta2.x * delta1.y;
        if (std::abs(determinant) < 1e-12f)
            continue;
        const float inverse = 1.0f / determinant;
        const Core::float3 tangent = (edge1 * delta2.y - edge2 * delta1.y) * inverse;
        const Core::float3 bitangent = (edge2 * delta1.x - edge1 * delta2.x) * inverse;
        for (const std::uint32_t vertex : {a, b, c}) {
            tangents[vertex] += tangent;
            bitangents[vertex] += bitangent;
        }
    }
    std::vector<Core::float4> result(count);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        const Core::float3 normal = data.normals[vertex];
        Core::float3 tangent = tangents[vertex] - normal * Core::dot(normal, tangents[vertex]);
        if (Core::dot(tangent, tangent) < 1e-20f) {
            const Core::float3 axis = std::abs(normal.x) < 0.9f ? Core::float3{1.0f, 0.0f, 0.0f} : Core::float3{0.0f, 1.0f, 0.0f};
            tangent = Core::cross(axis, normal);
        }
        tangent = Core::normalize(tangent);
        const float sign = Core::dot(Core::cross(normal, tangent), bitangents[vertex]) < 0.0f ? -1.0f : 1.0f;
        result[vertex] = {tangent.x, tangent.y, tangent.z, sign};
    }
    return result;
}

Scene::Mesh MeshBuilder::build(MeshData data) {
    const std::size_t count = data.positions.size();
    if (data.normals.size() != count)
        data.normals = computeNormals(data.positions, data.indices);
    if (data.tangents.size() != count && data.uvs[0].size() == count)
        data.tangents = computeTangents(data);

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
