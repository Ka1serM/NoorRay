#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Mesh.h"

namespace NoorRay::Import {

//! Unpacked vertex streams as importers read them. UVs use the Blender convention: v points up in the image.
struct MeshData {
    std::string name;
    std::vector<Core::float3> positions;
    std::vector<Core::float3> normals;
    std::vector<Core::float4> tangents;
    std::array<std::vector<Core::float2>, Scene::maxUvSets> uvs;
    std::vector<Core::float4> colors;
    std::vector<std::uint32_t> indices;
    std::vector<Scene::MeshSubset> subsets;

    void appendSubset(Core::Handle<Scene::Material> material, std::uint32_t firstIndex, std::uint32_t indexCount);
};

class MeshBuilder {
public:
    //! Fills missing normals with area-weighted normals and missing tangents from UV set 0, then packs the streams.
    static Scene::Mesh build(MeshData data);
    static std::vector<Core::float3> computeNormals(const std::vector<Core::float3>& positions, const std::vector<std::uint32_t>& indices);
    //! Tangents of UV set 0 with the bitangent sign in w; empty normals are flat, and corners with one vertex, normal and UV share a tangent like MikkTSpace.
    static std::vector<Core::float4> computeTangents(std::span<const Core::float3> positions, std::span<const std::uint32_t> indices,
                                                     std::span<const Core::float3> normals, Scene::MeshDomain normalDomain, std::span<const Core::float2> uvs,
                                                     Scene::MeshDomain uvDomain, Scene::MeshDomain tangentDomain);
};

}
