#pragma once

#include <array>
#include <cstdint>
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
    static std::vector<Core::float4> computeTangents(const MeshData& data);
};

}
