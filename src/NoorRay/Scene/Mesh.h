#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

struct Material;

inline constexpr std::size_t maxUvSets = 4;

struct MeshSubset {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    Core::Handle<Material> material;
};

//! Vertex streams are separate arrays in their GPU encoding: octahedral normals, packTangent tangents, half2 UVs, unorm4x8 colors.
struct Mesh {
    std::string name;
    std::vector<Core::float3> positions;
    std::vector<std::uint32_t> normals;
    std::vector<std::uint32_t> tangents;
    std::array<std::vector<std::uint32_t>, maxUvSets> uvs;
    std::vector<std::uint32_t> colors;
    std::vector<std::uint32_t> indices;
    std::vector<MeshSubset> subsets;
    Core::Bounds bounds;

    std::uint32_t getVertexCount() const { return static_cast<std::uint32_t>(positions.size()); }
    std::uint32_t getTriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
    std::uint32_t getUvSetCount() const {
        std::uint32_t count = 0;
        while (count < maxUvSets && !uvs[count].empty())
            ++count;
        return count;
    }
    void computeBounds() {
        bounds = {};
        for (const Core::float3 position : positions)
            bounds.grow(position);
    }
};

}
