#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

struct Material;
class MeshTopology;

inline constexpr std::size_t maxUvSets = 4;

struct MeshSubset {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    Core::Handle<Material> material;

    bool operator==(const MeshSubset&) const = default;
};

//! A named value of each vertex, which a material program reads.
struct MeshAttribute {
    std::string name;
    std::uint8_t components = 1;
    std::vector<float> values;
};

//! Vertex has one value per position; Corner has one value per triangle corner in the order of indices, like a Cycles corner attribute.
enum class MeshDomain : std::uint8_t { Vertex, Corner };

//! Streams are separate arrays in their GPU encoding: octahedral normals, packTangent tangents, half2 UVs, unorm4x8 colors. A mesh without normals is flat.
struct Mesh {
    std::string name;
    std::vector<Core::float3> positions;
    std::vector<std::uint32_t> normals;
    std::vector<std::uint32_t> tangents;
    std::array<std::vector<std::uint32_t>, maxUvSets> uvs;
    //! Bit n is set when UV set n is empty and reads UV set 0.
    std::uint32_t uvAliases = 0;
    std::vector<std::uint32_t> colors;
    MeshDomain normalDomain = MeshDomain::Vertex;
    MeshDomain tangentDomain = MeshDomain::Vertex;
    MeshDomain uvDomain = MeshDomain::Vertex;
    MeshDomain colorDomain = MeshDomain::Vertex;
    std::vector<std::uint32_t> indices;
    std::vector<MeshSubset> subsets;
    Core::Bounds bounds;
    std::shared_ptr<const MeshTopology> topology;
    std::vector<MeshAttribute> attributes;

    std::uint32_t getVertexCount() const { return static_cast<std::uint32_t>(positions.size()); }
    std::uint32_t getTriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
    std::size_t getStreamSize(const MeshDomain domain) const { return domain == MeshDomain::Vertex ? positions.size() : indices.size(); }
    std::uint32_t getUvSetCount() const {
        std::uint32_t count = 0;
        while (count < maxUvSets && !uvs[count].empty())
            ++count;
        return count;
    }
    //! True when each stream is empty or has one value per element of its domain.
    bool hasValidStreams() const {
        const auto valid = [&](const std::vector<std::uint32_t>& stream, const MeshDomain domain) {
            return stream.empty() || stream.size() == getStreamSize(domain);
        };
        for (const std::vector<std::uint32_t>& uv : uvs)
            if (!valid(uv, uvDomain))
                return false;
        return valid(normals, normalDomain) && valid(tangents, tangentDomain) && valid(colors, colorDomain);
    }
    bool hasSameTopology(const Mesh& other) const {
        const auto sameSize = [](const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) { return a.size() == b.size(); };
        for (std::size_t set = 0; set < maxUvSets; ++set)
            if (!sameSize(uvs[set], other.uvs[set]))
                return false;
        return positions.size() == other.positions.size() && sameSize(normals, other.normals) && sameSize(tangents, other.tangents)
               && sameSize(colors, other.colors) && normalDomain == other.normalDomain && tangentDomain == other.tangentDomain && uvDomain == other.uvDomain
               && colorDomain == other.colorDomain && uvAliases == other.uvAliases && subsets == other.subsets && indices == other.indices;
    }
    void computeBounds() {
        bounds = {};
        for (const Core::float3 position : positions)
            bounds.grow(position);
    }
};

}
