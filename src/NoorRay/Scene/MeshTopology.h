#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

struct Mesh;

enum class NormalSource : std::uint8_t { Vertex, Face, Stored };

//! The Blender faces over the vertices of a mesh; normals follow Blender (angle-weighted vertex normals, Newell face normals, flat without normals).
class MeshTopology {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NormalSource source, std::vector<std::uint32_t> cornerVertices, std::vector<std::uint32_t> faceOffsets,
                                                        const Mesh& mesh);

    NormalSource getNormalSource() const { return normalSource; }
    std::size_t getFaceCount() const { return faceOffsets.size() - 1; }
    std::span<const std::uint32_t> getCornerVertices() const { return cornerVertices; }
    std::span<const std::uint32_t> getFaceOffsets() const { return faceOffsets; }

    std::vector<Core::float3> computeFaceNormals(std::span<const Core::float3> positions) const;
    std::vector<Core::float3> computeVertexNormals(std::span<const Core::float3> positions, std::span<const Core::float3> faceNormals) const;

    struct Streams {
        std::vector<std::uint32_t> normals;
        std::vector<std::uint32_t> tangents;
    };
    //! The normals and tangents of new positions; Stored normals come with the positions, one per triangle corner.
    [[nodiscard]] std::expected<Streams, Core::Error> rebuild(std::span<const Core::float3> positions, std::span<const std::uint32_t> storedNormals,
                                                              const Mesh& mesh) const;

private:
    NormalSource normalSource = NormalSource::Vertex;
    std::vector<std::uint32_t> cornerVertices;
    std::vector<std::uint32_t> faceOffsets;
    std::vector<Core::float2> tangentUvs;
};

}
