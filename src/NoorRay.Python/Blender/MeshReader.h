#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay/Materials/Nodes/NodeAttributes.h"
#include "NoorRay/Scene/MeshTopology.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! What the materials of a mesh need from it; displaced meshes keep their source topology and the point attributes that the programs read.
struct MeshRequest {
    std::vector<Scene::MaterialHandle> slotMaterials;
    bool tangents = false;
    bool colors = false;
    Materials::UvStreams streams;
    std::string tangentMap;
    std::optional<std::vector<std::string>> attributes;

    bool operator==(const MeshRequest&) const = default;
};

//! What a full read of a mesh keeps, so that a later read of the same corners, faces, UVs and colours writes only positions, normals and turned quads.
struct MeshLayout {
    Scene::NormalSource normalSource{};
    std::size_t vertexCount = 0;
    std::vector<std::int32_t> cornerVertices;
    std::vector<std::uint32_t> faceStarts;
    std::vector<std::int32_t> faceSlots;
    //! One per face: 1 when the quad splits along its 1-3 diagonal.
    std::vector<std::uint8_t> quadFlips;
    //! The first triangle corner of each face, and the Blender corner of each triangle corner.
    std::vector<std::uint32_t> faceTriangles;
    std::vector<std::uint32_t> triangleCorners;
    std::array<std::vector<Core::float2>, Scene::maxUvSets> uvs;
    std::uint32_t uvAliases = 0;
    std::vector<std::uint32_t> colors;
    bool pointColors = false;
    //! The source positions and normals of the last read; the buffers stay between reads.
    std::vector<Core::float3> sourcePositions;
    std::vector<Core::float3> sourceNormals;
    //! The last read changed only what fill writes: positions, normals, and the six triangle corners from each patch on.
    bool reused = false;
    std::vector<std::uint32_t> patches;
};

//! Reads an evaluated Blender mesh like Cycles: Blender's vertices, its triangulation, corner UVs, and one subset per material slot.
class MeshReader {
public:
    explicit MeshReader(Reader& reader) : reader(reader) {}

    //! A mesh without triangles has no indices. When layout matches the mesh, the result is empty and layout.reused is set.
    Scene::Mesh read(const Object& mesh, const std::string& name, const MeshRequest& request, MeshLayout& layout);
    //! Writes a reused layout into the mesh of its last full read; false when the mesh does not have its sizes.
    static bool fill(const MeshLayout& layout, Scene::Mesh& mesh);

private:
    //! The values of a named attribute for a UV set, per corner; empty where the mesh does not have it.
    std::vector<float> derivedValues(const Object& mesh, const std::string& stream, std::span<const Core::float3> positions);
    std::vector<Core::float2> cornerStream(const Object& attributes, const Materials::UvStream& stream, std::size_t faceCount,
                                           std::span<const std::uint32_t> faceStarts, std::span<const std::int32_t> cornerVertices);

    Reader& reader;
};

}
