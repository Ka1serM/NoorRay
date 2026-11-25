#pragma once

#include <cstdint>
#include <vector>

#include "DiffRenderer/Methods/Mesh/Vector3.h"

namespace DiffRenderer::Methods {

//! A closed triangle mesh with a material on each vertex; triangle t has the corners 3t, 3t + 1 and 3t + 2.
struct SurfaceMesh {
    std::vector<Vector3> positions;
    //! The linear diffuse albedo kd of nvdiffrec.
    std::vector<Vector3> colors;
    //! ks of nvdiffrec: the occlusion, the roughness and the metallic value.
    std::vector<Vector3> materials;
    std::vector<std::uint32_t> triangles;

    std::size_t getVertexCount() const { return positions.size(); }
    std::size_t getTriangleCount() const { return triangles.size() / 3; }
    //! The corner across the edge that each corner faces, or -1 on a boundary or non-manifold edge.
    std::vector<std::int32_t> findOppositeCorners() const;
    //! The vertex across the edge that each corner faces, or -1, as the antialiasing finds silhouettes with it.
    std::vector<std::int32_t> findOppositeVertices() const;
    //! The neighbours of each vertex in compressed rows: offsets has one entry more than the vertices.
    void findNeighbors(std::vector<std::uint32_t>& offsets, std::vector<std::uint32_t>& neighbors) const;
    float getMeanEdgeLength() const;
    float getArea() const;
};

SurfaceMesh makeIcosphere(Vector3 center, float radius, int subdivisions);

}
