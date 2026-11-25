#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"

namespace DiffRenderer::Methods {

//! Isotropic remeshing after Botsch and Kobbelt 2004: split long edges, collapse short ones, flip towards valence 6 and relax tangentially.
class Remesher {
public:
    explicit Remesher(SurfaceMesh& mesh) : mesh(mesh) {}

    void remesh(float targetLength, int passes);

private:
    void link();
    void unlock();
    //! The corners of the fan around the vertex; false when the fan is open or the vertex is not manifold.
    bool collectFan(std::uint32_t vertex, std::vector<std::uint32_t>& fan) const;
    bool isLocked(std::span<const std::uint32_t> fan) const;
    void lock(std::span<const std::uint32_t> fan);
    int splitLongEdges(float high);
    bool collapse(std::uint32_t corner, float high);
    int collapseShortEdges(float low, float high);
    int flipEdges();
    void relax();
    void removeCollapsed();

    SurfaceMesh& mesh;
    std::vector<std::int32_t> opposites;
    std::vector<std::uint32_t> vertexCorners;
    std::vector<std::uint32_t> valences;
    std::vector<std::uint8_t> lockedVertices;
    std::vector<std::uint8_t> removedTriangles;
    std::vector<std::uint8_t> removedVertices;
    std::vector<std::uint32_t> fanA;
    std::vector<std::uint32_t> fanB;
};

}
