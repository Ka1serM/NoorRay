#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace DiffRenderer::Methods {

namespace {

std::uint32_t getNextCorner(const std::uint32_t corner) {
    return corner - corner % 3 + (corner + 1) % 3;
}

std::uint32_t getPreviousCorner(const std::uint32_t corner) {
    return corner - corner % 3 + (corner + 2) % 3;
}

std::uint64_t getEdgeKey(const std::uint32_t a, const std::uint32_t b) {
    return std::uint64_t{std::min(a, b)} << 32 | std::max(a, b);
}

}

std::vector<std::int32_t> SurfaceMesh::findOppositeCorners() const {
    std::vector<std::uint32_t> offsets(positions.size() + 1, 0);
    for (const std::uint32_t vertex : triangles)
        ++offsets[vertex + 1];
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
        offsets[vertex + 1] += offsets[vertex];
    std::vector<std::uint32_t> fill(offsets.begin(), offsets.end() - 1);
    std::vector<std::uint32_t> corners(triangles.size());
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner)
        corners[fill[triangles[corner]]++] = corner;
    std::vector<std::int32_t> opposites(triangles.size(), -1);
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner) {
        const std::uint32_t a = triangles[getNextCorner(corner)];
        const std::uint32_t b = triangles[getPreviousCorner(corner)];
        int matches = 0;
        std::uint32_t match = 0;
        for (std::uint32_t entry = offsets[b]; entry < offsets[b + 1]; ++entry) {
            if (triangles[getNextCorner(corners[entry])] != a)
                continue;
            match = getPreviousCorner(corners[entry]);
            ++matches;
        }
        if (matches == 1)
            opposites[corner] = static_cast<std::int32_t>(match);
    }
    return opposites;
}

std::vector<std::int32_t> SurfaceMesh::findOppositeVertices() const {
    std::vector<std::int32_t> opposites = findOppositeCorners();
    for (std::int32_t& opposite : opposites)
        if (opposite >= 0)
            opposite = static_cast<std::int32_t>(triangles[static_cast<std::size_t>(opposite)]);
    return opposites;
}

void SurfaceMesh::findNeighbors(std::vector<std::uint32_t>& offsets, std::vector<std::uint32_t>& neighbors) const {
    std::vector<std::uint64_t> edges;
    edges.reserve(triangles.size() * 2);
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner) {
        const std::uint32_t a = triangles[corner];
        const std::uint32_t b = triangles[getNextCorner(corner)];
        edges.push_back(std::uint64_t{a} << 32 | b);
        edges.push_back(std::uint64_t{b} << 32 | a);
    }
    std::ranges::sort(edges);
    const auto [first, last] = std::ranges::unique(edges);
    edges.erase(first, last);
    offsets.assign(positions.size() + 1, 0);
    neighbors.resize(edges.size());
    for (std::size_t index = 0; index < edges.size(); ++index) {
        ++offsets[(edges[index] >> 32) + 1];
        neighbors[index] = static_cast<std::uint32_t>(edges[index]);
    }
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
        offsets[vertex + 1] += offsets[vertex];
}

float SurfaceMesh::getMeanEdgeLength() const {
    double sum = 0.0;
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner)
        sum += (positions[triangles[corner]] - positions[triangles[getNextCorner(corner)]]).getLength();
    return triangles.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(triangles.size()));
}

float SurfaceMesh::getArea() const {
    double sum = 0.0;
    for (std::size_t corner = 0; corner < triangles.size(); corner += 3) {
        const Vector3 a = positions[triangles[corner]];
        sum += 0.5 * (positions[triangles[corner + 1]] - a).cross(positions[triangles[corner + 2]] - a).getLength();
    }
    return static_cast<float>(sum);
}

SurfaceMesh makeIcosphere(const Vector3 center, const float radius, const int subdivisions) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<Vector3> directions{{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                                    {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    std::vector<std::uint32_t> triangles{0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
                                         3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};
    for (int level = 0; level < subdivisions; ++level) {
        std::unordered_map<std::uint64_t, std::uint32_t> midpoints;
        const auto getMidpoint = [&](const std::uint32_t a, const std::uint32_t b) {
            const auto [found, inserted] = midpoints.try_emplace(getEdgeKey(a, b), static_cast<std::uint32_t>(directions.size()));
            if (inserted)
                directions.push_back((directions[a] + directions[b]) * 0.5f);
            return found->second;
        };
        std::vector<std::uint32_t> refined;
        refined.reserve(triangles.size() * 4);
        for (std::size_t corner = 0; corner < triangles.size(); corner += 3) {
            const std::uint32_t a = triangles[corner];
            const std::uint32_t b = triangles[corner + 1];
            const std::uint32_t c = triangles[corner + 2];
            const std::uint32_t ab = getMidpoint(a, b);
            const std::uint32_t bc = getMidpoint(b, c);
            const std::uint32_t ca = getMidpoint(c, a);
            refined.insert(refined.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        triangles = std::move(refined);
    }
    SurfaceMesh mesh;
    mesh.triangles = std::move(triangles);
    for (const Vector3 direction : directions)
        mesh.positions.push_back(center + direction.getNormalized() * radius);
    mesh.colors.assign(mesh.positions.size(), {0.5f, 0.5f, 0.5f});
    mesh.materials.assign(mesh.positions.size(), {0.0f, 0.5f, 0.0f});
    return mesh;
}

}
