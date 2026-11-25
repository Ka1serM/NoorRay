#include "DiffRenderer/Methods/SdfGrid/IsoSurface.h"

#include <array>
#include <cstdint>
#include <format>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

//! The Kuhn split of a cube into six tetrahedra around the diagonal from corner 0 to corner 7; corner bits are x, y, z.
constexpr std::array<std::array<std::uint32_t, 4>, 6> tetrahedra{{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};

struct Vertex {
    float x;
    float y;
    float z;
};

class Extractor {
public:
    Extractor(const std::span<const float> values, const SurfaceRendering::Grid& grid) : values(values), grid(grid) {}

    void run() {
        for (std::uint32_t z = 0; z + 1 < grid.size.z; ++z)
            for (std::uint32_t y = 0; y + 1 < grid.size.y; ++y)
                for (std::uint32_t x = 0; x + 1 < grid.size.x; ++x)
                    addCell(x, y, z);
    }

    std::span<const Vertex> getVertices() const { return vertices; }
    std::span<const std::array<std::uint32_t, 3>> getTriangles() const { return triangles; }

private:
    std::uint32_t getIndex(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) const { return (z * grid.size.y + y) * grid.size.x + x; }

    Vertex getPosition(const std::uint32_t node) const {
        const std::uint32_t x = node % grid.size.x;
        const std::uint32_t y = node / grid.size.x % grid.size.y;
        const std::uint32_t z = node / (grid.size.x * grid.size.y);
        return {grid.origin.x + static_cast<float>(x) * grid.origin.w, grid.origin.y + static_cast<float>(y) * grid.origin.w,
                grid.origin.z + static_cast<float>(z) * grid.origin.w};
    }

    std::uint32_t getEdgeVertex(const std::uint32_t inside, const std::uint32_t outside) {
        const std::uint64_t key = std::uint64_t{std::min(inside, outside)} << 32 | std::max(inside, outside);
        const auto [entry, added] = edgeVertices.try_emplace(key, static_cast<std::uint32_t>(vertices.size()));
        if (!added)
            return entry->second;
        const float a = values[inside];
        const float b = values[outside];
        const float t = a / (a - b);
        const Vertex p = getPosition(inside);
        const Vertex q = getPosition(outside);
        vertices.push_back({p.x + t * (q.x - p.x), p.y + t * (q.y - p.y), p.z + t * (q.z - p.z)});
        return entry->second;
    }

    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c, const std::uint32_t inside, const std::uint32_t outside) {
        const Vertex p = vertices[a];
        const Vertex q = vertices[b];
        const Vertex r = vertices[c];
        const Vertex u{q.x - p.x, q.y - p.y, q.z - p.z};
        const Vertex v{r.x - p.x, r.y - p.y, r.z - p.z};
        const Vertex normal{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
        const Vertex from = getPosition(inside);
        const Vertex to = getPosition(outside);
        if (normal.x * (to.x - from.x) + normal.y * (to.y - from.y) + normal.z * (to.z - from.z) < 0.0f)
            std::swap(b, c);
        triangles.push_back({a, b, c});
    }

    void addCell(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
        std::array<std::uint32_t, 8> corners{};
        std::uint32_t insideCount = 0;
        for (std::uint32_t corner = 0; corner < 8; ++corner) {
            corners[corner] = getIndex(x + (corner & 1), y + (corner >> 1 & 1), z + (corner >> 2));
            insideCount += values[corners[corner]] < 0.0f ? 1 : 0;
        }
        if (insideCount == 0 || insideCount == 8)
            return;
        for (const std::array<std::uint32_t, 4>& tetrahedron : tetrahedra)
            addTetrahedron({corners[tetrahedron[0]], corners[tetrahedron[1]], corners[tetrahedron[2]], corners[tetrahedron[3]]});
    }

    void addTetrahedron(const std::array<std::uint32_t, 4>& nodes) {
        std::array<std::uint32_t, 4> inside{};
        std::array<std::uint32_t, 4> outside{};
        std::uint32_t insideCount = 0;
        std::uint32_t outsideCount = 0;
        for (const std::uint32_t node : nodes)
            if (values[node] < 0.0f)
                inside[insideCount++] = node;
            else
                outside[outsideCount++] = node;
        if (insideCount == 1 || insideCount == 3) {
            const bool single = insideCount == 1;
            const std::uint32_t apex = single ? inside[0] : outside[0];
            const std::span<const std::uint32_t> others =
                single ? std::span<const std::uint32_t>(outside).first(3) : std::span<const std::uint32_t>(inside).first(3);
            const auto getVertex = [&](const std::uint32_t other) { return single ? getEdgeVertex(apex, other) : getEdgeVertex(other, apex); };
            addTriangle(getVertex(others[0]), getVertex(others[1]), getVertex(others[2]), single ? apex : others[0], single ? others[0] : apex);
            return;
        }
        if (insideCount != 2)
            return;
        const std::uint32_t a = getEdgeVertex(inside[0], outside[0]);
        const std::uint32_t b = getEdgeVertex(inside[0], outside[1]);
        const std::uint32_t c = getEdgeVertex(inside[1], outside[1]);
        const std::uint32_t d = getEdgeVertex(inside[1], outside[0]);
        addTriangle(a, b, c, inside[0], outside[0]);
        addTriangle(a, c, d, inside[0], outside[0]);
    }

    std::span<const float> values;
    const SurfaceRendering::Grid& grid;
    std::vector<Vertex> vertices;
    std::vector<std::array<std::uint32_t, 3>> triangles;
    std::unordered_map<std::uint64_t, std::uint32_t> edgeVertices;
};

#pragma pack(push, 1)
struct PlyFace {
    std::uint8_t count = 3;
    std::array<std::uint32_t, 3> indices{};
};
#pragma pack(pop)

}

std::expected<void, std::string> writeIsoSurface(const std::span<const float> values, const SurfaceRendering::Grid& grid, const std::filesystem::path& path) {
    if (values.size() != grid.size.w || grid.size.x < 2 || grid.size.y < 2 || grid.size.z < 2)
        return std::unexpected("the iso-surface needs a grid of at least 2 nodes on each axis and one value for each node");
    Extractor extractor(values, grid);
    extractor.run();
    std::ofstream file(path, std::ios::binary);
    if (!file)
        return std::unexpected(std::format("cannot open {}", path.string()));
    file << std::format("ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\n"
                        "element face {}\nproperty list uchar uint vertex_indices\nend_header\n",
                        extractor.getVertices().size(), extractor.getTriangles().size());
    file.write(reinterpret_cast<const char*>(extractor.getVertices().data()), static_cast<std::streamsize>(extractor.getVertices().size_bytes()));
    std::vector<PlyFace> faces;
    faces.reserve(extractor.getTriangles().size());
    for (const std::array<std::uint32_t, 3>& triangle : extractor.getTriangles())
        faces.push_back({3, triangle});
    file.write(reinterpret_cast<const char*>(faces.data()), static_cast<std::streamsize>(faces.size() * sizeof(PlyFace)));
    if (!file)
        return std::unexpected(std::format("cannot write {}", path.string()));
    return {};
}

}
