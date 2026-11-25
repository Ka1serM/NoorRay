#include "DiffRenderer/Methods/Mesh/VisualHull.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>
#include <unordered_map>

namespace DiffRenderer::Methods {

namespace {

constexpr float isoValue = 0.5f;
constexpr float boxMargin = 0.05f;
constexpr std::array<std::array<int, 3>, 6> axisOrders{{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

float transform(const NoorRHI::float4 row, const Vector3 point) {
    return row.x * point.x + row.y * point.y + row.z * point.z + row.w;
}

//! A point outside the image of a view or behind its camera keeps 1, since that view cannot carve it.
float sampleAlpha(const Scenes::View& view, const Vector3 point) {
    const Scenes::Camera& camera = view.camera;
    const float depth = transform(camera.worldToCamera[2], point);
    if (depth <= 0.0f)
        return 1.0f;
    const float u = camera.intrinsics.x * transform(camera.worldToCamera[0], point) / depth + camera.intrinsics.z - 0.5f;
    const float v = camera.intrinsics.y * transform(camera.worldToCamera[1], point) / depth + camera.intrinsics.w - 0.5f;
    const auto width = static_cast<float>(camera.width);
    const auto height = static_cast<float>(camera.height);
    if (u < 0.0f || v < 0.0f || u > width - 1.0f || v > height - 1.0f)
        return 1.0f;
    const auto x = std::min(static_cast<std::uint32_t>(u), camera.width - 2);
    const auto y = std::min(static_cast<std::uint32_t>(v), camera.height - 2);
    const float fx = u - static_cast<float>(x);
    const float fy = v - static_cast<float>(y);
    const std::size_t index = std::size_t{y} * camera.width + x;
    const float top = view.alpha[index] * (1.0f - fx) + view.alpha[index + 1] * fx;
    const float bottom = view.alpha[index + camera.width] * (1.0f - fx) + view.alpha[index + camera.width + 1] * fx;
    return top * (1.0f - fy) + bottom * fy;
}

class Grid {
public:
    Grid(const Vector3 low, const Vector3 high, const int resolution) : origin(low) {
        const Vector3 size = high - low;
        spacing = std::max({size.x, size.y, size.z}) / static_cast<float>(resolution);
        counts = {static_cast<int>(std::ceil(size.x / spacing)) + 1, static_cast<int>(std::ceil(size.y / spacing)) + 1,
                  static_cast<int>(std::ceil(size.z / spacing)) + 1};
        values.assign(static_cast<std::size_t>(counts[0]) * counts[1] * counts[2], 0.0f);
    }

    std::uint32_t getIndex(const int x, const int y, const int z) const { return static_cast<std::uint32_t>((z * counts[1] + y) * counts[0] + x); }
    Vector3 getPoint(const std::uint32_t index) const {
        const auto x = static_cast<int>(index % static_cast<std::uint32_t>(counts[0]));
        const auto y = static_cast<int>(index / static_cast<std::uint32_t>(counts[0]) % static_cast<std::uint32_t>(counts[1]));
        const auto z = static_cast<int>(index / static_cast<std::uint32_t>(counts[0] * counts[1]));
        return origin + Vector3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} * spacing;
    }
    bool isBorder(const int x, const int y, const int z) const {
        return x == 0 || y == 0 || z == 0 || x == counts[0] - 1 || y == counts[1] - 1 || z == counts[2] - 1;
    }

    void carve(const std::span<const Scenes::View> views) {
        const auto evaluateSlice = [&](const int z) {
            for (int y = 0; y < counts[1]; ++y)
                for (int x = 0; x < counts[0]; ++x) {
                    const std::uint32_t index = getIndex(x, y, z);
                    if (isBorder(x, y, z))
                        continue;
                    float value = 1.0f;
                    const Vector3 point = getPoint(index);
                    for (const Scenes::View& view : views) {
                        value = std::min(value, sampleAlpha(view, point));
                        if (value <= 0.0f)
                            break;
                    }
                    values[index] = value;
                }
        };
        const unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::jthread> threads;
        for (unsigned thread = 0; thread < threadCount; ++thread)
            threads.emplace_back([&, thread] {
                for (int z = static_cast<int>(thread); z < counts[2]; z += static_cast<int>(threadCount))
                    evaluateSlice(z);
            });
    }

    SurfaceMesh extract() const {
        SurfaceMesh mesh;
        std::unordered_map<std::uint64_t, std::uint32_t> edgeVertices;
        const auto getEdgeVertex = [&](std::uint32_t inside, std::uint32_t outside) {
            const std::uint32_t low = std::min(inside, outside);
            const std::uint32_t high = std::max(inside, outside);
            const auto [found, inserted] = edgeVertices.try_emplace(std::uint64_t{low} << 32 | high, static_cast<std::uint32_t>(mesh.positions.size()));
            if (inserted) {
                const float t = std::clamp((isoValue - values[low]) / (values[high] - values[low]), 0.01f, 0.99f);
                mesh.positions.push_back(getPoint(low) + (getPoint(high) - getPoint(low)) * t);
            }
            return found->second;
        };
        const auto addTriangle = [&](std::array<std::uint32_t, 3> corners, const Vector3 inward) {
            const Vector3 a = mesh.positions[corners[0]];
            if ((mesh.positions[corners[1]] - a).cross(mesh.positions[corners[2]] - a).dot(inward) > 0.0f)
                std::swap(corners[1], corners[2]);
            mesh.triangles.insert(mesh.triangles.end(), corners.begin(), corners.end());
        };
        for (int z = 0; z + 1 < counts[2]; ++z)
            for (int y = 0; y + 1 < counts[1]; ++y)
                for (int x = 0; x + 1 < counts[0]; ++x) {
                    std::array<std::uint32_t, 8> cube{};
                    int insideCount = 0;
                    for (int corner = 0; corner < 8; ++corner) {
                        cube[corner] = getIndex(x + (corner & 1), y + (corner >> 1 & 1), z + (corner >> 2 & 1));
                        insideCount += values[cube[corner]] > isoValue ? 1 : 0;
                    }
                    if (insideCount == 0 || insideCount == 8)
                        continue;
                    for (const std::array<int, 3>& order : axisOrders) {
                        const std::array<std::uint32_t, 4> tetrahedron{cube[0], cube[1 << order[0]], cube[(1 << order[0]) | (1 << order[1])], cube[7]};
                        std::array<std::uint32_t, 4> inside{};
                        std::array<std::uint32_t, 4> outside{};
                        std::size_t insides = 0;
                        std::size_t outsides = 0;
                        Vector3 inward;
                        for (const std::uint32_t point : tetrahedron) {
                            const bool isInside = values[point] > isoValue;
                            (isInside ? inside[insides++] : outside[outsides++]) = point;
                            inward += getPoint(point) * (isInside ? 1.0f : -1.0f);
                        }
                        if (insides == 1 || insides == 3) {
                            const bool single = insides == 1;
                            const std::uint32_t apex = single ? inside[0] : outside[0];
                            const std::array<std::uint32_t, 4>& others = single ? outside : inside;
                            addTriangle({getEdgeVertex(apex, others[0]), getEdgeVertex(apex, others[1]), getEdgeVertex(apex, others[2])}, inward);
                        }
                        if (insides != 2)
                            continue;
                        const std::uint32_t ac = getEdgeVertex(inside[0], outside[0]);
                        const std::uint32_t ad = getEdgeVertex(inside[0], outside[1]);
                        const std::uint32_t bd = getEdgeVertex(inside[1], outside[1]);
                        const std::uint32_t bc = getEdgeVertex(inside[1], outside[0]);
                        addTriangle({ac, ad, bd}, inward);
                        addTriangle({ac, bd, bc}, inward);
                    }
                }
        mesh.colors.assign(mesh.positions.size(), {0.5f, 0.5f, 0.5f});
        mesh.materials.assign(mesh.positions.size(), {0.0f, 0.5f, 0.0f});
        return mesh;
    }

private:
    Vector3 origin;
    float spacing = 0.0f;
    std::array<int, 3> counts{};
    std::vector<float> values;
};

}

std::optional<SurfaceMesh> carveVisualHull(const Scenes::Dataset& dataset, const int resolution) {
    if (dataset.train.empty() || dataset.points.positions.empty())
        return std::nullopt;
    if (std::ranges::any_of(dataset.train, [](const Scenes::View& view) { return view.alpha.empty(); }))
        return std::nullopt;
    Vector3 low{1e30f, 1e30f, 1e30f};
    Vector3 high{-1e30f, -1e30f, -1e30f};
    for (const NoorRHI::float3 point : dataset.points.positions) {
        low = {std::min(low.x, point.x), std::min(low.y, point.y), std::min(low.z, point.z)};
        high = {std::max(high.x, point.x), std::max(high.y, point.y), std::max(high.z, point.z)};
    }
    const Vector3 margin = (high - low) * boxMargin;
    Grid grid(low - margin, high + margin, resolution);
    grid.carve(dataset.train);
    SurfaceMesh mesh = grid.extract();
    if (mesh.triangles.empty())
        return std::nullopt;
    return mesh;
}

}
