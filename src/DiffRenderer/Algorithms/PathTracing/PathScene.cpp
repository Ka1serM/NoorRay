#include "DiffRenderer/Algorithms/PathTracing/PathScene.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr float objectRadius = 0.5f;
constexpr float objectHeight = 0.65f;

float3 operator-(const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 scaleTo(const float3 value, const float radius) {
    const float scale = radius / std::sqrt(dot(value, value));
    return {value.x * scale, value.y * scale, value.z * scale};
}

float getSignedOffset(Core::Random& random, const float low, const float high) {
    const float magnitude = random.uniform(low, high);
    return random.uniform(-1.0f, 1.0f) < 0.0f ? -magnitude : magnitude;
}

class Icosphere {
public:
    Icosphere() {
        const float g = (1.0f + std::sqrt(5.0f)) * 0.5f;
        for (const float3 corner : {float3{-1, g, 0}, float3{1, g, 0}, float3{-1, -g, 0}, float3{1, -g, 0}, float3{0, -1, g}, float3{0, 1, g},
                                    float3{0, -1, -g}, float3{0, 1, -g}, float3{g, 0, -1}, float3{g, 0, 1}, float3{-g, 0, -1}, float3{-g, 0, 1}})
            mesh.positions.push_back(scaleTo(corner, objectRadius));
        mesh.indices = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
                        3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};
        for (int level = 0; level < 3; ++level)
            subdivide();
        orientOutward();
    }

    PathMesh mesh;

private:
    std::uint32_t getMidpoint(const std::uint32_t a, const std::uint32_t b) {
        const std::pair key{std::min(a, b), std::max(a, b)};
        if (const auto found = midpoints.find(key); found != midpoints.end())
            return found->second;
        const float3 first = mesh.positions[a];
        const float3 second = mesh.positions[b];
        mesh.positions.push_back(scaleTo({first.x + second.x, first.y + second.y, first.z + second.z}, objectRadius));
        const auto index = static_cast<std::uint32_t>(mesh.positions.size() - 1);
        midpoints.emplace(key, index);
        return index;
    }

    void subdivide() {
        std::vector<std::uint32_t> indices;
        indices.reserve(mesh.indices.size() * 4);
        for (std::size_t face = 0; face < mesh.indices.size(); face += 3) {
            const std::uint32_t a = mesh.indices[face];
            const std::uint32_t b = mesh.indices[face + 1];
            const std::uint32_t c = mesh.indices[face + 2];
            const std::uint32_t ab = getMidpoint(a, b);
            const std::uint32_t bc = getMidpoint(b, c);
            const std::uint32_t ca = getMidpoint(c, a);
            indices.insert(indices.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        mesh.indices = std::move(indices);
    }

    void orientOutward() {
        for (std::size_t face = 0; face < mesh.indices.size(); face += 3) {
            const float3 a = mesh.positions[mesh.indices[face]];
            const float3 normal = cross(mesh.positions[mesh.indices[face + 1]] - a, mesh.positions[mesh.indices[face + 2]] - a);
            if (dot(normal, a) < 0.0f)
                std::swap(mesh.indices[face + 1], mesh.indices[face + 2]);
        }
    }

    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> midpoints;
};

std::vector<uint4> findEdges(const PathMesh& mesh) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, uint4> edges;
    for (std::uint32_t face = 0; face < mesh.indices.size() / 3; ++face)
        for (std::uint32_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t a = mesh.indices[3 * face + corner];
            const std::uint32_t b = mesh.indices[3 * face + (corner + 1) % 3];
            const std::pair key{std::min(a, b), std::max(a, b)};
            auto [entry, inserted] = edges.try_emplace(key, uint4{key.first, key.second, face, face});
            if (!inserted)
                entry->second.w = face;
        }
    std::vector<uint4> result;
    result.reserve(edges.size());
    for (const auto& [key, edge] : edges)
        result.push_back(edge);
    return result;
}

}

PathScene::PathScene(Core::Random& random) {
    ground.positions = {{-3.0f, 0.0f, -3.0f}, {-3.0f, 0.0f, 3.0f}, {3.0f, 0.0f, 3.0f}, {3.0f, 0.0f, -3.0f}};
    ground.indices = {0, 1, 2, 0, 2, 3};
    object = Icosphere().mesh;
    object.normals.reserve(object.positions.size());
    for (const float3 position : object.positions)
        object.normals.push_back(scaleTo(position, 1.0f));
    edges = findEdges(object);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        truth.albedos[channel] = random.uniform(0.15f, 0.9f);
        truth.albedos[3 + channel] = random.uniform(0.3f, 0.8f);
    }
    truth.translation = {random.uniform(-0.25f, 0.25f), objectHeight, random.uniform(-0.25f, 0.25f)};
}

PathParameters PathScene::perturb(Core::Random& random) const {
    PathParameters start;
    for (float& albedo : start.albedos)
        albedo = random.uniform(0.3f, 0.7f);
    start.translation = {truth.translation[0] + getSignedOffset(random, 0.15f, 0.3f), truth.translation[1] + getSignedOffset(random, 0.1f, 0.15f),
                         truth.translation[2] + getSignedOffset(random, 0.15f, 0.3f)};
    return start;
}

}
