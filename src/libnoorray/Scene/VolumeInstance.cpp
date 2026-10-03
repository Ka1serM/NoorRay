#include "Scene/VolumeInstance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <unordered_map>

#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include "Scene/Scene.h"

namespace
{

// Normals of coplanar faces differ by rounding alone.
constexpr float CoplanarTolerance = 1.0e-4f;
// Vertices closer than this fraction of the mesh's size are one corner.
constexpr float WeldFraction = 1.0e-5f;

struct Edge
{
    uint32_t from;
    uint32_t to;
    glm::vec3 normal;
    uint32_t faces;
    bool sharp;
};

}

VolumeOutline::VolumeOutline(const MeshGeometry& geometry, const float featureAngleDegrees)
{
    const std::span<const glm::vec3> positions = geometry.positions;
    const std::span<const uint32_t> indices = geometry.indices;

    // Faces split along normals or UV seams still share positions, and an
    // edge only pairs its two faces through the position.
    const float cell = std::max(glm::length(geometry.boundsMax - geometry.boundsMin) * WeldFraction,
        1.0e-6f);
    std::map<std::array<int64_t, 3>, uint32_t> corners;
    std::vector<uint32_t> cornerOfVertex(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        cornerOfVertex[i] = corners.try_emplace({std::llround(positions[i].x / cell),
            std::llround(positions[i].y / cell), std::llround(positions[i].z / cell)},
            static_cast<uint32_t>(corners.size())).first->second;

    const float flatDot = std::cos(glm::radians(featureAngleDegrees)) - CoplanarTolerance;
    std::unordered_map<uint64_t, Edge> edges;
    for (std::size_t triangle = 0; triangle + 2 < indices.size(); triangle += 3)
    {
        const std::array<uint32_t, 3> vertices{indices[triangle], indices[triangle + 1], indices[triangle + 2]};
        const glm::vec3 facing = glm::cross(positions[vertices[1]] - positions[vertices[0]],
            positions[vertices[2]] - positions[vertices[0]]);
        const float area = glm::length(facing);
        if (area == 0.0f)
            continue;
        const glm::vec3 normal = facing / area;
        for (std::size_t side = 0; side < 3; ++side)
        {
            const uint32_t from = vertices[side];
            const uint32_t to = vertices[(side + 1) % 3];
            const uint32_t a = cornerOfVertex[from];
            const uint32_t b = cornerOfVertex[to];
            if (a == b)
                continue;
            const uint64_t key = static_cast<uint64_t>(std::min(a, b)) << 32 | std::max(a, b);
            const auto [found, inserted] = edges.try_emplace(key, Edge{from, to, normal, 0, false});
            Edge& edge = found->second;
            if (!inserted)
                edge.sharp = edge.sharp || edge.faces > 1 || glm::dot(edge.normal, normal) < flatDot;
            ++edge.faces;
        }
    }

    for (const auto& [key, edge] : edges)
    {
        if (edge.faces != 1 && !edge.sharp)
            continue;
        segments_.push_back(positions[edge.from]);
        segments_.push_back(positions[edge.to]);
    }
}

const noorrhi::Buffer<glm::vec3>& VolumeOutline::upload(noorrhi::Device& device) const
{
    if (!buffer_ && !segments_.empty())
    {
        buffer_ = device.buffer<glm::vec3>(segments_.size());
        buffer_.upload(std::span<const glm::vec3>(segments_));
    }
    return buffer_;
}

VolumeInstance::VolumeInstance(Scene& scene, const std::string& name,
    std::shared_ptr<const VolumeOutline> outline, const Transform& transform,
    const glm::vec3& color)
    : SceneObject(scene, name, transform), outline(std::move(outline)), color(color)
{
}

VolumeInstance::VolumeInstance(const VolumeInstance& other)
    : SceneObject(other), outline(other.outline), color(other.color)
{
}

std::unique_ptr<SceneObject> VolumeInstance::clone() const
{
    return std::make_unique<VolumeInstance>(*this);
}

void VolumeInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    if (scene)
        scene->markVolumeChanged(slot);
}

void VolumeInstance::onVisibilityChanged()
{
    if (scene)
        scene->markVolumeChanged(slot);
}
