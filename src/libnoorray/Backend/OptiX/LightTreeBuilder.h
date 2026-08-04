#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/OptiX/ABI/SceneData.h"

namespace nr::light_tree
{

// Builds the compact binary hierarchy used by the GPU sampler. The split is
// a median split along the widest centroid axis: it is cheap to rebuild when
// lights move, remains balanced, and avoids the pointer-heavy CPU structures
// that are a poor fit for a hot GPU sampling loop.
inline void build(std::vector<DirectLightCandidate>& candidates,
    std::vector<LightTreeNode>& nodes)
{
    nodes.clear();
    for (auto& candidate : candidates)
        candidate.lightTreeLeaf = InvalidIndex;

    std::vector<uint32_t> active;
    active.reserve(candidates.size());
    for (uint32_t i = 0; i < candidates.size(); ++i)
    {
        const float weight = candidates[i].selectionWeight;
        if (std::isfinite(weight) && weight > 0.0f)
            active.push_back(i);
    }
    if (active.empty())
        return;

    const auto candidateCenter = [&](const uint32_t index) {
        return candidates[index].position;
    };
    const auto candidateRadius = [&](const uint32_t index) {
        return std::max(candidates[index].spatialRadius, 0.0f);
    };

    const auto setAggregate = [&](LightTreeNode& node,
                                  const LightTreeNode& left,
                                  const LightTreeNode& right) {
        const glm::vec3 leftMin = glm::vec3(left.sphere)
            - glm::vec3(left.sphere.w);
        const glm::vec3 leftMax = glm::vec3(left.sphere)
            + glm::vec3(left.sphere.w);
        const glm::vec3 rightMin = glm::vec3(right.sphere)
            - glm::vec3(right.sphere.w);
        const glm::vec3 rightMax = glm::vec3(right.sphere)
            + glm::vec3(right.sphere.w);
        const glm::vec3 minimum = glm::min(leftMin, rightMin);
        const glm::vec3 maximum = glm::max(leftMax, rightMax);
        const glm::vec3 center = 0.5f * (minimum + maximum);
        node.sphere = glm::vec4(center, glm::length(maximum - center));
        node.selectionWeight = left.selectionWeight + right.selectionWeight;
        node.flags = (left.flags | right.flags) & LightTreeHasDirectional;
    };

    const auto buildNode = [&](auto&& self, const size_t begin,
                               const size_t end, const uint32_t parent) -> uint32_t {
        const uint32_t nodeIndex = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        nodes[nodeIndex].parent = parent;

        if (end - begin == 1u)
        {
            const uint32_t candidateIndex = active[begin];
            const DirectLightCandidate& candidate = candidates[candidateIndex];
            nodes[nodeIndex].sphere = glm::vec4(
                candidateCenter(candidateIndex), candidateRadius(candidateIndex));
            nodes[nodeIndex].selectionWeight = candidate.selectionWeight;
            nodes[nodeIndex].childOrLightIndex = candidateIndex;
            nodes[nodeIndex].flags = LightTreeLeaf;
            if (candidate.type == DirectLightType::Directional)
                nodes[nodeIndex].flags |= LightTreeHasDirectional;
            candidates[candidateIndex].lightTreeLeaf = nodeIndex;
            return nodeIndex;
        }

        glm::vec3 minimum(std::numeric_limits<float>::infinity());
        glm::vec3 maximum(-std::numeric_limits<float>::infinity());
        for (size_t i = begin; i < end; ++i)
        {
            const uint32_t index = active[i];
            const glm::vec3 center = candidateCenter(index);
            minimum = glm::min(minimum, center);
            maximum = glm::max(maximum, center);
        }
        const glm::vec3 extent = maximum - minimum;
        uint32_t axis = 0u;
        if (extent.y > extent.x && extent.y >= extent.z)
            axis = 1u;
        else if (extent.z > extent.x && extent.z > extent.y)
            axis = 2u;
        const size_t middle = begin + (end - begin) / 2u;
        std::nth_element(active.begin() + begin, active.begin() + middle,
            active.begin() + end, [&](const uint32_t a, const uint32_t b) {
                return candidateCenter(a)[axis] < candidateCenter(b)[axis];
            });
        const uint32_t leftIndex = self(self, begin, middle, nodeIndex);
        const uint32_t rightIndex = self(self, middle, end, nodeIndex);
        // The contiguous preorder layout makes the left child implicit.
        (void)leftIndex;
        nodes[nodeIndex].childOrLightIndex = rightIndex;
        setAggregate(nodes[nodeIndex], nodes[nodeIndex + 1u],
            nodes[rightIndex]);
        return nodeIndex;
    };

    buildNode(buildNode, 0u, active.size(), InvalidIndex);
}

} // namespace nr::light_tree
