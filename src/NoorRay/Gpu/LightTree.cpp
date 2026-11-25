#include "NoorRay/Gpu/LightTree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace NoorRay::Gpu {

namespace {

constexpr std::uint32_t bucketCount = 12;
//! Below this depth the heuristic splits; deeper nodes split at the median count, so a tree of up to 2^32 lights stays within lightTreeMaxDepth.
constexpr std::uint32_t heuristicDepth = 32;

float safeAcos(const float value) {
    return std::acos(std::clamp(value, -1.0f, 1.0f));
}

//! The angle between two unit vectors, accurate for small and large angles.
float angleBetween(const Core::float3 a, const Core::float3 b) {
    if (Core::dot(a, b) < 0.0f)
        return Core::pi - 2.0f * std::asin(std::clamp(0.5f * Core::length(a + b), 0.0f, 1.0f));
    return 2.0f * std::asin(std::clamp(0.5f * Core::length(b - a), 0.0f, 1.0f));
}

//! Rodrigues' rotation of v about the unit axis.
Core::float3 rotate(const Core::float3 v, const Core::float3 axis, const float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return v * c + Core::cross(axis, v) * s + axis * (Core::dot(axis, v) * (1.0f - c));
}

//! The smallest cone of normals that holds both cones, as PBRT's DirectionCone::Union.
void uniteCones(LightBounds& result, const LightBounds& a, const LightBounds& b) {
    const float thetaA = safeAcos(a.cosThetaO);
    const float thetaB = safeAcos(b.cosThetaO);
    const float thetaD = angleBetween(a.axis, b.axis);
    if (std::min(thetaD + thetaB, Core::pi) <= thetaA) {
        result.axis = a.axis;
        result.cosThetaO = a.cosThetaO;
        return;
    }
    if (std::min(thetaD + thetaA, Core::pi) <= thetaB) {
        result.axis = b.axis;
        result.cosThetaO = b.cosThetaO;
        return;
    }
    const float thetaO = 0.5f * (thetaA + thetaD + thetaB);
    const Core::float3 pivot = Core::cross(a.axis, b.axis);
    result.axis = a.axis;
    result.cosThetaO = -1.0f;
    if (thetaO >= Core::pi || Core::dot(pivot, pivot) <= 0.0f)
        return;
    result.axis = Core::normalize(rotate(a.axis, Core::normalize(pivot), thetaO - thetaA));
    result.cosThetaO = std::cos(thetaO);
}

//! Lights in one plane that emit to the same side; the tolerance is relative to the plane offsets.
bool isCoplanar(const LightBounds& a, const LightBounds& b) {
    const float tolerance = 1e-4f * std::max({std::abs(a.planeOffset), std::abs(b.planeOffset), 1.0f});
    return a.planar && b.planar && Core::dot(a.axis, b.axis) >= 1.0f - 1e-6f && std::abs(a.planeOffset - b.planeOffset) <= tolerance;
}

LightBounds unite(const LightBounds& a, const LightBounds& b) {
    if (a.power <= 0.0f)
        return b;
    if (b.power <= 0.0f)
        return a;
    LightBounds result;
    result.planar = isCoplanar(a, b);
    result.planeOffset = a.planeOffset;
    result.bounds = a.bounds;
    result.bounds.grow(b.bounds);
    result.power = a.power + b.power;
    result.cosThetaE = std::min(a.cosThetaE, b.cosThetaE);
    result.twoSided = a.twoSided || b.twoSided;
    uniteCones(result, a, b);
    //! Tokuyoshi et al. 2024, Eqs. 2, 4 and 5: the flux-weighted means of the emission directions and positions, and the variance of the mixture.
    const float total = a.flux + b.flux;
    const float weightA = total > 0.0f ? a.flux / total : 0.5f;
    const float weightB = 1.0f - weightA;
    const Core::float3 apart = a.mean - b.mean;
    result.mean = a.mean * weightA + b.mean * weightB;
    result.variance = a.variance * weightA + b.variance * weightB + weightA * weightB * Core::dot(apart, apart);
    result.emission = a.emission * weightA + b.emission * weightB;
    result.flux = total;
    return result;
}

//! The vMF sharpness of a mean direction of the given length, as Banerjee et al. 2005, clamped to lightSharpnessMax.
float vmfSharpness(const float length) {
    if (length >= 1.0f)
        return Shaders::lightSharpnessMax;
    return std::min(length * (3.0f - length * length) / (1.0f - length * length), Shaders::lightSharpnessMax);
}

//! The integral over the sphere of the SG exp(sharpness * (cos - 1)).
float sgIntegral(const float sharpness) {
    if (sharpness < 1e-4f)
        return 4.0f * Core::pi;
    return 2.0f * Core::pi * -std::expm1(-2.0f * sharpness) / sharpness;
}

//! The SG light of a cluster (Tokuyoshi et al. 2024, Section 3.1); a planar cluster keeps its exact plane normal for the plane test.
Shaders::LightTreeNode makeNode(const LightBounds& total) {
    const float length = Core::length(total.emission);
    const Core::float3 axis = total.planar ? total.axis : (length > 0.0f ? total.emission / length : Core::float3{0.0f, 0.0f, 1.0f});
    const float sharpness = vmfSharpness(length);
    const Core::float3 reach = Core::max(total.bounds.max - total.mean, total.mean - total.bounds.min);
    Shaders::LightTreeNode node{};
    node.mean = {total.mean.x, total.mean.y, total.mean.z, total.variance};
    node.axis = {axis.x, axis.y, axis.z, sharpness};
    node.power = {total.flux / (2.0f * Core::pi * sgIntegral(sharpness)), Core::length(reach), total.planeOffset, 0.0f};
    node.meta.y = total.planar ? Shaders::lightNodePlanar : 0u;
    return node;
}

float surfaceArea(const Core::Bounds& bounds) {
    const Core::float3 e = bounds.extent();
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
}

//! PBRT's surface area orientation heuristic: power times the solid angle measure of the emission cone times the area, stretched along thin axes.
float splitCost(const LightBounds& bounds, const float stretch) {
    if (bounds.power <= 0.0f)
        return 0.0f;
    const float thetaO = safeAcos(bounds.cosThetaO);
    const float thetaE = safeAcos(bounds.cosThetaE);
    const float thetaW = std::min(thetaO + thetaE, Core::pi);
    const float sinThetaO = std::sqrt(std::max(1.0f - bounds.cosThetaO * bounds.cosThetaO, 0.0f));
    const float measure = 2.0f * Core::pi * (1.0f - bounds.cosThetaO)
                          + 0.5f * Core::pi * (2.0f * thetaW * sinThetaO - std::cos(thetaO - 2.0f * thetaW) - 2.0f * thetaO * sinThetaO + bounds.cosThetaO);
    return bounds.power * measure * stretch * surfaceArea(bounds.bounds);
}

float axisOf(const Core::float3 v, const int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

class Builder {
public:
    Builder(std::span<const LightBounds> lights, std::vector<Shaders::LightTreeNode>& nodes, std::span<Core::uint2> trails)
        : lights(lights), nodes(nodes), trails(trails) {}

    std::expected<std::uint32_t, Core::Error> build(const std::span<std::uint32_t> items, const std::uint32_t depth, const std::uint64_t trail) {
        if (depth >= Shaders::lightTreeMaxDepth)
            return Core::fail(Core::ErrorCode::InvalidState, "the light tree is deeper than its trails");
        LightBounds total;
        for (const std::uint32_t item : items)
            total = unite(total, lights[item]);
        const auto index = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(makeNode(total));
        if (items.size() == 1) {
            nodes[index].meta.x = items[0];
            nodes[index].meta.y |= Shaders::lightNodeLeaf;
            trails[items[0]] = {static_cast<std::uint32_t>(trail), static_cast<std::uint32_t>(trail >> 32)};
            return index;
        }
        const std::size_t middle = split(items, total, depth);
        if (auto first = build(items.first(middle), depth + 1, trail); !first)
            return first;
        auto second = build(items.subspan(middle), depth + 1, trail | std::uint64_t{1} << depth);
        if (!second)
            return second;
        nodes[index].meta.x = *second;
        return index;
    }

private:
    //! Orders the items into the two children and returns the size of the first; both children hold at least one light.
    std::size_t split(const std::span<std::uint32_t> items, const LightBounds& total, const std::uint32_t depth) {
        Core::Bounds centroids;
        for (const std::uint32_t item : items)
            centroids.grow(lights[item].bounds.center());
        const Core::float3 extent = centroids.extent();
        const int longest = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
        const auto median = [&](const int axis) {
            const std::size_t middle = items.size() / 2;
            std::ranges::nth_element(items, items.begin() + static_cast<std::ptrdiff_t>(middle), {},
                                     [&](const std::uint32_t item) { return axisOf(lights[item].bounds.center(), axis); });
            return middle;
        };
        if (depth >= heuristicDepth || axisOf(extent, longest) <= 0.0f)
            return median(longest);
        const Core::float3 size = total.bounds.extent();
        const float largest = Core::maxComponent(size);
        float bestCost = std::numeric_limits<float>::infinity();
        int bestAxis = -1;
        std::uint32_t bestBucket = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const float low = axisOf(centroids.min, axis);
            const float width = axisOf(extent, axis);
            if (width <= 0.0f)
                continue;
            std::array<LightBounds, bucketCount> buckets{};
            for (const std::uint32_t item : items)
                buckets[bucketOf(lights[item], axis, low, width)] = unite(buckets[bucketOf(lights[item], axis, low, width)], lights[item]);
            const float stretch = axisOf(size, axis) > 0.0f ? largest / axisOf(size, axis) : 1.0f;
            std::array<float, bucketCount> below{};
            LightBounds running;
            for (std::uint32_t bucket = 0; bucket + 1 < bucketCount; ++bucket) {
                running = unite(running, buckets[bucket]);
                below[bucket] = splitCost(running, stretch);
            }
            running = {};
            for (std::uint32_t bucket = bucketCount - 1; bucket > 0; --bucket) {
                running = unite(running, buckets[bucket]);
                const float cost = below[bucket - 1] + splitCost(running, stretch);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBucket = bucket;
                }
            }
        }
        if (bestAxis < 0)
            return median(longest);
        const float low = axisOf(centroids.min, bestAxis);
        const float width = axisOf(extent, bestAxis);
        const auto second = std::ranges::partition(items, [&](const std::uint32_t item) { return bucketOf(lights[item], bestAxis, low, width) < bestBucket; });
        const auto middle = static_cast<std::size_t>(second.begin() - items.begin());
        if (middle == 0 || middle == items.size())
            return median(longest);
        return middle;
    }

    static std::uint32_t bucketOf(const LightBounds& light, const int axis, const float low, const float width) {
        const float offset = (axisOf(light.bounds.center(), axis) - low) / width;
        return std::min(static_cast<std::uint32_t>(std::max(offset, 0.0f) * bucketCount), bucketCount - 1);
    }

    std::span<const LightBounds> lights;
    std::vector<Shaders::LightTreeNode>& nodes;
    std::span<Core::uint2> trails;
};

//! The frontier of the tree at the cut depth, left to right; a leaf above it stands for itself.
void collectCut(const std::span<const Shaders::LightTreeNode> nodes, const std::uint32_t index, const std::uint32_t depth, Shaders::LightTreeSet& set) {
    if (depth == Shaders::lightTreeCutDepth || (nodes[index].meta.y & Shaders::lightNodeLeaf) != 0) {
        set.cut[set.cutCount++] = index;
        return;
    }
    collectCut(nodes, index + 1, depth + 1, set);
    collectCut(nodes, nodes[index].meta.x, depth + 1, set);
}

}

std::expected<LightTrees, Core::Error> buildLightTrees(const std::span<const LightBounds> lights, const std::uint32_t setCount) {
    LightTrees trees;
    trees.trails.assign(std::max<std::size_t>(lights.size() * setCount, 1), Core::uint2{});
    std::vector<std::uint32_t> items;
    for (std::uint32_t set = 0; set < setCount; ++set) {
        Shaders::LightTreeSet record{};
        record.root = static_cast<std::uint32_t>(trees.nodes.size());
        items.clear();
        for (std::uint32_t light = 0; light < lights.size(); ++light)
            if (!lights[light].distant && lights[light].power > 0.0f && (lights[light].sets >> set & 1u) != 0)
                items.push_back(light);
        if (!items.empty()) {
            Builder builder(lights, trees.nodes, std::span(trees.trails).subspan(set * lights.size(), lights.size()));
            if (auto built = builder.build(items, 0, 0); !built)
                return std::unexpected(std::move(built.error()));
        }
        record.nodeCount = static_cast<std::uint32_t>(trees.nodes.size()) - record.root;
        if (record.nodeCount > 0)
            collectCut(trees.nodes, record.root, 0, record);
        record.firstDistant = static_cast<std::uint32_t>(trees.nodes.size());
        for (std::uint32_t light = 0; light < lights.size(); ++light) {
            const LightBounds& bounds = lights[light];
            if (!bounds.distant || bounds.power <= 0.0f || (bounds.sets >> set & 1u) == 0)
                continue;
            //! The disk of a distant light as a vMF about its direction, from the mean direction of a uniform cone.
            Shaders::LightTreeNode node{};
            node.axis = {bounds.axis.x, bounds.axis.y, bounds.axis.z, vmfSharpness(0.5f * (1.0f + bounds.cosThetaO))};
            node.power.x = bounds.power;
            node.meta = {light, Shaders::lightNodeLeaf, 0, 0};
            trees.nodes.push_back(node);
        }
        record.distantCount = static_cast<std::uint32_t>(trees.nodes.size()) - record.firstDistant;
        trees.sets.push_back(record);
    }
    return trees;
}

}
