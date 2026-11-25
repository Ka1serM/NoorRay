#include "DiffRenderer/Methods/GaussianSplatting/NearestNeighbours.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>

namespace DiffRenderer::Methods {

namespace {

constexpr std::size_t neighbourCount = 3;

float getAxis(const NoorRHI::float3& point, const std::size_t axis) {
    return axis == 0 ? point.x : (axis == 1 ? point.y : point.z);
}

class KdTree {
public:
    explicit KdTree(const std::span<const NoorRHI::float3> points) : points(points), order(points.size()) {
        std::iota(order.begin(), order.end(), 0u);
        build(0, order.size(), 0);
    }

    float getMeanSquaredDistance(const std::uint32_t self) const {
        std::array<float, neighbourCount> best;
        best.fill(std::numeric_limits<float>::infinity());
        search(0, order.size(), 0, self, best);
        float sum = 0.0f;
        std::size_t found = 0;
        for (const float distance : best) {
            if (distance == std::numeric_limits<float>::infinity())
                continue;
            sum += distance;
            ++found;
        }
        return found == 0 ? 0.0f : sum / static_cast<float>(found);
    }

private:
    void build(const std::size_t begin, const std::size_t end, const std::size_t axis) {
        if (end - begin <= 1)
            return;
        const std::size_t middle = (begin + end) / 2;
        std::nth_element(order.begin() + static_cast<std::ptrdiff_t>(begin), order.begin() + static_cast<std::ptrdiff_t>(middle),
                         order.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](const std::uint32_t a, const std::uint32_t b) { return getAxis(points[a], axis) < getAxis(points[b], axis); });
        build(begin, middle, (axis + 1) % 3);
        build(middle + 1, end, (axis + 1) % 3);
    }

    void search(const std::size_t begin, const std::size_t end, const std::size_t axis, const std::uint32_t self,
                std::array<float, neighbourCount>& best) const {
        if (begin >= end)
            return;
        const std::size_t middle = (begin + end) / 2;
        const NoorRHI::float3& query = points[self];
        const NoorRHI::float3& node = points[order[middle]];
        if (order[middle] != self) {
            const float dx = query.x - node.x;
            const float dy = query.y - node.y;
            const float dz = query.z - node.z;
            insert(dx * dx + dy * dy + dz * dz, best);
        }
        const float difference = getAxis(query, axis) - getAxis(node, axis);
        const std::size_t next = (axis + 1) % 3;
        if (difference < 0.0f)
            search(begin, middle, next, self, best);
        else
            search(middle + 1, end, next, self, best);
        if (difference * difference >= best.back())
            return;
        if (difference < 0.0f)
            search(middle + 1, end, next, self, best);
        else
            search(begin, middle, next, self, best);
    }

    static void insert(const float distance, std::array<float, neighbourCount>& best) {
        if (distance >= best.back())
            return;
        best.back() = distance;
        for (std::size_t index = neighbourCount - 1; index > 0 && best[index] < best[index - 1]; --index)
            std::swap(best[index], best[index - 1]);
    }

    std::span<const NoorRHI::float3> points;
    std::vector<std::uint32_t> order;
};

}

std::vector<float> getMeanSquaredNeighbourDistances(const std::span<const NoorRHI::float3> points) {
    const KdTree tree(points);
    std::vector<float> distances(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
        distances[index] = tree.getMeanSquaredDistance(static_cast<std::uint32_t>(index));
    return distances;
}

}
