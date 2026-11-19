#include "NoorRay/Gpu/AliasTable.h"

#include <cstdint>

namespace NoorRay::Gpu {

std::vector<Shaders::AliasEntry> buildAliasTable(const std::span<const float> weights) {
    if (weights.empty())
        return {Shaders::AliasEntry{1.0f, 0, 1.0f, 0}};
    const std::size_t count = weights.size();
    double total = 0.0;
    for (const float weight : weights)
        total += weight > 0.0f ? weight : 0.0f;
    std::vector<double> scaled(count);
    std::vector<Shaders::AliasEntry> entries(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double pdf = total > 0.0 ? (weights[i] > 0.0f ? weights[i] / total : 0.0) : 1.0 / static_cast<double>(count);
        scaled[i] = pdf * static_cast<double>(count);
        entries[i] = {1.0f, static_cast<std::uint32_t>(i), static_cast<float>(pdf), 0};
    }
    std::vector<std::uint32_t> small;
    std::vector<std::uint32_t> large;
    for (std::size_t i = 0; i < count; ++i)
        (scaled[i] < 1.0 ? small : large).push_back(static_cast<std::uint32_t>(i));
    while (!small.empty() && !large.empty()) {
        const std::uint32_t less = small.back();
        small.pop_back();
        const std::uint32_t more = large.back();
        entries[less].threshold = static_cast<float>(scaled[less]);
        entries[less].alias = more;
        scaled[more] += scaled[less] - 1.0;
        if (scaled[more] < 1.0) {
            large.pop_back();
            small.push_back(more);
        }
    }
    return entries;
}

}
