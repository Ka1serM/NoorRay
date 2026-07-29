#include "UI/MaterialXNodes/MaterialXGraphLayout.h"

#include <algorithm>
#include <functional>
#include <map>
#include <unordered_set>

namespace
{
constexpr float XSpacing = 250.0f;
constexpr float YGap = 40.0f;
constexpr float RootGap = 350.0f;
constexpr float Margin = 40.0f;
constexpr float DefaultHeight = 60.0f;
}

namespace MaterialXGraphLayout
{
namespace
{
float effectiveHeight(const LayoutNode& node)
{
    return node.height > 1.0f ? node.height : DefaultHeight;
}
}  // namespace

std::unordered_map<std::string, Vec2> autoLayout(
    const std::vector<LayoutNode>& nodes, const std::vector<Link>& links)
{
    // Reverse adjacency: which collected node consumes each node's output.
    std::unordered_map<std::string, std::vector<std::string>> consumers;
    std::unordered_set<std::string> known;
    for (const LayoutNode& node : nodes)
        known.insert(node.name);
    for (const Link& link : links) {
        if (known.count(link.first) && known.count(link.second))
            consumers[link.first].push_back(link.second);
    }

    // Level of a node = longest path from the roots (the nodes nothing in the
    // graph consumes), exactly the right-to-left layering the MaterialX graph
    // editor computes. Column 0 is the material, each connected input one
    // column further left.
    std::unordered_map<std::string, int> level;
    std::function<int(const std::string&, std::unordered_set<std::string>&)> deepest =
        [&](const std::string& name, std::unordered_set<std::string>& path) -> int {
        if (const auto found = level.find(name); found != level.end())
            return found->second;
        if (!path.insert(name).second)
            return 0;  // Back edge of a cycle: stop deepening.
        int best = 0;
        for (const std::string& consumer : consumers[name])
            best = std::max(best, deepest(consumer, path) + 1);
        path.erase(name);
        level[name] = best;
        return best;
    };
    std::unordered_set<std::string> path;
    for (const std::string& name : known)
        (void)deepest(name, path);

    std::unordered_map<std::string, LayoutNode> byName;
    for (const LayoutNode& node : nodes)
        byName.emplace(node.name, node);

    // Group into columns, keeping the order the nodes were passed in so the
    // stacking is deterministic.
    std::map<int, std::vector<std::string>> columns;
    for (const LayoutNode& node : nodes)
        columns[level[node.name]].push_back(node.name);
    const int maxLevel = columns.empty() ? 0 : columns.rbegin()->first;

    std::unordered_map<std::string, Vec2> result;
    for (const auto& [column, names] : columns) {
        const float x = Margin + static_cast<float>(maxLevel - column) * XSpacing;
        // Only the free nodes are stacked; anchored nodes stay where they are
        // and contribute to centering alone.
        float block = 0.0f;
        size_t pendingCount = 0;
        for (const std::string& name : names) {
            if (byName[name].hasPosition)
                continue;
            block += effectiveHeight(byName[name]);
            ++pendingCount;
        }
        if (pendingCount == 0)
            continue;
        block += static_cast<float>(pendingCount - 1) * YGap;

        float startY = Margin;
        if (column == 0) {
            // Roots start at the top; anchored roots already in the way push
            // the pending ones below them.
            float bottom = 0.0f;
            for (const std::string& name : names) {
                const LayoutNode& node = byName[name];
                if (!node.hasPosition)
                    continue;
                bottom = std::max(bottom, node.position.y + effectiveHeight(node));
            }
            if (bottom > 0.0f)
                startY = bottom + RootGap;
        } else {
            // Center this column vertically on the column to its right.
            const auto previous = columns.find(column - 1);
            if (previous != columns.end()) {
                float center = 0.0f;
                size_t count = 0;
                for (const std::string& name : previous->second) {
                    const LayoutNode& node = byName[name];
                    const float y = node.hasPosition ? node.position.y : result[name].y;
                    center += y + effectiveHeight(node) * 0.5f;
                    ++count;
                }
                if (count > 0)
                    startY = center / static_cast<float>(count) - block * 0.5f;
            }
        }

        float cursorY = startY;
        for (const std::string& name : names) {
            if (byName[name].hasPosition)
                continue;
            result[name] = {x, cursorY};
            cursorY += effectiveHeight(byName[name]) + YGap;
        }
    }
    return result;
}
}  // namespace MaterialXGraphLayout
