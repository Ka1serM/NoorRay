#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Depth-based auto layout of a MaterialX node graph. Mirrors the layout the
// MaterialX graph editor produces: the material/output nodes form column 0 on
// the right, every node wired into them sits one column further left, and each
// column is stacked and centered vertically on the column to its right.
//
// Kept as a self-contained pure function so it can be unit tested and reused:
// it knows nothing about ImNodeFlow or ImGui, only about node names, measured
// heights and the graph's connections.
namespace MaterialXGraphLayout
{
struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

// A node to be placed. hasPosition marks a node whose document records an
// xpos/ypos: it stays where it is and only anchors the nodes around it.
struct LayoutNode
{
    std::string name;
    float width = 210.0f;  // Measured width; <= 1 means "unknown", a default is used.
    float height = 60.0f;  // Measured height; <= 1 means "unknown", a default is used.
    bool hasPosition = false;
    Vec2 position{};       // Current position, the anchor when hasPosition is set.
};

// Connections as (source, target) pairs: the node named source feeds an input
// of the node named target.
using Link = std::pair<std::string, std::string>;

// Returns, for the nodes that are not anchored by a stored position, the grid
// position each should be placed at. Anchored nodes are not part of the result.
std::unordered_map<std::string, Vec2> autoLayout(
    const std::vector<LayoutNode>& nodes, const std::vector<Link>& links);
}  // namespace MaterialXGraphLayout
