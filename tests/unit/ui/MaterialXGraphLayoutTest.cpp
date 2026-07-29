#include <catch2/catch_test_macros.hpp>

#include "UI/MaterialXNodes/MaterialXGraphLayout.h"

#include <cmath>

namespace
{
// The same constants the layout implementation uses; the test pins the numbers
// so a change to the layout style is a deliberate, reviewed one.
constexpr float XSpacing = 250.0f;
constexpr float YGap = 40.0f;
constexpr float RootGap = 350.0f;
constexpr float Margin = 40.0f;

MaterialXGraphLayout::LayoutNode node(const std::string& name)
{
    MaterialXGraphLayout::LayoutNode layout;
    layout.name = name;
    return layout;
}

bool near(float a, float b)
{
    return std::fabs(a - b) < 0.01f;
}

const float& x(const std::unordered_map<std::string, MaterialXGraphLayout::Vec2>& layout,
    const std::string& name)
{
    return layout.at(name).x;
}

const float& y(const std::unordered_map<std::string, MaterialXGraphLayout::Vec2>& layout,
    const std::string& name)
{
    return layout.at(name).y;
}
}  // namespace

TEST_CASE("A chain is laid out in columns from the root", "[materialx][layout]")
{
    // A feeds B feeds C: C is the root and must sit rightmost, A leftmost, all
    // three vertically aligned.
    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout(
            {node("A"), node("B"), node("C")}, {{"A", "B"}, {"B", "C"}});

    REQUIRE(layout.size() == 3);
    REQUIRE(x(layout, "A") == Margin);
    REQUIRE(x(layout, "B") == Margin + XSpacing);
    REQUIRE(x(layout, "C") == Margin + 2 * XSpacing);
    REQUIRE(near(y(layout, "A"), y(layout, "B")));
    REQUIRE(near(y(layout, "B"), y(layout, "C")));
}

TEST_CASE("A diamond centers the shared input on its consumers", "[materialx][layout]")
{
    // A feeds both X and Y, which feed R. A sits two columns left of R and is
    // vertically centered on the X/Y pair.
    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout(
            {node("A"), node("X"), node("Y"), node("R")},
            {{"A", "X"}, {"A", "Y"}, {"X", "R"}, {"Y", "R"}});

    REQUIRE(x(layout, "R") == Margin + 2 * XSpacing);
    REQUIRE(x(layout, "X") == Margin + XSpacing);
    REQUIRE(x(layout, "Y") == Margin + XSpacing);
    REQUIRE(x(layout, "A") == Margin);

    const float xCenter = (y(layout, "X") + y(layout, "Y")) * 0.5f;
    REQUIRE(near(y(layout, "A"), xCenter));
}

TEST_CASE("Nodes on the same column share the column x", "[materialx][layout]")
{
    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout(
            {node("R1"), node("R2")}, {});

    REQUIRE(x(layout, "R1") == Margin);
    REQUIRE(x(layout, "R2") == Margin);
    // Roots stack vertically with a gap for the whole column.
    REQUIRE(y(layout, "R2") > y(layout, "R1") + 40.0f);
}

TEST_CASE("Unconnected nodes are treated as roots", "[materialx][layout]")
{
    // "source" feeds "sink", so both "sink" and the unconnected "lonely" node
    // have no consumers and land in the root column on the right.
    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout(
            {node("lonely"), node("source"), node("sink")}, {{"source", "sink"}});

    REQUIRE(layout.size() == 3);
    REQUIRE(x(layout, "sink") == Margin + XSpacing);
    REQUIRE(x(layout, "lonely") == Margin + XSpacing);
    REQUIRE(x(layout, "source") == Margin);
}

TEST_CASE("Anchored nodes keep their position and anchor the rest", "[materialx][layout]")
{
    MaterialXGraphLayout::LayoutNode anchored = node("C");
    anchored.hasPosition = true;
    anchored.position = {1000.0f, 500.0f};
    anchored.height = 100.0f;

    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout(
            {node("A"), node("B"), anchored}, {{"A", "B"}, {"B", "C"}});

    // The anchored node is not moved, the free chain hangs off it centered on
    // its middle.
    REQUIRE(layout.count("C") == 0);
    REQUIRE(layout.count("A") == 1);
    REQUIRE(layout.count("B") == 1);
    const float expectedCenter = 500.0f + 100.0f * 0.5f;
    REQUIRE(near(y(layout, "B") + 30.0f, expectedCenter));
    REQUIRE(near(y(layout, "A"), y(layout, "B")));
}

TEST_CASE("Anchored roots leave room for pending roots below", "[materialx][layout]")
{
    MaterialXGraphLayout::LayoutNode anchored = node("existing");
    anchored.hasPosition = true;
    anchored.position = {1000.0f, 100.0f};
    anchored.height = 80.0f;

    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> layout =
        MaterialXGraphLayout::autoLayout({anchored, node("fresh")}, {});

    REQUIRE(layout.count("existing") == 0);
    REQUIRE(layout.count("fresh") == 1);
    REQUIRE(y(layout, "fresh") > 100.0f + 80.0f + RootGap - 0.01f);
}
