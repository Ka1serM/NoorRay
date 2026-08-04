#include "Backend/OptiX/LightTreeBuilder.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("light tree is balanced, flat, and parent-addressable", "[light][tree]")
{
    std::vector<DirectLightCandidate> candidates(5);
    for (uint32_t i = 0; i < 5u; ++i)
    {
        candidates[i].type = DirectLightType::Point;
        candidates[i].position = glm::vec3(
            static_cast<float>(i) * 3.0f, static_cast<float>(i & 1u), 0.0f);
        candidates[i].selectionWeight = static_cast<float>(i + 1u);
    }
    candidates[4].selectionWeight = 0.0f;

    std::vector<LightTreeNode> nodes;
    nr::light_tree::build(candidates, nodes);

    REQUIRE(nodes.size() == 2u * 4u - 1u);
    CHECK(nodes.front().selectionWeight == 10.0f);
    CHECK(candidates[4].lightTreeLeaf == InvalidIndex);
    for (uint32_t i = 0; i < 4u; ++i)
    {
        const uint32_t leaf = candidates[i].lightTreeLeaf;
        REQUIRE(leaf < nodes.size());
        CHECK((nodes[leaf].flags & LightTreeLeaf) != 0u);
        CHECK(nodes[leaf].childOrLightIndex == i);
    }

    for (uint32_t i = 0; i < nodes.size(); ++i)
    {
        const LightTreeNode& node = nodes[i];
        if ((node.flags & LightTreeLeaf) != 0u)
            continue;
        REQUIRE(i + 1u < nodes.size());
        REQUIRE(node.childOrLightIndex < nodes.size());
        CHECK(nodes[i + 1u].parent == i);
        CHECK(nodes[node.childOrLightIndex].parent == i);
    }
}
