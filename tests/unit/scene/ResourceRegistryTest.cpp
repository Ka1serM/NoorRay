#include "Scene/ResourceRegistry.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {
struct TestResource
{
    int value{};
    void releaseResources() { value = -1; }
};

using TestRegistry =
    nr::ResourceRegistry<TestResource, std::vector<TestResource>>;
}

TEST_CASE("resource registry reserves import batches and revisions track membership",
    "[scene][registry]")
{
    TestRegistry registry;
    registry.reserveAdditional(8);
    REQUIRE(registry.storage().capacity() >= 8);
    const size_t capacity = registry.storage().capacity();

    std::vector<TestRegistry::Handle> handles;
    for (int value = 0; value < 8; ++value)
        handles.push_back(registry.emplace(TestResource{value}));

    CHECK(registry.storage().capacity() == capacity);
    CHECK(registry.revision() == 8);

    registry.clear();
    CHECK(registry.revision() == 9);
}

TEST_CASE("delayed release events cannot clear a reused slot",
    "[scene][registry]")
{
    TestRegistry registry;
    const TestRegistry::Handle first =
        registry.emplace(TestResource{1});
    REQUIRE(registry.release(first));

    // Reuse the same physical slot before the sidecar owner consumes the old
    // event. The generation filter must suppress that stale event.
    const TestRegistry::Handle replacement =
        registry.emplace(TestResource{2});
    REQUIRE(replacement.index() == first.index());
    REQUIRE(replacement != first);

    size_t consumed = 0;
    registry.consumeReleasedSlots(
        [&](uint32_t) { ++consumed; });
    CHECK(consumed == 0);

    REQUIRE(registry.release(replacement));
    registry.consumeReleasedSlots(
        [&](uint32_t slot) {
            CHECK(slot == replacement.index());
            ++consumed;
        });
    CHECK(consumed == 1);
}
