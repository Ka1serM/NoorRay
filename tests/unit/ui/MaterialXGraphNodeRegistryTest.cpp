#include <catch2/catch_test_macros.hpp>

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

// Node classes register themselves from their own files during static
// initialisation. Nothing references those files, so a registration that failed
// to run would not break the build -- every node would just quietly fall back
// to the generic class. These tests are what catches that.

TEST_CASE("Node classes register themselves for their categories", "[materialx]")
{
    const MaterialXGraphNodeRegistry& registry = MaterialXGraphNodeRegistry::instance();

    SECTION("image categories") {
        REQUIRE(registry.find("image", "color3") != nullptr);
        REQUIRE(registry.find("tiledimage", "color3") != nullptr);
        REQUIRE(registry.find("triplanarprojection", "color3") != nullptr);
    }

    SECTION("constant") {
        REQUIRE(registry.find("constant", "color3") != nullptr);
    }

    SECTION("maths categories") {
        REQUIRE(registry.find("add", "float") != nullptr);
        REQUIRE(registry.find("multiply", "vector3") != nullptr);
        REQUIRE(registry.find("crossproduct", "vector3") != nullptr);
    }
}

TEST_CASE("Shading roots are registered by output type", "[materialx]")
{
    const MaterialXGraphNodeRegistry& registry = MaterialXGraphNodeRegistry::instance();

    // The category of a shader varies per definition, so these match on what
    // the node produces instead.
    REQUIRE(registry.find("open_pbr_surface", "surfaceshader") != nullptr);
    REQUIRE(registry.find("standard_surface", "surfaceshader") != nullptr);
    REQUIRE(registry.find("surfacematerial", "material") != nullptr);
    REQUIRE(registry.find("displacement", "displacementshader") != nullptr);
}

TEST_CASE("Unregistered nodes fall back to the generic class", "[materialx]")
{
    const MaterialXGraphNodeRegistry& registry = MaterialXGraphNodeRegistry::instance();

    REQUIRE(registry.find("noise2d", "float") == nullptr);
    REQUIRE(registry.find("", "") == nullptr);
}

TEST_CASE("Category registration wins over output type", "[materialx]")
{
    const MaterialXGraphNodeRegistry& registry = MaterialXGraphNodeRegistry::instance();

    // "multiply" producing a surfaceshader would match both tables; the more
    // specific category entry is the one that must be used.
    const MaterialXGraphNodeCreator* byCategory = registry.find("multiply", "surfaceshader");
    const MaterialXGraphNodeCreator* byOutputType = registry.find("unknown", "surfaceshader");
    REQUIRE(byCategory != nullptr);
    REQUIRE(byOutputType != nullptr);
    REQUIRE(byCategory != byOutputType);
}
