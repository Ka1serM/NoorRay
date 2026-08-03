#include <catch2/catch_test_macros.hpp>

#include "UI/MaterialXNodeCatalog.h"

#include <algorithm>

namespace
{
const MaterialXNodeType* find(const std::vector<const MaterialXNodeType*>& types,
    const std::string& category, const std::string& outputType)
{
    const auto found = std::ranges::find_if(types,
        [&](const MaterialXNodeType* type) {
            return type->category == category && type->outputType == outputType;
        });
    return found == types.end() ? nullptr : *found;
}
}

TEST_CASE("Catalog loads the MaterialX standard libraries", "[materialx]")
{
    const MaterialXNodeCatalog& catalog = MaterialXNodeCatalog::instance();

    REQUIRE(catalog.loadError().empty());
    // The vendored MaterialX libraries currently expose roughly 800 typed
    // definitions. A much smaller count means a library family was omitted.
    REQUIRE(catalog.types().size() >= 800);
    REQUIRE(catalog.groups().size() > 5);
    REQUIRE(find(catalog.inGroup("pbr"), "disney_principled", "surfaceshader") != nullptr);
    REQUIRE(find(catalog.inGroup("spectral"), "noorray_sellmeier_ior", "float") != nullptr);
}

TEST_CASE("Every catalog entry is instantiable", "[materialx]")
{
    const MaterialXNodeCatalog& catalog = MaterialXNodeCatalog::instance();

    for (const MaterialXNodeType& type : catalog.types()) {
        INFO("node type " << type.nodeDefName);
        REQUIRE_FALSE(type.category.empty());
        REQUIRE_FALSE(type.group.empty());
        REQUIRE_FALSE(type.label.empty());
        // The add menu resolves the definition to copy its declared inputs, so
        // an entry whose definition cannot be found would add an empty node.
        REQUIRE(catalog.findNodeDef(type) != nullptr);
    }
}

TEST_CASE("Groups cover every entry exactly once", "[materialx]")
{
    const MaterialXNodeCatalog& catalog = MaterialXNodeCatalog::instance();

    size_t grouped = 0;
    for (const std::string& group : catalog.groups())
        grouped += catalog.inGroup(group).size();
    REQUIRE(grouped == catalog.types().size());
}

TEST_CASE("Search matches on category, type and group", "[materialx]")
{
    const MaterialXNodeCatalog& catalog = MaterialXNodeCatalog::instance();

    SECTION("an empty query lists everything") {
        REQUIRE(catalog.search("").size() == catalog.types().size());
    }

    SECTION("a category name finds its definitions") {
        const auto matches = catalog.search("image");
        REQUIRE_FALSE(matches.empty());
        REQUIRE(find(matches, "image", "color3") != nullptr);
    }

    SECTION("NoorRay spectral extensions are searchable") {
        REQUIRE(find(catalog.search("sellmeier"),
            "noorray_sellmeier_ior", "float") != nullptr);
    }

    SECTION("value aliases find MaterialX constant nodes") {
        const auto values = catalog.search("value");
        REQUIRE(find(values, "constant", "float") != nullptr);
        REQUIRE(find(values, "constant", "color3") != nullptr);
        REQUIRE(find(values, "constant", "vector3") != nullptr);
    }

    SECTION("plain type aliases find the corresponding constants") {
        REQUIRE(find(catalog.search("scalar"), "constant", "float") != nullptr);
        REQUIRE(find(catalog.search("color"), "constant", "color3") != nullptr);
        REQUIRE(find(catalog.search("vector"), "constant", "vector3") != nullptr);
    }

    SECTION("terms are combined, not alternated") {
        const auto matches = catalog.search("add color3");
        REQUIRE(find(matches, "add", "color3") != nullptr);
        REQUIRE(find(matches, "add", "float") == nullptr);
    }

    SECTION("matching ignores case") {
        REQUIRE(catalog.search("MULTIPLY").size() == catalog.search("multiply").size());
    }

    SECTION("a query that matches nothing returns nothing") {
        REQUIRE(catalog.search("nosuchnodetypeanywhere").empty());
    }
}
