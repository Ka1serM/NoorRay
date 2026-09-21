#include <catch2/catch_test_macros.hpp>

#include <MaterialXCore/Document.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Materials/MaterialX/SlangMaterialGenerator.h"

namespace
{
MaterialX::DocumentPtr standardMaterial()
{
    auto document = MaterialX::createDocument();
    auto surface = document->addNode("standard_surface", "surface", "surfaceshader");
    surface->setNodeDefString("ND_standard_surface_surfaceshader");
    surface->setInputValue("base_color", MaterialX::Color3(0.5f));
    auto material = document->addNode("surfacematerial", "material", "material");
    material->setConnectedNode("surfaceshader", surface);
    return document;
}
}

TEST_CASE("Slang generator supports default MaterialX surface nodes", "[materialx]")
{
    auto document = standardMaterial();
    auto normal = document->addNode("normalmap", "normal", "vector3");
    normal->setNodeDefString("ND_normalmap_float");
    normal->setInputValue("in", MaterialX::Vector3(0.5f, 0.5f, 1.0f));
    auto color = document->addNode("geompropvalue", "color", "color4");
    color->setNodeDefString("ND_geompropvalue_color4");
    color->setInputValue("geomprop", std::string("color"), "string");
    color->setInputValue("default", MaterialX::Color4(1.0f));

    // Keep both stock nodes live in the renderable graph.
    auto surface = document->getNode("surface");
    surface->setConnectedNode("normal", normal);
    surface->setConnectedNode("base_color", color);

    nr::materialx::SlangMaterialGenerator generator;
    const nr::materialx::SlangMaterial material = generator.generate(document);
    REQUIRE_FALSE(material.source.empty());
    REQUIRE(material.source.find("geometry.color") != std::string::npos);
}
