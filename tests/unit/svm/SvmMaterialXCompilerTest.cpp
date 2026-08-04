#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <bit>
#include <cstring>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Materials/SVM/SvmCompiler.h"
#include "Materials/SVM/normal_map.h"
#include "Materials/SVM/SvmTypes.h"

namespace
{
MaterialX::DocumentPtr unlitDocument()
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr material =
        document->addNode("surfacematerial", "material", "material");
    const MaterialX::NodePtr surface =
        document->addNode("surface_unlit", "surface", "surfaceshader");
    material->setConnectedNode("surfaceshader", surface);
    return document;
}

bool contains(const std::vector<std::uint32_t>& words, const nr::svm::NodeType type)
{
    const std::uint32_t encoded = static_cast<std::uint32_t>(type);
    return std::ranges::find(words, encoded) != words.end();
}
}

TEST_CASE("MaterialX unlit terminals lower to SVM closures", "[svm][materialx]")
{
    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(unlitDocument());

    REQUIRE_FALSE(program.bytecode.empty());
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureUniformEdf));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::SurfaceOutput));
}

TEST_CASE("SVM compiler attaches libraries to XML-round-tripped documents",
    "[svm][materialx][libraries]")
{
    const MaterialX::DocumentPtr source = nr::materialx::defaultMaterial();
    REQUIRE(source->getNode("nr_default_disney"));
    REQUIRE(source->getNode("nr_default_disney")->getCategory() == "disney_principled");
    const std::string xml = MaterialX::writeToXmlString(source);
    const MaterialX::DocumentPtr roundTripped = MaterialX::createDocument();
    MaterialX::readFromXmlString(roundTripped, xml);
    REQUIRE_FALSE(roundTripped->getDataLibrary());

    nr::svm::SvmCompiler compiler;
    REQUIRE_FALSE(compiler.compile(roundTripped).bytecode.empty());
    REQUIRE(roundTripped->getDataLibrary());
}

TEST_CASE("Default Principled material lowers through the composite closure",
    "[svm][materialx][principled]")
{
    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(
        nr::materialx::defaultMaterial());

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureOpenPbrSurface));
}

TEST_CASE("MaterialX four-channel values retain alpha through SVM", "[svm][materialx]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr combine =
        document->addNode("combine4", "rgba", "color4");
    combine->setInputValue("in1", 0.2f);
    combine->setInputValue("in2", 0.3f);
    combine->setInputValue("in3", 0.4f);
    combine->setInputValue("in4", 0.5f);
    const MaterialX::NodePtr premult =
        document->addNode("premult", "premult", "color4");
    premult->setConnectedNode("in", combine);
    surface->setConnectedNode("emission_color", premult);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::CombineColor4));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::Premultiply));
}

TEST_CASE("MaterialX vector4 arithmetic uses shared scalar SVM instructions", "[svm][materialx]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr a =
        document->addNode("constant", "a", "color4");
    a->setInputValue("value", MaterialX::Color4(0.1f, 0.2f, 0.3f, 0.4f));
    const MaterialX::NodePtr b =
        document->addNode("constant", "b", "color4");
    b->setInputValue("value", MaterialX::Color4(0.5f, 0.6f, 0.7f, 0.8f));
    const MaterialX::NodePtr add =
        document->addNode("add", "add", "color4");
    add->setConnectedNode("in1", a);
    add->setConnectedNode("in2", b);
    surface->setConnectedNode("emission_color", add);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(std::ranges::count(program.bytecode,
        static_cast<std::uint32_t>(nr::svm::NodeType::Math)) >= 4);
}

TEST_CASE("MaterialX scalar multiplication broadcasts across color3 components",
    "[svm][materialx][arithmetic]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr color = document->addNode("constant", "color", "color3");
    color->setInputValue("value", MaterialX::Color3(0.2f, 0.4f, 0.8f));
    const MaterialX::NodePtr separate =
        document->addNode("separate3", "separate", "color3");
    separate->setConnectedNode("in", color);
    const MaterialX::NodePtr multiply =
        document->addNode("multiply", "multiply", "color3");
    multiply->setConnectedNode("in1", color);
    const MaterialX::InputPtr scalar = multiply->addInput("in2", "float");
    scalar->setConnectedNode(separate);
    scalar->setOutputString("outg");
    surface->setConnectedNode("emission_color", multiply);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    bool foundBroadcast = false;
    constexpr std::size_t mathWords = sizeof(nr::svm::NodeMath) / sizeof(std::uint32_t);
    for (std::size_t i = 0; i + 1 + 3 * mathWords
        <= program.bytecode.size(); ++i) {
        if (program.bytecode[i] != static_cast<std::uint32_t>(nr::svm::NodeType::Math))
            continue;
        nr::svm::NodeMath first{};
        std::memcpy(&first, program.bytecode.data() + i + 1, sizeof(first));
        if (first.mathType != static_cast<std::uint32_t>(nr::svm::MathOp::Multiply))
            continue;
        if (first.resultOffset == nr::svm::InvalidOffset)
            continue;
        const auto* second = reinterpret_cast<const nr::svm::NodeMath*>(
            program.bytecode.data() + i + 2 + mathWords);
        const auto* third = reinterpret_cast<const nr::svm::NodeMath*>(
            program.bytecode.data() + i + 1 + 2 * (mathWords + 1));
        if (second->mathType == static_cast<std::uint32_t>(nr::svm::MathOp::Multiply)
            && third->mathType == static_cast<std::uint32_t>(nr::svm::MathOp::Multiply)
            && second->resultOffset == first.resultOffset + 1
            && third->resultOffset == first.resultOffset + 2
            && second->value2 == first.value2
            && third->value2 == first.value2) {
            foundBroadcast = true;
            break;
        }
    }
    REQUIRE(foundBroadcast);
}

TEST_CASE("MaterialX linked scalar inputs broadcast to a real color3 value",
    "[svm][materialx][conversion]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr scalar = document->addNode(
        "add", "scalar_value", "float");
    scalar->setInputValue("in1", 0.2f);
    scalar->setInputValue("in2", 0.3f);
    // surface_unlit/emission_color is a color3 terminal. The source is a
    // linked float, so this exercises the same scalar-to-color promotion used
    // by color inputs on closure nodes.
    surface->setConnectedNode("emission_color", scalar);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    bool foundBroadcast = false;
    constexpr std::size_t combineWords = sizeof(nr::svm::NodeCombineColor)
        / sizeof(std::uint32_t);
    for (std::size_t i = 0; i + 1 + combineWords <= program.bytecode.size(); ++i) {
        if (program.bytecode[i] != static_cast<std::uint32_t>(
                nr::svm::NodeType::CombineColor))
            continue;
        nr::svm::NodeCombineColor instruction{};
        std::memcpy(&instruction, program.bytecode.data() + i + 1,
            sizeof(instruction));
        if (instruction.x == instruction.y && instruction.y == instruction.z
            && instruction.resultOffset != nr::svm::InvalidOffset) {
            foundBroadcast = true;
            break;
        }
    }
    REQUIRE(foundBroadcast);
}

TEST_CASE("MaterialX compiler ignores unconnected authoring nodes",
    "[svm][materialx][graph]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr spareColor =
        document->addNode("constant", "spare_color", "color3");
    spareColor->setInputValue("value", MaterialX::Color3(0.2f, 0.4f, 0.8f));
    // This deliberately has no NodeDef. An editor may contain an incomplete
    // or newly authored node that has not been connected yet; it must not
    // affect compilation of the reachable material graph.
    document->addNode("authoring_placeholder", "spare_unknown", "color3");

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);
    REQUIRE_FALSE(program.bytecode.empty());
}

TEST_CASE("MaterialX vector2 combine does not overrun its SVM allocation", "[svm][materialx]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr combine =
        document->addNode("combine2", "uv", "vector2");
    combine->setInputValue("in1", 0.25f);
    combine->setInputValue("in2", 0.75f);
    surface->setConnectedNode("emission_color", combine);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::CombineColor2));
    REQUIRE_FALSE(contains(program.bytecode, nr::svm::NodeType::CombineColor));
}

TEST_CASE("MaterialX image color4 keeps alpha contiguous in the SVM stack",
    "[svm][materialx][image]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr image =
        document->addNode("image", "rgba_image", "color4");
    image->setInputValue("file", "test.png");
    const MaterialX::NodePtr premult =
        document->addNode("premult", "rgba_premult", "color4");
    premult->setConnectedNode("in", image);
    surface->setConnectedNode("emission_color", premult);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(
        document, {}, {{"test.png", 0}});
    const auto encoded = static_cast<std::uint32_t>(nr::svm::NodeType::ImageTexture);
    const auto found = std::ranges::find(program.bytecode, encoded);
    REQUIRE(found != program.bytecode.end());
    const std::size_t offset = static_cast<std::size_t>(
        std::distance(program.bytecode.begin(), found));
    REQUIRE(offset + 1 + sizeof(nr::svm::NodeImageTexture) / sizeof(std::uint32_t)
        <= program.bytecode.size());
    const auto& instruction = *reinterpret_cast<const nr::svm::NodeImageTexture*>(
        program.bytecode.data() + offset + 1);
    REQUIRE(instruction.resultAlphaOffset == instruction.resultColorOffset + 3);
}

TEST_CASE("Authoring albedo textures stay connected through SVM compilation",
    "[svm][materialx][image][authoring]")
{
    MaterialAuthoring material;
    material.albedo = {0.1f, 0.2f, 0.3f};
    material.albedoIndex = 7;
    const MaterialX::DocumentPtr document = nr::materialx::documentFromAuthoring(
        material, [](const int index) {
            return index == 7 ? std::string("grass004_diff") : std::string{};
        });

    const MaterialX::NodePtr principled = document->getNode("nr_synthetic_disney");
    REQUIRE(principled);
    const MaterialX::NodePtr image =
        principled->getInput("baseColor")->getConnectedNode();
    REQUIRE(image);
    REQUIRE(image->getCategory() == "image");
    REQUIRE(image->getInput("file")->getValueString() == "grass004_diff");
    REQUIRE(image->getInput("file")->getAttribute("colorspace") == "srgb_texture");

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(
        document, {}, {{"grass004_diff", 42}});
    REQUIRE(program.textureIndices == std::vector<std::uint32_t>{42});
}

TEST_CASE("MaterialX matrix and space transforms lower to typed SVM instructions",
    "[svm][materialx][transform]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr matrix =
        document->addNode("creatematrix", "matrix", "matrix44");
    matrix->setInputValue("in1", MaterialX::Vector3(1.0f, 0.0f, 0.0f));
    matrix->setInputValue("in2", MaterialX::Vector3(0.0f, 1.0f, 0.0f));
    matrix->setInputValue("in3", MaterialX::Vector3(0.0f, 0.0f, 1.0f));
    matrix->setInputValue("in4", MaterialX::Vector3(1.0f, 2.0f, 3.0f));
    const MaterialX::NodePtr transform =
        document->addNode("transformmatrix", "transform", "vector3");
    transform->setInputValue("in", MaterialX::Vector3(0.25f, 0.5f, 0.75f));
    transform->setConnectedNode("mat", matrix);
    surface->setConnectedNode("emission_color", transform);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::MatrixCompose));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::TransformMatrix));
}

TEST_CASE("MaterialX color4 adjustments retain their channel contract", "[svm][materialx]")
{
    nr::svm::SvmCompiler compiler;
    const MaterialX::DocumentPtr colorDocument = unlitDocument();
    const MaterialX::NodePtr colorSurface = colorDocument->getNode("surface");
    const MaterialX::NodePtr contrast =
        colorDocument->addNode("contrast", "color4_contrast", "color4");
    contrast->setInputValue("in", MaterialX::Color4(0.1f, 0.2f, 0.3f, 1.0f));
    colorSurface->setConnectedNode("emission_color", contrast);
    const nr::svm::CompiledSvmProgram program = compiler.compile(colorDocument);
    REQUIRE(std::ranges::count(program.bytecode,
        static_cast<std::uint32_t>(nr::svm::NodeType::Contrast)) == 4);
}

TEST_CASE("MaterialX geometry attributes lower through standard geompropvalue",
    "[svm][materialx][geometry]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr attribute =
        document->addNode("geompropvalue", "attribute", "color3");
    attribute->setInputValue("geomprop", "color");
    attribute->setInputValue("default", MaterialX::Color3(0.0f));
    surface->setConnectedNode("emission_color", attribute);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::TexCoord));
}

TEST_CASE("MaterialX matrix arithmetic lowers to typed matrix SVM opcodes",
    "[svm][materialx][matrix]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::Matrix44 identity(1.0f);
    const MaterialX::NodePtr add = document->addNode("add", "add_matrix", "matrix44");
    add->setInputValue("in1", identity);
    add->setInputValue("in2", identity);
    const MaterialX::NodePtr addScalar =
        document->addNode("add", "add_matrix_scalar", "matrix44");
    addScalar->setConnectedNode("in1", add);
    addScalar->setInputValue("in2", 1.0f);
    const MaterialX::NodePtr multiply =
        document->addNode("multiply", "multiply_matrix", "matrix44");
    multiply->setConnectedNode("in1", addScalar);
    multiply->setInputValue("in2", identity);
    const MaterialX::NodePtr divide =
        document->addNode("divide", "divide_matrix", "matrix44");
    divide->setConnectedNode("in1", multiply);
    divide->setInputValue("in2", identity);
    const MaterialX::NodePtr transpose =
        document->addNode("transpose", "transpose_matrix", "matrix44");
    transpose->setConnectedNode("in", divide);
    const MaterialX::NodePtr inverse =
        document->addNode("invertmatrix", "inverse_matrix", "matrix44");
    inverse->setConnectedNode("in", transpose);
    const MaterialX::NodePtr select =
        document->addNode("ifgreater", "select_matrix", "matrix44");
    select->setInputValue("value1", 1.0f);
    select->setInputValue("value2", 0.0f);
    select->setConnectedNode("in1", inverse);
    select->setInputValue("in2", identity);
    const MaterialX::NodePtr determinant =
        document->addNode("determinant", "determinant_matrix", "float");
    determinant->setConnectedNode("in", select);
    surface->setConnectedNode("emission_color", determinant);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::MatrixBinary));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::MatrixUnary));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::MatrixDeterminant));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::MatrixSelect));
}

TEST_CASE("MaterialX normal map preserves raw tangent-frame semantics", "[svm][materialx][normal]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 tangent(2.0f, 0.0f, 0.0f);
    const glm::vec3 bitangent(0.0f, 3.0f, 0.0f);
    const glm::vec3 encoded(0.75f, 0.5f, 1.0f);
    const glm::vec3 unpacked = encoded * 2.0f - glm::vec3(1.0f);
    const glm::vec3 expected = glm::normalize(
        tangent * unpacked.x + bitangent * unpacked.y + normal * unpacked.z);
    const glm::vec3 actual = nr::svm::detail::materialXNormalMap(
        encoded, glm::vec2(1.0f), normal, tangent, bitangent);

    CHECK(actual.x == Catch::Approx(expected.x).margin(1.0e-6f));
    CHECK(actual.y == Catch::Approx(expected.y).margin(1.0e-6f));
    CHECK(actual.z == Catch::Approx(expected.z).margin(1.0e-6f));
}

TEST_CASE("MaterialX standard geometry and camera direction lower to SVM",
    "[svm][materialx][geometry]")
{
    nr::svm::SvmCompiler compiler;

    const MaterialX::DocumentPtr objectDocument = unlitDocument();
    const MaterialX::NodePtr objectSurface = objectDocument->getNode("surface");
    const MaterialX::NodePtr objectInfo = objectDocument->addNode(
        "position", "position", "vector3");
    objectInfo->setInputValue("space", "object");
    objectInfo->setInputValue("default", MaterialX::Vector3(0.0f));
    objectSurface->setConnectedNode("emission_color", objectInfo);
    const nr::svm::CompiledSvmProgram objectProgram = compiler.compile(objectDocument);
    REQUIRE(contains(objectProgram.bytecode, nr::svm::NodeType::TexCoord));

    const MaterialX::DocumentPtr cameraDocument = unlitDocument();
    const MaterialX::NodePtr cameraSurface = cameraDocument->getNode("surface");
    const MaterialX::NodePtr viewDirection = cameraDocument->addNode(
        "viewdirection", "view_direction", "vector3");
    cameraSurface->setConnectedNode("emission_color", viewDirection);
    const nr::svm::CompiledSvmProgram cameraProgram = compiler.compile(cameraDocument);
    REQUIRE(contains(cameraProgram.bytecode, nr::svm::NodeType::TexCoord));
}

TEST_CASE("MaterialX surface geometry normal interfaces lower to SVM",
    "[svm][materialx][geometry][normal]")
{
    nr::svm::SvmCompiler compiler;
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr normal = document->addNode(
        "geometry_normal", "geometry_normal", "vector3");
    surface->setConnectedNode("emission_color", normal);

    const nr::svm::CompiledSvmProgram program = compiler.compile(document);
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::TexCoord));
}

TEST_CASE("MaterialX vector transforms support camera, object, and world spaces",
    "[svm][materialx][transform]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr transform = document->addNode(
        "transformvector", "camera_to_object", "vector3");
    transform->setInputValue("in", MaterialX::Vector3(0.25f, 0.5f, 0.75f));
    transform->setInputValue("fromspace", "camera");
    transform->setInputValue("tospace", "object");
    surface->setConnectedNode("emission_color", transform);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::Transform));
}

TEST_CASE("MaterialX white noise lowers to scalar and color SVM outputs",
    "[svm][materialx][texture]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr noise = document->addNode(
        "noise3d", "white_noise_equivalent", "float");
    noise->setInputValue("position", MaterialX::Vector3(1.0f, 2.0f, 3.0f));
    noise->setInputValue("amplitude", 1.0f);
    noise->setInputValue("pivot", 0.5f);
    surface->setConnectedNode("emission_color", noise);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::NoiseTexture));
}

TEST_CASE("MaterialX standard random nodes flatten into supported SVM noise",
    "[svm][materialx][texture]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr random = document->addNode(
        "randomfloat", "random", "float");
    random->setInputValue("in", 2.0f);
    random->setInputValue("min", 0.0f);
    random->setInputValue("max", 1.0f);
    random->setInputValue("seed", 0);
    surface->setConnectedNode("emission_color", random);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::CellNoiseTexture));
}

TEST_CASE("MaterialX metallic BSDF export target uses the standard conductor closure",
    "[svm][materialx][closure]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr material = document->getNode("material");
    const MaterialX::NodePtr surface = document->addNode(
        "surface", "surface_with_bsdf", "surfaceshader");
    material->setConnectedNode("surfaceshader", surface);
    const MaterialX::NodePtr metallic = document->addNode(
        "conductor_bsdf", "metallic_export_target", "BSDF");
    metallic->setInputValue("ior", MaterialX::Color3(0.2f, 0.9f, 1.4f));
    metallic->setInputValue("extinction", MaterialX::Color3(3.0f, 2.0f, 1.0f));
    metallic->setInputValue("roughness", 0.25f);
    surface->setConnectedNode("bsdf", metallic);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureConductorBsdf));
}

TEST_CASE("MaterialX exported diffuse graph compiles without graph flattening",
    "[svm][materialx][closure]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr diffuse = document->addNode(
        "oren_nayar_diffuse_bsdf", "n0", "BSDF");
    diffuse->setInputValue("color", MaterialX::Color3(0.08f, 0.52f, 0.82f));
    diffuse->setInputValue("roughness", 1.0f);
    diffuse->setInputValue("normal", MaterialX::Vector3(0.0f));
    const MaterialX::NodePtr surface = document->addNode(
        "surface", "n1", "surfaceshader");
    surface->setConnectedNode("bsdf", diffuse);
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "n2", "material");
    material->setConnectedNode("surfaceshader", surface);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureDiffuseBsdf));
}

TEST_CASE("MaterialX F82 metallic reflectivity lowers to a conductor closure",
    "[svm][materialx][closure]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr material = document->getNode("material");
    const MaterialX::NodePtr surface = document->addNode(
        "surface", "surface_with_f82_metal", "surfaceshader");
    material->setConnectedNode("surfaceshader", surface);
    const MaterialX::NodePtr metallic = document->addNode(
        "generalized_schlick_bsdf", "f82_metallic", "BSDF");
    metallic->setInputValue("color0", MaterialX::Color3(0.6f, 0.5f, 0.4f));
    metallic->setInputValue("color90", MaterialX::Color3(0.8f, 0.7f, 0.6f));
    metallic->setInputValue("roughness", MaterialX::Vector2(0.5f, 0.5f));
    metallic->setInputValue("scatter_mode", "R");
    surface->setConnectedNode("bsdf", metallic);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureConductorBsdf));
}

TEST_CASE("MaterialX unresolved node definitions fail before graph flattening",
    "[svm][materialx][diagnostics]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");
    const MaterialX::NodePtr unknown = document->addNode(
        "not_a_materialx_node", "unknown", "float");
    surface->setConnectedNode("emission_color", unknown);

    nr::svm::SvmCompiler compiler;
    REQUIRE_THROWS_AS(compiler.compile(document), nr::svm::SvmCompileError);
}

TEST_CASE("MaterialX roughness helper nodes are compiled natively",
    "[svm][materialx][roughness]")
{
    const MaterialX::DocumentPtr document = unlitDocument();
    const MaterialX::NodePtr surface = document->getNode("surface");

    const MaterialX::NodePtr anisotropic = document->addNode(
        "roughness_anisotropy", "specular_roughness", "vector2");
    anisotropic->setInputValue("roughness", 0.5f);
    anisotropic->setInputValue("anisotropy", 0.25f);
    surface->setConnectedNode("emission_color", anisotropic);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE_FALSE(program.bytecode.empty());
}

TEST_CASE("MaterialX default material ignores imported implementation graphs",
    "[svm][materialx][default]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr shader = document->addNode(
        "open_pbr_surface", "nr_default_material", "surfaceshader");
    shader->setInputValue("base_color", MaterialX::Color3(0.8f));
    shader->setInputValue("specular_roughness", 0.5f);
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "nr_default_material_material", "material");
    material->setConnectedNode("surfaceshader", shader);

    MaterialX::DocumentPtr libraries = MaterialX::createDocument();
    const MaterialX::FilePath path(NR_MATERIALX_STDLIB_DIR);
    MaterialX::FileSearchPath searchPath;
    searchPath.append(path);
    searchPath.append(path.getParentPath());
    REQUIRE_FALSE(MaterialX::loadLibraries(
        {path.getBaseName()}, searchPath, libraries).empty());
    document->importLibrary(libraries);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE_FALSE(program.bytecode.empty());
}

TEST_CASE("MaterialX Disney Principled controls affect compiled closure graph",
    "[svm][materialx][disney]")
{
    const auto makeDocument = [](const float metallic, const float specular,
                                  const float clearcoat, const float specTrans,
                                  const float roughness) {
        const MaterialX::DocumentPtr document = MaterialX::createDocument();
        MaterialX::DocumentPtr libraries = MaterialX::createDocument();
        const MaterialX::FilePath path(NR_MATERIALX_STDLIB_DIR);
        MaterialX::FileSearchPath searchPath;
        searchPath.append(path);
        searchPath.append(path.getParentPath());
        REQUIRE_FALSE(MaterialX::loadLibraries(
            {path.getBaseName()}, searchPath, libraries).empty());
        document->setDataLibrary(libraries);

        const MaterialX::NodePtr shader = document->addNode(
            "disney_principled", "disney", "surfaceshader");
        shader->setInputValue("baseColor", MaterialX::Color3(0.8f, 0.2f, 0.05f));
        shader->setInputValue("metallic", metallic);
        shader->setInputValue("specular", specular);
        shader->setInputValue("clearcoat", clearcoat);
        shader->setInputValue("specTrans", specTrans);
        shader->setInputValue("roughness", roughness);
        const MaterialX::NodePtr material = document->addNode(
            "surfacematerial", "material", "material");
        material->setConnectedNode("surfaceshader", shader);
        return document;
    };

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram dielectric = compiler.compile(
        makeDocument(0.0f, 0.5f, 0.0f, 0.0f, 0.5f));
    const nr::svm::CompiledSvmProgram metal = compiler.compile(
        makeDocument(1.0f, 0.5f, 0.0f, 0.0f, 0.5f));
    const nr::svm::CompiledSvmProgram coated = compiler.compile(
        makeDocument(0.0f, 0.5f, 1.0f, 0.0f, 0.5f));
    const nr::svm::CompiledSvmProgram metalRough = compiler.compile(
        makeDocument(1.0f, 0.5f, 0.0f, 0.0f, 0.1f));
    const nr::svm::CompiledSvmProgram metalSmooth = compiler.compile(
        makeDocument(1.0f, 0.5f, 0.0f, 0.0f, 0.9f));
    const nr::svm::CompiledSvmProgram transmissionRough = compiler.compile(
        makeDocument(0.0f, 0.5f, 0.0f, 1.0f, 0.1f));
    const nr::svm::CompiledSvmProgram transmissionSmooth = compiler.compile(
        makeDocument(0.0f, 0.5f, 0.0f, 1.0f, 0.9f));

    REQUIRE_FALSE(dielectric.bytecode.empty());
    REQUIRE_FALSE(metal.bytecode.empty());
    REQUIRE_FALSE(coated.bytecode.empty());
    REQUIRE(contains(dielectric.bytecode, nr::svm::NodeType::ClosureOpenPbrSurface));
    REQUIRE(contains(metal.bytecode, nr::svm::NodeType::ClosureOpenPbrSurface));
    REQUIRE(dielectric.bytecode != metal.bytecode);
    REQUIRE(dielectric.bytecode != coated.bytecode);
    REQUIRE(metalRough.bytecode != metalSmooth.bytecode);
    REQUIRE(transmissionRough.bytecode != transmissionSmooth.bytecode);

}

TEST_CASE("MaterialX surfaceshader mix emits both selectable branches",
    "[svm][materialx][mix]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    MaterialX::DocumentPtr libraries = MaterialX::createDocument();
    const MaterialX::FilePath path(NR_MATERIALX_STDLIB_DIR);
    MaterialX::FileSearchPath searchPath;
    searchPath.append(path);
    searchPath.append(path.getParentPath());
    REQUIRE_FALSE(MaterialX::loadLibraries(
        {path.getBaseName()}, searchPath, libraries).empty());
    document->setDataLibrary(libraries);

    const MaterialX::NodePtr disney = document->addNode(
        "open_pbr_surface", "disney_branch", "surfaceshader");
    disney->setInputValue("base_color", MaterialX::Color3(0.8f));
    const MaterialX::NodePtr glass = document->addNode(
        "dielectric_bsdf", "glass_bsdf", "BSDF");
    glass->setInputValue("scatter_mode", "RT");
    const MaterialX::NodePtr glassSurface = document->addNode(
        "surface", "glass_branch", "surfaceshader");
    glassSurface->setConnectedNode("bsdf", glass);
    const MaterialX::NodePtr factor = document->addNode(
        "constant", "mix_factor", "float");
    factor->setInputValue("value", 0.0f);
    const MaterialX::NodePtr mix = document->addNode(
        "mix", "surface_mix", "surfaceshader");
    mix->setConnectedNode("bg", disney);
    mix->setConnectedNode("fg", glassSurface);
    mix->setConnectedNode("mix", factor);
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "material", "material");
    material->setConnectedNode("surfaceshader", mix);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);

    REQUIRE(contains(program.bytecode, nr::svm::NodeType::JumpIfOne));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::JumpIfZero));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureOpenPbrSurface));
    REQUIRE(contains(program.bytecode, nr::svm::NodeType::ClosureDielectricBsdf));
}

TEST_CASE("MaterialX dielectric scatter mode preserves R, T, and RT",
    "[svm][materialx][dielectric]")
{
    const auto compileMode = [](const char* mode) {
        const MaterialX::DocumentPtr document = MaterialX::createDocument();
        const MaterialX::NodePtr material = document->addNode(
            "surfacematerial", "material", "material");
        const MaterialX::NodePtr surface = document->addNode(
            "surface", "surface", "surfaceshader");
        const MaterialX::NodePtr dielectric = document->addNode(
            "dielectric_bsdf", "dielectric", "BSDF");
        dielectric->setInputValue("scatter_mode", mode);
        surface->setConnectedNode("bsdf", dielectric);
        material->setConnectedNode("surfaceshader", surface);

        nr::svm::SvmCompiler compiler;
        return compiler.compile(document);
    };
    const auto modePayload = [](const nr::svm::CompiledSvmProgram& program) {
        const std::uint32_t opcode = static_cast<std::uint32_t>(
            nr::svm::NodeType::ClosureDielectricBsdf);
        for (std::size_t i = 0; i < program.bytecode.size(); ++i) {
            if (program.bytecode[i] != opcode)
                continue;
            REQUIRE(i + 1 + sizeof(nr::svm::NodeClosureDielectricBsdf)
                / sizeof(std::uint32_t) <= program.bytecode.size());
            nr::svm::NodeClosureDielectricBsdf payload{};
            std::memcpy(&payload, program.bytecode.data() + i + 1, sizeof(payload));
            return std::bit_cast<float>(payload.transmission);
        }
        FAIL("dielectric closure was not emitted");
        return -1.0f;
    };

    CHECK(modePayload(compileMode("R")) == 0.0f);
    CHECK(modePayload(compileMode("T")) == 1.0f);
    CHECK(modePayload(compileMode("RT")) == 2.0f);
}

TEST_CASE("MaterialX Sellmeier IOR reaches dielectric closure as spectral payload",
    "[svm][materialx][sellmeier]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "material", "material");
    const MaterialX::NodePtr surface = document->addNode(
        "surface", "surface", "surfaceshader");
    const MaterialX::NodePtr dielectric = document->addNode(
        "dielectric_bsdf", "dielectric", "BSDF");
    const MaterialX::NodePtr sellmeier = document->addNode(
        "noorray_sellmeier_ior", "sellmeier", "float");
    sellmeier->setInputValue("b1", 1.1f);
    sellmeier->setInputValue("b2", 0.2f);
    sellmeier->setInputValue("b3", 0.3f);
    sellmeier->setInputValue("c1", 0.01f);
    sellmeier->setInputValue("c2", 0.02f);
    sellmeier->setInputValue("c3", 100.0f);
    dielectric->setConnectedNode("ior", sellmeier);
    surface->setConnectedNode("bsdf", dielectric);
    material->setConnectedNode("surfaceshader", surface);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);
    const std::uint32_t opcode = static_cast<std::uint32_t>(
        nr::svm::NodeType::ClosureDielectricBsdf);
    for (std::size_t i = 0; i < program.bytecode.size(); ++i) {
        if (program.bytecode[i] != opcode)
            continue;
        REQUIRE(i + 1 + sizeof(nr::svm::NodeClosureDielectricBsdf)
            / sizeof(std::uint32_t) <= program.bytecode.size());
        nr::svm::NodeClosureDielectricBsdf payload{};
        std::memcpy(&payload, program.bytecode.data() + i + 1, sizeof(payload));
        CHECK(payload.sellmeier.enabled == 1);
        CHECK(std::bit_cast<float>(payload.sellmeier.b1) == 1.1f);
        CHECK(std::bit_cast<float>(payload.sellmeier.c3) == 100.0f);
        return;
    }
    FAIL("Sellmeier dielectric closure was not emitted");
}

TEST_CASE("MaterialX Sellmeier IOR reaches open PBR specular closure",
    "[svm][materialx][sellmeier][openpbr]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "material", "material");
    const MaterialX::NodePtr surface = document->addNode(
        "open_pbr_surface", "surface", "surfaceshader");
    const MaterialX::NodePtr sellmeier = document->addNode(
        "noorray_sellmeier_ior", "sellmeier", "float");
    sellmeier->setInputValue("c3", 100.0f);
    surface->setConnectedNode("specular_ior", sellmeier);
    material->setConnectedNode("surfaceshader", surface);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);
    const std::uint32_t opcode = static_cast<std::uint32_t>(
        nr::svm::NodeType::ClosureOpenPbrSurface);
    for (std::size_t i = 0; i < program.bytecode.size(); ++i) {
        if (program.bytecode[i] != opcode)
            continue;
        REQUIRE(i + 1 + sizeof(nr::svm::NodeClosureOpenPbrSurface)
            / sizeof(std::uint32_t) <= program.bytecode.size());
        nr::svm::NodeClosureOpenPbrSurface payload{};
        std::memcpy(&payload, program.bytecode.data() + i + 1, sizeof(payload));
        CHECK(payload.specularSellmeier.enabled == 1);
        CHECK(std::bit_cast<float>(payload.specularSellmeier.c3) == 100.0f);
        return;
    }
    FAIL("Sellmeier open PBR closure was not emitted");
}

TEST_CASE("MaterialX Sellmeier IOR reaches Disney Principled specular closure",
    "[svm][materialx][sellmeier][disney]")
{
    const MaterialX::DocumentPtr document = MaterialX::createDocument();
    const MaterialX::NodePtr material = document->addNode(
        "surfacematerial", "material", "material");
    const MaterialX::NodePtr disney = document->addNode(
        "disney_principled", "disney", "surfaceshader");
    const MaterialX::NodePtr sellmeier = document->addNode(
        "noorray_sellmeier_ior", "sellmeier", "float");
    sellmeier->setInputValue("c3", 100.0f);
    disney->setConnectedNode("ior", sellmeier);
    material->setConnectedNode("surfaceshader", disney);

    nr::svm::SvmCompiler compiler;
    const nr::svm::CompiledSvmProgram program = compiler.compile(document);
    const std::uint32_t opcode = static_cast<std::uint32_t>(
        nr::svm::NodeType::ClosureOpenPbrSurface);
    for (std::size_t i = 0; i < program.bytecode.size(); ++i) {
        if (program.bytecode[i] != opcode)
            continue;
        REQUIRE(i + 1 + sizeof(nr::svm::NodeClosureOpenPbrSurface)
            / sizeof(std::uint32_t) <= program.bytecode.size());
        nr::svm::NodeClosureOpenPbrSurface payload{};
        std::memcpy(&payload, program.bytecode.data() + i + 1, sizeof(payload));
        CHECK(payload.specularSellmeier.enabled == 1);
        CHECK(std::bit_cast<float>(payload.specularSellmeier.c3) == 100.0f);
        return;
    }
    FAIL("Sellmeier Disney Principled closure was not emitted");
}
