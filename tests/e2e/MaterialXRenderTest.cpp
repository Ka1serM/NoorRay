#include "RenderTestFixture.h"

#include "IO/BitmapReader.h"

#include <catch2/catch_test_macros.hpp>

#include <glm/vec3.hpp>

class MaterialXRenderTest : public RenderTestFixture {};

namespace
{

glm::vec3 averageColor(const Bitmap& bitmap)
{
    glm::vec3 sum{};
    for (const glm::vec4& pixel : bitmap.pixels())
        sum += glm::vec3(pixel);
    return sum / static_cast<float>(bitmap.pixels().size());
}

}

TEST_CASE_METHOD(MaterialXRenderTest, "NoorRay renders a MaterialX material to EXR", "[e2e][materialx]")
{
    const std::string output = render("materialx_noise_sphere.nrscene", 2, "materialx_noise_sphere.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 256);
    CHECK(bitmap.height() == 256);
    CHECK(bitmap.pixels().size() == 256 * 256);
}

TEST_CASE_METHOD(MaterialXRenderTest, "NoorRay renders a MaterialX image texture to EXR", "[e2e][materialx]")
{
    const std::string output = render("materialx_texture_sphere.nrscene", 2, "materialx_texture_sphere.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 256);
    CHECK(bitmap.height() == 256);
    CHECK(bitmap.pixels().size() == 256 * 256);
}

TEST_CASE_METHOD(MaterialXRenderTest,
    "MaterialX geomprop vertex color compiles and renders",
    "[e2e][materialx]")
{
    const std::string output =
        render("materialx_vertex_color.nrscene", 2, "materialx_vertex_color.exr");
    const Bitmap bitmap = BitmapReader::read(output);
    CHECK(bitmap.width() == 64);
    CHECK(bitmap.height() == 64);
}

// A compiled MaterialX base_color must affect the rendered pixels.
TEST_CASE_METHOD(MaterialXRenderTest, "MaterialX base_color actually reaches the rendered pixels", "[e2e][materialx]")
{
    const std::string redOutput = render("materialx_flat_red.nrscene", 16, "materialx_flat_red.exr");
    const std::string greenOutput = render("materialx_flat_green.nrscene", 16, "materialx_flat_green.exr");
    const Bitmap redBitmap = BitmapReader::read(redOutput);
    const Bitmap greenBitmap = BitmapReader::read(greenOutput);

    const glm::vec3 redAvg = averageColor(redBitmap);
    const glm::vec3 greenAvg = averageColor(greenBitmap);

    // The two renders must differ and each must retain its own dominant color.
    CHECK(redAvg != greenAvg);
    CHECK(redAvg.x > redAvg.y);
    CHECK(redAvg.x > redAvg.z);
    CHECK(greenAvg.y > greenAvg.x);
    CHECK(greenAvg.y > greenAvg.z);
}

// Partial opacity must blend the material instead of discarding it.
TEST_CASE_METHOD(MaterialXRenderTest,
    "MaterialX opacity below 1.0 blends instead of discarding the whole material",
    "[e2e][materialx]")
{
    const std::string opaqueOutput =
        render("materialx_flat_red.nrscene", 16, "materialx_flat_red_opaque.exr");
    const std::string halfOutput = render(
        "materialx_flat_red_half_opacity.nrscene", 16, "materialx_flat_red_half_opacity.exr");
    const Bitmap opaqueBitmap = BitmapReader::read(opaqueOutput);
    const Bitmap halfBitmap = BitmapReader::read(halfOutput);

    const glm::vec3 opaqueAvg = averageColor(opaqueBitmap);
    const glm::vec3 halfAvg = averageColor(halfBitmap);

    // The bug forced opacity to 0 outright, making the sphere fully
    // pass-through -- these scenes' default background is white, so every
    // camera ray would then land on it and G (near-zero for a red diffuse
    // sphere that fills the frame, per opaqueAvg) would jump to roughly 1.0.
    // A correct 50% blend instead lands roughly halfway between the two
    // (measured ~0.51: 50% of opaque's ~0.04 plus 50% of background's ~1.0),
    // clearly separated from the bug's ~1.0. R alone can't tell the two
    // apart here (the red material's own R and the white background's R are
    // both close to 1).
    CHECK(opaqueAvg.y < 0.1f);
    CHECK(halfAvg.y < 0.9f);
    // A real partial blend must differ from the fully opaque render.
    CHECK(halfAvg.x != opaqueAvg.x);
}

// Guards against a regression where open_pbr_surface's transmission_weight
// had no visible effect at all: its full closure tree structurally contains
// up to 9 lobe-consuming leaf BSDFs (diffuse, both subsurface thin-walled
// variants, three dielectric variants, two metal variants, coat --
// regardless of which weights are actually zero), but NoorRayCompositeBsdf
// only kept the first 6 the closure walk reached, silently dropping the
// rest -- and dielectric_transmission, buried several `layer`/`mix` levels
// deep, was reliably among the dropped ones (see CompositeBsdf.h's
// NoorRayMaxLobes comment). This material has zero diffuse weight, so with
// transmission_weight=0 it's dominated by a smooth Fresnel highlight (mostly
// dark against these scenes' white background); at transmission_weight=1
// most of that white background should show straight through instead.
TEST_CASE_METHOD(MaterialXRenderTest,
    "open_pbr_surface transmission_weight actually lets light through",
    "[e2e][materialx]")
{
    const std::string opaqueOutput = render(
        "materialx_open_pbr_transmission_0.nrscene", 16, "open_pbr_transmission_0.exr");
    const std::string glassOutput = render(
        "materialx_open_pbr_transmission_1.nrscene", 16, "open_pbr_transmission_1.exr");
    const Bitmap opaqueBitmap = BitmapReader::read(opaqueOutput);
    const Bitmap glassBitmap = BitmapReader::read(glassOutput);

    const glm::vec3 opaqueAvg = averageColor(opaqueBitmap);
    const glm::vec3 glassAvg = averageColor(glassBitmap);

    // With transmission dropped, transmission_weight=1 would render
    // identically to transmission_weight=0 (both fall back to the same
    // Fresnel-only reflection). A working transmission lobe must let the
    // white background through and measurably brighten the sphere.
    CHECK(glassAvg.y > opaqueAvg.y + 0.1f);
    CHECK(glassAvg.z > opaqueAvg.z + 0.1f);
}
