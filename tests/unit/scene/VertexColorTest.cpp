#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Mesh/VertexColor.h"

TEST_CASE("vertex colors use packed RGBA8 sRGB storage", "[vertex-color]")
{
    CHECK(nr::vertex_color::packLinear(glm::vec4(1.0f))
        == nr::vertex_color::White);
    CHECK(nr::vertex_color::packSrgb(glm::vec4(0.0f, 0.0f, 0.0f, 0.5f))
        == 0x80000000u);

    const uint32_t packed = nr::vertex_color::packLinear(
        glm::vec4(0.5f, 0.25f, 0.0f, 0.75f));
    const glm::vec4 decoded = nr::vertex_color::unpackLinear(packed);
    CHECK(decoded.r == Catch::Approx(0.5f).margin(0.005f));
    CHECK(decoded.g == Catch::Approx(0.25f).margin(0.005f));
    CHECK(decoded.b == Catch::Approx(0.0f));
    CHECK(decoded.a == Catch::Approx(0.75f).margin(1.0f / 255.0f));
}
