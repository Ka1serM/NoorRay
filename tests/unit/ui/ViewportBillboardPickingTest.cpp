#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include "Backend/Vulkan/Viewport/Viewport.h"

namespace {

constexpr uint32_t kWidth = 1920;
constexpr uint32_t kHeight = 1080;

// A camera at the origin looking down -Z, matching CameraInstance's plain
// glm::perspective (Y-up NDC, no Vulkan flip).
glm::mat4 viewProjection()
{
    const glm::mat4 projection = glm::perspective(glm::radians(45.0f),
        static_cast<float>(kWidth) / static_cast<float>(kHeight), 0.01f, 10000.0f);
    return projection * glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
}

// Mirrors ViewportPanel::screenToPixel: normalized cursor coordinates measured
// from the displayed image's top-left, converted to the renderer's bottom-left
// pixel space.
glm::vec2 clickAt(const float normalizedX, const float normalizedY)
{
    return {normalizedX * static_cast<float>(kWidth),
        (1.0f - normalizedY) * static_cast<float>(kHeight)};
}

}

TEST_CASE("a light above the camera projects to the upper half of the viewport",
    "[ui][viewport][billboard]")
{
    glm::vec2 pixel{};
    REQUIRE(projectViewportBillboard(viewProjection(),
        glm::vec3(0.0f, 2.0f, -10.0f), kWidth, kHeight, pixel));

    // The renderer's row order is bottom-left, so "up" must land above the
    // vertical midpoint. Inverting this is what made overlay clicks select the
    // light mirrored across the horizontal centre line instead of the one drawn
    // under the cursor.
    CHECK(pixel.y > static_cast<float>(kHeight) * 0.5f);
    CHECK(pixel.x == Catch::Approx(static_cast<float>(kWidth) * 0.5f));
}

TEST_CASE("a click on a drawn light overlay lands within the pick radius",
    "[ui][viewport][billboard]")
{
    const glm::mat4 camera = viewProjection();
    const glm::vec3 above(0.0f, 2.0f, -10.0f);
    const glm::vec3 below(0.0f, -2.0f, -10.0f);

    glm::vec2 abovePixel{};
    glm::vec2 belowPixel{};
    REQUIRE(projectViewportBillboard(camera, above, kWidth, kHeight, abovePixel));
    REQUIRE(projectViewportBillboard(camera, below, kWidth, kHeight, belowPixel));

    // Click the upper icon: it must win, and the mirrored one must not.
    const glm::vec2 click = clickAt(abovePixel.x / static_cast<float>(kWidth),
        1.0f - abovePixel.y / static_cast<float>(kHeight));
    const float radiusSq = ViewportBillboardPixelRadius * ViewportBillboardPixelRadius;
    CHECK(glm::dot(click - abovePixel, click - abovePixel) <= radiusSq);
    CHECK(glm::dot(click - belowPixel, click - belowPixel) > radiusSq);
}

TEST_CASE("a light behind the camera is not pickable", "[ui][viewport][billboard]")
{
    glm::vec2 pixel{};
    CHECK_FALSE(projectViewportBillboard(viewProjection(),
        glm::vec3(0.0f, 0.0f, 10.0f), kWidth, kHeight, pixel));
}
