#include "Rendering/ShadowTerminator.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{

nr::ShadowTerminatorTriangle curvedTriangle()
{
    return {
        {glm::vec3(1.0f, 0.0f, 0.0f),
         glm::vec3(0.0f, 1.0f, 0.0f),
         glm::vec3(0.0f, 0.0f, 1.0f)},
        {glm::vec3(1.0f, 0.0f, 0.0f),
         glm::vec3(0.0f, 1.0f, 0.0f),
         glm::vec3(0.0f, 0.0f, 1.0f)},
        glm::mat3(1.0f)};
}

}

TEST_CASE("smooth-surface offset lifts a curved triangle", "[shadow-terminator]")
{
    const nr::ShadowTerminatorTriangle triangle = curvedTriangle();
    const glm::vec3 geometricNormal = glm::normalize(glm::vec3(1.0f));
    const glm::vec3 offset = nr::shadowRaySmoothSurfaceOffset(
        triangle, 1.0f / 3.0f, 1.0f / 3.0f, geometricNormal);

    CHECK(offset.x == Catch::Approx(2.0f / 9.0f).margin(1e-6f));
    CHECK(offset.y == Catch::Approx(2.0f / 9.0f).margin(1e-6f));
    CHECK(offset.z == Catch::Approx(2.0f / 9.0f).margin(1e-6f));
}

TEST_CASE("shadow terminator offset only affects grazing rays", "[shadow-terminator]")
{
    const nr::ShadowTerminatorTriangle triangle = curvedTriangle();
    const glm::vec3 normal = glm::normalize(glm::vec3(1.0f));
    const glm::vec3 tangent = glm::normalize(glm::vec3(1.0f, -1.0f, 0.0f));

    const glm::vec3 grazingOffset = nr::shadowTerminatorOffset(
        triangle, 1.0f / 3.0f, 1.0f / 3.0f, normal, normal, tangent);
    CHECK(glm::length(grazingOffset) > 0.0f);

    const glm::vec3 facingOffset = nr::shadowTerminatorOffset(
        triangle, 1.0f / 3.0f, 1.0f / 3.0f, normal, normal, normal);
    CHECK(glm::length(facingOffset) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("flat vertex normals do not move the ray origin", "[shadow-terminator]")
{
    nr::ShadowTerminatorTriangle triangle{
        {glm::vec3(0.0f, 0.0f, 0.0f),
         glm::vec3(1.0f, 0.0f, 0.0f),
         glm::vec3(0.0f, 1.0f, 0.0f)},
        {glm::vec3(0.0f, 0.0f, 1.0f),
         glm::vec3(0.0f, 0.0f, 1.0f),
         glm::vec3(0.0f, 0.0f, 1.0f)},
        glm::mat3(1.0f)};
    const glm::vec3 offset = nr::shadowTerminatorOffset(
        triangle, 0.25f, 0.25f,
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(1.0f, 0.0f, 0.0f));

    CHECK(glm::length(offset) == Catch::Approx(0.0f).margin(1e-7f));
}
