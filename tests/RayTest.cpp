#include "Raytracing/Ray.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("ray owns its traversal interval", "[raytracing][ray]")
{
    const Ray ray(
        glm::vec3(1.0f, 2.0f, 3.0f),
        glm::vec3(0.0f, 0.0f, -1.0f),
        0.25f,
        12.0f);

    CHECK(ray.origin() == glm::vec3(1.0f, 2.0f, 3.0f));
    CHECK(ray.direction() == glm::vec3(0.0f, 0.0f, -1.0f));
    CHECK(ray.minDistance() == Catch::Approx(0.25f));
    CHECK(ray.maxDistance() == Catch::Approx(12.0f));
    CHECK(ray.at(2.0f) == glm::vec3(1.0f, 2.0f, 1.0f));
    CHECK(ray.hasTraversalInterval());
}

TEST_CASE("ray interval changes preserve its geometry", "[raytracing][ray]")
{
    const Ray original(
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(2.0f, 0.0f, 0.0f),
        0.1f,
        8.0f);
    const Ray clipped = original.withMinDistance(3.0f);

    CHECK(clipped.origin() == original.origin());
    CHECK(clipped.direction() == original.direction());
    CHECK(clipped.minDistance() == Catch::Approx(3.0f));
    CHECK(clipped.maxDistance() == Catch::Approx(8.0f));
    CHECK(clipped.closestDistanceTo(glm::vec3(5.0f, 0.0f, 0.0f))
        == Catch::Approx(2.0f));
    CHECK_FALSE(Ray::invalid().hasTraversalInterval());
}
