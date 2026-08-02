#include <catch2/catch_test_macros.hpp>

#include <glm/geometric.hpp>

#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/Lobes/DiffuseLobe.h"

TEST_CASE("Burley diffuse retains its roughness response", "[bsdf][diffuse]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view = normal;
    const glm::vec3 grazing = glm::normalize(glm::vec3(1.0f, 0.0f, 0.1f));

    nr::shading::lobes::DiffuseLobe lambert;
    nr::shading::lobes::DiffuseLobe smoothBurley;
    smoothBurley.burley = true;
    nr::shading::lobes::DiffuseLobe roughBurley;
    roughBurley.burley = true;
    roughBurley.roughness = 1.0f;

    const BsdfEvaluation lambertEval =
        lambert.eval(normal, view, grazing);
    const BsdfEvaluation smoothEval =
        smoothBurley.eval(normal, view, grazing);
    const BsdfEvaluation roughEval =
        roughBurley.eval(normal, view, grazing);

    REQUIRE(lambertEval.pdf > 0.0f);
    CHECK(smoothEval.pdf == lambertEval.pdf);
    CHECK(roughEval.pdf == lambertEval.pdf);
    CHECK(smoothEval.value[0] < lambertEval.value[0]);
    CHECK(roughEval.value[0] > lambertEval.value[0]);
}

// Guards against a regression where MaterialX's translucent_bsdf (pure
// diffuse *transmission*, e.g. paper/leaves letting light through to the
// far side) reused DiffuseLobe with no transmission handling at all, so it
// silently behaved as an ordinary diffuse *reflection* lobe -- same-side
// light only, nothing reaching the back face.
TEST_CASE("Translucent diffuse scatters into the transmission hemisphere", "[bsdf][diffuse][translucent]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view = normal;
    const glm::vec3 sameSide(0.0f, 0.0f, 1.0f);
    const glm::vec3 farSide(0.0f, 0.0f, -1.0f);

    nr::shading::lobes::DiffuseLobe translucent;
    translucent.translucent = true;

    // Reflection-side evaluation must be zero: all of the energy goes
    // through, none of it bounces back on the incident side.
    CHECK(translucent.eval(normal, view, sameSide).pdf == 0.0f);
    // Transmission-side evaluation is where a plain reflective DiffuseLobe
    // would incorrectly return zero.
    const BsdfEvaluation farEval = translucent.eval(normal, view, farSide);
    CHECK(farEval.pdf > 0.0f);
    CHECK(farEval.value[0] > 0.0f);

    RandomState rng = seedRandom(1);
    const BsdfSample sample = translucent.sample(normal, view, rng);
    REQUIRE(sample.pdf > 0.0f);
    CHECK(sample.event == BsdfEvent::Transmission);
    // The sampled direction must land on the far side of the normal, not the
    // incident side.
    CHECK(glm::dot(normal, sample.direction) < 0.0f);
}
