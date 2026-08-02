#include "Materials/Shading/CompositeBsdf.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("SVM closure pool rejects overflow instead of dropping lobes", "[bsdf][closure]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    nr::shading::NoorRayCompositeBsdf bsdf(normal, normal, normal);
    nr::shading::lobes::DiffuseLobe lobe;

    for (int i = 0; i < nr::shading::NoorRayMaxLobes; ++i)
        CHECK(bsdf.addDiffuse(SampledSpectrum(1.0f), lobe));

    CHECK_FALSE(bsdf.addDiffuse(SampledSpectrum(1.0f), lobe));
    CHECK_FALSE(bsdf.prepare());
}

TEST_CASE("SVM closure pool clamps and prunes weights", "[bsdf][closure]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    nr::shading::NoorRayCompositeBsdf bsdf(normal, normal, normal);
    nr::shading::lobes::DiffuseLobe lobe;

    CHECK(bsdf.addDiffuse(SampledSpectrum(-1.0f), lobe));
    CHECK(bsdf.addDiffuse(SampledSpectrum(1.0e-6f), lobe));
    CHECK(bsdf.addDiffuse(SampledSpectrum(1.0f), lobe));
    CHECK(bsdf.prepare());
    CHECK(bsdf.evaluate(normal).value[0] > 0.0f);
}

TEST_CASE("SVM composite clamps grazing shading normals", "[bsdf][normal]")
{
    const glm::vec3 geometricNormal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view = glm::normalize(glm::vec3(0.999f, 0.0f, 0.045f));
    const glm::vec3 divergentNormal = glm::normalize(glm::vec3(-0.12f, 0.0f, 0.993f));

    nr::shading::NoorRayCompositeBsdf bsdf(
        geometricNormal, divergentNormal, view);
    const glm::vec3 reflected = glm::reflect(-view, bsdf.shadingNormal());

    CHECK(glm::dot(geometricNormal, reflected) > 0.00999f);
}

TEST_CASE("SVM composite preserves translucent closure transmission", "[bsdf][transmission]")
{
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    nr::shading::NoorRayCompositeBsdf bsdf(normal, normal, normal);
    nr::shading::lobes::DiffuseLobe translucent;
    translucent.translucent = true;

    REQUIRE(bsdf.addDiffuse(SampledSpectrum(1.0f), translucent));
    REQUIRE(bsdf.prepare());
    const BsdfEvaluation evaluation = bsdf.evaluate(-normal);
    CHECK(evaluation.pdf > 0.0f);
    CHECK(evaluation.value[0] > 0.0f);
}
