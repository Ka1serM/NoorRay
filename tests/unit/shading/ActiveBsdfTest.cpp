#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <glm/geometric.hpp>

#include "Rendering/Sampling/HemisphereSampler.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/CompositeBsdf.h"
#include "Materials/Shading/Lobes/ConductorLobe.h"
#include "Materials/Shading/Lobes/DielectricLobe.h"

namespace
{

constexpr glm::vec3 Normal(0.0f, 0.0f, 1.0f);
constexpr glm::vec3 View(0.0f, 0.0f, 1.0f);

float integrateDielectric(
    const nr::shading::lobes::DielectricLobe& lobe,
    const glm::vec3 normal,
    const glm::vec3 view)
{
    constexpr int samples = 128;
    double energy = 0.0;
    for (int hemisphere = 0; hemisphere < 2; ++hemisphere)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                glm::vec3 light = nr::sampling::uniformHemisphere({
                    (y + 0.5f) / samples, (x + 0.5f) / samples});
                if (hemisphere != 0)
                    light = -light;
                const BsdfEvaluation evaluation = lobe.eval(
                    normal, view, light);
                const bool transmitted = glm::dot(normal, light) < 0.0f;
                const float eta = lobe.exiting ? 1.0f / lobe.ior : lobe.ior;
                const float radianceToEnergy = transmitted ? eta * eta : 1.0f;
                energy += evaluation.value[0] * fabsf(glm::dot(normal, light))
                    * radianceToEnergy;
            }
        }
    }
    return static_cast<float>(energy * 2.0 * BsdfPi
        / static_cast<double>(samples * samples));
}

float integrateComposite(const nr::shading::NoorRayShaderData& shader,
    const glm::vec3 normal, const glm::vec3 view)
{
    constexpr int samples = 512;
    double energy = 0.0;
    for (int hemisphere = 0; hemisphere < 2; ++hemisphere)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                glm::vec3 light = nr::sampling::uniformHemisphere({
                    (y + 0.5f) / samples, (x + 0.5f) / samples});
                if (hemisphere != 0)
                    light = -light;
                const float radianceToEnergy = glm::dot(normal, light) < 0.0f
                    ? 1.5f * 1.5f : 1.0f;
                energy += shader.evaluate(light).value[0]
                    * fabsf(glm::dot(normal, light)) * radianceToEnergy;
            }
        }
    }
    return static_cast<float>(energy * 2.0 * BsdfPi
        / static_cast<double>(samples * samples));
}

}

TEST_CASE("active GGX reflection PDF integrates to a probability", "[bsdf][ggx]")
{
    constexpr int samples = 128;
    double integral = 0.0;
    for (int y = 0; y < samples; ++y)
    {
        for (int x = 0; x < samples; ++x)
        {
            const glm::vec3 light = nr::sampling::uniformHemisphere({
                (y + 0.5f) / samples, (x + 0.5f) / samples});
            const glm::vec3 halfVector = glm::normalize(View + light);
            integral += nr::shading::ggx::reflectionPdf(
                Normal, View, halfVector, 0.35f);
        }
    }
    integral *= 2.0 * BsdfPi / static_cast<double>(samples * samples);
    CHECK(integral > 0.97);
    CHECK(integral <= 1.01);
}

TEST_CASE("active GGX retains the singular cutoff and finite transition", "[bsdf][ggx]")
{
    CHECK(nr::shading::ggx::isAlmostSpecular(0.003f));
    CHECK_FALSE(nr::shading::ggx::isAlmostSpecular(0.004f));
    const float pdf = nr::shading::ggx::reflectionPdf(
        Normal, View, Normal, 0.004f);
    CHECK(std::isfinite(pdf));
    CHECK(pdf == Catch::Approx(1.0f / (4.0f * BsdfPi
        * 0.004f * 0.004f * 0.004f * 0.004f)).epsilon(2.0e-4f));
}

TEST_CASE("active dielectric supports reflection, transmission, and Snell sampling",
    "[bsdf][dielectric]")
{
    nr::shading::lobes::DielectricLobe transmissionOnly;
    transmissionOnly.reflectionTint = SampledSpectrum(0.0f);
    transmissionOnly.transmissionTint = SampledSpectrum(1.0f);
    transmissionOnly.roughness = 0.0f;

    RandomState transmissionRng = seedRandom(0x7a11u);
    for (int i = 0; i < 256; ++i)
    {
        const BsdfSample sample = transmissionOnly.sample(
            Normal, View, transmissionRng);
        REQUIRE(sample.pdf > 0.0f);
        CHECK(sample.event == BsdfEvent::Transmission);
        CHECK(sample.direction.z < 0.0f);
    }

    nr::shading::lobes::DielectricLobe mixed;
    mixed.reflectionTint = SampledSpectrum(1.0f);
    mixed.transmissionTint = SampledSpectrum(1.0f);
    mixed.roughness = 0.0f;
    RandomState mixedRng = seedRandom(0x7a11u);
    int reflections = 0;
    int transmissions = 0;
    for (int i = 0; i < 4096; ++i)
    {
        const BsdfSample sample = mixed.sample(Normal, View, mixedRng);
        REQUIRE(sample.pdf > 0.0f);
        if (sample.event == BsdfEvent::Specular)
            ++reflections;
        else
        {
            REQUIRE(sample.event == BsdfEvent::Transmission);
            ++transmissions;
        }
    }
    CHECK(reflections > 0);
    CHECK(transmissions > 0);
}

TEST_CASE("active dielectric evaluates spectral IOR per sampled wavelength",
    "[bsdf][dielectric][dispersion]")
{
    nr::shading::lobes::DielectricLobe glass;
    glass.reflectionTint = SampledSpectrum(1.0f);
    glass.roughness = 0.25f;
    glass.spectralIor = SampledSpectrum{};
    glass.spectralIor[0] = 1.45f;
    glass.spectralIor[1] = 1.50f;
    glass.spectralIor[2] = 1.55f;
    glass.spectralIor[3] = 1.60f;
    glass.useSpectralIor = true;

    const BsdfEvaluation evaluation = glass.eval(
        Normal, View, glm::normalize(glm::vec3(0.3f, 0.1f, 0.95f)));
    CHECK(evaluation.value[0] != Catch::Approx(evaluation.value[3]));

    glass.reflectionTint = SampledSpectrum(0.0f);
    glass.transmissionTint = SampledSpectrum(1.0f);
    RandomState rng = seedRandom(0x51e11ae1u);
    const BsdfSample sample = glass.sample(Normal, View, rng);
    REQUIRE(sample.event == BsdfEvent::Transmission);
    CHECK(sample.dispersive);
}

TEST_CASE("active rough glass uses generated LUT energy compensation", "[bsdf][dielectric][furnace]")
{
    nr::shading::lobes::DielectricLobe glass;
    glass.reflectionTint = SampledSpectrum(1.0f);
    glass.transmissionTint = SampledSpectrum(1.0f);
    glass.roughness = 1.0f;
    glass.ior = 1.5f;

    for (const bool exiting : {false, true})
    {
        glass.exiting = exiting;
        const glm::vec3 normal = exiting ? -Normal : Normal;
        const glm::vec3 view = normal;
        const float energy = integrateDielectric(glass, normal, view);
        INFO("exiting=" << exiting << " energy=" << energy);
        CHECK(energy >= 0.98f);
        CHECK(energy <= 1.02f);
    }
}

TEST_CASE("Disney glass reflection and transmission conserve furnace energy",
    "[bsdf][dielectric][disney][furnace]")
{
    for (const float roughness : {0.25f, 0.5f, 1.0f})
    {
        nr::shading::NoorRayShaderData shader(Normal, Normal, View);

        nr::shading::lobes::DielectricLobe glass;
        glass.reflectionTint = SampledSpectrum(1.0f);
        glass.transmissionTint = SampledSpectrum(1.0f);
        glass.roughness = roughness;
        glass.ior = 1.5f;
        shader.addDielectric(SampledSpectrum(1.0f), glass);
        REQUIRE(shader.prepare());

        const float energy = integrateComposite(shader, Normal, View);
        const float lutDirectional = nr::shading::energy_lut::glassDirectionalAlbedo(
            nullptr, roughness, 1.0f, 1.5f);
        const float lutAverage = nr::shading::energy_lut::glassAverageAlbedo(
            nullptr, roughness, 1.5f);
        INFO("roughness=" << roughness << " energy=" << energy
            << " lutDirectional=" << lutDirectional
            << " lutAverage=" << lutAverage);
        CHECK(energy >= 0.98f);
        CHECK(energy <= 1.02f);
    }
}

TEST_CASE("active conductor uses LUT multi-scatter compensation", "[bsdf][conductor][furnace]")
{
    nr::shading::lobes::ConductorLobe metal;
    metal.eta = SampledSpectrum(0.2f);
    metal.extinction = SampledSpectrum(3.0f);
    metal.roughness = 0.5f;

    constexpr int samples = 128;
    double energy = 0.0;
    for (int y = 0; y < samples; ++y)
    {
        for (int x = 0; x < samples; ++x)
        {
            const glm::vec3 light = nr::sampling::uniformHemisphere({
                (y + 0.5f) / samples, (x + 0.5f) / samples});
            energy += metal.eval(Normal, View, light).value[0] * light.z;
        }
    }
    energy *= 2.0 * BsdfPi / static_cast<double>(samples * samples);
    CHECK(std::isfinite(static_cast<float>(energy)));
    CHECK(energy > 0.0);
    CHECK(energy <= 1.01);
}
