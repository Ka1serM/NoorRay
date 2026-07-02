#pragma once

#include "Mesh/Material.h"
#include "OpenPbr/OpenPbrSurface.h"
#include "Raytracing/Sellmeier.h"

#include <cmath>

namespace nr::bsdftest
{
inline constexpr float Pi = 3.14159265358979323846f;
}

class BsdfTestFixture
{
protected:
    static constexpr glm::vec3 normal{0.0f, 0.0f, 1.0f};
    static constexpr glm::vec3 tangent{1.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 view{0.0f, 0.0f, 1.0f};
    static constexpr glm::vec2 uv{0.0f, 0.0f};
    static inline const SampledSpectrum unitThroughput{1.0f};

    static SampledWavelengths wavelengths()
    {
        SampledWavelengths result{};
        constexpr float samples[NrSpectrumSamples] = {460.0f, 530.0f, 610.0f, 700.0f};
        for (int i = 0; i < NrSpectrumSamples; ++i) {
            result.lambda[i] = samples[i];
            result.pdf[i] = 1.0f;
        }
        return result;
    }

    static Material makeMaterial(
        const float albedo, const float metallic, const float specular,
        const float roughness, const float transmission = 0.0f,
        const float transmissionIor = 1.5f)
    {
        Material material{};
        material.albedo = glm::vec3(albedo);
        material.transmissionColor = glm::vec3(1.0f);
        material.metallic = metallic;
        material.roughness = roughness;
        material.specular = specular;
        material.transmission = transmission;
        if (transmission > 0.0f)
            material.sellmeier = constantIorSellmeier(transmissionIor);
        return material;
    }

    static BsdfSample sample(
        const Material& material, RandomState& rng, const SampledWavelengths& wl,
        const glm::vec3 viewDirection = view)
    {
        return nr::openpbr::sample(
            material, nullptr, uv, viewDirection, normal, tangent, rng, wl,
            1.0f, unitThroughput);
    }

    static SampledSpectrum evaluate(
        const Material& material, const glm::vec3 viewDirection,
        const glm::vec3 lightDirection, const SampledWavelengths& wl)
    {
        return nr::openpbr::evaluate(
            material, nullptr, uv, viewDirection, lightDirection, normal, tangent,
            wl, 1.0f, unitThroughput);
    }

    static float pdf(
        const Material& material, const glm::vec3 viewDirection,
        const glm::vec3 lightDirection)
    {
        return nr::openpbr::pdf(
            material, nullptr, uv, viewDirection, lightDirection, normal, tangent,
            unitThroughput);
    }

    static float integrateOpaque(
        const float albedo, const float metallic, const float specular, const float roughness,
        const glm::vec3 viewDirection = view)
    {
        constexpr int thetaSamples = 256;
        constexpr int phiSamples = 256;
        double integral = 0.0;
        const Material material = makeMaterial(albedo, metallic, specular, roughness);
        const SampledWavelengths wl = wavelengths();
        for (int y = 0; y < thetaSamples; ++y) {
            for (int x = 0; x < phiSamples; ++x) {
                const float z = (y + 0.5f) / thetaSamples;
                const float phi = 2.0f * nr::bsdftest::Pi * (x + 0.5f) / phiSamples;
                const float radius = std::sqrt(1.0f - z * z);
                const glm::vec3 light(radius * std::cos(phi), radius * std::sin(phi), z);
                integral += evaluate(material, viewDirection, light, wl)[0];
            }
        }
        return static_cast<float>(integral * (2.0 * nr::bsdftest::Pi)
            / static_cast<double>(thetaSamples * phiSamples));
    }
};
