#pragma once

#include "Bsdf/BsdfMaterial.h"
#include "Bsdf/Microfacet.h"

#include <cmath>

class BsdfTestFixture
{
protected:
    static constexpr glm::vec3 normal{0.0f, 0.0f, 1.0f};
    static constexpr glm::vec3 view{0.0f, 0.0f, 1.0f};
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

    static BsdfMaterialParameters makeMaterial(
        const float albedo, const float metallic, const float specular,
        const float roughness, const float transmission = 0.0f,
        const float transmissionIor = 1.5f)
    {
        BsdfMaterialParameters material{};
        material.baseColor = SampledSpectrum(albedo);
        material.transmissionColor = SampledSpectrum(1.0f);
        material.metalness = metallic;
        material.roughness = roughness;
        material.specular = specular;
        material.transmission = transmission;
        if (transmission > 0.0f)
            material.sellmeier = constantIorSellmeier(transmissionIor);
        return material;
    }

    static float integrateOpaque(const float albedo, const float metallic, const float specular, const float roughness,
                                 const glm::vec3 viewDirection = view)
    {
        constexpr int thetaSamples = 256;
        constexpr int phiSamples = 256;
        double integral = 0.0;
        const BsdfMaterialParameters material = makeMaterial(
            albedo, metallic, specular, roughness);
        const SampledWavelengths wl = wavelengths();
        RandomState rng = seedRandom(42);
        for (int y = 0; y < thetaSamples; ++y) {
            for (int x = 0; x < phiSamples; ++x) {
                const float z = (y + 0.5f) / thetaSamples;
                const float phi = 2.0f * nr::bsdf::Pi * (x + 0.5f) / phiSamples;
                const float radius = std::sqrt(1.0f - z * z);
                const glm::vec3 light(radius * std::cos(phi), radius * std::sin(phi), z);
                integral += nr::bsdf::evaluate(
                    material, normal, normal, viewDirection, light, wl, rng)[0];
            }
        }
        return static_cast<float>(integral * (2.0 * nr::bsdf::Pi)
            / static_cast<double>(thetaSamples * phiSamples));
    }
};
