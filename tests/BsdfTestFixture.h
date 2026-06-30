#pragma once

#include "Mesh/Material.h"
#include "Samplers/HemisphereSampler.h"

class BsdfTestFixture
{
protected:
    static constexpr glm::vec3 normal{0.0f, 0.0f, 1.0f};
    static constexpr glm::vec3 view{0.0f, 0.0f, 1.0f};

    static float integrateOpaque(const float albedo, const float metallic, const float specular, const float roughness,
                                 const glm::vec3 viewDirection = view)
    {
        constexpr int thetaSamples = 256;
        constexpr int phiSamples = 256;
        double integral = 0.0;
        const SampledSpectrum albedoSpectrum(albedo);
        for (int y = 0; y < thetaSamples; ++y) {
            for (int x = 0; x < phiSamples; ++x) {
                const glm::vec3 light = nr::sampling::uniformHemisphere({
                    (y + 0.5f) / thetaSamples, (x + 0.5f) / phiSamples});
                const SampledSpectrum value = Material::evaluateOpaqueSpectral(
                    normal, viewDirection, light, albedoSpectrum, metallic, specular, roughness);
                integral += value[0] * light.z;
            }
        }
        return static_cast<float>(integral * (2.0 * BsdfPi)
            / static_cast<double>(thetaSamples * phiSamples));
    }
};
