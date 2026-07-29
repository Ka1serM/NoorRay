#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Samplers/RandomSampler.h"
#include "Shading/BsdfClosure.h"
#include "Shading/Dielectric.h"
#include "Shading/EnergyLut.h"
#include "Shading/Ggx.h"
#include "Shading/Spectrum.h"

// Standalone GGX dielectric lobe (reflection + transmission through a
// single interface, scalar IOR -- MaterialX's dielectric_bsdf/
// generalized_schlick_bsdf/coat all reduce to this). No wavelength
// dispersion: MaterialX's ior input is a plain scalar, unlike the legacy
// Sellmeier-driven glass path.
namespace nr::shading::lobes
{

struct DielectricLobe
{
    SampledSpectrum reflectionTint{1.0f};
    SampledSpectrum transmissionTint{0.0f};
    float roughness{0.5f};
    float ior{1.5f};
    bool exiting{false};
    const nr::shading::energy_lut::Textures* energyLuts{};

    NR_CPU_GPU BsdfEvaluation eval(
        const glm::vec3 normal, const glm::vec3 view,
        const glm::vec3 outgoing) const
    {
        const float normalOutgoing = glm::dot(normal, outgoing);
        if (normalOutgoing > 0.0f)
            return evalReflection(normal, view, outgoing, normalOutgoing);
        if (normalOutgoing < 0.0f && transmissionTint.maxComponent() > 0.0f)
            return evalTransmission(normal, view, outgoing, normalOutgoing);
        return {};
    }

    NR_CPU_GPU BsdfSample sample(
        const glm::vec3 normal, const glm::vec3 view, RandomState& rng) const
    {
        const float normalView = glm::dot(normal, view);
        if (normalView <= 0.0f)
            return {};

        const bool smooth = nr::shading::ggx::isAlmostSpecular(roughness);
        const float etaPath = pathEta();
        const glm::vec3 halfVector = smooth
            ? normal
            : nr::shading::ggx::sampleVisibleNormal(normal, view, roughness,
                  glm::vec2(randomFloat(rng), randomFloat(rng)));
        const float viewHalfCosine = fabsf(glm::dot(view, halfVector));
        if (viewHalfCosine <= 0.0f)
            return {};

        const float fresnel = nr::shading::dielectric::fresnel(
            viewHalfCosine, 1.0f, etaPath);
        const bool hasTransmission = transmissionTint.maxComponent() > 0.0f;
        const glm::vec3 refracted = glm::refract(
            -view, halfVector, 1.0f / etaPath);
        const bool totalInternalReflection =
            glm::dot(refracted, refracted) < 1.0e-10f;

        if (!hasTransmission || totalInternalReflection
            || randomFloat(rng) < fresnel)
        {
            BsdfSample result;
            result.event = BsdfEvent::Specular;
            result.direction = glm::reflect(-view, halfVector);
            if (glm::dot(normal, result.direction) <= 0.0f)
                return {};
            if (smooth)
            {
                result.singular = true;
                const float branchProbability =
                    hasTransmission ? fresnel : 1.0f;
                result.pdf = branchProbability * nr::shading::ggx::SingularPdf;
                result.weight = reflectionTint
                    * (fresnel / fmaxf(branchProbability, BsdfEpsilon));
                return result;
            }
            const BsdfEvaluation evaluation = eval(
                normal, view, result.direction);
            const float outgoingCosine = glm::dot(normal, result.direction);
            if (evaluation.pdf <= 0.0f || outgoingCosine <= 0.0f)
                return {};
            result.weight = evaluation.value
                * (outgoingCosine / evaluation.pdf);
            result.pdf = evaluation.pdf;
            return result;
        }

        const glm::vec3 direction = glm::normalize(refracted);
        if (glm::dot(normal, direction) >= 0.0f)
            return {};

        BsdfSample result;
        result.event = BsdfEvent::Transmission;
        result.eta = etaPath;
        result.direction = direction;
        if (smooth || fabsf(etaPath - 1.0f) < 1.0e-4f)
        {
            result.singular = true;
            result.pdf = (1.0f - fresnel) * nr::shading::ggx::SingularPdf;
            result.weight = transmissionTint * (1.0f / (etaPath * etaPath));
            return result;
        }
        const BsdfEvaluation evaluation = eval(normal, view, direction);
        const float outgoingCosine = -glm::dot(normal, direction);
        if (evaluation.pdf <= 0.0f || outgoingCosine <= 0.0f)
            return {};
        result.weight = evaluation.value * (outgoingCosine / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    NR_CPU_GPU SampledSpectrum albedoEstimate(
        const glm::vec3 normal, const glm::vec3 view) const
    {
        const float normalView = fmaxf(glm::dot(normal, view), 0.0f);
        const float f0 = nr::shading::dielectric::fresnel(1.0f, 1.0f, ior);
        const float directional =
            nr::shading::dielectric::fresnelFromNormalReflectance(
                normalView, f0);
        return reflectionTint * directional + transmissionTint * (1.0f - directional);
    }

private:
    NR_CPU_GPU float pathEta() const
    {
        return exiting ? 1.0f / fmaxf(ior, nr::shading::dielectric::DenominatorEpsilon)
                        : ior;
    }

    NR_CPU_GPU BsdfEvaluation evalReflection(
        const glm::vec3 normal, const glm::vec3 view,
        const glm::vec3 outgoing, const float normalOutgoing) const
    {
        BsdfEvaluation result;
        if (nr::shading::ggx::isAlmostSpecular(roughness))
            return result;

        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateReflection(normal, view, outgoing, roughness);
        if (microfacet.cosineWeighted <= 0.0f)
            return result;

        const float normalView = glm::dot(normal, view);
        const float f0 = nr::shading::dielectric::fresnel(1.0f, 1.0f, ior);
        const float viewAlbedo = nr::shading::energy_lut::dielectricDirectionalAlbedo(
            energyLuts, roughness, normalView, f0);
        const float lightAlbedo = nr::shading::energy_lut::dielectricDirectionalAlbedo(
            energyLuts, roughness, normalOutgoing, f0);
        const float averageAlbedo = nr::shading::energy_lut::dielectricAverageAlbedo(
            energyLuts, roughness, f0);
        const float averageFresnelValue = nr::shading::dielectric::averageFresnel(ior);
        const float geometryFactor = microfacet.cosineWeighted / normalOutgoing;
        const float missingAverage = 1.0f - averageAlbedo;
        const float msScale = missingAverage > nr::shading::ggx::DenominatorEpsilon
            ? (1.0f - viewAlbedo) * (1.0f - lightAlbedo)
                / (BsdfPi * missingAverage)
            : 0.0f;
        const float ms = msScale * nr::shading::energy_lut::multipleScatterFresnel(
            averageFresnelValue, averageAlbedo);

        const float fresnel = nr::shading::dielectric::fresnel(
            microfacet.viewHalfCosine, 1.0f, ior);
        result.pdf = microfacet.pdf;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result.value[i] = reflectionTint[i] * (fresnel * geometryFactor + ms);
        return result;
    }

    NR_CPU_GPU BsdfEvaluation evalTransmission(
        const glm::vec3 normal, const glm::vec3 view,
        const glm::vec3 outgoing, const float normalOutgoing) const
    {
        BsdfEvaluation result;
        const float etaPath = pathEta();
        if (nr::shading::ggx::isAlmostSpecular(roughness)
            || fabsf(etaPath - 1.0f) < 1.0e-4f)
            return result;

        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateTransmission(
                normal, view, outgoing, roughness, etaPath);
        const float outgoingCosine = -normalOutgoing;
        if (microfacet.cosineWeighted <= 0.0f || outgoingCosine <= 0.0f)
            return result;

        const float base = microfacet.cosineWeighted
            / outgoingCosine / (etaPath * etaPath);
        const float fresnel = nr::shading::dielectric::fresnel(
            microfacet.viewHalfCosine, 1.0f, etaPath);
        result.pdf = (1.0f - fresnel) * microfacet.pdf;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result.value[i] = transmissionTint[i] * (1.0f - fresnel) * base;
        return result;
    }
};

}
