#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/BsdfClosure.h"
#include "Materials/Shading/Dielectric.h"
#include "Materials/Shading/EnergyLut.h"
#include "Materials/Shading/Ggx.h"
#include "Materials/Shading/Spectrum.h"

// Standalone GGX dielectric lobe (reflection + transmission through a
// single interface). Direction and branch selection use the hero wavelength;
// Fresnel and spectral throughput use the per-wavelength IOR when supplied.
namespace nr::shading::lobes
{

struct DielectricLobe
{
    SampledSpectrum reflectionTint{1.0f};
    SampledSpectrum transmissionTint{0.0f};
    float roughness{0.5f};
    float ior{1.5f};
    SampledSpectrum spectralIor{1.5f};
    bool useSpectralIor{false};
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

    template <typename Rng>
    NR_CPU_GPU BsdfSample sample(
        const glm::vec3 normal, const glm::vec3 view, Rng& rng) const
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

        const SampledSpectrum fresnelValues = fresnelSpectrum(viewHalfCosine);
        const float fresnel = fresnelValues[0];
        const bool hasReflection = reflectionTint.maxComponent() > 0.0f;
        const bool hasTransmission = transmissionTint.maxComponent() > 0.0f;
        const glm::vec3 refracted = glm::refract(
            -view, halfVector, 1.0f / etaPath);
        const bool totalInternalReflection =
            glm::dot(refracted, refracted) < 1.0e-10f;

        if (hasReflection && (!hasTransmission || totalInternalReflection
            || randomFloat(rng) < fresnel))
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
                result.weight = reflectionTint * (fresnelValues
                    * (1.0f / fmaxf(branchProbability, BsdfEpsilon)));
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

        if (!hasTransmission || totalInternalReflection)
            return {};

        const glm::vec3 direction = nr::safeNormalize(refracted);
        if (!nr::isFinite(direction)
            || glm::dot(direction, direction) <= 1.0e-12f)
            return {};
        if (glm::dot(normal, direction) >= 0.0f)
            return {};

        BsdfSample result;
        result.event = BsdfEvent::Transmission;
        result.eta = etaPath;
        result.dispersive = useSpectralIor;
        result.direction = direction;
        if (smooth || fabsf(etaPath - 1.0f) < 1.0e-4f)
        {
            result.singular = true;
            result.pdf = (1.0f - fresnel) * nr::shading::ggx::SingularPdf;
            SampledSpectrum etaScale;
            for (int i = 0; i < NrSpectrumSamples; ++i) {
                const float eta = pathEtaAt(i);
                etaScale[i] = 1.0f / (eta * eta);
            }
            result.weight = transmissionTint * etaScale;
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
        SampledSpectrum result;
        for (int i = 0; i < NrSpectrumSamples; ++i) {
            const float eta = pathEtaAt(i);
            const float f0 = nr::shading::dielectric::fresnel(1.0f, 1.0f, eta);
            // For a rough reflection-only lobe, the relevant estimate is the
            // directional GGX albedo, not smooth-interface Fresnel.  The
            // latter underestimates/overestimates the amount that the LUT
            // compensated lobe contributes at oblique views and makes a
            // composite energy guard scale Disney stacks incorrectly.
            const float singleScatter =
                nr::shading::ggx::isAlmostSpecular(roughness)
                ? nr::shading::dielectric::fresnelFromNormalReflectance(
                    normalView, f0)
                : nr::shading::energy_lut::dielectricDirectionalAlbedo(
                    energyLuts, roughness, normalView, f0);
            const float averageGeometryAlbedo =
                nr::shading::ggx::isAlmostSpecular(roughness)
                ? 1.0f
                : nr::shading::energy_lut::ggxAverageAlbedo(
                    energyLuts, roughness);
            const float averageFresnel =
                nr::shading::dielectric::averageFresnel(eta);
            const float multipleScatter = nr::shading::ggx::isAlmostSpecular(roughness)
                ? 0.0f
                : nr::shading::energy_lut::multipleScatterFresnel(
                    averageFresnel, averageGeometryAlbedo);
            // Match the directional integral of evalReflection(), including
            // the Kulla-Conty multiple-scatter term.  Using only the LUT's
            // single-scatter value made the composite guard vary with view
            // angle and left a visible white-furnace gradient.
            const float geometryDirectional =
                nr::shading::ggx::isAlmostSpecular(roughness)
                ? 1.0f
                : nr::shading::energy_lut::ggxDirectionalAlbedo(
                    energyLuts, roughness, normalView);
            const float directional = singleScatter
                + (1.0f - geometryDirectional) * multipleScatter;
            result[i] = reflectionTint[i] * directional
                + transmissionTint[i] * (1.0f - directional);
        }
        return result;
    }

private:
    NR_CPU_GPU float iorAt(const int index) const
    {
        return fmaxf(useSpectralIor ? spectralIor[index] : ior, 1.0f);
    }

    NR_CPU_GPU float pathEtaAt(const int index) const
    {
        const float materialIor = iorAt(index);
        return exiting ? 1.0f / fmaxf(materialIor,
            nr::shading::dielectric::DenominatorEpsilon) : materialIor;
    }

    NR_CPU_GPU float pathEta() const
    {
        return pathEtaAt(0);
    }

    NR_CPU_GPU SampledSpectrum fresnelSpectrum(const float cosine) const
    {
        SampledSpectrum result;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result[i] = nr::shading::dielectric::fresnel(
                cosine, 1.0f, pathEtaAt(i));
        return result;
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
        const float etaPath = pathEta();
        const float f0 = nr::shading::dielectric::fresnel(
            1.0f, 1.0f, etaPath);
        const SampledSpectrum fresnelValues = fresnelSpectrum(
            microfacet.viewHalfCosine);
        if (transmissionTint.maxComponent() > 0.0f)
        {
            const float base = microfacet.cosineWeighted / normalOutgoing;
            const SampledSpectrum scale = glassEnergyScale(
                normalView, etaPath);
            result.pdf = microfacet.pdf * fresnelValues[0];
            for (int i = 0; i < NrSpectrumSamples; ++i)
                result.value[i] = reflectionTint[i] * fresnelValues[i]
                    * base * scale[i];
            return result;
        }
        // The dielectric table is the single-scatter Fresnel-weighted
        // response. Missing GGX energy is controlled by the geometry-only
        // masking albedo, shared with the conductor compensation.
        const float viewAlbedo = nr::shading::energy_lut::ggxDirectionalAlbedo(
            energyLuts, roughness, normalView);
        const float lightAlbedo = nr::shading::energy_lut::ggxDirectionalAlbedo(
            energyLuts, roughness, normalOutgoing);
        const float averageAlbedo = nr::shading::energy_lut::ggxAverageAlbedo(
            energyLuts, roughness);
        const float averageFresnelValue = nr::shading::dielectric::averageFresnel(etaPath);
        const float geometryFactor = microfacet.cosineWeighted / normalOutgoing;
        const float missingAverage = 1.0f - averageAlbedo;
        const float msScale = missingAverage > nr::shading::ggx::DenominatorEpsilon
            ? (1.0f - viewAlbedo) * (1.0f - lightAlbedo)
                / (BsdfPi * missingAverage)
            : 0.0f;
        const float ms = msScale * nr::shading::energy_lut::multipleScatterFresnel(
            averageFresnelValue, averageAlbedo);

        // sample() picks the reflection branch with probability
        // fresnel(pathEta) whenever this lobe also transmits, so the density
        // has to carry that factor -- evalTransmission()'s (1 - fresnel) is
        // its mirror image. Leaving it out made every rough reflection
        // weight come back a factor of F too small.
        result.pdf = microfacet.pdf;
        if (transmissionTint.maxComponent() > 0.0f)
            result.pdf *= nr::shading::dielectric::fresnel(
                microfacet.viewHalfCosine, 1.0f, pathEta());
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result.value[i] = reflectionTint[i]
                * (fresnelValues[i] * geometryFactor + ms);
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

        const SampledSpectrum fresnelValues = fresnelSpectrum(
            microfacet.viewHalfCosine);
        result.pdf = (1.0f - fresnelValues[0]) * microfacet.pdf;
        const SampledSpectrum scale = glassEnergyScale(
            glm::dot(normal, view), etaPath);
        for (int i = 0; i < NrSpectrumSamples; ++i) {
            const float eta = pathEtaAt(i);
            const float base = microfacet.cosineWeighted
                / outgoingCosine / (eta * eta);
            result.value[i] = transmissionTint[i] * (1.0f - fresnelValues[i])
                * base * scale[i];
        }
        return result;
    }

    NR_CPU_GPU SampledSpectrum glassEnergyScale(
        const float normalView, const float etaPath) const
    {
        SampledSpectrum result(1.0f);
        if (nr::shading::ggx::isAlmostSpecular(roughness)
            || roughness < nr::shading::ggx::MinimumLutRoughness)
            return result;

        const float viewAlbedo = fmaxf(
            nr::shading::energy_lut::glassDirectionalAlbedo(
                energyLuts, roughness, normalView, etaPath), 0.05f);
        const float averageAlbedo = fmaxf(
            nr::shading::energy_lut::glassAverageAlbedo(
                energyLuts, roughness, etaPath), 0.05f);
        const float averageFresnel =
            nr::shading::dielectric::averageFresnel(etaPath);
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float tint = fminf(fmaxf(transmissionTint[i], 0.0f), 1.0f);
            const float multiBounceTint = tint > BsdfEpsilon
                ? tint : averageFresnel;
            const float multipleScatter = multiBounceTint * averageAlbedo
                / fmaxf(1.0f - multiBounceTint * (1.0f - averageAlbedo),
                    nr::shading::ggx::DenominatorEpsilon);
            result[i] = 1.0f
                + multipleScatter * (1.0f - viewAlbedo) / viewAlbedo;
        }
        return result;
    }
};

}
