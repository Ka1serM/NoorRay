#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"
#include "Shading/BsdfClosure.h"
#include "Shading/Dielectric.h"
#include "Shading/EnergyLut.h"
#include "Shading/Ggx.h"
#include "Shading/Sellmeier.h"
#include "Shading/Spectrum.h"

// Unified principled BSDF: handles opaque GGX (metallic/dielectric + diffuse)
// and spectral glass (transmission) in a single class.

#if defined(NR_GPU_CODE)
struct Material;
struct Surface;
struct GpuSceneData;
struct Ray;
#endif

class Bsdf
{
public:
    NR_CPU_GPU Bsdf(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const glm::vec3 view,
        const SampledSpectrum albedo,
        const float metallic,
        const float specular,
        const float roughness,
        const float transmission,
        const SampledSpectrum transmissionColor,
        const SellmeierCoefficients& sellmeier,
        const SampledWavelengths& wavelengths)
        : Bsdf(geometricNormal, shadingNormal, view,
               albedo, metallic, specular, roughness,
               transmission, transmissionColor,
               sellmeier, wavelengths, nullptr)
    {}

    NR_CPU_GPU Bsdf(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const glm::vec3 view,
        const SampledSpectrum albedo,
        const float metallic,
        const float specular,
        const float roughness,
        const float transmission,
        const SampledSpectrum transmissionColor,
        const SellmeierCoefficients& sellmeier,
        const SampledWavelengths& wavelengths,
        const nr::shading::energy_lut::Textures& energyLuts)
        : Bsdf(geometricNormal, shadingNormal, view,
               albedo, metallic, specular, roughness,
               transmission, transmissionColor,
               sellmeier, wavelengths, &energyLuts)
    {}

#if defined(NR_GPU_CODE)
    // Construct from a material hit at a surface.
    NR_GPU Bsdf(
        const ::Material& material,
        const ::Surface& surface,
        const ::GpuSceneData& scene,
        const ::Ray& incident,
        const SampledWavelengths& wavelengths);
#endif

    NR_CPU_GPU BsdfEvaluation evaluate(const glm::vec3 outgoing) const
    {
        if (!accepts(outgoing))
            return {};

        const float normalOutgoing = glm::dot(
            shadingNormal_, outgoing);

        if (normalOutgoing > 0.0f)
        {
            if (transmission_ <= 0.0f)
                return evaluateOpaque(outgoing, normalOutgoing);

            if (transmission_ >= 1.0f)
                return evaluateGlassReflection(outgoing, normalOutgoing);

            const float trans = transmission_;
            const float invTrans = 1.0f - trans;
            BsdfEvaluation result = evaluateOpaque(
                outgoing, normalOutgoing);
            result.value *= invTrans;
            result.pdf *= invTrans;

            const BsdfEvaluation glass = evaluateGlassReflection(
                outgoing, normalOutgoing);
            result.value += glass.value * trans;
            result.pdf += glass.pdf * trans;
            return result;
        }

        if (normalOutgoing < 0.0f && transmission_ > 0.0f)
        {
            BsdfEvaluation result = evaluateGlassTransmission(
                outgoing, normalOutgoing);
            result.value *= transmission_;
            result.pdf *= transmission_;
            return result;
        }
        return {};
    }

    NR_CPU_GPU BsdfSample sample(RandomState& rng) const
    {
        const bool useGlass = transmission_ > 0.0f
            && randomFloat(rng) < transmission_;
        BsdfSample result = useGlass
            ? proposeGlass(rng)
            : proposeOpaque(rng);

        if (result.singular
            || glm::dot(result.direction, result.direction) == 0.0f)
            return result;

        const BsdfEvaluation evaluation = evaluate(result.direction);
        const float outgoingCosine = cosine(result.direction);
        if (evaluation.pdf <= 0.0f || outgoingCosine <= 0.0f)
            return {};

        result.weight = evaluation.value
            * (outgoingCosine / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    NR_CPU_GPU SampledSpectrum evaluateDirect(
        const glm::vec3 outgoing,
        const SampledSpectrum radiance) const
    {
        const BsdfEvaluation evaluation = evaluate(outgoing);
        return evaluation.value * radiance * cosine(outgoing);
    }

    NR_CPU_GPU float cosine(const glm::vec3 direction) const
    {
        return fabsf(glm::dot(shadingNormal_, direction));
    }

    NR_CPU_GPU const glm::vec3& shadingNormal() const
    {
        return shadingNormal_;
    }

    NR_CPU_GPU float roughness() const
    {
        return roughness_;
    }

    NR_CPU_GPU static glm::vec3 clampShadingNormal(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const glm::vec3 view)
    {
        const glm::vec3 reflected = glm::reflect(-view, shadingNormal);
        const float normalView = fmaxf(
            glm::dot(geometricNormal, view), 0.0f);
        const float threshold = fminf(0.9f * normalView, 0.01f);
        if (glm::dot(geometricNormal, reflected) >= threshold)
            return shadingNormal;

        const glm::vec3 rawTangent = geometricNormal * normalView - view;
        const float tangentLengthSquared = glm::dot(rawTangent, rawTangent);
        if (tangentLengthSquared < 1.0e-8f)
            return shadingNormal;
        const glm::vec3 tangent = rawTangent
            * (1.0f / sqrtf(tangentLengthSquared));

        const float thresholdSine = sqrtf(fmaxf(
            1.0f - threshold * threshold, 0.0f));
        const glm::vec3 correctedReflection = geometricNormal * threshold
            + tangent * thresholdSine;
        const glm::vec3 clamped = view + correctedReflection;
        const float clampedLengthSquared = glm::dot(clamped, clamped);
        if (clampedLengthSquared < 1.0e-12f)
            return geometricNormal;
        return clamped * (1.0f / sqrtf(clampedLengthSquared));
    }

private:
    // ── Constructor implementation ──

    NR_CPU_GPU Bsdf(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const glm::vec3 view,
        const SampledSpectrum albedo,
        const float metallic,
        const float specular,
        const float roughness,
        const float transmission,
        const SampledSpectrum transmissionColor,
        const SellmeierCoefficients& sellmeier,
        const SampledWavelengths& wavelengths,
        const nr::shading::energy_lut::Textures* energyLuts)
        : geometricNormal_(glm::dot(geometricNormal, view) < 0.0f
              ? -geometricNormal : geometricNormal),
          shadingNormal_(glm::dot(shadingNormal, view) < 0.0f
              ? -shadingNormal : shadingNormal),
          view_(view),
          exiting_(glm::dot(-view, geometricNormal) > 0.0f),
          transmission_(fminf(fmaxf(transmission, 0.0f), 1.0f)),
          albedo_(albedo),
          metallic_(metallic),
          roughness_(fminf(fmaxf(roughness, 0.0f), 1.0f)),
          dielectricF0_(fminf(fmaxf(0.08f * specular, 0.0f), 1.0f)),
          energyLuts_(energyLuts),
          transmissionColor_(transmissionColor)
    {
        shadingNormal_ = clampShadingNormal(
            geometricNormal_, shadingNormal_, view_);
        if (transmission_ < 1.0f)
            initOpaque();
        if (transmission_ > 0.0f)
            initGlass(sellmeier, wavelengths);
    }

    // ── Opaque GGX initialisation ──

    NR_CPU_GPU void initOpaque()
    {
        const float viewCosine = fmaxf(glm::dot(
            shadingNormal_, view_), 0.0f);

        energy_ = makeEnergyContext(
            energyLuts_, roughness_, viewCosine, dielectricF0_);
        specularProbability_ = makeSpecularProbability();
    }

    NR_CPU_GPU float makeSpecularProbability() const
    {
        if (metallic_ >= 1.0f - BsdfEpsilon)
            return 1.0f;

        const float cosine = fabsf(glm::dot(
            shadingNormal_, view_));
        float maximumFresnel = 0.0f;
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float f0 = dielectricF0_
                + (albedo_[i] - dielectricF0_) * metallic_;
            maximumFresnel = fmaxf(maximumFresnel,
                nr::shading::dielectric::fresnelFromNormalReflectance(
                    cosine, f0));
        }

        const float diffuseWeight = (1.0f - metallic_)
            * albedo_.maxComponent() * (1.0f - maximumFresnel);
        const float totalWeight = maximumFresnel + diffuseWeight;
        if (totalWeight <= 0.0f)
            return metallic_ > 0.0f ? 1.0f : 0.0f;
        return fminf(fmaxf(maximumFresnel / totalWeight, 0.0f), 1.0f);
    }

    // ── Glass initialisation ──

    NR_CPU_GPU void initGlass(
        const SellmeierCoefficients& sellmeier,
        const SampledWavelengths& wavelengths)
    {
        for (int i = 0; i < NrSpectrumSamples; ++i)
            iors_[i] = sellmeierIor(sellmeier, wavelengths[i]);
        const float incident = exiting_
            ? fmaxf(iors_[0], 1.0f) : 1.0f;
        const float transmitted = exiting_
            ? 1.0f : fmaxf(iors_[0], 1.0f);
        etaPath_ = transmitted / incident;
        makeEnergyScale();
    }

    NR_CPU_GPU void makeEnergyScale()
    {
        energyScale_ = SampledSpectrum(1.0f);
        if (nr::shading::ggx::isAlmostSpecular(roughness_))
            return;

        const float viewCosine = fminf(fmaxf(
            fabsf(glm::dot(shadingNormal_, view_)),
            0.0f), 1.0f);
        const bool smoothLut = roughness_
            < nr::shading::ggx::MinimumLutRoughness;
        const float eps = nr::shading::ggx::DenominatorEpsilon;

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float relativeIor = exiting_
                ? 1.0f / fmaxf(iors_[i], BsdfEpsilon) : iors_[i];

            const float viewAlbedo = smoothLut
                ? 1.0f
                : nr::shading::energy_lut::glassDirectionalAlbedo(
                    energyLuts_, roughness_, viewCosine, relativeIor);
            const float averageAlbedo = smoothLut
                ? 1.0f
                : nr::shading::energy_lut::glassAverageAlbedo(
                    energyLuts_, roughness_, relativeIor);

        const float v = fminf(fmaxf(viewAlbedo, 0.05f), 1.0f);
        const float average = fminf(fmaxf(averageAlbedo, 0.05f), 1.0f);
            const float tint = fminf(fmaxf(transmissionColor_[i], 0.0f), 1.0f);
            const float averageFresnel =
                nr::shading::dielectric::averageFresnel(relativeIor);
            const float multiBounceTint = tint > eps ? tint : averageFresnel;
            const float denom = fmaxf(
                1.0f - multiBounceTint * (1.0f - average), eps);
            const float multipleScatter = multiBounceTint * average / denom;
            energyScale_[i] = 1.0f
                + multipleScatter * (1.0f - v) / v;
        }
    }

    // ── Evaluate helpers ──

    NR_CPU_GPU BsdfEvaluation evaluateOpaque(
        const glm::vec3 outgoing,
        const float normalOutgoing) const
    {
        BsdfEvaluation result;

        const float diffusePdf = normalOutgoing / BsdfPi;
        if (metallic_ <= 0.0f && dielectricF0_ <= 0.0f)
        {
            result.value = albedo_ * (1.0f / BsdfPi);
            result.pdf = diffusePdf;
            return result;
        }

        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateReflection(
                shadingNormal_, view_, outgoing, roughness_);
        result.pdf = specularProbability_ * microfacet.pdf
            + (1.0f - specularProbability_) * diffusePdf;

        const float geometryFactor = microfacet.cosineWeighted
            / normalOutgoing;
        const float lightAlbedo = directionalAlbedo(
            energyLuts_, normalOutgoing, roughness_);
        const float dielectricLightReflectance =
            dielectricDirectionalReflectance(
                energyLuts_, normalOutgoing, roughness_, dielectricF0_,
                energy_.averageAlbedo);

        const float diffuseEnergy =
            (1.0f - energy_.dielectricViewReflectance)
            * (1.0f - dielectricLightReflectance)
            / fmaxf(1.0f - energy_.dielectricAverageReflectance,
                BsdfEpsilon);

        const float dielectricFresnel = nr::shading::dielectric::fresnel(
            microfacet.viewHalfCosine, 1.0f,
            nr::shading::dielectric::iorFromNormalReflectance(dielectricF0_));

        const float missingAverage = 1.0f - energy_.averageAlbedo;
        const float msScale = missingAverage
                > nr::shading::ggx::DenominatorEpsilon
            ? (1.0f - energy_.viewAlbedo) * (1.0f - lightAlbedo)
                / (BsdfPi * missingAverage)
            : 0.0f;

        const float dielectricAvg = nr::shading::dielectric::averageFresnel(
            nr::shading::dielectric::iorFromNormalReflectance(dielectricF0_));

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float fresnel = nr::shading::ggx::fresnelBlend(
                dielectricFresnel, microfacet.viewHalfCosine,
                albedo_[i], metallic_);
            const float conductorAvg = nr::shading::dielectric::averageFresnel(
                nr::shading::dielectric::iorFromNormalReflectance(albedo_[i]));
            const float ms = msScale
                * multipleScatterFresnel(
                    dielectricAvg + (conductorAvg - dielectricAvg) * metallic_,
                    energy_.averageAlbedo);
            const float diffuse = diffuseEnergy * (1.0f - metallic_)
                * albedo_[i] / BsdfPi;
            result.value[i] = diffuse + fresnel * geometryFactor + ms;
        }
        return result;
    }

    NR_CPU_GPU BsdfEvaluation evaluateGlassReflection(
        const glm::vec3 outgoing,
        const float normalOutgoing) const
    {
        BsdfEvaluation result;
        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateReflection(
                shadingNormal_, view_, outgoing, roughness_);
        if (microfacet.cosineWeighted <= 0.0f)
            return result;

        const float base = microfacet.cosineWeighted / normalOutgoing;
        const float heroFresnel = nr::shading::dielectric::fresnel(
            microfacet.viewHalfCosine, 1.0f, etaPath_);
        result.pdf = heroFresnel * microfacet.pdf;

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float spectralFresnel =
                nr::shading::dielectric::fresnel(
                    microfacet.viewHalfCosine,
                    exiting_ ? iors_[i] : 1.0f,
                    exiting_ ? 1.0f : iors_[i]);
            result.value[i] = spectralFresnel * base * energyScale_[i];
        }
        return result;
    }

    NR_CPU_GPU BsdfEvaluation evaluateGlassTransmission(
        const glm::vec3 outgoing,
        const float normalOutgoing) const
    {
        BsdfEvaluation result;
        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateTransmission(
                shadingNormal_, view_, outgoing, roughness_, etaPath_);
        const float outgoingCosine = -normalOutgoing;
        if (microfacet.cosineWeighted <= 0.0f
            || outgoingCosine <= 0.0f)
            return result;

        const float base = microfacet.cosineWeighted
            / outgoingCosine / (etaPath_ * etaPath_);
        const float heroFresnel = nr::shading::dielectric::fresnel(
            microfacet.viewHalfCosine, 1.0f, etaPath_);
        result.pdf = (1.0f - heroFresnel) * microfacet.pdf;

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float spectralFresnel =
                nr::shading::dielectric::fresnel(
                    microfacet.viewHalfCosine,
                    exiting_ ? iors_[i] : 1.0f,
                    exiting_ ? 1.0f : iors_[i]);
            result.value[i] = transmissionColor_[i]
                * (1.0f - spectralFresnel) * base * energyScale_[i];
        }
        return result;
    }

    // ── Energy compensation helpers ──

    struct EnergyContext
    {
        float viewAlbedo{1.0f};
        float averageAlbedo{1.0f};
        float dielectricViewReflectance{};
        float dielectricAverageReflectance{};
    };

    NR_CPU_GPU static float directionalAlbedo(
        const nr::shading::energy_lut::Textures* luts,
        const float cosine,
        const float roughness)
    {
        if (roughness < nr::shading::ggx::MinimumLutRoughness)
            return 1.0f;
        return nr::shading::energy_lut::ggxDirectionalAlbedo(
            luts, roughness, cosine);
    }

    NR_CPU_GPU static float averageDirectionalAlbedo(
        const nr::shading::energy_lut::Textures* luts,
        const float roughness)
    {
        if (roughness < nr::shading::ggx::MinimumLutRoughness)
            return 1.0f;
        return nr::shading::energy_lut::ggxAverageAlbedo(luts, roughness);
    }

    NR_CPU_GPU static float dielectricSingleScatterDirectionalAlbedo(
        const nr::shading::energy_lut::Textures* luts,
        const float cosine,
        const float roughness,
        const float normalReflectance)
    {
        const float f0 = fminf(fmaxf(normalReflectance, 0.0f),
            nr::shading::energy_lut::MaximumDielectricF0);
        if (f0 <= 0.0f)
            return 0.0f;
        if (roughness < nr::shading::ggx::MinimumLutRoughness)
            return nr::shading::dielectric::fresnelFromNormalReflectance(
                cosine, f0);
        return nr::shading::energy_lut::dielectricDirectionalAlbedo(
            luts, roughness, cosine, f0);
    }

    NR_CPU_GPU static float dielectricSingleScatterAverageAlbedo(
        const nr::shading::energy_lut::Textures* luts,
        const float roughness,
        const float normalReflectance)
    {
        const float f0 = fminf(fmaxf(normalReflectance, 0.0f),
            nr::shading::energy_lut::MaximumDielectricF0);
        if (f0 <= 0.0f)
            return 0.0f;
        if (roughness < nr::shading::ggx::MinimumLutRoughness)
            return nr::shading::dielectric::averageFresnel(
                nr::shading::dielectric::iorFromNormalReflectance(f0));
        return nr::shading::energy_lut::dielectricAverageAlbedo(
            luts, roughness, f0);
    }

    NR_CPU_GPU static float multipleScatterFresnel(
        const float averageFresnel,
        const float averageAlbedo)
    {
        const float fresnel = fminf(fmaxf(averageFresnel, 0.0f), 1.0f);
        const float albedo = fminf(fmaxf(averageAlbedo, 0.0f), 1.0f);
        return fresnel * fresnel * albedo
            / fmaxf(1.0f - fresnel * (1.0f - albedo),
                nr::shading::ggx::DenominatorEpsilon);
    }

    NR_CPU_GPU static float dielectricDirectionalReflectance(
        const nr::shading::energy_lut::Textures* luts,
        const float cosine,
        const float roughness,
        const float normalReflectance,
        const float averageBaseAlbedo)
    {
        if (roughness < nr::shading::ggx::MinimumLutRoughness)
            return nr::shading::dielectric::fresnelFromNormalReflectance(
                cosine, normalReflectance);

        const float baseAlbedo = directionalAlbedo(luts, cosine, roughness);
        const float singleScatter = dielectricSingleScatterDirectionalAlbedo(
            luts, cosine, roughness, normalReflectance);
        const float averageFresnel = nr::shading::dielectric::averageFresnel(
            nr::shading::dielectric::iorFromNormalReflectance(
                normalReflectance));
        return fminf(fmaxf(singleScatter
            + multipleScatterFresnel(averageFresnel, averageBaseAlbedo)
                * (1.0f - baseAlbedo), 0.0f), 1.0f);
    }

    NR_CPU_GPU static EnergyContext makeEnergyContext(
        const nr::shading::energy_lut::Textures* luts,
        const float roughness,
        const float viewCosine,
        const float dielectricNormalReflectance)
    {
        EnergyContext result;
        result.viewAlbedo = directionalAlbedo(luts, viewCosine, roughness);
        result.averageAlbedo = averageDirectionalAlbedo(luts, roughness);
        result.dielectricViewReflectance = dielectricDirectionalReflectance(
            luts, viewCosine, roughness, dielectricNormalReflectance,
            result.averageAlbedo);
        const float averageFresnel = nr::shading::dielectric::averageFresnel(
            nr::shading::dielectric::iorFromNormalReflectance(
                dielectricNormalReflectance));
        result.dielectricAverageReflectance =
            dielectricSingleScatterAverageAlbedo(
                luts, roughness, dielectricNormalReflectance)
            + multipleScatterFresnel(averageFresnel, result.averageAlbedo)
                * (1.0f - result.averageAlbedo);
        result.dielectricAverageReflectance = fminf(fmaxf(
            result.dielectricAverageReflectance, 0.0f), 1.0f);
        return result;
    }

    // ── Sample helpers ──

    NR_CPU_GPU static glm::vec3 sampleDiffuseDirection(
        const glm::vec3 normal, RandomState& rng)
    {
        return nr::sampling::cosineHemisphere(normal,
            glm::vec2(randomFloat(rng), randomFloat(rng)));
    }

    NR_CPU_GPU BsdfSample proposeOpaque(RandomState& rng) const
    {
        BsdfSample result;
        const float normalView = fmaxf(
            glm::dot(shadingNormal_, view_), 0.0f);
        if (normalView <= 0.0f)
            return result;

        if (metallic_ <= 0.0f && dielectricF0_ <= 0.0f)
        {
            result.event = BsdfEvent::Diffuse;
            result.direction = sampleDiffuseDirection(
                shadingNormal_, rng);
            return acceptsReflection(result.direction)
                ? result : BsdfSample{};
        }

        if (randomFloat(rng) < specularProbability_)
        {
            result.event = BsdfEvent::Specular;
            const bool smooth = nr::shading::ggx::isAlmostSpecular(
                roughness_);
            if (smooth)
            {
                result.singular = true;
                result.direction = glm::reflect(
                    -view_, shadingNormal_);
                const float dielectricFresnel =
                    nr::shading::dielectric::fresnel(
                        normalView, 1.0f,
                        nr::shading::dielectric::iorFromNormalReflectance(
                            dielectricF0_));
                for (int i = 0; i < NrSpectrumSamples; ++i)
                    result.weight[i] = nr::shading::ggx::fresnelBlend(
                        dielectricFresnel, normalView,
                        albedo_[i], metallic_) / specularProbability_;
                result.pdf = specularProbability_ * nr::shading::ggx::SingularPdf;
            }
            else
            {
                const glm::vec3 halfVector =
                    nr::shading::ggx::sampleVisibleNormal(
                        shadingNormal_, view_,
                        roughness_,
                        glm::vec2(randomFloat(rng), randomFloat(rng)));
                result.direction = glm::reflect(
                    -view_, halfVector);
            }
        }
        else
        {
            result.event = BsdfEvent::Diffuse;
            result.direction = sampleDiffuseDirection(
                shadingNormal_, rng);
        }

        return acceptsReflection(result.direction)
            ? result : BsdfSample{};
    }

    NR_CPU_GPU BsdfSample proposeGlass(RandomState& rng) const
    {
        const float normalView = glm::dot(
            shadingNormal_, view_);
        if (normalView <= 0.0f)
            return {};

        const glm::vec3 incident = -view_;
        const bool smooth = nr::shading::ggx::isAlmostSpecular(roughness_);

        const glm::vec3 halfVector = smooth
            ? shadingNormal_
            : nr::shading::ggx::sampleVisibleNormal(
                shadingNormal_, view_, roughness_,
                glm::vec2(randomFloat(rng), randomFloat(rng)));
        const float viewHalfCosine = fabsf(
            glm::dot(view_, halfVector));
        if (viewHalfCosine <= 0.0f)
            return {};

        const float heroFresnel = nr::shading::dielectric::fresnel(
            viewHalfCosine, 1.0f, etaPath_);
        const glm::vec3 refracted = glm::refract(
            incident, halfVector, 1.0f / etaPath_);
        const bool totalInternalReflection =
            glm::dot(refracted, refracted) < 1.0e-10f;

        if (totalInternalReflection || randomFloat(rng) < heroFresnel)
        {
            if (smooth)
                return makeSingularReflection(halfVector,
                    viewHalfCosine, heroFresnel,
                    totalInternalReflection ? 1.0f : heroFresnel);

            BsdfSample result;
            result.event = BsdfEvent::Specular;
            result.direction = glm::reflect(-view_, halfVector);
            return acceptsReflection(result.direction)
                ? result : BsdfSample{};
        }

        const glm::vec3 direction = glm::normalize(refracted);
        if (!acceptsTransmission(direction))
            return {};

        if (smooth || fabsf(etaPath_ - 1.0f) < 1.0e-4f)
            return makeSingularTransmission(
                direction, heroFresnel);

        BsdfSample result;
        result.event = BsdfEvent::Transmission;
        result.direction = direction;
        result.eta = etaPath_;
        return result;
    }

    NR_CPU_GPU BsdfSample makeSingularReflection(
        const glm::vec3 halfVector,
        const float viewHalfCosine,
        const float heroFresnel,
        const float branchProbability) const
    {
        BsdfSample result;
        result.event = BsdfEvent::Specular;
        result.singular = true;
        result.direction = glm::reflect(-view_, halfVector);
        if (!acceptsReflection(result.direction))
            return {};
        result.pdf = branchProbability * nr::shading::ggx::SingularPdf;
        const float invHeroFresnel = 1.0f / fmaxf(heroFresnel, BsdfEpsilon);
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float spectralFresnel =
                nr::shading::dielectric::fresnel(
                    viewHalfCosine,
                    exiting_ ? iors_[i] : 1.0f,
                    exiting_ ? 1.0f : iors_[i]);
            result.weight[i] = spectralFresnel * invHeroFresnel;
        }
        return result;
    }

    NR_CPU_GPU BsdfSample makeSingularTransmission(
        const glm::vec3 direction,
        const float heroFresnel) const
    {
        BsdfSample result;
        result.event = BsdfEvent::Transmission;
        result.singular = true;
        result.direction = direction;
        if (!acceptsTransmission(result.direction))
            return {};
        result.eta = etaPath_;
        result.pdf = (1.0f - heroFresnel) * nr::shading::ggx::SingularPdf;
        result.weight = transmissionColor_
            * (1.0f / (etaPath_ * etaPath_));
        return result;
    }

    // ── Helpers ──

    NR_CPU_GPU bool accepts(const glm::vec3 direction) const
    {
        const float geometricCosine = glm::dot(geometricNormal_, direction);
        const float shadingCosine = glm::dot(shadingNormal_, direction);
        return geometricCosine != 0.0f && shadingCosine != 0.0f
            && (geometricCosine > 0.0f) == (shadingCosine > 0.0f);
    }

    NR_CPU_GPU bool acceptsReflection(const glm::vec3 direction) const
    {
        return glm::dot(geometricNormal_, direction) > 0.0f
            && glm::dot(shadingNormal_, direction) > 0.0f;
    }

    NR_CPU_GPU bool acceptsTransmission(const glm::vec3 direction) const
    {
        return glm::dot(geometricNormal_, direction) < 0.0f
            && glm::dot(shadingNormal_, direction) < 0.0f;
    }

    // ── State ──

    glm::vec3 geometricNormal_{};
    glm::vec3 shadingNormal_{};
    glm::vec3 view_{};
    bool exiting_{};
    float transmission_{};

    // Opaque / GGX state
    SampledSpectrum albedo_{};
    float metallic_{};
    float roughness_{};
    float dielectricF0_{};
    const nr::shading::energy_lut::Textures* energyLuts_{};
    EnergyContext energy_{};
    float specularProbability_{};

    // Glass state
    SampledSpectrum transmissionColor_{};
    SampledSpectrum iors_{};
    SampledSpectrum energyScale_{1.0f};
    float etaPath_{1.0f};
};

#if defined(NR_GPU_CODE)

#include "Raytracing/Gpu/Geometry.h"
#include "Raytracing/Gpu/SceneData.h"
#include "Shading/Material.h"
#include "Shading/RgbToSpectrum.h"

NR_GPU inline Bsdf::Bsdf(
    const Material& material,
    const Surface& surface,
    const GpuSceneData& scene,
    const Ray& incident,
    const SampledWavelengths& wavelengths)
{
    glm::vec3 shadingNormal = material.shadingNormalAt(
        scene.textures, surface.uv, surface.tangent, surface.normal);
    if (glm::dot(shadingNormal, incident.direction()) > 0.0f)
        shadingNormal = -shadingNormal;
    const glm::vec3 viewDirection = -incident.direction();

    const SampledSpectrum albedoSpec = material.sampleDiffuse(
        scene.textures, surface.uv, wavelengths,
        scene.spectrumTableScale, scene.spectrumTableCoeffs,
        glm::vec3(surface.color));
    const float met = material.sampleMetallic(scene.textures, surface.uv);
    const float rough = material.sampleRoughness(scene.textures, surface.uv);
    const float trans = material.sampleTransmission(scene.textures, surface.uv);
    const float spec = material.sampleSpecular(scene.textures, surface.uv);
    SampledSpectrum transmissionColorSpec{};
    if (trans > 0.0f)
        transmissionColorSpec = rgbAlbedoToSpectrum(
            material.transmissionColor, wavelengths,
            scene.spectrumTableScale, scene.spectrumTableCoeffs);

    *this = Bsdf(surface.geometricNormal, shadingNormal, viewDirection,
        albedoSpec, met, spec, rough, trans,
        transmissionColorSpec, material.sellmeier, wavelengths,
        scene.energyLuts);
}

#endif
