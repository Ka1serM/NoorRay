#pragma once

#include <cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Mesh/OpenPbrEnergy.h"
#include "Raytracing/Sellmeier.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

static constexpr float BsdfPi = 3.14159265358979323846f;
static constexpr float BsdfEpsilon = 1e-5f;

static constexpr float BsdfSmoothAlphaThreshold = 1e-3f;
static constexpr float BsdfDeltaPdf = 1.0e8f;

enum class BsdfEvent : uint32_t
{
    Diffuse,
    Specular,
    Transmission,
};

struct BsdfSample
{
    glm::vec3 direction{};
    SampledSpectrum weight{};
    float pdf{};
    float eta{1.0f};
    BsdfEvent event{BsdfEvent::Diffuse};
};

// Spectral GGX/Lambert BSDF for a single surface interaction. Holds only the
// state shared across evaluate()/sample()/pdf() (surface geometry and the
// already texture-resolved material parameters); anything only one method
// needs (rng, the light direction) stays a method parameter.
class Bsdf
{
public:
    NR_CPU_GPU Bsdf(
        const glm::vec3& geometricNormal,
        const glm::vec3& shadingNormal,
        const glm::vec3& view,
        const SampledSpectrum& albedo,
        const float metallic,
        const float specular,
        const float roughness,
        const float transmission,
        const SampledSpectrum& transmissionColor,
        const SellmeierCoefficients& sellmeier,
        const SampledWavelengths& wavelengths,
        const nr::openpbr::EnergyLutTextures& openPbrLuts)
        : geometricNormal(geometricNormal),
          shadingNormal(shadingNormal),
          view(view),
          albedo(albedo),
          metallic(metallic),
          specular(specular),
          // Material resolution handles the physical [0,1] range. The BSDF owns only its
          // numerical GGX floor while preserving exact zero for delta reflection.
          roughness(roughness == 0.0f ? 0.0f : fmaxf(roughness, 0.045f)),
          transmission(transmission),
          transmissionColor(transmissionColor),
          sellmeier(sellmeier),
          wavelengths(wavelengths),
          openPbrLuts(openPbrLuts)
    {}

    // Evaluates the BSDF for an explicit light direction (used for direct
    // light sampling / MIS). The sampled direction from sample() does not go
    // through here — its weight already carries the evaluate()/pdf() ratio.
    NR_CPU_GPU SampledSpectrum evaluate(const glm::vec3& light) const
    {
        const SampledSpectrum opaque = evaluateOpaqueSpectral(
            shadingNormal, view, light, albedo, metallic, specular, roughness, openPbrLuts);
        const float normalCorrection = shadingNormalCorrection(
            geometricNormal, shadingNormal, view, light);
        if (transmission <= 0.0f)
            return opaque * normalCorrection;
        if (glm::dot(shadingNormal, view) <= 0.0f || glm::dot(shadingNormal, light) <= 0.0f)
            return opaque * ((1.0f - transmission) * normalCorrection);

        const glm::vec3 halfVector = glm::normalize(view + light);
        const float D = distributionGgx(shadingNormal, halfVector, roughness);
        const float G = geometrySmith(shadingNormal, view, light, roughness);
        const float ndv = fmaxf(glm::dot(shadingNormal, view), 0.0f);
        const float ndl = fmaxf(glm::dot(shadingNormal, light), 0.0f);
        SampledSpectrum result = opaque * (1.0f - transmission);
        const float geometry = D * G / fmaxf(4.0f * ndv * ndl, BsdfEpsilon);
        const float cosTheta = fmaxf(glm::dot(view, halfVector), 0.0f);
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result[i] += fresnelDielectric(cosTheta, 1.0f, sellmeierIor(sellmeier, wavelengths[i]))
                * geometry * transmission;
        return result * normalCorrection;
    }

    // Importance-samples an outgoing direction and returns its throughput
    // weight (already divided by pdf).
    NR_CPU_GPU BsdfSample sample(RandomState& rng) const
    {
        BsdfSample result{};

        glm::vec3 sampleNormal = shadingNormal;
        if (glm::dot(sampleNormal, view) < 0.0f)
            sampleNormal = -sampleNormal;

        if (randomFloat(rng) < transmission)
        {
            // The hero wavelength selects the dispersive refraction direction.
            const float heroIor = sellmeierIor(sellmeier, wavelengths[0]);
            const BsdfSample dielectric = sampleDielectric(
                view, geometricNormal, sampleNormal, roughness, heroIor, rng);
            result.direction = dielectric.direction;
            result.event = dielectric.event;
            result.pdf = dielectric.pdf;
            result.eta = dielectric.eta;

            const glm::vec3 reflectionHalf = dielectric.event == BsdfEvent::Specular
                ? glm::normalize(view + dielectric.direction) : glm::vec3(0.0f);
            const float cosTheta = dielectric.event == BsdfEvent::Specular
                ? fabsf(glm::dot(view, reflectionHalf)) : 0.0f;
            const bool exitingDielectric = glm::dot(-view, geometricNormal) > 0.0f;
            const float heroFresnel = dielectric.event == BsdfEvent::Specular
                ? fmaxf(fresnelDielectric(cosTheta,
                    exitingDielectric ? heroIor : 1.0f,
                    exitingDielectric ? 1.0f : heroIor), BsdfEpsilon)
                : 1.0f;
            for (int i = 0; i < NrSpectrumSamples; ++i)
            {
                const float wavelengthIor = sellmeierIor(sellmeier, wavelengths[i]);
                result.weight[i] = dielectric.event == BsdfEvent::Transmission
                    ? transmissionColor[i] * dielectric.weight[0]
                    : dielectric.weight[0] * fresnelDielectric(
                        cosTheta,
                        exitingDielectric ? wavelengthIor : 1.0f,
                        exitingDielectric ? 1.0f : wavelengthIor) / heroFresnel;
            }
        }
        else if (metallic <= 0.0f && specular <= 0.0f)
        {
            result.event = BsdfEvent::Diffuse;
            result.direction = sampleDiffuseDirection(sampleNormal, rng);
            const float ndl = fmaxf(glm::dot(sampleNormal, result.direction), 0.0f);
            if (ndl <= 0.0f)
                return result;
            result.pdf = ndl / BsdfPi;
            result.weight = albedo * shadingNormalCorrection(
                geometricNormal, sampleNormal, view, result.direction);
        }
        else
        {
            // A perfectly smooth, fully metallic surface is a single delta lobe. Do not run it
            // through the diffuse/specular mixture or the GGX PDF: the continuous GGX density is
            // undefined in this limit, and selecting the (zero-energy) diffuse lobe produces dark
            // highlight samples. Its throughput is simply conductor Fresnel.
            if (metallic >= 1.0f - BsdfEpsilon && isSmoothAlpha(roughness))
            {
                const float ndv = fmaxf(glm::dot(sampleNormal, view), 0.0f);
                if (ndv <= 0.0f)
                    return result;
                result.event = BsdfEvent::Specular;
                result.direction = glm::reflect(-view, sampleNormal);
                result.pdf = 0.0f; // delta distribution: no finite solid-angle PDF
                for (int i = 0; i < NrSpectrumSamples; ++i)
                    result.weight[i] = fresnelConductor(ndv, albedo[i]);
                result.weight *= shadingNormalCorrection(
                    geometricNormal, sampleNormal, view, result.direction);
                return result;
            }

            const float specProb = specularSamplingProbability();

            glm::vec3 halfVector{};
            if (randomFloat(rng) < specProb)
            {
                result.event = BsdfEvent::Specular;
                if (isSmoothAlpha(roughness))
                {
                    const float ndv = fmaxf(glm::dot(sampleNormal, view), 0.0f);
                    if (ndv <= 0.0f)
                        return result;
                    result.direction = glm::reflect(-view, sampleNormal);
                    const float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);
                    const float dielectricIor = f0ToIor(dielectricF0);
                    const float dielectricF = fresnelDielectric(ndv, 1.0f, dielectricIor);
                    for (int i = 0; i < NrSpectrumSamples; ++i)
                    {
                        const float conductorF = fresnelConductor(ndv, albedo[i]);
                        const float F = dielectricF + (conductorF - dielectricF) * metallic;
                        result.weight[i] = F / specProb;
                    }
                    result.pdf = BsdfDeltaPdf;
                    result.weight *= shadingNormalCorrection(
                        geometricNormal, sampleNormal, view, result.direction);
                    return result;
                }
                halfVector = sampleGgxHalfVector(view, sampleNormal, roughness, rng);
                result.direction = glm::reflect(-view, halfVector);
            }
            else
            {
                result.event = BsdfEvent::Diffuse;
                result.direction = sampleDiffuseDirection(sampleNormal, rng);
                halfVector = glm::normalize(view + result.direction);
            }

            const float ndl = fmaxf(glm::dot(sampleNormal, result.direction), 0.0f);
            if (ndl <= 0.0f)
                return result;

            const SampledSpectrum brdfSpec = evaluateOpaqueSpectral(
                sampleNormal, view, result.direction, albedo, metallic, specular, roughness, openPbrLuts);
            const float specPdf = pdfGgxReflection(view, sampleNormal, halfVector, roughness);
            const float diffusePdf = ndl / BsdfPi;
            const float pdf = fmaxf(
                specProb * specPdf + (1.0f - specProb) * diffusePdf, BsdfEpsilon);
            result.pdf = pdf;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                result.weight[i] = brdfSpec[i] * (ndl / pdf);
            result.weight *= shadingNormalCorrection(
                geometricNormal, sampleNormal, view, result.direction);
        }
        return result;
    }

    // PDF of sample() having produced `light` via the opaque (diffuse+specular)
    // lobe. Zero for any transmissive surface, matching sample()'s exclusive
    // transmission/opaque branch — direct light sampling never targets the
    // dielectric lobe.
    NR_CPU_GPU float pdf(const glm::vec3& light) const
    {
        if (transmission > 0.0f)
            return 0.0f;
        if (metallic >= 1.0f - BsdfEpsilon && isSmoothAlpha(roughness))
            return 0.0f;
        const float ndl = fmaxf(glm::dot(shadingNormal, light), 0.0f);
        if (ndl <= 0.0f)
            return 0.0f;
        glm::vec3 orientedGeometricNormal = geometricNormal;
        if (glm::dot(orientedGeometricNormal, view) < 0.0f)
            orientedGeometricNormal = -orientedGeometricNormal;
        if (glm::dot(orientedGeometricNormal, light) <= 0.0f)
            return 0.0f;
        const glm::vec3 halfVector = glm::normalize(view + light);
        const float specularPdf = pdfGgxReflection(view, shadingNormal, halfVector, roughness);
        const float diffusePdf = ndl / BsdfPi;
        const float specProb = specularSamplingProbability();
        return specProb * specularPdf + (1.0f - specProb) * diffusePdf;
    }

    // Rotates a shading normal minimally toward the geometric normal so that
    // reflecting `view` about it does not punch below the geometric
    // hemisphere. This is the "ensure valid reflection" technique (as used in
    // Blender Cycles) that fixes energy loss from smooth-normal divergence
    // near silhouettes of coarsely tessellated meshes, complementing (not
    // replacing) shadingNormalCorrection's hard zero cutoff below.
    // geometricNormal/shadingNormal must already satisfy dot(N, view) >= 0.
    // Static and public: callers correct the shading normal before a Bsdf
    // for the (corrected) surface exists.
    NR_CPU_GPU static glm::vec3 clampShadingNormal(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const glm::vec3 view)
    {
        const glm::vec3 reflected = glm::reflect(-view, shadingNormal);
        const float ndv = fmaxf(glm::dot(geometricNormal, view), 0.0f);
        const float threshold = fminf(0.9f * ndv, 0.01f);
        if (glm::dot(geometricNormal, reflected) >= threshold)
            return shadingNormal;

        const glm::vec3 rawTangent = geometricNormal * ndv - view;
        const float tangentLenSq = glm::dot(rawTangent, rawTangent);
        if (tangentLenSq < 1e-8f)
            return shadingNormal;
        const glm::vec3 tangent = rawTangent * (1.0f / sqrtf(tangentLenSq));

        const float sinThreshold = sqrtf(fmaxf(1.0f - threshold * threshold, 0.0f));
        const glm::vec3 correctedReflection = geometricNormal * threshold + tangent * sinThreshold;
        const glm::vec3 clamped = view + correctedReflection;
        const float clampedLenSq = glm::dot(clamped, clamped);
        if (clampedLenSq < 1e-12f)
            return geometricNormal;
        return clamped * (1.0f / sqrtf(clampedLenSq));
    }

    // Per-wavelength spectral BSDF evaluation of the opaque (diffuse +
    // dielectric/conductor GGX) lobe only, with no transmission mixing or
    // shading-normal correction. Exposed (public static, pure function of its
    // arguments) so CPU-side tests can exercise it directly — e.g. furnace and
    // PDF-normalization tests — without constructing a full Bsdf.
    NR_CPU_GPU static SampledSpectrum evaluateOpaqueSpectral(
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light,
        const SampledSpectrum& albedo,
        const float metallic,
        const float specular,
        const float roughness,
        const nr::openpbr::EnergyLutTextures& openPbrLuts = {})
    {
        const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
        const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
        if (ndv <= 0.0f || ndl <= 0.0f)
            return SampledSpectrum(0.0f);
        if (metallic <= 0.0f && specular <= 0.0f)
            return albedo * (1.0f / BsdfPi);
        const glm::vec3 halfVector = glm::normalize(view + light);
        const float vdh = fmaxf(glm::dot(view, halfVector), 0.0f);
        const float D = distributionGgx(normal, halfVector, roughness);
        const float G = geometrySmith(normal, view, light, roughness);
        const float geomFactor = D * G / fmaxf(4.0f * ndv * ndl, BsdfEpsilon);
        const float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);
        const float sqrtF0 = sqrtf(dielectricF0);
        const float dielectricIor = (1.0f + sqrtF0)
            / fmaxf(1.0f - sqrtF0, BsdfEpsilon);
        const float alpha = roughness * roughness;
        const float availableView = nr::openpbr::opaqueDielectricEnergyComplement(
            openPbrLuts, dielectricIor, alpha, ndv);
        const float availableLight = nr::openpbr::opaqueDielectricEnergyComplement(
            openPbrLuts, dielectricIor, alpha, ndl);
        const float averageAvailable = nr::openpbr::opaqueDielectricAverageEnergyComplement(
            openPbrLuts, dielectricIor, alpha);
        const float diffuseEnergy = availableView * availableLight
            / fmaxf(averageAvailable, BsdfEpsilon);
        const float idealMissingView = nr::openpbr::idealDielectricEnergyComplement(
            openPbrLuts, dielectricIor, alpha, ndv);
        const float idealMissingLight = nr::openpbr::idealDielectricEnergyComplement(
            openPbrLuts, dielectricIor, alpha, ndl);
        const float idealAverageMissing = nr::openpbr::idealDielectricAverageEnergyComplement(
            openPbrLuts, dielectricIor, alpha);
        const float dielectricReflectionRatio = nr::openpbr::idealDielectricReflectionRatio(
            openPbrLuts, dielectricIor, alpha);
        const float dielectricMultipleScatter = idealMissingView * idealMissingLight
            * dielectricReflectionRatio
            / fmaxf(idealAverageMissing * BsdfPi, BsdfEpsilon);

        SampledSpectrum result;
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float dielectricF = fresnelDielectric(vdh, 1.0f, dielectricIor);
            const float conductorF = fresnelConductor(vdh, albedo[i]);
            const float F = dielectricF + (conductorF - dielectricF) * metallic;
            const float specBrdf = F * geomFactor;
            // OpenPBR's reciprocal layered compensation accounts for light
            // passing through the rough dielectric interface in both
            // directions. Its tabulated hemispherical average normalizes the
            // layer so a white substrate passes the furnace test.
            const float diffuse = diffuseEnergy * (1.0f - metallic)
                * albedo[i] / BsdfPi;
            result[i] = diffuse + specBrdf
                + (1.0f - metallic) * dielectricMultipleScatter;
        }
        return result;
    }

    // GGX-VNDF reflection-lobe PDF. Exposed for the same reason as
    // evaluateOpaqueSpectral above (PDF-normalization tests).
    NR_CPU_GPU static float pdfGgxReflection(
        const glm::vec3 view,
        const glm::vec3 normal,
        const glm::vec3 halfVector,
        const float roughness)
    {
        const float ndv = fmaxf(glm::dot(normal, view), BsdfEpsilon);
        return smithG1Ggx(ndv, roughness)
            * distributionGgx(normal, halfVector, roughness) / (4.0f * ndv);
    }

    // Samples a rough-dielectric (Fresnel-weighted reflect/refract) direction.
    // Exposed for CPU-side reflection/transmission sampling tests.
    NR_CPU_GPU static BsdfSample sampleDielectric(
        const glm::vec3 view,
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        const float roughness,
        const float ior,
        RandomState& rng)
    {
        glm::vec3 orientedShadingNormal = shadingNormal;
        if (glm::dot(orientedShadingNormal, view) < 0.0f)
            orientedShadingNormal = -orientedShadingNormal;

        const glm::vec3 incident = -view;
        const bool exiting = glm::dot(incident, geometricNormal) > 0.0f;
        const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
        const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);
        const float ndv = fmaxf(fabsf(glm::dot(orientedShadingNormal, view)), BsdfEpsilon);

        if (isSmoothAlpha(roughness))
        {
            const float fresnel = fresnelDielectric(ndv, etaI, etaT);
            const glm::vec3 refracted = glm::refract(incident, orientedShadingNormal, etaI / etaT);
            const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;

            BsdfSample sample{};
            if (totalInternalReflection || randomFloat(rng) < fresnel)
            {
                sample.event = BsdfEvent::Specular;
                sample.direction = glm::reflect(-view, orientedShadingNormal);
                sample.pdf = BsdfDeltaPdf;
                sample.weight = SampledSpectrum(1.0f);
            }
            else
            {
                sample.event = BsdfEvent::Transmission;
                sample.direction = glm::normalize(refracted);
                const float etaPath = etaT / etaI;
                sample.eta = etaPath;
                sample.pdf = BsdfDeltaPdf;
                sample.weight = SampledSpectrum(1.0f / (etaPath * etaPath));
            }
            return sample;
        }

        const glm::vec3 halfVector = sampleGgxHalfVector(
            view, orientedShadingNormal, roughness, rng);
        const float vdh = fabsf(glm::dot(view, halfVector));
        const float fresnel = fresnelDielectric(vdh, etaI, etaT);
        const glm::vec3 refracted = glm::refract(incident, halfVector, etaI / etaT);
        const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;
        const float distribution = distributionGgx(
            orientedShadingNormal, halfVector, roughness);
        const float wmPdf = distribution * smithG1Ggx(ndv, roughness) * vdh / ndv;

        BsdfSample sample{};
        if (totalInternalReflection || randomFloat(rng) < fresnel)
        {
            sample.event = BsdfEvent::Specular;
            sample.direction = glm::reflect(-view, halfVector);
            if (glm::dot(geometricNormal, view)
                    * glm::dot(geometricNormal, sample.direction) <= 0.0f)
                return {};
            const float ndl = fabsf(glm::dot(orientedShadingNormal, sample.direction));
            sample.pdf = wmPdf * fresnel / fmaxf(4.0f * vdh, BsdfEpsilon);
            // weight = f*ndl/pdf. f and pdf both carry a 1/(4*vdh*ndv-ish) VNDF
            // Jacobian and the branch-selection `fresnel` factor; algebraically
            // those cancel completely (this is Heitz's VNDF-sampling identity,
            // weight = G1(wi) for reflection). Computing the ratio directly —
            // instead of separately clamping f's and pdf's denominators with
            // BsdfEpsilon on *different* expressions (4*ndv*ndl vs 4*vdh) — is
            // what makes that cancellation exact instead of breaking down (and
            // crushing the weight toward zero) at grazing angles, where either
            // clamp can dominate independently.
            sample.weight = SampledSpectrum(smithG1Ggx(ndl, roughness));
        }
        else
        {
            sample.event = BsdfEvent::Transmission;
            sample.direction = glm::normalize(refracted);
            if (glm::dot(geometricNormal, view)
                    * glm::dot(geometricNormal, sample.direction) >= 0.0f)
                return {};
            const float etaPath = etaT / etaI;
            sample.eta = etaPath;
            const float wiDotM = glm::dot(sample.direction, halfVector);
            const float woDotM = glm::dot(view, halfVector);
            const float cosI = fabsf(glm::dot(orientedShadingNormal, sample.direction));
            const float denominator = wiDotM + woDotM / etaPath;
            const float denominator2 = denominator * denominator;
            const float dWmDWi = fabsf(wiDotM) / fmaxf(denominator2, BsdfEpsilon);
            sample.pdf = wmPdf * dWmDWi * (1.0f - fresnel);
            // Same cancellation as the reflection branch above (weight =
            // G1(wi)/etaPath^2 here), computed directly rather than via f/pdf
            // with mismatched epsilon clamps on the shared `denominator2` term
            // (BsdfEpsilon alone vs. cosI*ndv*BsdfEpsilon). At IOR ~ 1,
            // `denominator` is ~0 for every sample regardless of angle, so the
            // old clamp mismatch wasn't a rare edge case — it fired on every
            // transmissive sample and crushed brightness accordingly.
            sample.weight = SampledSpectrum(
                smithG1Ggx(cosI, roughness) / (etaPath * etaPath));
        }
        return sample;
    }

private:
    NR_CPU_GPU static bool isSmoothAlpha(const float roughness)
    {
        const float alpha = roughness * roughness;
        return alpha < BsdfSmoothAlphaThreshold;
    }

    NR_CPU_GPU static glm::vec3 sampleGgxVndfLocal(
        const glm::vec3 view, const float roughness, const glm::vec2 sample)
    {
        const float alpha = roughness * roughness;
        const glm::vec3 stretched = glm::normalize(glm::vec3(alpha * view.x, alpha * view.y, view.z));
        const float phi = 2.0f * BsdfPi * sample.x;
        const float z = (1.0f - sample.y) * (1.0f + stretched.z) - stretched.z;
        const float sinTheta = sqrtf(fminf(fmaxf(1.0f - z * z, 0.0f), 1.0f));
        const glm::vec3 c(sinTheta * cosf(phi), sinTheta * sinf(phi), z);
        const glm::vec3 halfStretched = c + stretched;
        return glm::normalize(glm::vec3(
            alpha * halfStretched.x, alpha * halfStretched.y, halfStretched.z));
    }

    NR_CPU_GPU static glm::vec3 sampleGgxHalfVector(
        const glm::vec3 view, const glm::vec3 normal, const float roughness, RandomState& rng)
    {
        glm::vec3 tangent{}, bitangent{};
        nr::sampling::buildBasis(normal, tangent, bitangent);
        const glm::vec3 localView(
            glm::dot(view, tangent), glm::dot(view, bitangent), glm::dot(view, normal));
        const glm::vec3 localHalf = sampleGgxVndfLocal(
            localView, roughness, glm::vec2(randomFloat(rng), randomFloat(rng)));
        return glm::normalize(
            tangent * localHalf.x + bitangent * localHalf.y + normal * localHalf.z);
    }

    NR_CPU_GPU static float distributionGgx(
        const glm::vec3 normal, const glm::vec3 halfVector, const float roughness)
    {
        const float alpha = roughness * roughness;
        const float alpha2 = alpha * alpha;
        const float ndh = fmaxf(glm::dot(normal, halfVector), 0.0f);
        const float denominator = ndh * ndh * (alpha2 - 1.0f) + 1.0f;
        return alpha2 / fmaxf(BsdfPi * denominator * denominator, BsdfEpsilon);
    }

    NR_CPU_GPU static float smithG1Ggx(const float nd, const float roughness)
    {
        const float cosTheta = fminf(fmaxf(nd, 0.0f), 1.0f);
        if (cosTheta <= 0.0f)
            return 0.0f;
        const float alpha = roughness * roughness;
        const float alpha2 = alpha * alpha;
        const float root = sqrtf(alpha2 + (1.0f - alpha2) * cosTheta * cosTheta);
        return 2.0f * cosTheta / fmaxf(cosTheta + root, BsdfEpsilon);
    }

    NR_CPU_GPU static float shadingNormalCorrection(
        glm::vec3 geometricNormal,
        glm::vec3 shadingNormal,
        const glm::vec3 view,
        const glm::vec3 light)
    {
        if (glm::dot(geometricNormal, view) < 0.0f)
            geometricNormal = -geometricNormal;
        if (glm::dot(shadingNormal, view) < 0.0f)
            shadingNormal = -shadingNormal;
        if (glm::dot(geometricNormal, light) <= 0.0f)
            return 0.0f;
        return glm::dot(shadingNormal, light) > 0.0f ? 1.0f : 0.0f;
    }

    NR_CPU_GPU static float geometrySmith(
        const glm::vec3 normal, const glm::vec3 view, const glm::vec3 light, const float roughness)
    {
        return smithG1Ggx(fmaxf(glm::dot(normal, view), 0.0f), roughness)
            * smithG1Ggx(fmaxf(glm::dot(normal, light), 0.0f), roughness);
    }

    NR_CPU_GPU static float fresnelDielectric(float cosThetaI, float etaI, float etaT)
    {
        if (cosThetaI < 0.0f)
        {
            const float swap = etaI;
            etaI = etaT;
            etaT = swap;
            cosThetaI = -cosThetaI;
        }
        const float eta = etaI / etaT;
        const float sin2ThetaT = eta * eta * fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
        if (sin2ThetaT >= 1.0f)
            return 1.0f;
        const float cosThetaT = sqrtf(1.0f - sin2ThetaT);
        const float parallelNumerator = etaT * cosThetaI - etaI * cosThetaT;
        const float parallelDenominator = etaT * cosThetaI + etaI * cosThetaT;
        const float perpendicularNumerator = etaI * cosThetaI - etaT * cosThetaT;
        const float perpendicularDenominator = etaI * cosThetaI + etaT * cosThetaT;
        const float parallel = parallelNumerator
            / fmaxf(fabsf(parallelDenominator), BsdfEpsilon);
        const float perpendicular = perpendicularNumerator
            / fmaxf(fabsf(perpendicularDenominator), BsdfEpsilon);
        return 0.5f * (parallel * parallel + perpendicular * perpendicular);
    }

    NR_CPU_GPU static float f0ToIor(const float f0)
    {
        const float sqrtF0 = sqrtf(fminf(fmaxf(f0, 0.0f), 0.999999f));
        return (1.0f + sqrtF0) / fmaxf(1.0f - sqrtF0, BsdfEpsilon);
    }

    NR_CPU_GPU static float fresnelConductor(const float cosTheta, const float f0)
    {
        return fresnelDielectric(cosTheta, 1.0f, f0ToIor(f0));
    }

    NR_CPU_GPU static glm::vec3 sampleDiffuseDirection(
        const glm::vec3 normal, RandomState& rng)
    {
        const float u1 = randomFloat(rng);
        const float u2 = randomFloat(rng);
        return nr::sampling::cosineHemisphere(normal, glm::vec2(u1, u2));
    }

    // Probability of the specular (vs. diffuse) lobe used to importance-sample
    // the opaque path, and consumed by pdf() to keep MIS consistent with
    // sample(). Deterministic in shared state only, so both methods agree
    // without needing to thread a value from sample() into pdf().
    NR_CPU_GPU float specularSamplingProbability() const
    {
        const float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 1.0f);
        const float cosTheta = fabsf(glm::dot(shadingNormal, view));
        float maxFresnel = 0.0f;
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float f0 = glm::mix(dielectricF0, albedo[i], metallic);
            maxFresnel = fmaxf(maxFresnel, fresnelConductor(cosTheta, f0));
        }
        return fminf(fmaxf(maxFresnel, 0.02f), 0.98f);
    }

public:
    // Read-only surface/material state this Bsdf was constructed with.
    // Public (rather than accessor methods) since callers such as Shade.cu
    // need e.g. `transmission` to steer light-sampling strategy.
    const glm::vec3 geometricNormal;
    const glm::vec3 shadingNormal;
    const glm::vec3 view;
    const SampledSpectrum albedo;
    const float metallic;
    const float specular;
    const float roughness;
    const float transmission;
    const SampledSpectrum transmissionColor;
    const SellmeierCoefficients sellmeier;
    const SampledWavelengths wavelengths;
    const nr::openpbr::EnergyLutTextures openPbrLuts;
};
