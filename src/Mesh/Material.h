#pragma once

#include <cmath>
#include <cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Texture.h"
#include "Mesh/OpenPbrEnergy.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Raytracing/Sellmeier.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

class Material
{
public:
    glm::vec3 albedo{1.0f};
    int albedoIndex{-1};
    float specular{0.5f};
    float metallic{};
    float roughness{};
    SellmeierCoefficients sellmeier{};
    int specularIndex{-1};
    int metallicIndex{-1};
    int roughnessIndex{-1};
    int normalIndex{-1};
    glm::vec3 transmissionColor{1.0f};
    float transmission{};
    glm::vec3 emission{1.0f};
    float emissionStrength{};
    int emissionIndex{-1};
    int transmissionIndex{-1};
    int opacityIndex{-1};
    float opacity{1.0f};

    // BSDF math below is exposed (not private) so CPU-side tests
    // (tests/BsdfTests.cpp) can exercise it directly, e.g. furnace and
    // PDF-normalization tests, without going through the GPU-only spectral
    // wrappers further down.
    NR_CPU_GPU static void buildBasis(
        const glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
    {
        nr::sampling::buildBasis(normal, tangent, bitangent);
    }

    NR_CPU_GPU static glm::vec3 sampleDiffuseDirection(const glm::vec3 normal, RandomState& rng)
    {
        const float u1 = randomFloat(rng);
        const float u2 = randomFloat(rng);
        return nr::sampling::cosineHemisphere(normal, glm::vec2(u1, u2));
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
        buildBasis(normal, tangent, bitangent);
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

    // Rotates a shading normal minimally toward the geometric normal so that
    // reflecting `view` about it does not punch below the geometric
    // hemisphere. This is the "ensure valid reflection" technique (as used in
    // Blender Cycles) that fixes energy loss from smooth-normal divergence
    // near silhouettes of coarsely tessellated meshes, complementing (not
    // replacing) shadingNormalCorrection's hard zero cutoff below.
    // geometricNormal/shadingNormal must already satisfy dot(N, view) >= 0.
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

    NR_CPU_GPU static glm::vec3 fresnelSchlick(const float cosTheta, const glm::vec3 f0)
    {
        const float oneMinusCos = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
        const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
        return f0 + (glm::vec3(1.0f) - f0) * factor;
    }

    NR_CPU_GPU static float fresnelSchlickS(const float cosTheta, const float f0)
    {
        const float omc = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
        const float f = omc * omc * omc * omc * omc;
        return f0 + (1.0f - f0) * f;
    }

    // Per-wavelength spectral BSDF evaluation (opaque PBR).
    NR_CPU_GPU static SampledSpectrum evaluateOpaqueSpectral(
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light,
        const SampledSpectrum& albedoSpec,
        const float metallic,
        const float specular,
        const float roughness,
        const nr::openpbr::EnergyLutTextures& openPbrLuts = {})
    {
        const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
        const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
        if (ndv <= 0.0f || ndl <= 0.0f)
            return SampledSpectrum(0.0f);
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
            const float conductorF = fresnelSchlickS(vdh, albedoSpec[i]);
            const float F = dielectricF + (conductorF - dielectricF) * metallic;
            const float specBrdf = F * geomFactor;
            // OpenPBR's reciprocal layered compensation accounts for light
            // passing through the rough dielectric interface in both
            // directions. Its tabulated hemispherical average normalizes the
            // layer so a white substrate passes the furnace test.
            const float diffuse = diffuseEnergy * (1.0f - metallic)
                * albedoSpec[i] / BsdfPi;
            result[i] = diffuse + specBrdf
                + (1.0f - metallic) * dielectricMultipleScatter;
        }
        return result;
    }

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
        const glm::vec3 halfVector = sampleGgxHalfVector(
            view, orientedShadingNormal, roughness, rng);
        const float vdh = fabsf(glm::dot(view, halfVector));

        const glm::vec3 incident = -view;
        const bool exiting = glm::dot(incident, geometricNormal) > 0.0f;
        const float etaI = exiting ? fmaxf(ior, 1.0f) : 1.0f;
        const float etaT = exiting ? 1.0f : fmaxf(ior, 1.0f);
        const float fresnel = fresnelDielectric(vdh, etaI, etaT);
        const glm::vec3 refracted = glm::refract(incident, halfVector, etaI / etaT);
        const bool totalInternalReflection = glm::dot(refracted, refracted) < 1e-10f;
        const float ndv = fmaxf(fabsf(glm::dot(orientedShadingNormal, view)), BsdfEpsilon);
        const float distribution = distributionGgx(
            orientedShadingNormal, halfVector, roughness);
        const float wmPdf = distribution * smithG1Ggx(ndv, roughness) * vdh / ndv;

        BsdfSample sample{};
        if (totalInternalReflection || randomFloat(rng) < fresnel)
        {
            sample.event     = BsdfEvent::Specular;
            sample.direction = glm::reflect(-view, halfVector);
            const float ndl = fabsf(glm::dot(orientedShadingNormal, sample.direction));
            sample.pdf = wmPdf * fresnel / fmaxf(4.0f * vdh, BsdfEpsilon);
            const float f = distribution
                * geometrySmith(orientedShadingNormal, view, sample.direction, roughness)
                * fresnel / fmaxf(4.0f * ndv * ndl, BsdfEpsilon);
            sample.weight = SampledSpectrum(f * ndl / fmaxf(sample.pdf, BsdfEpsilon));
        }
        else
        {
            sample.event     = BsdfEvent::Transmission;
            sample.direction = glm::normalize(refracted);
            const float etaPath = etaT / etaI;
            sample.eta = etaPath;
            const float wiDotM = glm::dot(sample.direction, halfVector);
            const float woDotM = glm::dot(view, halfVector);
            const float cosI = fabsf(glm::dot(orientedShadingNormal, sample.direction));
            const float denominator = wiDotM + woDotM / etaPath;
            const float denominator2 = denominator * denominator;
            const float dWmDWi = fabsf(wiDotM) / fmaxf(denominator2, BsdfEpsilon);
            sample.pdf = wmPdf * dWmDWi * (1.0f - fresnel);
            const float transmissionGeometry = smithG1Ggx(
                fabsf(glm::dot(orientedShadingNormal, view)), roughness)
                * smithG1Ggx(fabsf(glm::dot(
                    orientedShadingNormal, sample.direction)), roughness);
            float f = (1.0f - fresnel) * distribution
                * transmissionGeometry
                * fabsf(wiDotM * woDotM
                    / fmaxf(cosI * ndv * denominator2, BsdfEpsilon));
            f /= etaPath * etaPath;
            sample.weight = SampledSpectrum(f * cosI / fmaxf(sample.pdf, BsdfEpsilon));
        }
        return sample;
    }

public:
#if defined(NR_GPU_CODE)
    // ── Spectral BSDF (GPU only) ──────────────────────────────────────────

    NR_GPU BsdfSample sampleBsdfSpectral(
        const CudaTexture* textures,
        const glm::vec2 uv,
        const glm::vec3 view,
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormal,
        RandomState& rng,
        const SampledWavelengths& wl,
        const float* spectrumScale,
        const float* spectrumCoeffs,
        const float* d65,
        const nr::openpbr::EnergyLutTextures& openPbrLuts) const
    {
        BsdfSample result{};

        // --- Albedo (spectral) ---
        glm::vec3 rgbAlbedo = albedo;
        if (albedoIndex >= 0)
            rgbAlbedo *= glm::vec3(textures[albedoIndex].sample(uv));
        result.albedo = rgbAlbedoToSpectrum(rgbAlbedo, wl, spectrumScale, spectrumCoeffs);

        // --- Emission (spectral) ---
        glm::vec3 rgbEmission = emission;
        if (emissionIndex >= 0)
            rgbEmission *= glm::vec3(textures[emissionIndex].sample(uv));
        rgbEmission *= emissionStrength;
        result.emission = rgbIlluminantToSpectrum(
            rgbEmission, wl, spectrumScale, spectrumCoeffs, d65);

        float met = metallic;
        if (metallicIndex >= 0)
            met *= textures[metallicIndex].sample(uv).z;
        met = fminf(fmaxf(met, 0.0f), 1.0f);

        float rough = roughness;
        if (roughnessIndex >= 0)
            rough *= textures[roughnessIndex].sample(uv).y;
        rough = fminf(fmaxf(rough, 0.045f), 1.0f);

        float trans = transmission;
        if (transmissionIndex >= 0)
            trans *= textures[transmissionIndex].sample(uv).x;
        trans = fminf(fmaxf(trans, 0.0f), 1.0f);

        float spec = specular;
        if (specularIndex >= 0)
            spec *= textures[specularIndex].sample(uv).x;
        spec = fminf(fmaxf(spec, 0.0f), 1.0f);

        result.metallic    = met;
        result.roughness   = rough;
        result.specular    = spec;
        result.transmission = trans;

        if (randomFloat(rng) < trans)
        {
            // The hero wavelength selects the dispersive refraction direction.
            const float heroIor = sellmeierIor(sellmeier, wl[0]);
            const BsdfSample dielectric = sampleDielectric(
                view, geometricNormal, shadingNormal, rough, heroIor, rng);
            result.direction = dielectric.direction;
            result.event = dielectric.event;
            result.pdf = dielectric.pdf;
            result.eta = dielectric.eta;
            // Transmission weight: tint the spectral albedo of the transmission color.
            const SampledSpectrum transSpec = rgbAlbedoToSpectrum(
                transmissionColor, wl, spectrumScale, spectrumCoeffs);
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
                const float wavelengthIor = sellmeierIor(sellmeier, wl[i]);
                result.weight[i] = dielectric.event == BsdfEvent::Transmission
                    ? transSpec[i] * dielectric.weight[0]
                    : dielectric.weight[0] * fresnelDielectric(
                        cosTheta,
                        exitingDielectric ? wavelengthIor : 1.0f,
                        exitingDielectric ? 1.0f : wavelengthIor) / heroFresnel;
            }
        }
        else
        {
            // Opaque path — sample direction from RGB luminance albedo for importance sampling.
            const glm::vec3 dielectricF0v(fminf(fmaxf(0.08f * spec, 0.0f), 1.0f));
            const glm::vec3 samplingFresnel = fresnelSchlick(
                fabsf(glm::dot(shadingNormal, view)),
                glm::mix(dielectricF0v, rgbAlbedo, met));
            const float specProb = fminf(fmaxf(
                fmaxf(samplingFresnel.x, fmaxf(samplingFresnel.y, samplingFresnel.z)),
                0.02f), 0.98f);

            glm::vec3 halfVector{};
            glm::vec3 sampleNormal = shadingNormal;
            if (glm::dot(sampleNormal, view) < 0.0f)
                sampleNormal = -sampleNormal;

            if (randomFloat(rng) < specProb)
            {
                result.event = BsdfEvent::Specular;
                halfVector = sampleGgxHalfVector(view, sampleNormal, rough, rng);
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
                sampleNormal, view, result.direction, result.albedo, met, spec, rough, openPbrLuts);
            const float specPdf    = pdfGgxReflection(view, sampleNormal, halfVector, rough);
            const float diffusePdf = ndl / BsdfPi;
            const float pdf = fmaxf(
                specProb * specPdf + (1.0f - specProb) * diffusePdf, BsdfEpsilon);
            result.pdf = pdf;
            result.samplingSpecularProbability = specProb;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                result.weight[i] = brdfSpec[i] * (ndl / pdf);
            result.weight *= shadingNormalCorrection(
                geometricNormal, sampleNormal, view, result.direction);
        }
        return result;
    }

    NR_GPU SampledSpectrum evaluateDirectSpectral(
        const BsdfSample& bsdf,
        const glm::vec3 geometricNormal,
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light,
        const SampledWavelengths& wl,
        const nr::openpbr::EnergyLutTextures& openPbrLuts) const
    {
        const SampledSpectrum opaque = evaluateOpaqueSpectral(
            normal, view, light, bsdf.albedo, bsdf.metallic, bsdf.specular, bsdf.roughness, openPbrLuts);
        const float normalCorrection = shadingNormalCorrection(
            geometricNormal, normal, view, light);
        if (bsdf.transmission <= 0.0f)
            return opaque * normalCorrection;
        if (glm::dot(normal, view) <= 0.0f || glm::dot(normal, light) <= 0.0f)
            return opaque * ((1.0f - bsdf.transmission) * normalCorrection);
        const glm::vec3 halfVector = glm::normalize(view + light);
        const float D = distributionGgx(normal, halfVector, bsdf.roughness);
        const float G = geometrySmith(normal, view, light, bsdf.roughness);
        const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
        const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
        SampledSpectrum result = opaque * (1.0f - bsdf.transmission);
        const float geometry = D * G / fmaxf(4.0f * ndv * ndl, BsdfEpsilon);
        const float cosTheta = fmaxf(glm::dot(view, halfVector), 0.0f);
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result[i] += fresnelDielectric(
                cosTheta, 1.0f, sellmeierIor(sellmeier, wl[i]))
                * geometry * bsdf.transmission;
        return result * normalCorrection;
    }

    NR_GPU float pdfDirectSpectral(
        const BsdfSample& bsdf,
        const glm::vec3 geometricNormal,
        const glm::vec3 normal,
        const glm::vec3 view,
        const glm::vec3 light) const
    {
        if (bsdf.transmission > 0.0f)
            return 0.0f;
        const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
        if (ndl <= 0.0f)
            return 0.0f;
        glm::vec3 orientedGeometricNormal = geometricNormal;
        if (glm::dot(orientedGeometricNormal, view) < 0.0f)
            orientedGeometricNormal = -orientedGeometricNormal;
        if (glm::dot(orientedGeometricNormal, light) <= 0.0f)
            return 0.0f;
        const glm::vec3 halfVector = glm::normalize(view + light);
        const float specularPdf = pdfGgxReflection(
            view, normal, halfVector, bsdf.roughness);
        const float diffusePdf = ndl / BsdfPi;
        return bsdf.samplingSpecularProbability * specularPdf
            + (1.0f - bsdf.samplingSpecularProbability) * diffusePdf;
    }
#endif
};
