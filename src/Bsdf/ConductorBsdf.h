#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Bsdf/Fresnel.h"
#include "Bsdf/Microfacet.h"
#include "CUDA/Annotations.h"
#include "Raytracing/Bsdf.h"
#include "Raytracing/Spectrum.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{

// Single-scatter GGX + Smith conductor BSDF evaluation (f * cosθ_i).
// Uses the artist-friendly F0-metallic workflow: the specular F0 at normal
// incidence is the base color (albedo), and angular falloff follows the
// exact dielectric Fresnel equations from iorFromF0.
NR_CPU_GPU inline SampledSpectrum evaluateConductorSingleScatter(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    float roughness,
    const SampledSpectrum& albedo,
    const SampledWavelengths& wl)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return SampledSpectrum(0.0f);

    const glm::vec3 halfVector = glm::normalize(view + light);
    const float D = distributionGgx(normal, halfVector, roughness);
    const float G = geometrySmith(normal, view, light, roughness);
    const float geometry = D * G / fmaxf(4.0f * ndv * ndl, Epsilon);
    const float cosHalf = fmaxf(glm::dot(view, halfVector), 0.0f);

    SampledSpectrum result;
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        const float F = fresnelFromF0(cosHalf, albedo[i]);
        result[i] = F * geometry;
    }
    return result;
}

// VNDF sampling PDF in the reflected-direction solid angle.
NR_CPU_GPU inline float pdfConductorReflection(
    const glm::vec3 view,
    const glm::vec3 normal,
    const glm::vec3 halfVector,
    float roughness)
{
    return pdfGgxReflection(view, normal, halfVector, roughness);
}

// Multi-bounce Smith conductor sampling (Bitterli & d'Eon 2023).
// Uses the random-walk estimator for unbiased multi-scattering GGX.
// The conductor Fresnel is evaluated from the base color via fresnelFromF0.
NR_CPU_GPU inline BsdfSample sampleConductorMultiBounce(
    const glm::vec3 view,
    const glm::vec3 geometricNormal,
    const glm::vec3 shadingNormal,
    float roughness,
    const SampledSpectrum& albedo,
    const SampledWavelengths& wl,
    RandomState& rng)
{
    BsdfSample result{};
    result.event = BsdfEvent::Specular;

    glm::vec3 sampleNormal = shadingNormal;
    if (glm::dot(sampleNormal, view) < 0.0f)
        sampleNormal = -sampleNormal;

    SampledSpectrum throughput(1.0f);
    glm::vec3 currentDir = view;
    float ndv = fmaxf(glm::dot(sampleNormal, currentDir), Epsilon);

    constexpr int maxBounces = 100;
    for (int bounce = 0; bounce < maxBounces; ++bounce)
    {
        const glm::vec3 halfVector = sampleGgxHalfVector(
            currentDir, sampleNormal, roughness, rng);
        const float cosHalf = fmaxf(glm::dot(currentDir, halfVector), Epsilon);

        const glm::vec3 reflected = glm::reflect(-currentDir, halfVector);
        const float ndr = fabsf(glm::dot(sampleNormal, reflected));
        if (ndr <= 0.0f)
            break;

        SampledSpectrum F;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            F[i] = fresnelFromF0(cosHalf, albedo[i]);

        const float G1_r = smithG1Ggx(ndr, roughness);

        for (int i = 0; i < NrSpectrumSamples; ++i)
            result.weight[i] += throughput[i] * F[i] * G1_r;

        if (randomFloat(rng) < G1_r)
        {
            result.direction = reflected;
            result.pdf = pdfGgxReflection(currentDir, sampleNormal, halfVector, roughness);

            if (glm::dot(geometricNormal, result.direction) <= 0.0f &&
                glm::dot(geometricNormal, view) >= 0.0f)
                result.weight = SampledSpectrum(0.0f);
            if (glm::dot(geometricNormal, result.direction) >= 0.0f &&
                glm::dot(geometricNormal, view) <= 0.0f)
                result.weight = SampledSpectrum(0.0f);

            break;
        }

        for (int i = 0; i < NrSpectrumSamples; ++i)
            throughput[i] *= F[i];
        currentDir = reflected;
        ndv = ndr;

        float maxThroughput = 0.0f;
        for (int i = 0; i < NrSpectrumSamples; ++i)
            maxThroughput = fmaxf(maxThroughput, throughput[i]);
        if (maxThroughput < 1e-4f)
            break;
    }

    return result;
}

// Multi-bounce BSDF evaluation using the position-free path tracing estimator
// from Wang et al. 2021 ("Position-free Multiple-bounce Computations for Smith
// Microfacet BSDFs"). Returns f * cosθ_o for the given (light, view) pair.
// The single-scatter contribution is computed analytically; higher-order
// bounces are estimated via Monte Carlo with next-event estimation at each
// bounce of a VNDF random walk.
NR_CPU_GPU inline SampledSpectrum evaluateConductorMultiBounce(
    const glm::vec3 normal,
    const glm::vec3 view,
    const glm::vec3 light,
    float roughness,
    const SampledSpectrum& albedo,
    const SampledWavelengths& wl,
    RandomState& rng)
{
    const float ndv = fmaxf(glm::dot(normal, view), 0.0f);
    const float ndl = fmaxf(glm::dot(normal, light), 0.0f);
    if (ndv <= 0.0f || ndl <= 0.0f)
        return SampledSpectrum(0.0f);

    SampledSpectrum result(0.0f);

    // Single-scatter (analytic)
    {
        const glm::vec3 h = glm::normalize(view + light);
        const float D = distributionGgx(normal, h, roughness);
        const float G = geometrySmith(normal, view, light, roughness);
        const float geom = D * G / fmaxf(4.0f * ndv * ndl, Epsilon);
        const float cosH = fmaxf(glm::dot(view, h), 0.0f);
        for (int i = 0; i < NrSpectrumSamples; ++i)
            result[i] = fresnelFromF0(cosH, albedo[i]) * geom;
    }

    // Multi-scatter (Wang et al. 2021 PT estimator)
    constexpr int Mc = 4;
    constexpr int maxBounces = 10;

    for (int s = 0; s < Mc; ++s)
    {
        // Cumulative path weight = (throughput * vertex * segment / VNDF_pdf)
        SampledSpectrum pathW(1.0f);
        float pdfPath = 1.0f;
        glm::vec3 curDir = -light;
        float ndCur = ndl;

        for (int bounce = 0; bounce < maxBounces; ++bounce)
        {
            // Sample VNDF
            const glm::vec3 hS = sampleGgxHalfVector(curDir, normal, roughness, rng);
            const float cosHS = fabsf(glm::dot(curDir, hS));

            const glm::vec3 nextDir = glm::reflect(-curDir, hS);
            const float ndNext = fabsf(glm::dot(normal, nextDir));

            // VNDF PDF: D(h) * G1(ndCur) / (4 * ndCur)
            const float DhS = distributionGgx(normal, hS, roughness);
            const float G1c = smithG1Ggx(ndCur, roughness);
            const float pdfDir = DhS * G1c / fmaxf(4.0f * ndCur, Epsilon);

            // Fresnel per wavelength
            SampledSpectrum F;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                F[i] = fresnelFromF0(cosHS, albedo[i]);

            const float G1n = smithG1Ggx(ndNext, roughness);
            const float G1v = smithG1Ggx(ndv, roughness);

            // NEE: connect nextDir to view
            const glm::vec3 hNee = glm::normalize(nextDir + view);
            const float cosNeeN = fabsf(glm::dot(nextDir, hNee));
            const float cosNeeV = fabsf(glm::dot(view, hNee));
            if (cosNeeN > Epsilon && cosNeeV > Epsilon)
            {
                const float DhNee = distributionGgx(normal, hNee, roughness);
                const float denomNee = fmaxf(4.0f * cosNeeN * cosNeeV, Epsilon);

                SampledSpectrum FNee;
                for (int i = 0; i < NrSpectrumSamples; ++i)
                    FNee[i] = fresnelFromF0(cosNeeN, albedo[i]);

                // Vertex term for the NEE connection
                SampledSpectrum vNee = FNee * SampledSpectrum(DhNee / denomNee);

                // The NEE contribution at bounce k terminates the random walk
                // with probability G1(ω_k·n).  The path weight is the cumulative
                // throughput from previous bounces times F_k (Fresnel at this
                // bounce) times G1_k (termination probability), divided by the
                // cumulative VNDF path PDF.
                //
                // Full contribution:
                //   Π_{j<k} [F_j*(1-G1_j)]  *  F_k * G1_k
                //   *  F_nee * D_nee * G1_o / (4*|ω_k·h_nee|*|ω_o·h_nee|)
                //   /  Π_{j≤k} pdfDir_j
                //
                // where pdfDir_j = D(h_j) * G1(ω_{j-1}·n) / (4 * |ω_{j-1}·n|)
                const float segFactor = G1n;
                const float pdfTotal = fmaxf(pdfPath * pdfDir, Epsilon);

                for (int i = 0; i < NrSpectrumSamples; ++i)
                    result[i] += pathW[i] * F[i] * segFactor * vNee[i] * G1v / pdfTotal;
            }

            // Russian roulette: terminate with probability G1n, continue with 1-G1n
            if (randomFloat(rng) < G1n)
                break;

            // Continue walk: throughput multiplies by F * (1-G1n)
            // Path PDF multiplies by the VNDF PDF (pdfDir)
            for (int i = 0; i < NrSpectrumSamples; ++i)
                pathW[i] = pathW[i] * F[i] * (1.0f - G1n);
            pdfPath *= pdfDir;

            curDir = nextDir;
            ndCur = ndNext;

            // Early termination
            float maxW = 0.0f;
            for (int i = 0; i < NrSpectrumSamples; ++i)
                maxW = fmaxf(maxW, pathW[i]);
            if (maxW < 1e-6f)
                break;
        }
    }

    result *= (1.0f / static_cast<float>(Mc));
    return result;
}

} // namespace nr::bsdf
