#pragma once

// A 1:1 port of Materials/Shading/Lobes/DiffuseLobe.h.
//
// Diffuse reflection shared by MaterialX's Oren-Nayar and Burley closures.
// Burley mode adds the Disney diffuse grazing/retroreflection response used by
// MaterialX Standard Surface and OpenPBR.

#include "BsdfClosure.slang"
#include "Sampling.slang"

float diffusePow5(float value)
{
    const float squared = value * value;
    return squared * squared * value;
}

struct DiffuseLobe
{
    SampledSpectrum albedo;
    float roughness;
    bool burley;
    // MaterialX's translucent_bsdf: pure Lambertian *transmission* (light
    // entering one side scatters back out the other, e.g. paper/leaves) -
    // physically a different lobe from diffuse reflection, not a coat of the
    // same math. Scatters into the hemisphere behind the shading normal
    // instead of in front of it; never combined with `burley`.
    bool translucent;

    BsdfEvaluation eval(float3 normal, float3 view, float3 outgoing)
    {
        const float3 effectiveNormal = translucent ? -normal : normal;
        const float normalOutgoing = dot(effectiveNormal, outgoing);
        if (normalOutgoing <= 0.0)
            return makeBsdfEvaluation();
        float diffuse = 1.0 / BsdfPi;
        if (burley)
        {
            const float normalView = max(dot(normal, view), 0.0);
            const float3 halfSum = view + outgoing;
            const float halfLengthSquared = dot(halfSum, halfSum);
            const float lightHalf = halfLengthSquared > BsdfEpsilon
                ? max(dot(outgoing, halfSum * (1.0 / sqrt(halfLengthSquared))), 0.0)
                : 0.0;
            const float fd90 = 0.5 + 2.0 * roughness * lightHalf * lightHalf;
            const float lightGrazing = diffusePow5(1.0 - normalOutgoing);
            const float viewGrazing = diffusePow5(1.0 - normalView);
            diffuse *= (1.0 + (fd90 - 1.0) * lightGrazing)
                * (1.0 + (fd90 - 1.0) * viewGrazing);
        }
        BsdfEvaluation result;
        result.value = albedo * diffuse;
        result.pdf = normalOutgoing / BsdfPi;
        return result;
    }

    BsdfSample sample(float3 normal, float3 view, inout Rng rng)
    {
        BsdfSample result = makeBsdfSample();
        result.event = translucent ? BsdfEventTransmission : BsdfEventDiffuse;
        const float3 sampleNormal = translucent ? -normal : normal;
        result.direction = cosineHemisphere(sampleNormal, rng.nextFloat2());
        const BsdfEvaluation evaluation = eval(normal, view, result.direction);
        if (evaluation.pdf <= 0.0)
            return makeBsdfSample();
        result.weight = evaluation.value
            * (abs(dot(normal, result.direction)) / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    SampledSpectrum albedoEstimate(float3 normal, float3 view)
    {
        if (burley && !translucent)
        {
            // Closed-form polynomial of the Disney diffuse furnace integral
            // (x = 1 - N.V), avoiding a per-hit 2D quadrature while retaining
            // the grazing correction the composite energy guard needs.
            const float normalView = max(dot(normal, view), 0.0);
            const float x = 1.0 - min(normalView, 1.0);
            const float x2 = x * x;
            const float x3 = x2 * x;
            const float x4 = x3 * x;
            const float x5 = x4 * x;
            const float x6 = x5 * x;
            const float a0 = 0.976190476 - 0.488095238 * x5;
            const float a1 = 0.059523810 - 0.011904762 * x
                + 1.607142857 * x5 - 0.654760863 * x6;
            const float a2 = -0.000001844 + 0.000212882 * x
                - 0.003366655 * x2 + 0.020045492 * x3
                - 0.057259231 * x4 + 0.159789532 * x5
                - 0.049970620 * x6;
            const float roughnessSquared = roughness * roughness;
            const float burleyAlbedo = max(
                a0 + roughness * a1 + roughnessSquared * a2, 0.0);
            return albedo * burleyAlbedo;
        }
        return albedo;
    }
};

DiffuseLobe makeDiffuseLobe()
{
    DiffuseLobe lobe;
    lobe.albedo = SampledSpectrum(1.0);
    lobe.roughness = 0.0;
    lobe.burley = false;
    lobe.translucent = false;
    return lobe;
}
