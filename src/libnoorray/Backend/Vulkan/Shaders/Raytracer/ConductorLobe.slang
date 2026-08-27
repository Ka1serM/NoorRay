#pragma once

// A 1:1 port of Materials/Shading/Lobes/ConductorLobe.h.
//
// Spectral conductor Fresnel with its own Kulla-Conty multi-scatter
// compensation - no coupling to any co-existing diffuse/dielectric lobe,
// matching how BSDL/testrender's composite sums independently-compensated
// lobes.

#include "BsdfClosure.slang"
#include "Dielectric.slang"
#include "EnergyLut.slang"
#include "Ggx.slang"
#include "Sampling.slang"

struct ConductorLobe
{
    SampledSpectrum eta;
    SampledSpectrum extinction;
    float roughness;

    BsdfEvaluation eval(ByteAddressBuffer energyLuts, float3 normal, float3 view,
        float3 outgoing)
    {
        BsdfEvaluation result = makeBsdfEvaluation();
        const float normalOutgoing = dot(normal, outgoing);
        const float normalView = dot(normal, view);
        if (normalOutgoing <= 0.0 || normalView <= 0.0)
            return result;

        // A perfectly smooth conductor is a delta lobe: it contributes nothing
        // to an arbitrary (non-mirror) outgoing direction.
        if (isAlmostSpecular(roughness))
            return makeBsdfEvaluation();

        const GgxEvaluation microfacet =
            evaluateGgxReflection(normal, view, outgoing, roughness);
        if (microfacet.cosineWeighted <= 0.0)
            return result;

        result.pdf = microfacet.pdf;
        const float geometryFactor = microfacet.cosineWeighted / normalOutgoing;

        const float viewAlbedo = ggxDirectionalAlbedo(energyLuts, roughness, normalView);
        const float lightAlbedo = ggxDirectionalAlbedo(energyLuts, roughness, normalOutgoing);
        const float averageAlbedo = ggxAverageAlbedo(energyLuts, roughness);
        const float missingAverage = 1.0 - averageAlbedo;
        const float msScale = missingAverage > GgxDenominatorEpsilon
            ? (1.0 - viewAlbedo) * (1.0 - lightAlbedo) / (BsdfPi * missingAverage)
            : 0.0;

        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float fresnel = conductorFresnel(
                microfacet.viewHalfCosine, eta[i], extinction[i]);
            const float normalFresnel = conductorFresnel(1.0, eta[i], extinction[i]);
            const float conductorAvg = averageFresnel(
                iorFromNormalReflectance(normalFresnel));
            const float ms = msScale * multipleScatterFresnel(conductorAvg, averageAlbedo);
            result.value[i] = fresnel * geometryFactor + ms;
        }
        return result;
    }

    BsdfSample sample(ByteAddressBuffer energyLuts, float3 normal, float3 view,
        inout Rng rng)
    {
        BsdfSample result = makeBsdfSample();
        const float normalView = max(dot(normal, view), 0.0);
        if (normalView <= 0.0)
            return result;

        result.event = BsdfEventSpecular;
        if (isAlmostSpecular(roughness))
        {
            result.singular = true;
            result.direction = reflect(-view, normal);
            if (dot(normal, result.direction) <= 0.0)
                return makeBsdfSample();
            [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
                result.weight[i] = conductorFresnel(normalView, eta[i], extinction[i]);
            result.pdf = BsdfSingularPdf;
            return result;
        }

        const float3 halfVector = sampleVisibleNormal(
            normal, view, roughness, rng.nextFloat2());
        result.direction = reflect(-view, halfVector);
        if (dot(normal, result.direction) <= 0.0)
            return makeBsdfSample();

        const BsdfEvaluation evaluation = eval(energyLuts, normal, view, result.direction);
        const float outgoingCosine = dot(normal, result.direction);
        if (evaluation.pdf <= 0.0 || outgoingCosine <= 0.0)
            return makeBsdfSample();
        result.weight = evaluation.value * (outgoingCosine / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    SampledSpectrum albedoEstimate(ByteAddressBuffer energyLuts, float3 normal,
        float3 view)
    {
        const float normalView = max(dot(normal, view), 0.0);
        SampledSpectrum result = SampledSpectrum(0.0);
        if (isAlmostSpecular(roughness))
        {
            [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
                result[i] = conductorFresnel(normalView, eta[i], extinction[i]);
            return result;
        }

        // The composite energy guard must estimate the same compensated
        // directional albedo as eval(). In particular a white Disney metal must
        // stay exactly white at every roughness: its missing single-scatter
        // energy is returned by the Kulla-Conty term.
        const float directionalGeometry =
            ggxDirectionalAlbedo(energyLuts, roughness, normalView);
        const float averageGeometry = ggxAverageAlbedo(energyLuts, roughness);
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float normalFresnel = conductorFresnel(1.0, eta[i], extinction[i]);
            const float averageFresnelValue = averageFresnel(
                iorFromNormalReflectance(normalFresnel));
            const float multipleScatter =
                multipleScatterFresnel(averageFresnelValue, averageGeometry);
            result[i] = conductorFresnel(normalView, eta[i], extinction[i])
                * directionalGeometry
                + (1.0 - directionalGeometry) * multipleScatter;
        }
        return result;
    }
};

ConductorLobe makeConductorLobe()
{
    ConductorLobe lobe;
    lobe.eta = SampledSpectrum(0.0);
    lobe.extinction = SampledSpectrum(0.0);
    lobe.roughness = 0.5;
    return lobe;
}
