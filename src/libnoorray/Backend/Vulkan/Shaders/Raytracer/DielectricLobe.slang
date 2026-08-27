#pragma once

// A 1:1 port of Materials/Shading/Lobes/DielectricLobe.h.
//
// Direction and branch selection use the hero wavelength; Fresnel and spectral
// throughput use the per-wavelength IOR when supplied.

#include "BsdfClosure.slang"
#include "Dielectric.slang"
#include "EnergyLut.slang"
#include "Ggx.slang"
#include "Sampling.slang"

struct DielectricLobe
{
    SampledSpectrum reflectionTint;
    SampledSpectrum transmissionTint;
    float roughness;
    float ior;
    SampledSpectrum spectralIor;
    bool useSpectralIor;
    bool exiting;

    float iorAt(int index)
    {
        return max(useSpectralIor ? spectralIor[index] : ior, 1.0);
    }

    float pathEtaAt(int index)
    {
        const float materialIor = iorAt(index);
        return exiting ? 1.0 / max(materialIor, DielectricDenominatorEpsilon)
                       : materialIor;
    }

    float pathEta() { return pathEtaAt(0); }

    SampledSpectrum fresnelSpectrum(float cosine)
    {
        SampledSpectrum result;
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
            result[i] = dielectricFresnel(cosine, 1.0, pathEtaAt(i));
        return result;
    }

    SampledSpectrum glassEnergyScale(ByteAddressBuffer energyLuts, float normalView,
        float etaPath)
    {
        SampledSpectrum result = SampledSpectrum(1.0);
        if (isAlmostSpecular(roughness) || roughness < GgxMinimumLutRoughness)
            return result;

        const float viewAlbedo = max(
            glassDirectionalAlbedo(energyLuts, roughness, normalView, etaPath), 0.05);
        const float averageAlbedo = max(
            glassAverageAlbedo(energyLuts, roughness, etaPath), 0.05);
        const float averageFresnelValue = averageFresnel(etaPath);
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float tint = clamp(transmissionTint[i], 0.0, 1.0);
            const float multiBounceTint = tint > BsdfEpsilon ? tint : averageFresnelValue;
            const float multipleScatter = multiBounceTint * averageAlbedo
                / max(1.0 - multiBounceTint * (1.0 - averageAlbedo),
                    GgxDenominatorEpsilon);
            result[i] = 1.0 + multipleScatter * (1.0 - viewAlbedo) / viewAlbedo;
        }
        return result;
    }

    BsdfEvaluation evalReflection(ByteAddressBuffer energyLuts, float3 normal,
        float3 view, float3 outgoing, float normalOutgoing)
    {
        BsdfEvaluation result = makeBsdfEvaluation();
        if (isAlmostSpecular(roughness))
            return result;

        const GgxEvaluation microfacet =
            evaluateGgxReflection(normal, view, outgoing, roughness);
        if (microfacet.cosineWeighted <= 0.0)
            return result;

        const float normalView = dot(normal, view);
        const float etaPath = pathEta();
        const SampledSpectrum fresnelValues = fresnelSpectrum(microfacet.viewHalfCosine);
        if (spectrumMaxComponent(transmissionTint) > 0.0)
        {
            const float base = microfacet.cosineWeighted / normalOutgoing;
            const SampledSpectrum scale = glassEnergyScale(energyLuts, normalView, etaPath);
            result.pdf = microfacet.pdf * fresnelValues[0];
            [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
                result.value[i] = reflectionTint[i] * fresnelValues[i] * base * scale[i];
            return result;
        }
        // The dielectric table is the single-scatter Fresnel-weighted response.
        // Missing GGX energy is controlled by the geometry-only masking albedo,
        // shared with the conductor compensation.
        const float viewAlbedo = ggxDirectionalAlbedo(energyLuts, roughness, normalView);
        const float lightAlbedo = ggxDirectionalAlbedo(energyLuts, roughness, normalOutgoing);
        const float averageAlbedo = ggxAverageAlbedo(energyLuts, roughness);
        const float averageFresnelValue = averageFresnel(etaPath);
        const float geometryFactor = microfacet.cosineWeighted / normalOutgoing;
        const float missingAverage = 1.0 - averageAlbedo;
        const float msScale = missingAverage > GgxDenominatorEpsilon
            ? (1.0 - viewAlbedo) * (1.0 - lightAlbedo) / (BsdfPi * missingAverage)
            : 0.0;
        const float ms = msScale * multipleScatterFresnel(averageFresnelValue, averageAlbedo);

        result.pdf = microfacet.pdf;
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
            result.value[i] = reflectionTint[i] * (fresnelValues[i] * geometryFactor + ms);
        return result;
    }

    BsdfEvaluation evalTransmission(ByteAddressBuffer energyLuts, float3 normal,
        float3 view, float3 outgoing, float normalOutgoing)
    {
        BsdfEvaluation result = makeBsdfEvaluation();
        const float etaPath = pathEta();
        if (isAlmostSpecular(roughness) || abs(etaPath - 1.0) < 1.0e-4)
            return result;

        const GgxEvaluation microfacet =
            evaluateGgxTransmission(normal, view, outgoing, roughness, etaPath);
        const float outgoingCosine = -normalOutgoing;
        if (microfacet.cosineWeighted <= 0.0 || outgoingCosine <= 0.0)
            return result;

        const SampledSpectrum fresnelValues = fresnelSpectrum(microfacet.viewHalfCosine);
        result.pdf = (1.0 - fresnelValues[0]) * microfacet.pdf;
        const SampledSpectrum scale = glassEnergyScale(
            energyLuts, dot(normal, view), etaPath);
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float eta = pathEtaAt(i);
            const float base = microfacet.cosineWeighted / outgoingCosine / (eta * eta);
            result.value[i] = transmissionTint[i] * (1.0 - fresnelValues[i])
                * base * scale[i];
        }
        return result;
    }

    BsdfEvaluation eval(ByteAddressBuffer energyLuts, float3 normal, float3 view,
        float3 outgoing)
    {
        const float normalOutgoing = dot(normal, outgoing);
        if (normalOutgoing > 0.0)
            return evalReflection(energyLuts, normal, view, outgoing, normalOutgoing);
        if (normalOutgoing < 0.0 && spectrumMaxComponent(transmissionTint) > 0.0)
            return evalTransmission(energyLuts, normal, view, outgoing, normalOutgoing);
        return makeBsdfEvaluation();
    }

    BsdfSample sample(ByteAddressBuffer energyLuts, float3 normal, float3 view,
        inout Rng rng)
    {
        const float normalView = dot(normal, view);
        if (normalView <= 0.0)
            return makeBsdfSample();

        const bool smooth = isAlmostSpecular(roughness);
        const float etaPath = pathEta();
        const float3 halfVector = smooth
            ? normal
            : sampleVisibleNormal(normal, view, roughness, rng.nextFloat2());
        const float viewHalfCosine = abs(dot(view, halfVector));
        if (viewHalfCosine <= 0.0)
            return makeBsdfSample();

        const SampledSpectrum fresnelValues = fresnelSpectrum(viewHalfCosine);
        const float fresnel = fresnelValues[0];
        const bool hasReflection = spectrumMaxComponent(reflectionTint) > 0.0;
        const bool hasTransmission = spectrumMaxComponent(transmissionTint) > 0.0;
        const float3 refracted = refract(-view, halfVector, 1.0 / etaPath);
        const bool totalInternalReflection = dot(refracted, refracted) < 1.0e-10;

        if (hasReflection && (!hasTransmission || totalInternalReflection
            || rng.nextFloat() < fresnel))
        {
            BsdfSample result = makeBsdfSample();
            result.event = BsdfEventSpecular;
            result.direction = reflect(-view, halfVector);
            if (dot(normal, result.direction) <= 0.0)
                return makeBsdfSample();
            if (smooth)
            {
                result.singular = true;
                const float branchProbability = hasTransmission ? fresnel : 1.0;
                result.pdf = branchProbability * BsdfSingularPdf;
                result.weight = reflectionTint
                    * (fresnelValues * (1.0 / max(branchProbability, BsdfEpsilon)));
                return result;
            }
            const BsdfEvaluation evaluation =
                eval(energyLuts, normal, view, result.direction);
            const float outgoingCosine = dot(normal, result.direction);
            if (evaluation.pdf <= 0.0 || outgoingCosine <= 0.0)
                return makeBsdfSample();
            result.weight = evaluation.value * (outgoingCosine / evaluation.pdf);
            result.pdf = evaluation.pdf;
            return result;
        }

        if (!hasTransmission || totalInternalReflection)
            return makeBsdfSample();

        const float3 direction = safeNormalize(refracted, float3(0.0));
        if (any(isnan(direction)) || any(isinf(direction))
            || dot(direction, direction) <= 1.0e-12)
            return makeBsdfSample();
        if (dot(normal, direction) >= 0.0)
            return makeBsdfSample();

        BsdfSample result = makeBsdfSample();
        result.event = BsdfEventTransmission;
        result.eta = etaPath;
        result.dispersive = useSpectralIor;
        result.direction = direction;
        if (smooth || abs(etaPath - 1.0) < 1.0e-4)
        {
            result.singular = true;
            result.pdf = (1.0 - fresnel) * BsdfSingularPdf;
            SampledSpectrum etaScale;
            [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
            {
                const float eta = pathEtaAt(i);
                etaScale[i] = 1.0 / (eta * eta);
            }
            result.weight = transmissionTint * etaScale;
            return result;
        }
        const BsdfEvaluation evaluation = eval(energyLuts, normal, view, direction);
        const float outgoingCosine = -dot(normal, direction);
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
        SampledSpectrum result;
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float eta = pathEtaAt(i);
            const float f0 = dielectricFresnel(1.0, 1.0, eta);
            // For a rough reflection-only lobe the relevant estimate is the
            // directional GGX albedo, not smooth-interface Fresnel.
            const float singleScatter = isAlmostSpecular(roughness)
                ? fresnelFromNormalReflectance(normalView, f0)
                : dielectricDirectionalAlbedo(energyLuts, roughness, normalView, f0);
            const float averageGeometryAlbedo = isAlmostSpecular(roughness)
                ? 1.0 : ggxAverageAlbedo(energyLuts, roughness);
            const float averageFresnelValue = averageFresnel(eta);
            const float multipleScatter = isAlmostSpecular(roughness)
                ? 0.0
                : multipleScatterFresnel(averageFresnelValue, averageGeometryAlbedo);
            // Match the directional integral of evalReflection(), including the
            // Kulla-Conty term; using only the LUT's single-scatter value made
            // the composite guard vary with view angle.
            const float geometryDirectional = isAlmostSpecular(roughness)
                ? 1.0 : ggxDirectionalAlbedo(energyLuts, roughness, normalView);
            const float directional = singleScatter
                + (1.0 - geometryDirectional) * multipleScatter;
            result[i] = reflectionTint[i] * directional
                + transmissionTint[i] * (1.0 - directional);
        }
        return result;
    }
};

DielectricLobe makeDielectricLobe()
{
    DielectricLobe lobe;
    lobe.reflectionTint = SampledSpectrum(1.0);
    lobe.transmissionTint = SampledSpectrum(0.0);
    lobe.roughness = 0.5;
    lobe.ior = 1.5;
    lobe.spectralIor = SampledSpectrum(1.5);
    lobe.useSpectralIor = false;
    lobe.exiting = false;
    return lobe;
}
