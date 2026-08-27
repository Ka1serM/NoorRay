#pragma once

// A 1:1 port of Materials/Shading/CompositeBsdf.h.
//
// Bounded closure storage with combined sampling and evaluation across all
// active lobes. Slang has no unions, so NoorRayShaderClosure::Storage becomes
// a tagged record whose spectra are reused by the three lobe kinds - the
// occupancy matches the material union rather than tripling it.

#include "ConductorLobe.slang"
#include "DielectricLobe.slang"
#include "DiffuseLobe.slang"

static const uint NoorRayClosureDiffuse = 0u;
static const uint NoorRayClosureConductor = 1u;
static const uint NoorRayClosureDielectric = 2u;

// The hardware hit-stage fast pool covers every lobe a single OpenPBR/Disney
// terminal can emit (base, subsurface, metal, reflection, transmission, coat,
// and fuzz). Larger arbitrary closure trees are rejected explicitly; they
// require the streaming/two-pass evaluator rather than a multi-kilobyte local
// array in every closest-hit invocation.
static const int NoorRayMaxLobes = 8;
static const float NoorRayClosureWeightCutoff = 1.0e-5;

struct ClosureStorage
{
    // diffuse.albedo | conductor.eta | dielectric.reflectionTint
    SampledSpectrum a;
    // conductor.extinction | dielectric.transmissionTint
    SampledSpectrum b;
    // dielectric.spectralIor
    SampledSpectrum c;
    float roughness;
    float ior;
    bool flag0;  // diffuse.burley | dielectric.useSpectralIor
    bool flag1;  // diffuse.translucent | dielectric.exiting
};

struct NoorRayShaderClosure
{
    uint kind;
    float3 normal;
    // The authored weight this lobe was assigned while walking the closure
    // tree (add()/mul()/layer() composition), already converted to spectral.
    SampledSpectrum weight;
    // Sample closures proportional to closure_sample_weight, the average of the
    // clamped spectral weight. The BSDF's albedo belongs in evaluation, not in
    // closure selection probability.
    float sampleWeight;
    ClosureStorage storage;

    DiffuseLobe asDiffuse()
    {
        DiffuseLobe lobe = makeDiffuseLobe();
        lobe.albedo = storage.a;
        lobe.roughness = storage.roughness;
        lobe.burley = storage.flag0;
        lobe.translucent = storage.flag1;
        return lobe;
    }

    ConductorLobe asConductor()
    {
        ConductorLobe lobe = makeConductorLobe();
        lobe.eta = storage.a;
        lobe.extinction = storage.b;
        lobe.roughness = storage.roughness;
        return lobe;
    }

    DielectricLobe asDielectric()
    {
        DielectricLobe lobe = makeDielectricLobe();
        lobe.reflectionTint = storage.a;
        lobe.transmissionTint = storage.b;
        lobe.spectralIor = storage.c;
        lobe.roughness = storage.roughness;
        lobe.ior = storage.ior;
        lobe.useSpectralIor = storage.flag0;
        lobe.exiting = storage.flag1;
        return lobe;
    }

    BsdfEvaluation eval(ByteAddressBuffer energyLuts, float3 view, float3 outgoing)
    {
        switch (kind)
        {
        case NoorRayClosureDiffuse:
            return asDiffuse().eval(normal, view, outgoing);
        case NoorRayClosureConductor:
            return asConductor().eval(energyLuts, normal, view, outgoing);
        default:
            return asDielectric().eval(energyLuts, normal, view, outgoing);
        }
    }

    BsdfSample sample(ByteAddressBuffer energyLuts, float3 view, inout Rng rng)
    {
        switch (kind)
        {
        case NoorRayClosureDiffuse:
            return asDiffuse().sample(normal, view, rng);
        case NoorRayClosureConductor:
            return asConductor().sample(energyLuts, normal, view, rng);
        default:
            return asDielectric().sample(energyLuts, normal, view, rng);
        }
    }

    SampledSpectrum albedoEstimate(ByteAddressBuffer energyLuts, float3 view)
    {
        switch (kind)
        {
        case NoorRayClosureDiffuse:
            return asDiffuse().albedoEstimate(normal, view);
        case NoorRayClosureConductor:
            return asConductor().albedoEstimate(energyLuts, normal, view);
        default:
            return asDielectric().albedoEstimate(energyLuts, normal, view);
        }
    }

    float transmissionEstimate()
    {
        return kind == NoorRayClosureDielectric
            ? spectrumMaxComponent(storage.b) : 0.0;
    }
};

// Rotate a shading normal minimally toward the geometric normal so that its
// reflection stays above the geometric surface. This is the Blender/Cycles
// grazing-angle correction the SVM composite needs for normal-mapped surfaces.
float3 clampShadingNormal(float3 geometricNormal, float3 shadingNormal, float3 view)
{
    const float3 reflected = reflect(-view, shadingNormal);
    const float normalView = max(dot(geometricNormal, view), 0.0);
    const float threshold = min(0.9 * normalView, 0.01);
    if (dot(geometricNormal, reflected) >= threshold)
        return shadingNormal;

    const float3 rawTangent = geometricNormal * normalView - view;
    const float tangentLengthSquared = dot(rawTangent, rawTangent);
    if (tangentLengthSquared < 1.0e-8)
        return shadingNormal;
    const float3 tangent = rawTangent * (1.0 / sqrt(tangentLengthSquared));

    const float thresholdSine = sqrt(max(1.0 - threshold * threshold, 0.0));
    const float3 correctedReflection = geometricNormal * threshold + tangent * thresholdSine;
    const float3 clamped = view + correctedReflection;
    const float clampedLengthSquared = dot(clamped, clamped);
    if (clampedLengthSquared < 1.0e-12)
        return geometricNormal;
    return clamped * (1.0 / sqrt(clampedLengthSquared));
}

// Fixed closure array with admission and mixture-PDF calculation.
struct NoorRayShaderData
{
    float3 geometricNormal_;
    float3 shadingNormal_;
    float3 view_;
    NoorRayShaderClosure lobes_[NoorRayMaxLobes];
    float selectionPdf_[NoorRayMaxLobes];
    SampledSpectrum energyScale_;
    int count_;
    bool overflowed_;

    [mutating] void setShadingNormal(float3 geometricNormal, float3 shadingNormal)
    {
        const float3 safeGeometricNormal = safeNormalize(geometricNormal, float3(0.0, 1.0, 0.0));
        const float3 safeShadingNormal = safeNormalize(shadingNormal, safeGeometricNormal);
        geometricNormal_ = dot(safeGeometricNormal, view_) < 0.0
            ? -safeGeometricNormal : safeGeometricNormal;
        shadingNormal_ = dot(safeShadingNormal, view_) < 0.0
            ? -safeShadingNormal : safeShadingNormal;
        shadingNormal_ = clampShadingNormal(geometricNormal_, shadingNormal_, view_);
    }

    // Clamp weights and skip negligible closures at allocation time.
    [mutating] bool add(uint kind, SampledSpectrum weight, float3 requestedNormal,
        ClosureStorage storage)
    {
        const SampledSpectrum nonNegativeWeight = clampZero(weight);
        const float sampleWeight = abs(spectrumAverage(nonNegativeWeight));
        if (sampleWeight < NoorRayClosureWeightCutoff)
            return true;
        if (count_ >= NoorRayMaxLobes)
        {
            overflowed_ = true;
            return false;
        }
        lobes_[count_].kind = kind;
        const float3 normal = clampShadingNormal(geometricNormal_,
            safeNormalize(requestedNormal, shadingNormal_), view_);
        lobes_[count_].normal = dot(normal, view_) < 0.0 ? -normal : normal;
        lobes_[count_].weight = nonNegativeWeight;
        lobes_[count_].sampleWeight = sampleWeight;
        lobes_[count_].storage = storage;
        ++count_;
        return true;
    }

    [mutating] bool addDiffuse(SampledSpectrum weight, DiffuseLobe lobe, float3 normal)
    {
        ClosureStorage storage;
        storage.a = lobe.albedo;
        storage.b = SampledSpectrum(0.0);
        storage.c = SampledSpectrum(0.0);
        storage.roughness = lobe.roughness;
        storage.ior = 1.0;
        storage.flag0 = lobe.burley;
        storage.flag1 = lobe.translucent;
        return add(NoorRayClosureDiffuse, weight, normal, storage);
    }

    [mutating] bool addConductor(SampledSpectrum weight, ConductorLobe lobe, float3 normal)
    {
        ClosureStorage storage;
        storage.a = lobe.eta;
        storage.b = lobe.extinction;
        storage.c = SampledSpectrum(0.0);
        storage.roughness = lobe.roughness;
        storage.ior = 1.0;
        storage.flag0 = false;
        storage.flag1 = false;
        return add(NoorRayClosureConductor, weight, normal, storage);
    }

    [mutating] bool addDielectric(SampledSpectrum weight, DielectricLobe lobe, float3 normal)
    {
        ClosureStorage storage;
        storage.a = lobe.reflectionTint;
        storage.b = lobe.transmissionTint;
        storage.c = lobe.spectralIor;
        storage.roughness = lobe.roughness;
        storage.ior = lobe.ior;
        storage.flag0 = lobe.useSpectralIor;
        storage.flag1 = lobe.exiting;
        return add(NoorRayClosureDielectric, weight, normal, storage);
    }

    // Set lobe selection probabilities before sample()/evaluate().
    [mutating] bool prepare(ByteAddressBuffer energyLuts)
    {
        if (overflowed_)
            return false;
        float total = 0.0;
        SampledSpectrum estimatedEnergy = SampledSpectrum(0.0);
        for (int i = 0; i < count_; ++i)
        {
            const SampledSpectrum albedo = lobes_[i].albedoEstimate(energyLuts, view_);
            estimatedEnergy += lobes_[i].weight * albedo;
            // Select closures by expected contribution, not just authored
            // closure weight: a Disney diffuse+specular material has ~1 unit of
            // diffuse energy and 0.02-0.08 of specular, and a 50/50 selector
            // creates large variance. The mixture stays unbiased because the
            // full mixture PDF is still evaluated after sampling.
            selectionPdf_[i] = lobes_[i].sampleWeight * max(spectrumAverage(albedo), 0.0);
            total += selectionPdf_[i];
        }
        for (int i = 0; i < count_; ++i)
            selectionPdf_[i] = total > 0.0
                ? selectionPdf_[i] / total
                : (count_ > 0 ? 1.0 / float(count_) : 0.0);

        // Closure trees use additive evaluation for efficient mixture sampling,
        // while MaterialX `layer` nodes are physically energy sharing, so a
        // stack of independently compensated lobes can exceed one in a white
        // furnace. Apply a spectral guard once per prepared surface so the BSDF
        // stays non-negative and energy conserving without changing branch PDFs.
        energyScale_ = SampledSpectrum(1.0);
        [unroll] for (int i = 0; i < NrSpectrumSamples; ++i)
            if (estimatedEnergy[i] > 1.0)
                energyScale_[i] = 1.0 / estimatedEnergy[i];
        return true;
    }

    BsdfEvaluation evaluate(ByteAddressBuffer energyLuts, float3 outgoing)
    {
        BsdfEvaluation result = makeBsdfEvaluation();
        for (int i = 0; i < count_; ++i)
        {
            const BsdfEvaluation lobeEval = lobes_[i].eval(energyLuts, view_, outgoing);
            result.value = result.value + energyScale_ * (lobes_[i].weight * lobeEval.value);
            result.pdf += selectionPdf_[i] * lobeEval.pdf;
        }
        return result;
    }

    int chooseLobe(inout Rng rng)
    {
        const float u = rng.nextFloat();
        float cumulative = 0.0;
        for (int i = 0; i < count_; ++i)
        {
            cumulative += selectionPdf_[i];
            if (u < cumulative)
                return i;
        }
        return count_ - 1;
    }

    BsdfSample sample(ByteAddressBuffer energyLuts, inout Rng rng)
    {
        if (count_ == 0)
            return makeBsdfSample();

        const int chosen = chooseLobe(rng);
        BsdfSample proposal = lobes_[chosen].sample(energyLuts, view_, rng);
        if (proposal.singular || dot(proposal.direction, proposal.direction) == 0.0)
        {
            // A delta lobe has zero density under every other (smooth) lobe's
            // pdf - cross-lobe MIS doesn't apply, only its own selection
            // probability does.
            if (selectionPdf_[chosen] <= 0.0)
                return makeBsdfSample();
            proposal.weight = proposal.weight * lobes_[chosen].weight
                * energyScale_ * (1.0 / selectionPdf_[chosen]);
            proposal.pdf *= selectionPdf_[chosen];
            return proposal;
        }

        // The selected lobe already evaluated f*cos/pdf while producing this
        // sample. Reconstruct its f from that result and only evaluate the
        // other lobes for the mixture MIS terms, removing one expensive
        // GGX/dielectric evaluation on every non-delta sample.
        const float selectedPdf = proposal.pdf;
        const float selectedCosine = abs(dot(lobes_[chosen].normal, proposal.direction));
        if (selectedPdf <= 0.0 || selectedCosine <= 0.0)
            return makeBsdfSample();
        const SampledSpectrum selectedValue = proposal.weight * (selectedPdf / selectedCosine);
        BsdfEvaluation combined = makeBsdfEvaluation();
        combined.value = lobes_[chosen].weight * selectedValue;
        combined.pdf = selectionPdf_[chosen] * selectedPdf;
        for (int i = 0; i < count_; ++i)
        {
            if (i == chosen)
                continue;
            const BsdfEvaluation other = lobes_[i].eval(energyLuts, view_, proposal.direction);
            combined.value += lobes_[i].weight * other.value;
            combined.pdf += selectionPdf_[i] * other.pdf;
        }
        if (!combined.isFinite())
            return makeBsdfSample();
        if (combined.pdf <= 0.0)
            return makeBsdfSample();
        proposal.weight = energyScale_ * combined.value * (selectedCosine / combined.pdf);
        proposal.pdf = combined.pdf;
        return proposal;
    }

    float cosine(float3 direction)
    {
        return abs(dot(shadingNormal_, direction));
    }

    SampledSpectrum evaluateDirect(ByteAddressBuffer energyLuts, float3 outgoing,
        SampledSpectrum radiance)
    {
        const BsdfEvaluation evaluation = evaluate(energyLuts, outgoing);
        return evaluation.value * radiance * cosine(outgoing);
    }

    float3 shadingNormal() { return shadingNormal_; }

    float transmissionEstimate()
    {
        float result = 0.0;
        for (int i = 0; i < count_; ++i)
            result += selectionPdf_[i] * lobes_[i].transmissionEstimate();
        return clamp(result, 0.0, 1.0);
    }
};

NoorRayShaderData makeShaderData(float3 geometricNormal, float3 shadingNormal, float3 view)
{
    NoorRayShaderData data;
    data.view_ = view;
    data.energyScale_ = SampledSpectrum(1.0);
    data.count_ = 0;
    data.overflowed_ = false;
    data.setShadingNormal(geometricNormal, shadingNormal);
    return data;
}
