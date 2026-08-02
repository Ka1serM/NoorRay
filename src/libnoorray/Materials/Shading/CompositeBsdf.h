#pragma once

#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/BsdfClosure.h"
#include "Materials/Shading/Lobes/ConductorLobe.h"
#include "Materials/Shading/Lobes/DielectricLobe.h"
#include "Materials/Shading/Lobes/DiffuseLobe.h"
#include "Materials/Shading/Spectrum.h"

// Bounded closure storage with combined sampling and evaluation across all
// active lobes.
namespace nr::shading
{

// Closure type stored in the fixed closure pool.
enum class NoorRayClosureType : uint8_t
{
    Diffuse,
    Conductor,
    Dielectric,
};

// MaterialX closure graphs can contain more leaves than a single surface
// model, so the pool follows the SVM closure limit rather than a model limit.
inline constexpr int NoorRayMaxLobes = 64;
inline constexpr float NoorRayClosureWeightCutoff = 1.0e-5f;

struct NoorRayShaderClosure
{
    NoorRayClosureType kind;
    glm::vec3 normal;
    // The authored weight this lobe was assigned while walking the closure
    // tree (add()/mul()/layer() composition), already converted to spectral.
    SampledSpectrum weight;
    // Sample closures proportional to closure_sample_weight,
    // which is the average of the clamped spectral weight. The BSDF's albedo
    // belongs in evaluation, not in closure selection probability.
    float sampleWeight;

    union Storage
    {
        nr::shading::lobes::DiffuseLobe diffuse;
        nr::shading::lobes::ConductorLobe conductor;
        nr::shading::lobes::DielectricLobe dielectric;
        NR_CPU_GPU Storage() {}
    } storage;

    NR_CPU_GPU BsdfEvaluation eval(const glm::vec3 view,
        const glm::vec3 outgoing) const
    {
        switch (kind)
        {
        case NoorRayClosureType::Diffuse:
            return storage.diffuse.eval(normal, view, outgoing);
        case NoorRayClosureType::Conductor:
            return storage.conductor.eval(normal, view, outgoing);
        case NoorRayClosureType::Dielectric:
            return storage.dielectric.eval(normal, view, outgoing);
        }
        return {};
    }

    template <typename Rng>
    NR_CPU_GPU BsdfSample sample(const glm::vec3 view, Rng& rng) const
    {
        switch (kind)
        {
        case NoorRayClosureType::Diffuse:
            return storage.diffuse.sample(normal, view, rng);
        case NoorRayClosureType::Conductor:
            return storage.conductor.sample(normal, view, rng);
        case NoorRayClosureType::Dielectric:
            return storage.dielectric.sample(normal, view, rng);
        }
        return {};
    }

    NR_CPU_GPU SampledSpectrum albedoEstimate(const glm::vec3 view) const
    {
        switch (kind)
        {
        case NoorRayClosureType::Diffuse:
            return storage.diffuse.albedoEstimate(normal, view);
        case NoorRayClosureType::Conductor:
            return storage.conductor.albedoEstimate(normal, view);
        case NoorRayClosureType::Dielectric:
            return storage.dielectric.albedoEstimate(normal, view);
        }
        return {};
    }

    NR_CPU_GPU float transmissionEstimate() const
    {
        return kind == NoorRayClosureType::Dielectric
            ? storage.dielectric.transmissionTint.maxComponent() : 0.0f;
    }
};

// Fixed closure array with admission and mixture-PDF calculation.
class NoorRayShaderData
{
public:
    NR_CPU_GPU NoorRayShaderData(
        const glm::vec3 geometricNormal,
        const glm::vec3 shadingNormalIn,
        const glm::vec3 view)
        : view_(view)
    {
        setShadingNormal(geometricNormal, shadingNormalIn);
    }

    NR_CPU_GPU void setShadingNormal(
        const glm::vec3 geometricNormal, const glm::vec3 shadingNormal)
    {
        const glm::vec3 safeGeometricNormal = nr::safeNormalize(
            geometricNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::vec3 safeShadingNormal = nr::safeNormalize(
            shadingNormal, safeGeometricNormal);
        geometricNormal_ = glm::dot(safeGeometricNormal, view_) < 0.0f
            ? -safeGeometricNormal : safeGeometricNormal;
        shadingNormal_ = glm::dot(safeShadingNormal, view_) < 0.0f
            ? -safeShadingNormal : safeShadingNormal;
        shadingNormal_ = clampShadingNormal(
            geometricNormal_, shadingNormal_, view_);
    }

    // Rotate a shading normal minimally toward the geometric normal so that
    // its reflection stays above the geometric surface. This is the
    // Blender/Cycles grazing-angle correction used by the legacy Bsdf path;
    // the SVM composite needs the same correction for normal-mapped surfaces.
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

    // Record overflow for prepare() instead of silently dropping lobes.
    NR_CPU_GPU bool addDiffuse(const SampledSpectrum weight,
        const nr::shading::lobes::DiffuseLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayClosureType::Diffuse, weight, normal,
            [&](NoorRayShaderClosure::Storage& s) { s.diffuse = lobe; });
    }

    NR_CPU_GPU bool addConductor(const SampledSpectrum weight,
        const nr::shading::lobes::ConductorLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayClosureType::Conductor, weight, normal,
            [&](NoorRayShaderClosure::Storage& s) { s.conductor = lobe; });
    }

    NR_CPU_GPU bool addDielectric(const SampledSpectrum weight,
        const nr::shading::lobes::DielectricLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayClosureType::Dielectric, weight, normal,
            [&](NoorRayShaderClosure::Storage& s) { s.dielectric = lobe; });
    }

    // Set lobe selection probabilities before sample()/evaluate().
    NR_CPU_GPU bool prepare()
    {
        if (overflowed_)
            return false;
        float total = 0.0f;
        for (int i = 0; i < count_; ++i)
        {
            selectionPdf_[i] = lobes_[i].sampleWeight;
            total += selectionPdf_[i];
        }
        for (int i = 0; i < count_; ++i)
            selectionPdf_[i] = total > 0.0f
                ? selectionPdf_[i] / total
                : (count_ > 0 ? 1.0f / static_cast<float>(count_) : 0.0f);
        return true;
    }

    NR_CPU_GPU BsdfEvaluation evaluate(const glm::vec3 outgoing) const
    {
        BsdfEvaluation result;
        for (int i = 0; i < count_; ++i)
        {
            const BsdfEvaluation lobeEval = lobes_[i].eval(view_, outgoing);
            result.value = result.value + lobes_[i].weight * lobeEval.value;
            result.pdf += selectionPdf_[i] * lobeEval.pdf;
        }
        return result;
    }

    template <typename Rng>
    NR_CPU_GPU BsdfSample sample(Rng& rng) const
    {
        if (count_ == 0)
            return {};

        const int chosen = chooseLobe(rng);
        BsdfSample proposal = lobes_[chosen].sample(view_, rng);
        if (proposal.singular
            || glm::dot(proposal.direction, proposal.direction) == 0.0f)
        {
            // A delta lobe has zero density under every other (smooth)
            // lobe's pdf -- cross-lobe MIS doesn't apply, only this lobe's
            // own selection probability does.
            if (selectionPdf_[chosen] <= 0.0f)
                return {};
            proposal.weight = proposal.weight * lobes_[chosen].weight
                * (1.0f / selectionPdf_[chosen]);
            proposal.pdf *= selectionPdf_[chosen];
            return proposal;
        }

        const BsdfEvaluation combined = evaluate(proposal.direction);
        if (!combined.isFinite())
            return {};
        const float outgoingCosine = fabsf(
            glm::dot(lobes_[chosen].normal, proposal.direction));
        if (combined.pdf <= 0.0f || outgoingCosine <= 0.0f)
            return {};
        proposal.weight = combined.value * (outgoingCosine / combined.pdf);
        proposal.pdf = combined.pdf;
        return proposal;
    }

    NR_CPU_GPU SampledSpectrum evaluateDirect(
        const glm::vec3 outgoing, const SampledSpectrum radiance) const
    {
        const BsdfEvaluation evaluation = evaluate(outgoing);
        return evaluation.value * radiance * cosine(outgoing);
    }

    NR_CPU_GPU float cosine(const glm::vec3 direction) const
    {
        return fabsf(glm::dot(shadingNormal_, direction));
    }

    NR_CPU_GPU const glm::vec3& shadingNormal() const { return shadingNormal_; }

    NR_CPU_GPU float transmissionEstimate() const
    {
        float result = 0.0f;
        for (int i = 0; i < count_; ++i)
            result += selectionPdf_[i] * lobes_[i].transmissionEstimate();
        return fminf(fmaxf(result, 0.0f), 1.0f);
    }

private:
    template <typename Assign>
    NR_CPU_GPU bool add(const NoorRayClosureType kind, const SampledSpectrum weight,
        const glm::vec3 requestedNormal,
        Assign assign)
    {
        // Clamp weights and skip negligible closures at allocation time.
        const SampledSpectrum nonNegativeWeight = clampZero(weight);
        const float sampleWeight = fabsf(nonNegativeWeight.average());
        if (sampleWeight < NoorRayClosureWeightCutoff)
            return true;
        if (count_ >= NoorRayMaxLobes)
        {
            overflowed_ = true;
            return false;
        }
        lobes_[count_].kind = kind;
        const glm::vec3 normal = clampShadingNormal(
            geometricNormal_,
            nr::safeNormalize(requestedNormal, shadingNormal_),
            view_);
        lobes_[count_].normal = glm::dot(normal, view_) < 0.0f ? -normal : normal;
        lobes_[count_].weight = nonNegativeWeight;
        lobes_[count_].sampleWeight = sampleWeight;
        assign(lobes_[count_].storage);
        ++count_;
        return true;
    }

    template <typename Rng>
    NR_CPU_GPU int chooseLobe(Rng& rng) const
    {
        const float u = randomFloat(rng);
        float cumulative = 0.0f;
        for (int i = 0; i < count_; ++i)
        {
            cumulative += selectionPdf_[i];
            if (u < cumulative)
                return i;
        }
        return count_ - 1;
    }

    glm::vec3 geometricNormal_{};
    glm::vec3 shadingNormal_{};
    glm::vec3 view_{};

    // Active entries are completely assigned by add(); prepare() assigns
    // selectionPdf_ for every active entry. Clearing all 64 slots on every
    // hit only creates a multi-kilobyte local-memory memset.
    NoorRayShaderClosure lobes_[NoorRayMaxLobes];
    float selectionPdf_[NoorRayMaxLobes];
    int count_{};
    bool overflowed_{};
};

// Compatibility aliases for existing call sites.
using NoorRayCompositeBsdf = NoorRayShaderData;
using NoorRayLobeEntry = NoorRayShaderClosure;
using NoorRayLobeKind = NoorRayClosureType;

}
