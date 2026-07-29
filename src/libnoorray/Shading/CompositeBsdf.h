#pragma once

#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Samplers/RandomSampler.h"
#include "Shading/BsdfClosure.h"
#include "Shading/Lobes/ConductorLobe.h"
#include "Shading/Lobes/DielectricLobe.h"
#include "Shading/Lobes/DiffuseLobe.h"
#include "Shading/Spectrum.h"

// A closure tree can mix an arbitrary number of independent BSDFs (two
// conductors, a diffuse add()ed to a conductor rather than metalness-
// blended, sheen/coat as their own lobes, ...). NoorRayCompositeBsdf holds a
// bounded array of them and combines sampling/evaluation across all active
// lobes, generalizing the exact "propose a direction from one branch, then
// call the full evaluate() to recombine" pattern Bsdf already uses for its
// own diffuse/specular/glass branches (Shading/Bsdf.h) -- just over N lobes
// instead of a hardcoded few. Each lobe carries its own, independent
// multi-scatter energy compensation (no cross-lobe redistribution), matching
// how BSDL/testrender's own CompositeBSDF composes lobes.
namespace nr::shading
{

enum class NoorRayLobeKind : uint8_t
{
    Diffuse,
    Conductor,
    Dielectric,
};

// open_pbr_surface's full closure tree (bxdf/open_pbr_surface.mtlx) can
// structurally contain up to 9 lobe-consuming leaf BSDFs simultaneously --
// diffuse_bsdf, subsurface_thin_walled_reflection_bsdf,
// subsurface_thin_walled_transmission_bsdf, dielectric_transmission,
// dielectric_reflection, dielectric_reflection_tf, metal_bsdf,
// metal_bsdf_tf and coat_bsdf -- regardless of which weights are actually
// nonzero, since MaterialX doesn't prune the tree statically. At the old
// cap of 6, whichever leaves the iterative walk reaches last (in practice
// dielectric_transmission, buried under layer(coat, mix(substrate,
// opaque_base)) several levels deeper than diffuse/specular/metal) were
// silently dropped by add()'s bounds check -- observed as "transmission
// weight has no effect" with an otherwise fully compiled, correct program.
inline constexpr int NoorRayMaxLobes = 10;

struct NoorRayLobeEntry
{
    NoorRayLobeKind kind{};
    glm::vec3 normal{};
    // The authored weight this lobe was assigned while walking the closure
    // tree (add()/mul()/layer() composition), already converted to spectral.
    SampledSpectrum weight{};

    union Storage
    {
        nr::shading::lobes::DiffuseLobe diffuse;
        nr::shading::lobes::ConductorLobe conductor;
        nr::shading::lobes::DielectricLobe dielectric;
        NR_CPU_GPU Storage() {}
    } storage{};

    NR_CPU_GPU BsdfEvaluation eval(const glm::vec3 view,
        const glm::vec3 outgoing) const
    {
        switch (kind)
        {
        case NoorRayLobeKind::Diffuse:
            return storage.diffuse.eval(normal, view, outgoing);
        case NoorRayLobeKind::Conductor:
            return storage.conductor.eval(normal, view, outgoing);
        case NoorRayLobeKind::Dielectric:
            return storage.dielectric.eval(normal, view, outgoing);
        }
        return {};
    }

    NR_CPU_GPU BsdfSample sample(const glm::vec3 view, RandomState& rng) const
    {
        switch (kind)
        {
        case NoorRayLobeKind::Diffuse:
            return storage.diffuse.sample(normal, view, rng);
        case NoorRayLobeKind::Conductor:
            return storage.conductor.sample(normal, view, rng);
        case NoorRayLobeKind::Dielectric:
            return storage.dielectric.sample(normal, view, rng);
        }
        return {};
    }

    NR_CPU_GPU SampledSpectrum albedoEstimate(const glm::vec3 view) const
    {
        switch (kind)
        {
        case NoorRayLobeKind::Diffuse:
            return storage.diffuse.albedoEstimate(normal, view);
        case NoorRayLobeKind::Conductor:
            return storage.conductor.albedoEstimate(normal, view);
        case NoorRayLobeKind::Dielectric:
            return storage.dielectric.albedoEstimate(normal, view);
        }
        return {};
    }

    NR_CPU_GPU float transmissionEstimate() const
    {
        return kind == NoorRayLobeKind::Dielectric
            ? storage.dielectric.transmissionTint.maxComponent() : 0.0f;
    }
};

class NoorRayCompositeBsdf
{
public:
    NR_CPU_GPU NoorRayCompositeBsdf(
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
        geometricNormal_ = glm::dot(geometricNormal, view_) < 0.0f
            ? -geometricNormal : geometricNormal;
        shadingNormal_ = glm::dot(shadingNormal, view_) < 0.0f
            ? -shadingNormal : shadingNormal;
    }

    // Bounds-checked: silently drops the lobe past NoorRayMaxLobes rather
    // than corrupting memory (the OSL-era closure bridge's own closure-pool
    // bump allocator used the same overflow policy).
    NR_CPU_GPU bool addDiffuse(const SampledSpectrum weight,
        const nr::shading::lobes::DiffuseLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayLobeKind::Diffuse, weight, normal,
            [&](NoorRayLobeEntry::Storage& s) { s.diffuse = lobe; });
    }

    NR_CPU_GPU bool addConductor(const SampledSpectrum weight,
        const nr::shading::lobes::ConductorLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayLobeKind::Conductor, weight, normal,
            [&](NoorRayLobeEntry::Storage& s) { s.conductor = lobe; });
    }

    NR_CPU_GPU bool addDielectric(const SampledSpectrum weight,
        const nr::shading::lobes::DielectricLobe& lobe,
        const glm::vec3 normal = glm::vec3(0.0f))
    {
        return add(NoorRayLobeKind::Dielectric, weight, normal,
            [&](NoorRayLobeEntry::Storage& s) { s.dielectric = lobe; });
    }

    // Must be called once after all lobes are added and before
    // sample()/evaluate(): sets each lobe's selection probability
    // proportional to weight * albedo(view), matching testrender's
    // CompositeBSDF::prepare().
    NR_CPU_GPU void prepare()
    {
        float total = 0.0f;
        for (int i = 0; i < count_; ++i)
        {
            const SampledSpectrum albedo =
                lobes_[i].albedoEstimate(view_);
            selectionPdf_[i] = fmaxf(
                (lobes_[i].weight * albedo).maxComponent(), 0.0f);
            total += selectionPdf_[i];
        }
        for (int i = 0; i < count_; ++i)
            selectionPdf_[i] = total > 0.0f
                ? selectionPdf_[i] / total
                : (count_ > 0 ? 1.0f / static_cast<float>(count_) : 0.0f);
    }

    NR_CPU_GPU BsdfEvaluation evaluate(const glm::vec3 outgoing) const
    {
        if (!accepts(outgoing))
            return {};
        BsdfEvaluation result;
        for (int i = 0; i < count_; ++i)
        {
            const BsdfEvaluation lobeEval = lobes_[i].eval(view_, outgoing);
            result.value = result.value + lobes_[i].weight * lobeEval.value;
            result.pdf += selectionPdf_[i] * lobeEval.pdf;
        }
        return result;
    }

    NR_CPU_GPU BsdfSample sample(RandomState& rng) const
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
    NR_CPU_GPU bool add(const NoorRayLobeKind kind, const SampledSpectrum weight,
        const glm::vec3 requestedNormal,
        Assign assign)
    {
        if (count_ >= NoorRayMaxLobes)
            return false;
        lobes_[count_].kind = kind;
        const glm::vec3 normal = glm::dot(requestedNormal, requestedNormal) > 1.0e-12f
            ? glm::normalize(requestedNormal) : shadingNormal_;
        lobes_[count_].normal = glm::dot(normal, view_) < 0.0f ? -normal : normal;
        lobes_[count_].weight = weight;
        assign(lobes_[count_].storage);
        ++count_;
        return true;
    }

    NR_CPU_GPU int chooseLobe(RandomState& rng) const
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

    NR_CPU_GPU bool accepts(const glm::vec3 direction) const
    {
        const float geometricCosine = glm::dot(geometricNormal_, direction);
        const float shadingCosine = glm::dot(shadingNormal_, direction);
        return geometricCosine != 0.0f && shadingCosine != 0.0f
            && (geometricCosine > 0.0f) == (shadingCosine > 0.0f);
    }

    glm::vec3 geometricNormal_{};
    glm::vec3 shadingNormal_{};
    glm::vec3 view_{};

    NoorRayLobeEntry lobes_[NoorRayMaxLobes]{};
    float selectionPdf_[NoorRayMaxLobes]{};
    int count_{};
};

}
