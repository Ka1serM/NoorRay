#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/Host/Platform.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/BsdfClosure.h"
#include "Materials/Shading/Dielectric.h"
#include "Materials/Shading/EnergyLut.h"
#include "Materials/Shading/Ggx.h"
#include "Materials/Shading/Spectrum.h"

// Standalone GGX conductor lobe: full complex-IOR Fresnel (n/k per spectral
// channel), with its own Kulla-Conty multi-scatter compensation -- no
// coupling to any co-existing diffuse/dielectric lobe, matching how
// BSDL/testrender's composite sums independently-compensated lobes.
namespace nr::shading::lobes
{

struct ConductorLobe
{
    SampledSpectrum eta{};
    SampledSpectrum extinction{};
    float roughness{0.5f};
    const nr::shading::energy_lut::Textures* energyLuts{};

    NR_CPU_GPU BsdfEvaluation eval(
        const glm::vec3 normal, const glm::vec3 view,
        const glm::vec3 outgoing) const
    {
        BsdfEvaluation result;
        const float normalOutgoing = glm::dot(normal, outgoing);
        const float normalView = glm::dot(normal, view);
        if (normalOutgoing <= 0.0f || normalView <= 0.0f)
            return result;

        if (nr::shading::ggx::isAlmostSpecular(roughness))
            return evalSingular(normal, view, outgoing);

        const nr::shading::ggx::Evaluation microfacet =
            nr::shading::ggx::evaluateReflection(normal, view, outgoing, roughness);
        if (microfacet.cosineWeighted <= 0.0f)
            return result;

        result.pdf = microfacet.pdf;
        const float geometryFactor = microfacet.cosineWeighted / normalOutgoing;

        const float viewAlbedo = nr::shading::energy_lut::ggxDirectionalAlbedo(
            energyLuts, roughness, normalView);
        const float lightAlbedo = nr::shading::energy_lut::ggxDirectionalAlbedo(
            energyLuts, roughness, normalOutgoing);
        const float averageAlbedo = nr::shading::energy_lut::ggxAverageAlbedo(
            energyLuts, roughness);
        const float missingAverage = 1.0f - averageAlbedo;
        const float msScale = missingAverage > nr::shading::ggx::DenominatorEpsilon
            ? (1.0f - viewAlbedo) * (1.0f - lightAlbedo)
                / (BsdfPi * missingAverage)
            : 0.0f;

        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float fresnel = nr::shading::dielectric::conductorFresnel(
                microfacet.viewHalfCosine, eta[i], extinction[i]);
            const float normalFresnel = nr::shading::dielectric::conductorFresnel(
                1.0f, eta[i], extinction[i]);
            const float conductorAvg = nr::shading::dielectric::averageFresnel(
                nr::shading::dielectric::iorFromNormalReflectance(normalFresnel));
            const float ms = msScale
                * nr::shading::energy_lut::multipleScatterFresnel(
                    conductorAvg, averageAlbedo);
            result.value[i] = fresnel * geometryFactor + ms;
        }
        return result;
    }

    template <typename Rng>
    NR_CPU_GPU BsdfSample sample(
        const glm::vec3 normal, const glm::vec3 view, Rng& rng) const
    {
        BsdfSample result;
        const float normalView = fmaxf(glm::dot(normal, view), 0.0f);
        if (normalView <= 0.0f)
            return result;

        result.event = BsdfEvent::Specular;
        if (nr::shading::ggx::isAlmostSpecular(roughness))
        {
            result.singular = true;
            result.direction = glm::reflect(-view, normal);
            if (glm::dot(normal, result.direction) <= 0.0f)
                return {};
            for (int i = 0; i < NrSpectrumSamples; ++i)
                result.weight[i] = nr::shading::dielectric::conductorFresnel(
                    normalView, eta[i], extinction[i]);
            result.pdf = nr::shading::ggx::SingularPdf;
            return result;
        }

        const glm::vec3 halfVector = nr::shading::ggx::sampleVisibleNormal(
            normal, view, roughness,
            glm::vec2(randomFloat(rng), randomFloat(rng)));
        result.direction = glm::reflect(-view, halfVector);
        if (glm::dot(normal, result.direction) <= 0.0f)
            return {};

        const BsdfEvaluation evaluation = eval(normal, view, result.direction);
        const float outgoingCosine = glm::dot(normal, result.direction);
        if (evaluation.pdf <= 0.0f || outgoingCosine <= 0.0f)
            return {};
        result.weight = evaluation.value * (outgoingCosine / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    NR_CPU_GPU SampledSpectrum albedoEstimate(
        const glm::vec3 normal, const glm::vec3 view) const
    {
        const float normalView = fmaxf(glm::dot(normal, view), 0.0f);
        SampledSpectrum result{};
        if (nr::shading::ggx::isAlmostSpecular(roughness))
        {
            for (int i = 0; i < NrSpectrumSamples; ++i)
                result[i] = nr::shading::dielectric::conductorFresnel(
                    normalView, eta[i], extinction[i]);
            return result;
        }

        // The composite energy guard must estimate the same compensated
        // directional albedo as eval(), rather than only sampling Fresnel at
        // the view angle.  In particular, a white Disney metal must remain
        // exactly white at every roughness: its missing single-scatter energy
        // is returned by the Kulla--Conty term.
        const float directionalGeometry =
            nr::shading::energy_lut::ggxDirectionalAlbedo(
                energyLuts, roughness, normalView);
        const float averageGeometry =
            nr::shading::energy_lut::ggxAverageAlbedo(energyLuts, roughness);
        for (int i = 0; i < NrSpectrumSamples; ++i)
        {
            const float normalFresnel =
                nr::shading::dielectric::conductorFresnel(
                    1.0f, eta[i], extinction[i]);
            const float averageFresnel =
                nr::shading::dielectric::averageFresnel(
                    nr::shading::dielectric::iorFromNormalReflectance(
                        normalFresnel));
            const float multipleScatter =
                nr::shading::energy_lut::multipleScatterFresnel(
                    averageFresnel, averageGeometry);
            result[i] = nr::shading::dielectric::conductorFresnel(
                    normalView, eta[i], extinction[i])
                * directionalGeometry
                + (1.0f - directionalGeometry) * multipleScatter;
        }
        return result;
    }

private:
    NR_CPU_GPU BsdfEvaluation evalSingular(
        const glm::vec3, const glm::vec3, const glm::vec3) const
    {
        // A perfectly smooth conductor is a delta lobe: it contributes
        // nothing to an arbitrary (non-mirror) outgoing direction, exactly
        // like the existing combined Bsdf's smooth-specular handling.
        return {};
    }
};

}
