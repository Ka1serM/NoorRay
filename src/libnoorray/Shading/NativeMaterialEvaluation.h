#pragma once

#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Shading/CompositeBsdf.h"
#include "Shading/Dielectric.h"
#include "Shading/EnergyLut.h"
#include "Shading/Lobes/DielectricLobe.h"
#include "Shading/Lobes/DiffuseLobe.h"
#include "Shading/Material.h"
#include "Shading/MaterialEvaluation.h"
#include "Shading/RgbToSpectrum.h"
#include "Shading/Spectrum.h"

// Native (non-OSL) fallback for a material with no compiled MaterialX/OSL
// program (Material::materialxProgramIndex < 0): either a material with no
// MaterialX network at all, or one whose first compile hasn't landed yet.
// Going through OSL for a graph this simple costs roughly the same as a
// full multi-lobe uber-shader regardless of what it actually computes --
// see docs/MaterialX.md's performance notes -- so this shades Material's
// plain fields directly instead: diffuse + a dielectric specular lobe that
// tints toward the base color as metallic increases. That tint is an
// artist-friendly Schlick-style approximation (the same one
// MX_GENERALIZED_SCHLICK_ID already uses via iorFromNormalReflectance,
// see MaterialClosureBridge.h), not full complex-IOR conductor Fresnel --
// deliberately, since real conductor tables aren't part of Material's
// authoring fields. A material that actually needs that (or transmission,
// coat, sheen, subsurface, textures) should get a real MaterialX network
// instead; this exists for the common untextured/placeholder case, not as
// a second full shading model. The dielectric IOR is reduced from the
// material's own sellmeier coefficients at the path's hero wavelength
// (DielectricLobe's ior is a plain scalar, no dispersion -- same reduction
// MaterialX's own scalar ior inputs already go through); a material that
// needs true spectral dispersion should use hasDispersiveIor's full
// per-wavelength sellmeierIor instead of this fallback.
NR_GPU inline void noorrayEvaluateNativeMaterial(
    const Material& material,
    const SampledWavelengths& wavelengths,
    const float* spectrumScale, const float* spectrumCoeffs,
    const nr::shading::energy_lut::Textures& energyLuts,
    const bool exiting,
    MaterialEvaluation& out,
    nr::shading::NoorRayCompositeBsdf& bsdf)
{
    const float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
    const SampledSpectrum albedoSpectrum = rgbAlbedoToSpectrum(
        material.albedo, wavelengths, spectrumScale, spectrumCoeffs);

    if (metallic < 1.0f)
    {
        nr::shading::lobes::DiffuseLobe diffuse;
        diffuse.albedo = albedoSpectrum;
        diffuse.roughness = material.roughness;
        diffuse.burley = true;
        bsdf.addDiffuse(SampledSpectrum(1.0f - metallic), diffuse);
    }

    const float specularWeight = fmaxf(material.specular, metallic);
    if (specularWeight > 0.0f)
    {
        const float albedoLuma =
            (material.albedo.x + material.albedo.y + material.albedo.z)
            / 3.0f;
        const float metalIor =
            nr::shading::dielectric::iorFromNormalReflectance(albedoLuma);
        const float dielectricIor = sellmeierIor(material.sellmeier, wavelengths[0]);

        nr::shading::lobes::DielectricLobe specular;
        specular.reflectionTint =
            SampledSpectrum(1.0f - metallic) + albedoSpectrum * metallic;
        specular.transmissionTint = SampledSpectrum(0.0f);
        specular.roughness = material.roughness;
        specular.ior = dielectricIor * (1.0f - metallic) + metalIor * metallic;
        specular.exiting = exiting;
        specular.energyLuts = &energyLuts;
        bsdf.addDielectric(SampledSpectrum(specularWeight), specular);
    }

    out.opacity = material.opacity;
    out.albedo = material.albedo;
    if (material.emissionStrength > 0.0f)
    {
        out.emission = material.emission;
        out.emissionStrength = material.emissionStrength;
        out.set(MaterialEvaluationFlags::HasEmission);
    }
    if (material.opacity < 1.0f)
        out.set(MaterialEvaluationFlags::HasCutout);

    bsdf.prepare();
}
