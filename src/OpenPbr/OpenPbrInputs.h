#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/Texture.h"
#include "Mesh/Material.h"

// OpenPbrInterop.h supplies GLM-backed vec2/vec3/vec4 (OPENPBR_USE_CUSTOM_VEC_TYPES)
// and forces OPENPBR_LANGUAGE_TARGET_CUDA; the vendor's own
// interop/openpbr_interop_cuda.h (auto-included by openpbr.h) provides everything else.
#include "OpenPbr/OpenPbrInterop.h"
#include "OpenPbr/OpenPbrLuts.h"

#include "../../external/openpbr-bsdf/openpbr.h"

// The vendor's CUDA interop (interop/openpbr_interop_cuda.h) defines several
// GLSL-style helpers as unscoped, unguarded function-like macros (mix, equal,
// notEqual, greaterThan, greaterThanEqual, all, any). Left defined, they
// collide with unrelated identically-named identifiers pulled in by later
// includes in the same translation unit (e.g. std::equal via <algorithm>).
// They're only needed while openpbr.h's own headers are being parsed above,
// so undefine them here rather than patching the vendored header.
#undef mix
#undef equal
#undef notEqual
#undef greaterThan
#undef greaterThanEqual
#undef all
#undef any

namespace nr::openpbr {

// Build an OpenPBR_ResolvedInputs from Material + textures.
// ior: IOR at the relevant (hero) wavelength.
// opacity: pre-resolved from texture (0–1); pass 1.0 if already handled.
NR_GPU inline OpenPBR_ResolvedInputs buildInputs(
    const Material& material,
    const CudaTexture* textures,
    glm::vec2 uv,
    glm::vec3 shadingNormal,
    glm::vec3 tangent,
    float opacity,
    float ior)
{
    // --- Resolve textures ---
    auto sampleTexture = [&](int index, float fallback) -> float {
        if (index < 0) return fallback;
        return textures[index].sample(uv).x;
    };
    auto sampleTextureRgb = [&](int index, glm::vec3 fallback) -> glm::vec3 {
        if (index < 0) return fallback;
        return glm::vec3(textures[index].sample(uv));
    };

    const glm::vec3 baseColor = sampleTextureRgb(material.albedoIndex, material.albedo);
    const float metallic = fminf(fmaxf(
        sampleTexture(material.metallicIndex, material.metallic), 0.0f), 1.0f);
    const float roughness = fminf(fmaxf(
        sampleTexture(material.roughnessIndex, material.roughness), 0.0f), 1.0f);
    const float specular = fminf(fmaxf(
        sampleTexture(material.specularIndex, material.specular), 0.0f), 1.0f);
    const float transmission = fminf(fmaxf(
        sampleTexture(material.transmissionIndex, material.transmission), 0.0f), 1.0f);
    const glm::vec3 emissionColor = sampleTextureRgb(material.emissionIndex, material.emission);
    const float emissionStrength = material.emissionStrength;

    // --- Build the basis ---
    glm::vec3 n = shadingNormal;
    glm::vec3 t = glm::normalize(tangent - glm::dot(tangent, n) * n);
    glm::vec3 b = glm::cross(n, t);

    OpenPBR_Basis basis;
    basis.n = vec3(n.x, n.y, n.z);
    basis.t = vec3(t.x, t.y, t.z);
    basis.b = vec3(b.x, b.y, b.z);

    // --- Fill resolved inputs ---
    // Use the OpenPBR default initializer and override what matters.
    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();

    inputs.base_weight = 1.0f;
    inputs.base_color = vec3(baseColor.x, baseColor.y, baseColor.z);
    inputs.base_diffuse_roughness = roughness;
    inputs.base_metalness = metallic;

    inputs.specular_weight = specular;
    inputs.specular_color = vec3(1.0f);
    inputs.specular_roughness = roughness;
    inputs.specular_roughness_anisotropy = 0.0f;
    inputs.specular_ior = ior;
    inputs.specular_anisotropy_rotation_cos_sin = vec2(1.0f, 0.0f);

    inputs.transmission_weight = transmission;
    inputs.transmission_color = vec3(
        material.transmissionColor.x,
        material.transmissionColor.y,
        material.transmissionColor.z);
    inputs.transmission_dispersion_scale = 0.0f;

    inputs.emission_luminance = emissionStrength;
    inputs.emission_color = vec3(emissionColor.x, emissionColor.y, emissionColor.z);

    inputs.geometry_opacity = opacity;
    inputs.geometry_thin_walled = false;

    inputs.geometry_basis = basis;
    inputs.geometry_coat_basis = basis;

    return inputs;
}

} // namespace nr::openpbr
