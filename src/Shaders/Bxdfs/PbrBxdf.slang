#ifndef PBR_BXDF_SLANG
#define PBR_BXDF_SLANG

#include "../Core/Core.slang"
#include "Brdf.slang"
#include "IBxdf.slang"

struct PbrBxdf : IBxdf
{
    BxdfEval evalOpaque(
        float3 viewDir,
        float3 perturbedNormal,
        float3 geometricNormal,
        float3 lightDir,
        float3 albedo,
        float  specular,
        float  metallic,
        float  roughness,
        float  swSpecular)
    {
        BxdfEval result;
        result.f   = float3(0.0);
        result.pdf = 0.0;

        if (max(dot(geometricNormal, lightDir), 0.0) <= 0.0)
            return result;

        float3 N = perturbedNormal;
        float3 V = viewDir;
        float3 L = lightDir;
        float3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float VdotL = dot(V, L);

        if (NdotL <= 0.0 || NdotV <= 0.0)
            return result;

        float  dielectricF0 = dielectricF0FromSpecular(specular);
        float3 F0           = lerp(float3(dielectricF0), albedo, metallic);
        // Proper Fresnel: exact dielectric for non-metals, Schlick for conductors.
        float3 F = fresnelMaterial(F0, metallic, VdotH, specular);

        float3 diffuseBRDF  = (1.0 - F) * evaluateDiffuseBRDF(albedo, metallic, roughness, NdotL, NdotV, VdotL);
        float3 specularBRDF = evaluateSpecularBRDF(N, V, L, F, roughness, H);

        float specPdf = pdfSpecular(V, N, H, roughness);
        float diffPdf = pdfDiffuse(N, L);
        result.pdf = max(swSpecular * specPdf + (1.0 - swSpecular) * diffPdf, EPSILON);
        result.f   = diffuseBRDF + specularBRDF;
        return result;
    }
};

#endif // PBR_BXDF_SLANG
