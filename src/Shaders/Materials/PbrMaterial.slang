#ifndef PBR_MATERIAL_SLANG
#define PBR_MATERIAL_SLANG

#include "../Bindings.slang"

float3 sampleMaterialAlbedo(Material material, float2 uv)
{
    float3 albedo = material.albedo;
    if (material.albedoIndex != -1)
        albedo *= textureSamplers[material.albedoIndex].Sample(uv).rgb;
    return albedo;
}

float3 sampleMaterialEmission(Material material, float2 uv)
{
    float3 emission = material.emission * material.emissionStrength;
    if (material.emissionIndex != -1)
        emission *= textureSamplers[material.emissionIndex].Sample(uv).rgb;
    return emission;
}

float sampleMaterialMetallic(Material material, float2 uv)
{
    float metallic = material.metallic;
    if (material.metallicIndex != -1)
        metallic *= textureSamplers[material.metallicIndex].Sample(uv).r;
    return metallic;
}

float sampleMaterialSpecular(Material material, float2 uv)
{
    float specular = material.specular;
    if (material.specularIndex != -1)
        specular *= textureSamplers[material.specularIndex].Sample(uv).r;
    return specular;
}

float sampleMaterialRoughness(Material material, float2 uv)
{
    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
        roughness *= textureSamplers[material.roughnessIndex].Sample(uv).r;
    return roughness;
}

float sampleMaterialOpacity(Material material, float2 uv)
{
    float opacity = material.opacity;
    if (material.opacityIndex != -1)
        opacity *= textureSamplers[material.opacityIndex].Sample(uv).a;
    return clamp(opacity, 0.0, 1.0);
}

float sampleMaterialTransmission(Material material, float2 uv)
{
    float transmission = material.transmission;
    if (material.transmissionIndex != -1)
        transmission *= textureSamplers[material.transmissionIndex].Sample(uv).r;
    return clamp(transmission, 0.0, 1.0);
}

void sampleMaterialPbrInputs(
    in Material material,
    in float2 uv,
    out float3 albedo,
    out float3 emission,
    out float metallic,
    out float specular,
    out float roughness)
{
    albedo = sampleMaterialAlbedo(material, uv);
    emission = sampleMaterialEmission(material, uv);
    metallic = sampleMaterialMetallic(material, uv);
    specular = clamp(sampleMaterialSpecular(material, uv) * 2.0, 0.0, 1.0);
    roughness = clamp(sampleMaterialRoughness(material, uv), 0.02, 1.0);
}

void samplePbrMaterial(
    in Material material,
    in float2 uv,
    out float3 albedo,
    out float3 emission,
    out float metallic,
    out float specular,
    out float roughness,
    out float opacity,
    out float transmission)
{
    sampleMaterialPbrInputs(material, uv, albedo, emission, metallic, specular, roughness);
    opacity = sampleMaterialOpacity(material, uv);
    transmission = sampleMaterialTransmission(material, uv);
}

#endif // PBR_MATERIAL_SLANG
