#include "Bindings.glsl" 

void shadeClosestHit(in vec3 worldPosition, in vec3 geometricNormal, in vec2 interpolatedUV, in vec3 worldRayDirection, in Material material, inout Payload payload) {
    // --- Payload Initialization ---
    vec3 normal = geometricNormal;
    payload.color = material.emission;
    payload.position = worldPosition;
    payload.normal = normal;

    // --- Material Property Fetching ---
    vec3 albedo = material.albedo;
    if (material.albedoIndex != -1)
    albedo *= texture(textureSamplers[material.albedoIndex], interpolatedUV).rgb;

    float metallic = material.metallic;
    if (material.metallicIndex != -1)
    metallic *= texture(textureSamplers[material.metallicIndex], interpolatedUV).r;
    metallic = clamp(metallic, 0.05, 0.99);

    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
    roughness *= texture(textureSamplers[material.roughnessIndex], interpolatedUV).r;
    roughness = clamp(roughness, 0.05, 0.99);

    float specular = material.specular;
    if (material.specularIndex != -1)
    specular *= texture(textureSamplers[material.specularIndex], interpolatedUV).r;
    specular *= 2.0; // scale to 0 - 2

    // --- Refraction Logic ---
    float transmissionFactor = (material.transmission.r + material.transmission.g + material.transmission.b) / 3.0;
    if (transmissionFactor > 0.0 && rand(payload.rngState) < transmissionFactor) {
        vec3 I = normalize(worldRayDirection);
        float etaI = 1.0, etaT = material.ior;
        if (dot(I, normal) > 0.0) {
            normal = -normal;
            float tmp = etaI; etaI = etaT; etaT = tmp;
        }
        vec3 refracted = refract(I, normal, etaI / etaT);
        if (length(refracted) < 1e-5)
        payload.nextDirection = reflect(I, normal);
        else
        payload.nextDirection = refracted;

        payload.throughput *= material.transmission;
        payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        return;
    }

    // --- Diffuse + Specular PBR Logic ---
    vec3 viewDir = normalize(-worldRayDirection);
    if (dot(normal, viewDir) < 0.0) normal = -normal;

    float NdotV = max(dot(normal, viewDir), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float fresnelAtNdotV = fresnelSchlick(NdotV, F0).r;
    float diffuseEnergy = (1.0 - metallic) * (1.0 - fresnelAtNdotV);
    float specularEnergy = max(fresnelAtNdotV, 0.04) * max(1.0 - roughness * roughness, 0.05);
    float sumEnergy = diffuseEnergy + specularEnergy + 1e-6;
    float probDiffuse = diffuseEnergy / sumEnergy;

    bool choseDiffuse = bool(rand(payload.rngState) < probDiffuse);
    vec3 sampledDir = choseDiffuse ? sampleDiffuse(normal, payload.rngState) : sampleSpecular(viewDir, normal, roughness, payload.rngState);

    float pdfDiffuseVal = max(pdfDiffuse(normal, sampledDir), 1e-6);
    float pdfSpecularVal = max(pdfSpecular(viewDir, normal, roughness, sampledDir), 1e-6);
    vec3 diffuseBRDF = evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    float wDiffuse = probDiffuse * pdfDiffuseVal;
    float wSpecular = (1.0 - probDiffuse) * pdfSpecularVal;
    float misWeight = choseDiffuse ? (wDiffuse * wDiffuse) : (wSpecular * wSpecular);
    misWeight /= (wDiffuse * wDiffuse + wSpecular * wSpecular + 1e-6);

    float pdfCombined = probDiffuse * pdfDiffuseVal + (1.0 - probDiffuse) * pdfSpecularVal;
    float NoL = max(dot(normal, sampledDir), 0.0);

    payload.throughput *= (diffuseBRDF + specularBRDF) * NoL * misWeight / pdfCombined;
    payload.nextDirection = normalize(sampledDir);
    payload.bounceType = choseDiffuse ? BOUNCE_TYPE_DIFFUSE : BOUNCE_TYPE_SPECULAR;
}