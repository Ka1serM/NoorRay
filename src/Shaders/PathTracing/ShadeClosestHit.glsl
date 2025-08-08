#include "Bindings.glsl"

void shadeClosestHit(
in vec3 worldPosition,
in vec3 interpolatedNormal,
in vec3 interpolatedTangent,
in vec2 interpolatedUV,
in vec3 worldRayDirection,
in Material material,
inout Payload payload
) {
    // --- Initialize ---
    vec3 normal = interpolatedNormal;
    payload.position = worldPosition;
    payload.color = material.emission * material.emissionStrength;

    // --- Normal Mapping ---
    if (material.normalIndex != -1) {
        vec3 tangentNormal = texture(textureSamplers[material.normalIndex], interpolatedUV).xyz;
        tangentNormal = normalize(tangentNormal * 2.0 - 1.0); // [0,1] → [-1,1]

        vec3 T = normalize(interpolatedTangent);
        vec3 B = normalize(cross(normal, T));
        vec3 N = normalize(normal);
        mat3 TBN = mat3(T, B, N);

        normal = normalize(TBN * tangentNormal); // World-space normal
    }
    payload.normal = normal;

    // --- Albedo ---
    vec3 albedo = material.albedo;
    if (material.albedoIndex != -1)
    albedo *= texture(textureSamplers[material.albedoIndex], interpolatedUV).rgb;
    
    payload.albedo = albedo;

    // --- Metallic ---
    float metallic = material.metallic;
    if (material.metallicIndex != -1)
    metallic *= texture(textureSamplers[material.metallicIndex], interpolatedUV).r;
    metallic = clamp(metallic, 0.0, 1.0);

    // --- Roughness ---
    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
    roughness *= texture(textureSamplers[material.roughnessIndex], interpolatedUV).r;
    roughness = clamp(roughness, 0.05, 0.99);

    // --- Specular ---
    float specular = material.specular;
    if (material.specularIndex != -1)
    specular *= texture(textureSamplers[material.specularIndex], interpolatedUV).r;
    specular *= 2.0;
    
    // --- Transmission (Glass, Water) ---
    if (material.transmissionStrength > 0.0 && rand(payload.rngState) < material.transmissionStrength) {
        payload.throughput *= material.transmission;
        vec3 I = normalize(worldRayDirection);
        vec3 N = normal;

        float etaI = 1.0;
        float etaT = material.ior;

        if (dot(I, N) > 0.0) {
            N = -N;
            float tmp = etaI; etaI = etaT; etaT = tmp;
        }

        float cosI = abs(dot(I, N));
        float r0 = (etaI - etaT) / (etaI + etaT);
        r0 *= r0;
        float reflectance = r0 + (1.0 - r0) * pow(1.0 - cosI, 5.0);

        vec3 refractedDir = refract(I, N, etaI / etaT);
        bool cannotRefract = length(refractedDir) < 1e-5;

        if (cannotRefract || rand(payload.rngState) < reflectance) {
            payload.nextDirection = reflect(I, N);
            payload.bounceType = BOUNCE_TYPE_SPECULAR;
        } else {
            payload.nextDirection = refractedDir;
            payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        }
        return;
    }

    // --- Opaque PBR Lighting ---
    vec3 viewDir = normalize(-worldRayDirection);
    if (dot(normal, viewDir) < 0.0)
    normal = -normal;

    float NdotV = max(dot(normal, viewDir), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 Fs = fresnelSchlick(NdotV, F0);

    float physicalSpecularWeight = max(max(Fs.r, Fs.g), Fs.b);
    float artisticSpecularWeight = physicalSpecularWeight * specular;
    float diffuseWeight = (1.0 - physicalSpecularWeight) * (1.0 - metallic) * max(max(albedo.r, albedo.g), albedo.b);
    float totalWeight = artisticSpecularWeight + diffuseWeight;
    float probDiffuse = (totalWeight < EPSILON) ? 0.0 : diffuseWeight / totalWeight;

    // --- Importance Sampling ---
    vec3 sampledDir;
    bool choseDiffuse = rand(payload.rngState) < probDiffuse;

    if (choseDiffuse) {
        sampledDir = sampleDiffuse(normal, payload.rngState);
        payload.bounceType = BOUNCE_TYPE_DIFFUSE;
    } else {
        sampledDir = sampleSpecular(viewDir, normal, roughness, payload.rngState);
        payload.bounceType = BOUNCE_TYPE_SPECULAR;
    }

    float pdfDiffuseVal = pdfDiffuse(normal, sampledDir);
    float pdfSpecularVal = pdfSpecular(viewDir, normal, roughness, sampledDir);
    vec3 diffuseBRDF = evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    float combinedPDF = probDiffuse * pdfDiffuseVal + (1.0 - probDiffuse) * pdfSpecularVal;
    float NoL = max(dot(normal, sampledDir), 0.0);

    if (combinedPDF > EPSILON) {
        payload.throughput *= (diffuseBRDF + specularBRDF) * NoL / combinedPDF;
    }

    payload.nextDirection = normalize(sampledDir);
}