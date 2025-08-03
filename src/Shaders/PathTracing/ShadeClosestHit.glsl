#include "Bindings.glsl"

void shadeClosestHit(in vec3 worldPosition, in vec3 geometricNormal, in vec2 interpolatedUV, in vec3 worldRayDirection, in Material material, inout Payload payload) {
    vec3 normal = geometricNormal;
    payload.color = material.emission * material.emissionStrength;
    payload.position = worldPosition;
    payload.normal = normal;

    // Early exit for strong emitters (like light sources)
    if (material.emissionStrength > 1.0) {
        payload.throughput *= material.emission * material.emissionStrength;
        payload.bounceType = BOUNCE_TYPE_DIFFUSE;
        payload.nextDirection = vec3(0.0);
        return;
    }

    // --- Material Property Fetching ---
    vec3 albedo = material.albedo;
    if (material.albedoIndex != -1)
        albedo *= texture(textureSamplers[material.albedoIndex], interpolatedUV).rgb;

    float metallic = material.metallic;
    if (material.metallicIndex != -1)
        metallic *= texture(textureSamplers[material.metallicIndex], interpolatedUV).r;
    metallic = clamp(metallic, 0.0, 1.0);

    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
        roughness *= texture(textureSamplers[material.roughnessIndex], interpolatedUV).r;
    roughness = clamp(roughness, 0.05, 0.99);

    float specular = material.specular;
    if (material.specularIndex != -1)
        specular *= texture(textureSamplers[material.specularIndex], interpolatedUV).r;
    specular *= 2.0; // scale to 0 - 2, as per original artistic control

    // --- Transmissive (Glass/Water) BSDF with Fresnel ---
    if (material.transmissionStrength > 0.0 && rand(payload.rngState) < material.transmissionStrength) {
        payload.throughput *= material.transmission;
        vec3 I = normalize(worldRayDirection);
        vec3 N = normal;
        float etaI = 1.0;
        float etaT = material.ior;

        // Check if ray is entering or exiting the medium and adjust normal/IOR accordingly
        if (dot(I, N) > 0.0) {
            N = -N; // Ray is inside the medium, so we look "out"
            float tmp = etaI; etaI = etaT; etaT = tmp; // Swap IORs
        }

        float cosI = abs(dot(I, N));

        // Calculate reflectance using Schlick's approximation for the Fresnel term
        float r0 = (etaI - etaT) / (etaI + etaT);
        r0 *= r0;
        float reflectance = r0 + (1.0 - r0) * pow(1.0 - cosI, 5.0);

        // Check for Total Internal Reflection (TIR)
        vec3 refractedDir = refract(I, N, etaI / etaT);
        bool cannotRefract = length(refractedDir) < 1e-5;

        // Stochastically choose between reflection and refraction
        if (cannotRefract || rand(payload.rngState) < reflectance) {
            // --- Specular Reflection Path ---
            payload.nextDirection = reflect(I, N);
            payload.bounceType = BOUNCE_TYPE_SPECULAR;
        } else {
            // --- Refraction Path ---
            payload.nextDirection = refractedDir;
            payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        }
        return; // Path is determined, exit shader
    }

    // --- Opaque PBR BSDF (Diffuse + Specular) ---
    vec3 viewDir = normalize(-worldRayDirection);
    if (dot(normal, viewDir) < 0.0) normal = -normal; // Handle two-sided materials for opaque objects

    float NdotV = max(dot(normal, viewDir), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 Fs = fresnelSchlick(NdotV, F0);

    // --- Physically-based Lobe Selection Probability ---
    // The physical specular reflectance based on the Fresnel term.
    float physicalSpecularWeight = max(max(Fs.r, Fs.g), Fs.b);

    // The artistic specular weight includes the 'specular' material parameter.
    // This is crucial for keeping the MIS balanced.
    float artisticSpecularWeight = physicalSpecularWeight * specular;

    // The diffuse weight is based on the PHYSICAL energy split (1 - Fs), 
    // representing the energy available for diffuse reflection.
    float diffuseWeight = (1.0 - physicalSpecularWeight) * (1.0 - metallic) * max(max(albedo.r, albedo.g), albedo.b);

    float totalWeight = artisticSpecularWeight + diffuseWeight;
    float probDiffuse = (totalWeight < EPSILON) ? 0.0 : diffuseWeight / totalWeight;

    // --- Multiple Importance Sampling (MIS) ---
    vec3 sampledDir;
    bool choseDiffuse = rand(payload.rngState) < probDiffuse;
    if (choseDiffuse) {
        sampledDir = sampleDiffuse(normal, payload.rngState);
        payload.bounceType = BOUNCE_TYPE_DIFFUSE;
    } else {
        sampledDir = sampleSpecular(viewDir, normal, roughness, payload.rngState);
        payload.bounceType = BOUNCE_TYPE_SPECULAR;
    }

    // Evaluate PDFs and BRDFs for the chosen direction
    float pdfDiffuseVal = pdfDiffuse(normal, sampledDir);
    float pdfSpecularVal = pdfSpecular(viewDir, normal, roughness, sampledDir);
    vec3 diffuseBRDF = evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    // Update throughput using the correct estimator for mixed importance sampling.
    float combinedPDF = probDiffuse * pdfDiffuseVal + (1.0 - probDiffuse) * pdfSpecularVal;
    float NoL = max(dot(normal, sampledDir), 0.0);

    if (combinedPDF > EPSILON) {
        payload.throughput *= (diffuseBRDF + specularBRDF) * NoL / combinedPDF;
    }

    payload.nextDirection = normalize(sampledDir);
}