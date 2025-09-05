#ifndef CLOSEST_HIT
#define CLOSEST_HIT

#include "../Common.glsl"

void buildCoordinateSystem(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.z) < 0.999)
        T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    else
        T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    B = cross(T, N);
}

vec3 sampleDiffuse(vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;
    vec3 local = vec3(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u1)));
    vec3 T, B;
    buildCoordinateSystem(N, T, B);
    return normalize(T * local.x + B * local.y + N * local.z);
}

vec3 evaluateDiffuseBRDF(vec3 albedo, float metallic) {
    return (1 - metallic) * (albedo / PI);
}

float pdfDiffuse(vec3 N, vec3 L) {
    return max(dot(N, L), 0.0) / PI;
}

vec3 sampleGGXVNDF_local(vec3 Vlocal, float roughness, vec2 u) {
    float a = roughness * roughness;
    vec3 Vstretched = normalize(vec3(a * Vlocal.x, a * Vlocal.y, Vlocal.z));
    float phi = 2.0 * PI * u.x;
    float z = (1.0 - u.y) * (1.0 + Vstretched.z) - Vstretched.z;
    float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    vec3 c = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    vec3 Hstretched = c + Vstretched;
    return normalize(vec3(a * Hstretched.x, a * Hstretched.y, Hstretched.z));
}

vec3 sampleHalfVector(vec3 V, vec3 N, float roughness, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    vec3 T, B;
    buildCoordinateSystem(N, T, B);
    mat3 TBN = mat3(T, B, N);
    vec3 Vlocal = transpose(TBN) * V;
    vec3 Hlocal = sampleGGXVNDF_local(Vlocal, roughness, vec2(u1, u2));
    return TBN * Hlocal;
}

float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return NdotV / max(NdotV * (1.0 - k) + k, EPSILON);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 evaluateSpecularBRDF(vec3 normal, vec3 viewDir, vec3 sampledDir, vec3 F, float roughness, vec3 H) {
    float D = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, viewDir, sampledDir, roughness);
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, sampledDir), 0.0);
    return (D * G * F) / max(4.0 * NdotV * NdotL, EPSILON);
}

float pdfSpecular(vec3 V, vec3 N, vec3 H, float roughness) {
    float NdotV = max(dot(N, V), EPSILON);
    float D = distributionGGX(N, H, roughness);
    float G1_V = geometrySchlickGGX(NdotV, roughness);
    return (G1_V * D) / (4.0 * NdotV);
}

float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    // Branchless swap if ray is inside the surface
    float entering = step(0.0, cosThetaI); // 1 if cosThetaI > 0, else 0
    float newEtaI = mix(etaT, etaI, entering);
    float newEtaT = mix(etaI, etaT, entering);
    cosThetaI = mix(-cosThetaI, cosThetaI, entering);

    // Ratio of indices
    float eta = newEtaI / newEtaT;

    // sin^2(theta_t) using Snell's law
    float cosThetaI2 = cosThetaI * cosThetaI;
    float sin2ThetaT = eta * eta * max(0.0, 1.0 - cosThetaI2);

    // Total internal reflection
    if (sin2ThetaT >= 1.0)
        return 1.0;

    float cosThetaT = sqrt(1.0 - sin2ThetaT);

    // Fresnel equations
    float A = newEtaT * cosThetaI;
    float B = newEtaI * cosThetaT;
    float Rs = (A - B) / (A + B);
    float Rp = (A * cosThetaT - B * cosThetaI) / (A * cosThetaT + B * cosThetaI);

    return 0.5 * (Rs * Rs + Rp * Rp);
}

void handleDielectricBSDF(vec3 viewDir, vec3 geometricNormal, vec3 shadingNormal, float roughness, float ior, vec3 transmissionColor, inout Payload payload) {
    payload.flags |= BOUNCE_TRANSMIT;

    vec3 Ns_shading = shadingNormal;
    if (dot(Ns_shading, viewDir) < 0.0)
        Ns_shading = -Ns_shading;

    vec3 H = sampleHalfVector(viewDir, Ns_shading, roughness, payload.rngState);
    float VdotH = max(dot(viewDir, H), 0.0);
    vec3 I = normalize(-viewDir);
    float etaI = 1.0, etaT = ior;
    
    vec3 N_geo = geometricNormal;
    bool exiting = dot(I, N_geo) > 0.0;
    if (exiting) {
        N_geo = -N_geo;
        etaI = ior; etaT = 1.0;
    }

    float reflectProb = max(fresnelDielectric(VdotH, 1.0, ior), EPSILON);
    vec3 refractedDir = refract(I, H, etaI / etaT);
    bool cannotRefract = length(refractedDir) < 1e-5;

    if (cannotRefract || rand(payload.rngState) < reflectProb) {
        vec3 reflectedDir = reflect(-viewDir, H);
        vec3 brdf = evaluateSpecularBRDF(Ns_shading, viewDir, reflectedDir, vec3(reflectProb), roughness, H);
        float pdf = max(pdfSpecular(viewDir, Ns_shading, H, roughness), EPSILON);

        payload.attenuation = (brdf * max(dot(Ns_shading, reflectedDir), 0.0)) / (pdf * reflectProb);
        payload.nextDirection = reflectedDir;

    } else {
        payload.attenuation = transmissionColor / max(1.0 - reflectProb, EPSILON);
        payload.nextDirection = refractedDir;
    }
}

void handleOpaqueBSDF(vec3 viewDir, vec3 geometricNormal, vec3 shadingNormal, vec3 albedo, float metallic, float specular, float roughness, inout Payload payload) {
    float NdotV = dot(shadingNormal, viewDir);

    vec3 normal = shadingNormal;
    if (NdotV < 0.0)
        normal = -normal;

    vec3 geoNormal = geometricNormal;
    if (dot(geometricNormal, viewDir) < 0.0)
            geoNormal = - geoNormal;

    float probSpecular = mix(fresnelDielectric(abs(NdotV), 1.0, 1.5), 1.0, metallic);

    vec3 sampledDir;
    vec3 halfVector;
    if (rand(payload.rngState) < probSpecular) {
        payload.flags |= BOUNCE_SPECULAR;
        halfVector = sampleHalfVector(viewDir, normal, roughness, payload.rngState);
        sampledDir = reflect(-viewDir, halfVector);
    } else {
        payload.flags |= BOUNCE_DIFFUSE;
        sampledDir = sampleDiffuse(normal, payload.rngState);
        halfVector = normalize(viewDir + sampledDir); // Fake half-vector only for MIS/Fresnel weighting
    }
    payload.nextDirection = sampledDir;
        
    // Mix with metallic
    vec3 F_dielectric = vec3(fresnelDielectric(max(dot(viewDir, halfVector), 0.0), 1.0, 1.5));
    vec3 F = mix(F_dielectric, albedo, metallic);

    // BRDF evaluation
    vec3 diffuseBRDF  = evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = evaluateSpecularBRDF(normal, viewDir, sampledDir, F, roughness, halfVector);
    vec3 combinedBRDF = diffuseBRDF + specularBRDF;

    // PDFs
    float pdfSpecular = pdfSpecular(viewDir, normal, halfVector, roughness);
    float pdfDiffuse = pdfDiffuse(normal, sampledDir);
    
    float combinedPdf = max(probSpecular * pdfSpecular + (1.0 - probSpecular) * pdfDiffuse, EPSILON);
    
    payload.attenuation = combinedBRDF * max(dot(normal, sampledDir), 0.0) / combinedPdf;
     
    payload.position += geoNormal * 0.001;
}

void shadeClosestHit(in vec3 worldPosition, in vec3 geometricNormal, in vec3 shadingNormal, in vec3 interpolatedTangent, in vec2 interpolatedUV, in vec3 worldRayDirection, in Material material, inout Payload payload) {
    payload.position = worldPosition;

    float opacity = material.opacity;
    if (material.opacityIndex != -1)
        opacity *= texture(textureSamplers[material.opacityIndex], interpolatedUV).a;

    if (rand(payload.rngState) > opacity) {
        payload.flags |= RAY_TRANSPARENT;
        return;
    }

    vec3 albedo = material.albedo;
    if (material.albedoIndex != -1)
        albedo *= texture(textureSamplers[material.albedoIndex], interpolatedUV).rgb;

    vec3 shadingNormalTextured = shadingNormal;
    if (material.normalIndex != -1) {
        vec3 tangentNormal = texture(textureSamplers[material.normalIndex], interpolatedUV).xyz * 2.0 - 1.0;
        vec3 T = normalize(interpolatedTangent);
        vec3 B = normalize(cross(shadingNormal, T));
        mat3 TBN = mat3(T, B, shadingNormal);
        shadingNormalTextured = normalize(TBN * tangentNormal);
    }

    vec3 emission = material.emission * material.emissionStrength;
    if (material.emissionIndex != -1)
        emission *= texture(textureSamplers[material.emissionIndex], interpolatedUV).rgb;

    float metallic = material.metallic;
    if (material.metallicIndex != -1)
        metallic *= texture(textureSamplers[material.metallicIndex], interpolatedUV).r;

    float specular = material.specular;
    if (material.specularIndex != -1)
        specular *= texture(textureSamplers[material.specularIndex], interpolatedUV).r;
    specular *= 2.0;

    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
        roughness *= texture(textureSamplers[material.roughnessIndex], interpolatedUV).r;
    roughness = clamp(roughness, 0.02, 1.0);

    float transmission = material.transmission;
    if (material.transmissionIndex != -1)
        transmission *= texture(textureSamplers[material.transmissionIndex], interpolatedUV).r;


    payload.albedo = albedo;
    payload.normal = shadingNormalTextured * 0.5 + 0.5;
    payload.emission = emission;

    vec3 viewDir = -worldRayDirection;
    if (rand(payload.rngState) < transmission)
        handleDielectricBSDF(viewDir, geometricNormal, shadingNormalTextured, roughness, material.ior, material.transmissionColor, payload);
    else
        handleOpaqueBSDF(viewDir, geometricNormal, shadingNormalTextured, albedo, metallic, specular, roughness, payload);
}

#endif