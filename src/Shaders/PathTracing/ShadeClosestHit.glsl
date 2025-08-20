#include "../Common.glsl"

// Build a local orthonormal coordinate system from a normal.
void buildCoordinateSystem(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.z) < 0.999)
        T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    else
        T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    B = cross(T, N);
}

// D(h): GGX Normal Distribution.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / max(denom, EPSILON);
}

// G: Schlick-GGX geometry (for Smith joint).
float geometrySchlickGGX(float NdotV, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return NdotV / max(NdotV * (1.0 - k) + k, EPSILON);
}

// G(v, l): Smith joint shadowing.
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

// Full Fresnel equations for dielectrics (non-metals).
float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);

    // Swap on entering
    if (!(cosThetaI > 0.0)) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = -cosThetaI;
    }

    // Compute sin^2 θt directly
    float eta = etaI / etaT;
    float sin2ThetaT = eta * eta * max(0.0, 1.0 - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sin2ThetaT >= 1.0)
        return 1.0;

    float cosThetaT = sqrt(1.0 - sin2ThetaT);

    // Fresnel terms
    float A = etaT * cosThetaI;
    float B = etaI * cosThetaT;
    float Rs = (A - B) / (A + B);
    float Rp = (A * cosThetaT - B * cosThetaI) / (A * cosThetaT + B * cosThetaI);

    return 0.5 * (Rs * Rs + Rp * Rp);
}

// --- SAMPLING FUNCTIONS ---

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

vec3 sampleH(vec3 V, vec3 N, float roughness, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    vec3 T, B;
    buildCoordinateSystem(N, T, B);
    mat3 TBN = mat3(T, B, N);
    vec3 Vlocal = transpose(TBN) * V;
    vec3 Hlocal = sampleGGXVNDF_local(Vlocal, roughness, vec2(u1, u2));
    return TBN * Hlocal;
}


// --- PDF & BRDF EVALUATION ---

float pdfDiffuse(vec3 N, vec3 L) {
    return max(dot(N, L), 0.0) / PI;
}

float pdfSpecular(vec3 V, vec3 N, vec3 H, float roughness) {
    float D = distributionGGX(N, H, roughness);
    float NdotV = max(dot(N, V), 0.0);
    float G1_V = geometrySchlickGGX(NdotV, roughness);
    return (D * G1_V) / max(4.0 * NdotV, EPSILON);
}

vec3 evaluateDiffuseBRDF(vec3 albedo, float metallic, vec3 F) {
    return (1.0 - metallic) * (albedo / PI) * (vec3(1.0) - F);
}

vec3 evaluateSpecularBRDF(vec3 normal, vec3 viewDir, vec3 sampledDir, vec3 F, float roughness, vec3 H) {
    float D = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, viewDir, sampledDir, roughness);
    float NdotV = max(dot(normal, viewDir), 0.0);
    float NdotL = max(dot(normal, sampledDir), 0.0);
    return (D * G * F) / max(4.0 * NdotV * NdotL, EPSILON);
}


void handleDielectricBSDF(vec3 viewDir, vec3 shadingNormal, float roughness, float ior, vec3 transmissionColor, inout Payload payload) {
    vec3 Ns_shading = shadingNormal;
    //  forward facing
    if (dot(Ns_shading, viewDir) < 0.0)
        Ns_shading = -Ns_shading;

    payload.bounceType = BOUNCE_TYPE_TRANSMISSION;

    // Sample the microfacet normal H
    vec3 H = sampleH(viewDir, Ns_shading, roughness, payload.rngState);
    float VdotH = max(dot(viewDir, H), 0.0);

    vec3 I = normalize(-viewDir);
    float etaI = 1.0, etaT = ior;
    vec3 N_geo = shadingNormal;
    if (dot(I, N_geo) > 0.0) { // Exiting the medium
       N_geo = -N_geo;
       etaI = ior; etaT = 1.0;
    }

    float reflectProb = fresnelDielectric(VdotH, 1.0, ior);

    // Attempt to refract
    vec3 refractedDir = refract(I, H, etaI / etaT);
    bool cannotRefract = length(refractedDir) < 1e-5; // TIR

    if (cannotRefract || rand(payload.rngState) < reflectProb) {
        // --- Handle Reflection ---
        vec3 reflectedDir = reflect(-viewDir, H);
        // For dielectrics, Fresnel is a scalar (uncolored reflection)
        vec3 F = vec3(reflectProb);
        vec3 brdf = evaluateSpecularBRDF(Ns_shading, viewDir, reflectedDir, F, roughness, H);
        float pdf = pdfSpecular(viewDir, Ns_shading, H, roughness);

        if (pdf > EPSILON && reflectProb > EPSILON) {
            float NdotL = max(dot(Ns_shading, reflectedDir), 0.0);
            payload.throughput *= (brdf * NdotL) / (pdf * reflectProb);
            payload.nextDirection = reflectedDir;
        } else
            payload.throughput = vec3(0.0);
    } else {
        // --- Handle Refraction ---
        float transProb = 1.0 - reflectProb;
        if (transProb > EPSILON) {
            payload.throughput *= transmissionColor / transProb; // Unbiased
            payload.nextDirection = refractedDir;
        } else
            payload.throughput = vec3(0.0);
    }
}
void handleOpaqueBSDF(
    vec3 viewDir,
    vec3 shadingNormal,
    vec3 albedo,
    float metallic,
    float roughness,
inout Payload payload
) {
    // Face-forward the normal
    vec3 N = shadingNormal;
    if (dot(N, viewDir) < 0.0)
        N = -N;

    // --- Fresnel for lobe selection (scalar) ---
    float VdotN = max(dot(viewDir, N), 0.0);
    float F_dielectric_scalar = fresnelDielectric(VdotN, 1.0, 1.5);
    float F_scalar = mix(F_dielectric_scalar, 1.0, metallic); // metals are mostly specular

    float specularWeight = F_scalar;
    float diffuseWeight  = (1.0 - F_scalar) * (1.0 - metallic);
    float totalWeight    = specularWeight + diffuseWeight;
    float probSpecular   = (totalWeight < EPSILON) ? 0.5 : specularWeight / totalWeight;

    // --- Sample direction ---
    vec3 sampledDir;
    vec3 H; // half-vector
    if (rand(payload.rngState) < probSpecular) {
        payload.bounceType = BOUNCE_TYPE_SPECULAR;
        H = sampleH(viewDir, N, roughness, payload.rngState);
        sampledDir = reflect(-viewDir, H);
    } else {
        payload.bounceType = BOUNCE_TYPE_DIFFUSE;
        sampledDir = sampleDiffuse(N, payload.rngState);
        H = normalize(viewDir + sampledDir);
    }

    // --- Evaluate BRDFs (non-colored conductor, tinted by albedo) ---
    float VdotH = max(dot(viewDir, H), 0.0);
    vec3 F_dielectric = vec3(fresnelDielectric(VdotH, 1.0, 1.5));
    vec3 F = mix(F_dielectric, albedo, metallic);

    vec3 diffuseBRDF  = evaluateDiffuseBRDF(albedo, metallic, F);
    vec3 specularBRDF = evaluateSpecularBRDF(N, viewDir, sampledDir, F, roughness, H);
    vec3 bsdf = diffuseBRDF + specularBRDF;

    // --- Compute MIS PDF ---
    float p_spec = pdfSpecular(viewDir, N, H, roughness);
    float p_diff = pdfDiffuse(N, sampledDir);
    float mis_pdf = probSpecular * p_spec + (1.0 - probSpecular) * p_diff;

    // --- Update payload ---
    if (mis_pdf > EPSILON) {
        float NdotL = max(dot(N, sampledDir), 0.0);
        payload.throughput *= bsdf * NdotL / mis_pdf;
        payload.nextDirection = sampledDir;
    } else
        payload.throughput = vec3(0.0);
}

void handleBSDF(vec3 viewDir, vec3 albedo, float metallic, float roughness, vec3 normal, float ior, vec3 transmissionColor, float transmissionStrength, vec3 emission, inout Payload payload) {
    payload.color = emission;
    payload.albedo = albedo;
    payload.normal = normal;
    
    if (rand(payload.rngState) < transmissionStrength)
        handleDielectricBSDF(viewDir, normal, roughness, ior, transmissionColor, payload);
    else
        handleOpaqueBSDF(viewDir, normal, albedo, metallic, roughness, payload);
}

void shadeClosestHit(in vec3 worldPosition, in vec3 interpolatedNormal, in vec3 interpolatedTangent, in vec2 interpolatedUV, in vec3 worldRayDirection, in Material material, inout Payload payload) {
    payload.position = worldPosition;
    
    vec3 normal = normalize(interpolatedNormal);
    // Apply normal map if it exists
    if (material.normalIndex != -1) {
        vec3 tangentNormal = texture(textureSamplers[material.normalIndex], interpolatedUV).xyz * 2.0 - 1.0;
        vec3 T = normalize(interpolatedTangent);
        vec3 B = normalize(cross(normal, T));
        mat3 TBN = mat3(T, B, normal);
        normal = normalize(TBN * tangentNormal);
    }

    vec3  albedo    = material.albedo;
    if (material.albedoIndex != -1)
        albedo *= texture(textureSamplers[material.albedoIndex], interpolatedUV).rgb;

    float metallic  = material.metallic;
    if (material.metallicIndex != -1)
        metallic *= texture(textureSamplers[material.metallicIndex], interpolatedUV).r;

    float roughness = material.roughness;
    if (material.roughnessIndex != -1)
        roughness *= texture(textureSamplers[material.roughnessIndex], interpolatedUV).r;
    roughness = clamp(roughness, 0.005, 1.0);

    vec3 viewDir = normalize(-worldRayDirection);
    handleBSDF(viewDir, albedo, metallic, roughness, normal, material.ior,  material.transmission, material.transmissionStrength,  material.emission * material.emissionStrength, payload);
}