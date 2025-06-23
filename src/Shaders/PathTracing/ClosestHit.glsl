#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Common.glsl"

// Buffer reference types
layout(buffer_reference, scalar) buffer VertexBuffer { Vertex data[]; };
layout(buffer_reference, scalar) buffer IndexBuffer { uint data[]; };
layout(buffer_reference, scalar) buffer FaceBuffer { Face data[]; };
layout(buffer_reference, scalar) buffer MaterialBuffer { Material data[]; };

layout(set = 0, binding = 2) buffer MeshAddressesBuffer { MeshAddresses instances[]; };
layout(set = 0, binding = 3) buffer PointLights { PointLight pointLights[]; };
layout(set = 0, binding = 4) uniform sampler2D textureSamplers[];

// Ray payload and attributes
layout(location = 0) rayPayloadInEXT PrimaryRayPayload payload;
hitAttributeEXT vec3 attribs;

// Push constants
layout(push_constant) uniform PushConstants {
    int frame, isPathtracing, isMoving, _pad1;
    vec3 position; int _pad2;
    vec3 direction; int _pad3;
    vec3 horizontal; int _pad4;
    vec3 vertical; int _pad5;
} pushConstants;

// --- Helper: build tangent frame from normal ---
void buildTangentFrame(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.z) < 0.999) {
        T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    } else {
        T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    }
    B = cross(T, N);
}

// --- Sample diffuse direction (cosine-weighted hemisphere) ---
vec3 sampleDiffuse(vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);

    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - u1);

    vec3 T, B;
    buildTangentFrame(N, T, B);

    return normalize(x * T + y * B + z * N);
}

// --- PDF for diffuse direction ---
float pdfDiffuse(vec3 N, vec3 L) {
    float NoL = max(dot(N, L), 0.0);
    return NoL / PI;
}

// --- Evaluate Lambertian Diffuse BRDF ---
vec3 evaluateDiffuseBRDF(vec3 albedo, float metallic)
{
    // Metals have no diffuse component, so scale by (1 - metallic)
    return albedo / PI * (1.0 - metallic);
}

// --- Sample GGX half vector ---
vec3 sampleGGX(float roughness, vec3 N, inout uint rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);

    float a = roughness * roughness;

    float phi = 2.0 * PI * u1;
    float cosTheta = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    vec3 Ht = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 T, B;
    buildTangentFrame(N, T, B);

    return normalize(Ht.x * T + Ht.y * B + Ht.z * N);
}

// --- GGX normal distribution function (NDF) ---
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// --- Sample specular direction ---
vec3 sampleSpecular(vec3 viewDir, vec3 N, float roughness, inout uint rngState) {
    vec3 H = sampleGGX(roughness, N, rngState);

    // Flip H if it's in the wrong hemisphere
    if (dot(H, N) < 0.0) {
        H = -H;
    }

    vec3 sampledDir = reflect(-viewDir, H);
    return sampledDir;
}

// --- PDF for specular direction ---
float pdfSpecular(vec3 viewDir, vec3 N, float roughness, vec3 L) {
    vec3 H = normalize(viewDir + L);
    float NoH = max(dot(N, H), 1e-4);
    float VoH = max(dot(viewDir, H), 1e-4);

    float D = distributionGGX(N, H, roughness);
    return max((D * NoH) / (4.0 * VoH), 1e-6); // Clamp to avoid near-zero
}

// --- Fresnel Schlick approximation ---
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// --- Schlick-GGX geometry function ---
float geometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness * roughness) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k + 1e-6);
}

// --- Smith geometry term (combined for view and light) ---
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);

    return ggxV * ggxL;
}

// --- Evaluate specular BRDF ---
vec3 evaluateSpecularBRDF(vec3 viewDir, vec3 N, vec3 albedo, float metallic, float roughness, vec3 L)
{
    vec3 H = normalize(viewDir + L);

    float NoV = max(dot(N, viewDir), 0.0);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(viewDir, H), 0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, viewDir, L, roughness);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(VoH, F0);

    vec3 numerator = F * D * G;
    float denominator = max(4.0 * NoV * NoL, 1e-6);

    return numerator / denominator;
}

void main() {
    const MeshAddresses mesh = instances[gl_InstanceCustomIndexEXT];
    VertexBuffer vertexBuf = VertexBuffer(mesh.vertexAddress);
    IndexBuffer indexBuf = IndexBuffer(mesh.indexAddress);
    FaceBuffer faceBuf = FaceBuffer(mesh.faceAddress);
    MaterialBuffer materialBuf = MaterialBuffer(mesh.materialAddress);

    const uint primitiveIndex = gl_PrimitiveID;
    const Face face = faceBuf.data[primitiveIndex];
    const Material material = materialBuf.data[face.materialIndex];

    const uint i0 = indexBuf.data[3 * primitiveIndex + 0];
    const uint i1 = indexBuf.data[3 * primitiveIndex + 1];
    const uint i2 = indexBuf.data[3 * primitiveIndex + 2];

    const Vertex v0 = vertexBuf.data[i0];
    const Vertex v1 = vertexBuf.data[i1];
    const Vertex v2 = vertexBuf.data[i2];

    const vec3 bary = calculateBarycentric(attribs);
    vec3 localPosition = interpolateBarycentric(bary, v0.position, v1.position, v2.position);
    vec3 localNormal = normalize(interpolateBarycentric(bary, v0.normal, v1.normal, v2.normal));
    vec2 interpolatedUV = interpolateBarycentric(bary, v0.uv, v1.uv, v2.uv);

    vec3 worldPosition = (gl_ObjectToWorldEXT * vec4(localPosition, 1.0)).xyz;
    mat3 objectToWorld3x3 = mat3(gl_ObjectToWorldEXT);
    mat3 normalMatrix = transpose(inverse(objectToWorld3x3));

    vec3 geometricNormalWorld = normalize(normalMatrix * localNormal);
    vec3 normal = geometricNormalWorld;

    payload.color = material.emission;
    payload.position = worldPosition;
    payload.normal = normal;

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
    specular *= 2; //scale to 0 - 2

    //Refraction
    float transmissionFactor = (material.transmission.r + material.transmission.g + material.transmission.b) / 3.0;
    if (transmissionFactor > 0.0 && rand(payload.rngState) < transmissionFactor) {
        vec3 I = normalize(gl_WorldRayDirectionEXT);

        float etaI = 1.0;
        float etaT = material.ior;
        float eta;

        if (dot(I, normal) > 0.0) {
            normal = -normal;
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }

        eta = etaI / etaT;

        vec3 refracted = refract(I, normal, eta);
        if (length(refracted) < 1e-5)
            payload.nextDirection = reflect(I, normal);
        else
            payload.nextDirection = refracted;

        payload.throughput *= material.transmission;
        return;
    }

    // diffuse + specular with MIS, lobe weights from average energy
    vec3 viewDir = normalize(-gl_WorldRayDirectionEXT);

    if (dot(normal, viewDir) < 0.0)
    normal = -normal;

    float NdotV = max(dot(normal, viewDir), 0.0);

    // Fresnel at normal incidence
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Average diffuse reflectance (energy)
    float fresnelAtNdotV = fresnelSchlick(NdotV, F0).r;
    float diffuseEnergy = (1.0 - metallic) * (1.0 - fresnelAtNdotV);

    // Average specular reflectance (energy)
    float specularEnergy = max(fresnelAtNdotV, 0.04);
    specularEnergy *= max(1.0 - roughness * roughness, 0.05);

    // Normalize weights to sum to 1
    float sumEnergy = diffuseEnergy + specularEnergy + 1e-6;
    float probDiffuse = diffuseEnergy / sumEnergy;
    float probSpecular = specularEnergy / sumEnergy;

    // Choose lobe to sample
    bool choseDiffuse = rand(payload.rngState) < probDiffuse;

    vec3 sampledDir;
    if (choseDiffuse)
        sampledDir = sampleDiffuse(normal, payload.rngState);
    else
        sampledDir = sampleSpecular(viewDir, normal, roughness, payload.rngState);

    // PDFs
    float pdfDiffuseVal = max(pdfDiffuse(normal, sampledDir), 1e-6);
    float pdfSpecularVal = max(pdfSpecular(viewDir, normal, roughness, sampledDir), 1e-6);
    
    vec3 diffuseBRDF = evaluateDiffuseBRDF(albedo, metallic);
    vec3 specularBRDF = evaluateSpecularBRDF(viewDir, normal, albedo, metallic, roughness, sampledDir) * specular;

    // Combined PDF
    float pdfCombined = probDiffuse * pdfDiffuseVal + probSpecular * pdfSpecularVal;

    //MIS with power heuristic
    float wDiffuse = probDiffuse * pdfDiffuseVal;
    float wSpecular = probSpecular * pdfSpecularVal;

    float misWeight;
    if (choseDiffuse)
        misWeight = (wDiffuse * wDiffuse) / (wDiffuse * wDiffuse + wSpecular * wSpecular + 1e-6);
    else
        misWeight = (wSpecular * wSpecular) / (wDiffuse * wDiffuse + wSpecular * wSpecular + 1e-6);

    float NoL = max(dot(normal, sampledDir), 0.0);
    vec3 totalBRDF = diffuseBRDF + specularBRDF;

    payload.throughput *= totalBRDF * NoL * misWeight / pdfCombined;
    payload.nextDirection = normalize(sampledDir);
}
