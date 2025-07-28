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
layout(set = 0, binding = 3) uniform sampler2D textureSamplers[];

// Ray payload and attributes
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

// Push constants
layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

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

        payload.color *= albedo;
        payload.throughput *= material.transmission;
        
        payload.bounceType = BOUNCE_TYPE_TRANSMISSION;
        return;
    }

    // diffuse + specular with MIS, lobe weights from average energy
    vec3 viewDir = normalize(-gl_WorldRayDirectionEXT);

    if (dot(normal, viewDir) < 0.0) //two sided lighting
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
    
    payload.bounceType = choseDiffuse ? BOUNCE_TYPE_DIFFUSE : BOUNCE_TYPE_SPECULAR;
}