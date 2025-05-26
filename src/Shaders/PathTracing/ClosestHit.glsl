#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Common.glsl"

// Buffer reference types
layout(buffer_reference, scalar) buffer VertexBuffer { Vertex data[]; };
layout(buffer_reference, scalar) buffer IndexBuffer  { uint data[]; };
layout(buffer_reference, scalar) buffer FaceBuffer   { Face data[]; };

layout(set = 0, binding = 2) buffer MeshAddressesBuffer { MeshAddresses instances[]; };
layout(set = 0, binding = 3) buffer Materials { Material materials[]; };
layout(set = 0, binding = 4) buffer PointLights { PointLight pointLights[]; };
layout(set = 0, binding = 5) uniform sampler2D textureSamplers[];

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

void main() {
    //Instance-based mesh selection
    const uint instIdx = gl_InstanceCustomIndexEXT;
    const MeshAddresses mesh = instances[instIdx];

    VertexBuffer vertexBuf = VertexBuffer(mesh.vertexAddress);
    IndexBuffer  indexBuf  = IndexBuffer(mesh.indexAddress);
    FaceBuffer   faceBuf   = FaceBuffer(mesh.faceAddress);

    const uint primitiveIndex = gl_PrimitiveID;
    const Face face = faceBuf.data[primitiveIndex];
    const Material material = materials[face.materialIndex];

    const uint i0 = indexBuf.data[3 * primitiveIndex + 0];
    const uint i1 = indexBuf.data[3 * primitiveIndex + 1];
    const uint i2 = indexBuf.data[3 * primitiveIndex + 2];

    const Vertex v0 = vertexBuf.data[i0];
    const Vertex v1 = vertexBuf.data[i1];
    const Vertex v2 = vertexBuf.data[i2];

    // Interpolate barycentric attributes
    const vec3 bary = calculateBarycentric(attribs);
    const vec3 position = interpolateBarycentric(bary, v0.position, v1.position, v2.position);
    const vec3 normal = normalize(interpolateBarycentric(bary, v0.normal, v1.normal, v2.normal));
    const vec2 uv = interpolateBarycentric(bary, v0.uv, v1.uv, v2.uv);

    // Material color
    vec3 albedo = material.albedo;
    if (material.albedoIndex != -1)
    albedo *= texture(textureSamplers[material.albedoIndex], uv).rgb;

    // Output payload
    payload.position = position;
    payload.normal = normal;
    payload.color = material.emission;

    // Path tracing BRDF sample
    uint seedX = i0 + pushConstants.frame;
    uint seedY = i1 + pushConstants.frame + 1;
    vec3 sampledDir = sampleDirection(rand(seedX), rand(seedY), normal);

    float cosTheta = max(0.0, dot(sampledDir, normal));
    float pdf = cosTheta / PI;
    vec3 brdf = albedo / PI;

    payload.throughput *= brdf * cosTheta / max(pdf, 0.00001);
}