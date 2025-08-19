#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Bindings.glsl"
#include "../Pathtracing/ShadeClosestHit.glsl"

// Payload and Attributes
layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

void main() {
 // --- Unpack Hit Data (RTX Pipeline Specific) ---
 const MeshAddresses mesh = instances[gl_InstanceCustomIndexEXT];
 VertexBuffer vertexBuf = VertexBuffer(mesh.vertexAddress);
 IndexBuffer indexBuf = IndexBuffer(mesh.indexAddress);
 FaceBuffer faceBuf = FaceBuffer(mesh.faceAddress);
 MaterialBuffer materialBuf = MaterialBuffer(mesh.materialAddress);

 const Face face = faceBuf.data[gl_PrimitiveID];
 const Material material = materialBuf.data[face.materialIndex];

 const uint i0 = indexBuf.data[3 * gl_PrimitiveID + 0];
 const uint i1 = indexBuf.data[3 * gl_PrimitiveID + 1];
 const uint i2 = indexBuf.data[3 * gl_PrimitiveID + 2];

 const Vertex v0 = vertexBuf.data[i0], v1 = vertexBuf.data[i1], v2 = vertexBuf.data[i2];
 const vec3 bary = calculateBarycentric(attribs);
 vec3 localPosition = interpolateBarycentric(bary, v0.position, v1.position, v2.position);
 vec3 localNormal = normalize(interpolateBarycentric(bary, v0.normal, v1.normal, v2.normal));
 vec3 localTangent = normalize(interpolateBarycentric(bary, v0.tangent, v1.tangent, v2.tangent));
 vec2 interpolatedUV = interpolateBarycentric(bary, v0.uv, v1.uv, v2.uv);

 vec3 worldPosition = (gl_ObjectToWorldEXT * vec4(localPosition, 1.0)).xyz;
 mat3 normalMatrix = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
 vec3 geometricNormalWorld = normalize(normalMatrix * localNormal);
 vec3 tangentWorld = normalize(mat3(gl_ObjectToWorldEXT) * localTangent);
 
  shadeClosestHit(worldPosition, geometricNormalWorld, tangentWorld, interpolatedUV, gl_WorldRayDirectionEXT, material, payload);
  payload.objectIndex = gl_InstanceID;
}