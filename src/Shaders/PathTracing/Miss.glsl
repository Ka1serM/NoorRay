#version 460
#pragma shader_stage(miss)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Common.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;
layout(set = 0, binding = 3) uniform sampler2D textureSamplers[];

layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

void main()
{
    vec3 viewDir = normalize(gl_WorldRayDirectionEXT);

    //spherical projection
    vec2 uv;
    uv.x = atan(viewDir.z, viewDir.x) / (2.0 * PI) + 0.5;
    uv.y = 1 - acos(clamp(viewDir.y, -1.0, 1.0)) / PI; //flip y

    payload.color = texture(textureSamplers[pushConstants.pushData.hdriTexture], uv).rgb * 2;
    payload.done = true;
}