#version 460
#pragma shader_stage(miss)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Common.glsl"
#include "../ShadeMiss.glsl"

// Payload and Bindings
layout(location = 0) rayPayloadInEXT Payload payload;
layout(set = 0, binding = 3) uniform sampler2D textureSamplers[];

// Push Constants
layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

void main()
{
    shadeMiss(gl_WorldRayDirectionEXT, pushConstants.pushData, payload);
}
