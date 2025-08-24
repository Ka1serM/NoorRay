#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing: enable
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_scalar_block_layout: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

#include "../Common.glsl"
#include "../SharedStructs.h"
#include "../Bindings.glsl"

layout (push_constant) uniform PushConstants {
    PushConstantsData pushConstants;
};

layout (location = 0) rayPayloadEXT Payload payload;

#include "../Pathtracing/PrimaryRayGen.glsl"

void main() {
    const ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    const ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);
    primaryRayGen(pixelCoord, screenSize);
}