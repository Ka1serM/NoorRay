#version 460
#pragma shader_stage(miss)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "../SharedStructs.h"
#include "../Common.glsl"

layout(location = 0) rayPayloadInEXT PrimaryRayPayload payload;

void main()
{
    payload.color = vec3(0.0, 0.0, 0.0);
    payload.done = true;
}