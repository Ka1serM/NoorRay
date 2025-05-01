#version 460
#pragma shader_stage(miss)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "../SharedStructs.h"
#include "../Common.glsl"

layout(location = 0) rayPayloadInEXT ShadowRayPayload payload;

void main() {
    payload.hit = false;
}