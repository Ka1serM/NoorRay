#version 460
#pragma shader_stage(compute)

#extension GL_GOOGLE_include_directive : enable

#include "Agx.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba8) writeonly uniform image2D outputImage;

void main() {
    imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), vec4(agx(imageLoad(inputImage, ivec2(gl_GlobalInvocationID.xy)).rgb), 1.0));
}