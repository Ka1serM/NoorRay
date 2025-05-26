#version 460
#pragma shader_stage(compute)

#extension GL_GOOGLE_include_directive : enable

#include "Agx.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba8) writeonly uniform image2D outputImage;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 flippedCoord = ivec2(coord.x, imageSize(inputImage).y - 1 - coord.y);
    imageStore(outputImage, coord, vec4(agx(imageLoad(inputImage, flippedCoord).rgb), 1.0));
}