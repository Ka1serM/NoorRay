#version 460
#pragma shader_stage(compute)

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) readonly uniform image2D inputImage;
layout(binding = 1, rgba8) writeonly uniform image2D outputImage;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    vec4 color = imageLoad(inputImage, coord);
    vec4 normalizedColor = clamp(color, 0.0, 1.0);
    imageStore(outputImage, coord, normalizedColor);
}
