#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_scalar_block_layout: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

#include "../SharedStructs.h"
#include "../Common.glsl"
#include "../PrimaryRayGen.glsl"

// Bindings
layout (set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 1, rgba32f) uniform image2D outputImage;

// Push Constants
layout (push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

// Ray Payload
layout (location = 0) rayPayloadEXT Payload payload;

void main() {
    const int SAMPLES_PER_PIXEL = 1;

    const ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    const ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);

    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.pushData.frame * 16777619));
    uint rngStateX = seed.x;
    uint rngStateY = seed.y;

    vec3 accumulatedColor = vec3(0.0);

    for (int i = 0; i < SAMPLES_PER_PIXEL; ++i) {
        vec3 rayOrigin, rayDirection;
        // Generate the primary ray using the shared function
        generatePrimaryRay(pixelCoord, screenSize, pushConstants.camera, rngStateX, rngStateY, rayOrigin, rayDirection);

        // Initialize payload for the path
        payload.throughput = vec3(1.0);
        payload.done = false;
        payload.rngState = rngStateX;
        rand(payload.rngState); // Advance RNG for first hit

        vec3 pathContribution = vec3(0.0);

        // Path tracing loop
        int diffuseBounces = 0, specularBounces = 0, transmissionBounces = 0;
        const int maxDiffuseBounces = 4, maxSpecularBounces = 6, maxTransmissionBounces = 12;

        for (int totalDepth = 0; totalDepth < 12; ++totalDepth) {
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayOrigin, 0.001, rayDirection, 10000.0, 0);

            // Clamp indirect light to prevent fireflies.
            vec3 bounceContribution = payload.throughput * payload.color;
            if (totalDepth > 0)
                bounceContribution = clamp(bounceContribution, 0.0, 50.0);
            pathContribution += bounceContribution;
            
            // Check bounce limits
            switch (payload.bounceType) {
                case BOUNCE_TYPE_DIFFUSE: if (++diffuseBounces > maxDiffuseBounces) payload.done = true; break;
                case BOUNCE_TYPE_SPECULAR: if (++specularBounces > maxSpecularBounces) payload.done = true; break;
                case BOUNCE_TYPE_TRANSMISSION: if (++transmissionBounces > maxTransmissionBounces) payload.done = true; break;
            }

            if (payload.done)
                break;

            // Prepare for next bounce
            rayOrigin = payload.position;
            rayDirection = payload.nextDirection;
        }

        rngStateX = payload.rngState;
        rand(rngStateX); // Advance for next sample
        accumulatedColor += pathContribution;
    }

    // Accumulate and store final color
    vec3 finalColor = accumulatedColor / float(SAMPLES_PER_PIXEL);
    vec3 previousColor = imageLoad(outputImage, pixelCoord).rgb;
    finalColor = (finalColor + previousColor * float(pushConstants.pushData.frame)) / float(pushConstants.pushData.frame + 1);
    imageStore(outputImage, pixelCoord, vec4(finalColor, 1.0));
}