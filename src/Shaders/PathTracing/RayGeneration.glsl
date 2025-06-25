#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../SharedStructs.h"
#include "../Common.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba32f) uniform image2D outputImage;

layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

layout(location = 0) rayPayloadEXT Payload payload;

void main() {
    ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);
    
    // Initialize RNG state using pcg2d, use both components
    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.pushData.frame * 16777619));
    uint rngStateX = seed.x;
    uint rngStateY = seed.y;

    // Jitter for anti-aliasing, only after first frame
    vec2 jitter = (vec2(rand(rngStateX), rand(rngStateY)) - 0.5) * min(float(pushConstants.pushData.frame), 1.0);

    // Compute normalized UV coordinates (with jitter)
    vec2 uv = (vec2(pixelCoord) + jitter) / vec2(screenSize);
    uv.y = 1.0 - uv.y;  // flip y

    // Map UV from [0,1] to [-0.5, 0.5]
    vec2 sensorOffset = uv - 0.5;

    // Camera parameters from push constants
    const vec3 camPos   = pushConstants.camera.position;
    const vec3 camDir   = normalize(pushConstants.camera.direction);
    const vec3 horizontal = pushConstants.camera.horizontal; // full sensor width vector in meters
    const vec3 vertical = pushConstants.camera.vertical;     // full sensor height vector in meters

    // Convert focal length from mm to meters
    float focalLength = pushConstants.camera.focalLength * 0.001;

    // Compute image plane center at focal length in front of camera
    vec3 imagePlaneCenter = camPos + camDir * focalLength;

    // Compute image plane point offset by sensor coords
    vec3 imagePlanePoint = imagePlaneCenter
    + horizontal * sensorOffset.x
    + vertical * sensorOffset.y;

    // Start with pinhole camera ray (no DOF)
    vec3 rayOrigin = camPos;
    vec3 rayDirection = normalize(imagePlanePoint - rayOrigin);

    // Apply depth of field if aperture is not pinhole
    if (pushConstants.camera.aperture > 0.0) {
        // Calculate aperture radius: convert focal length back to mm for f-stop calculation, then to meters
        float apertureRadius = (pushConstants.camera.focalLength / pushConstants.camera.aperture) * 0.5 * 0.001;

        // Sample point on aperture disk
        vec2 lensSample = roundBokeh(rand(rngStateX), rand(rngStateY), pushConstants.camera.bokehBias) * apertureRadius;

        // Create orthonormal basis for the lens plane
        vec3 lensU = normalize(horizontal);
        vec3 lensV = normalize(vertical);

        // Offset ray origin on lens plane
        vec3 rayOriginDOF = camPos + lensU * lensSample.x + lensV * lensSample.y;

        // Calculate focus point: where the original pinhole ray intersects the focus plane
        float focusDistance = pushConstants.camera.focusDistance; // already in meters
        vec3 focusPoint = rayOrigin + rayDirection * focusDistance;

        // New ray direction from lens sample point to focus point
        vec3 rayDirectionDOF = normalize(focusPoint - rayOriginDOF);

        // Use DOF ray
        rayOrigin = rayOriginDOF;
        rayDirection = rayDirectionDOF;
    }

    // Initialize color
    vec3 color = vec3(0.0);
    // Initialize payload
    payload.color = vec3(0.0);
    payload.normal = vec3(0.0);
    payload.position = vec3(0.0);
    payload.throughput = vec3(1.0);
    payload.nextDirection = rayDirection;
    payload.done = false;
    payload.rngState = rngStateX;

    // Advance RNG state for next use
    rand(payload.rngState);
    
    const int maxDiffuseBounces = 4;
    const int maxSpecularBounces = 6;
    const int maxTransmissionBounces = 12;
    int diffuseBounces = 0;
    int specularBounces = 0;
    int transmissionBounces = 0;
    
    int totalDepth = 0;
    while (totalDepth < 24) {
        traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayOrigin, 0.001, rayDirection, 10000.0, 0);
        
        color += payload.throughput * payload.color;
        
        if (payload.done)
            break;

        //Termination condition based on bounce type
        if (payload.bounceType == BOUNCE_TYPE_DIFFUSE)
            diffuseBounces++;
        else if (payload.bounceType == BOUNCE_TYPE_SPECULAR)
            specularBounces++;
        else if (payload.bounceType == BOUNCE_TYPE_TRANSMISSION)
            transmissionBounces++;
        
        if ((payload.bounceType == BOUNCE_TYPE_DIFFUSE && diffuseBounces > maxDiffuseBounces)
        || (payload.bounceType == BOUNCE_TYPE_SPECULAR && specularBounces > maxSpecularBounces)
        || (payload.bounceType == BOUNCE_TYPE_TRANSMISSION && transmissionBounces > maxTransmissionBounces))
            break;
        
        rayOrigin = payload.position;
        rayDirection = payload.nextDirection;
        
        ++totalDepth;
    }
    
    // Accumulate color over frames
    color = (color + imageLoad(outputImage, pixelCoord).rgb * float(pushConstants.pushData.frame)) / float(pushConstants.pushData.frame + 1);

    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}