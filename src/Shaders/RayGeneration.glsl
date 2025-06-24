#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "SharedStructs.h"
#include "Common.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba32f) uniform image2D outputImage;
layout(set = 0, binding = 3) buffer Materials { Material materials[]; };
layout(set = 0, binding = 4) buffer PointLights { PointLight pointLights[]; };
layout(set = 0, binding = 5) uniform sampler2D textureSamplers[];

layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

layout(location = 0) rayPayloadEXT PrimaryRayPayload payload;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadowPayload;

void main() {
    ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);

    // Initialize RNG state
    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.pushData.frame * 16777619));
    uint rngState = seed.x;

    // Jitter for anti-aliasing
    vec2 jitter = (vec2(rand(rngState), rand(rngState)) - 0.5) * 1.5 * min(pushConstants.pushData.frame, 1.0);

    // Compute normalized UV coordinates (with jitter)
    vec2 uv = (vec2(pixelCoord) + jitter) / vec2(screenSize);
    uv.y = 1.0 - uv.y;  // flip y

    // Map UV to sensor offset in [-1, 1]
    vec2 sensorOffset = uv * 2.0 - 1.0;

    // Camera parameters from push constants
    vec3 camPos   = pushConstants.camera.position;
    vec3 camDir   = normalize(pushConstants.camera.direction);
    vec3 camRight = pushConstants.camera.horizontal; // half sensor width vector (meters)
    vec3 camUp    = pushConstants.camera.vertical;   // half sensor height vector (meters)

    float focalLength = pushConstants.camera.focalLength * 0.001; // convert mm to meters
    float focusDist = pushConstants.camera.focusDistance;         // meters
    float aperture = pushConstants.camera.aperture;               // f-number

    // Compute point on image plane at focalLength in front of camera
    vec3 imagePlaneCenter = camPos + camDir * focalLength;

    // Compute image plane point by offsetting center with half sensor size vectors scaled by sensorOffset
    vec3 imagePlanePoint = imagePlaneCenter
    + camRight * sensorOffset.x
    + camUp * sensorOffset.y;

    // Compute aperture radius (lens radius) in meters
    float apertureRadius = (focalLength / aperture) * 0.5;

    // Sample point on lens disk (uniform in [-apertureRadius, apertureRadius])
    vec2 lensSample = (vec2(rand(rngState), rand(rngState)) - 0.5) * 2.0 * apertureRadius;

    // Offset ray origin on lens plane using normalized directions of horizontal and vertical vectors
    vec3 lensRight = normalize(camRight);
    vec3 lensUp = normalize(camUp);
    vec3 rayOrigin = camPos + lensRight * lensSample.x + lensUp * lensSample.y;

    // Compute focus point along ray direction at focusDist
    vec3 focusPoint = camPos + normalize(imagePlanePoint - camPos) * focusDist;

    // Compute final ray direction from offset origin to focus point
    vec3 rayDirection = normalize(focusPoint - rayOrigin);

    // Initialize payload
    payload.color = vec3(0.0);
    payload.normal = vec3(0.0);
    payload.position = vec3(0.0);
    payload.throughput = vec3(1.0);
    payload.nextDirection = rayDirection;
    payload.done = false;

    payload.rngState = rngState;
    // Advance RNG for next use
    rand(payload.rngState);

    vec3 color = vec3(0.0);
    
    if (bool(pushConstants.pushData.isRayTracing)) {
        traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayOrigin, 0.001, rayDirection, 10000.0, 0);

        vec3 lighting = vec3(1.0);
        int lightCount = pointLights.length();
        if (lightCount == 1)
            lighting = vec3(1.0); // full brightness for dummy light only
        
        else if (lightCount > 1) {
            
            lighting = vec3(0.0);
            for (int i = 1; i < lightCount; ++i) {

                PointLight light = pointLights[i];
                vec3 lightVec = light.position - payload.position;
                float dist = length(lightVec);
                vec3 lightDir = normalize(lightVec);

                shadowPayload.hit = false;
                traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 2, 0, 2, payload.position, 0.01, lightDir, dist, 1);
                if (shadowPayload.hit)
                    continue;

                float NdotL = max(dot(payload.normal, lightDir), 0.0);
                lighting += NdotL * max(0.01, dist) * light.color;
            }
        }

        color = payload.color * lighting;
    }
    else {
        int maxDepth = 12;
        for (int depth = 0; depth < maxDepth; ++depth) {
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 1, 0, 1, rayOrigin, 0.001, rayDirection, 10000.0, 0);

            color += payload.throughput * payload.color;

            if (payload.done)
                break;

            rayOrigin = payload.position;
            rayDirection = payload.nextDirection;
        }
    }

    vec3 prevColor = imageLoad(outputImage, pixelCoord).rgb;
    color = (color + prevColor * float(pushConstants.pushData.frame)) / float(pushConstants.pushData.frame + 1);

    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}