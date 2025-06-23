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

    uvec2 seed = pcg2d(uvec2(pixelCoord) ^ uvec2(pushConstants.pushData.frame * 16777619));
    vec2 jitter = (vec2(rand(seed.x), rand(seed.y)) - 0.5) * 1.5 * min(pushConstants.pushData.frame, 1.0);

    vec2 uv = (vec2(pixelCoord) + jitter) / vec2(screenSize);
    uv.y = 1.0 - uv.y;

    vec2 sensorOffset = uv * 2.0 - 1.0;  // [-1,1]

    vec3 camPos   = pushConstants.camera.position;
    vec3 camDir   = pushConstants.camera.direction;
    vec3 camRight = pushConstants.camera.horizontal;
    vec3 camUp    = pushConstants.camera.vertical;

    float focalLength = pushConstants.camera.focalLength * 0.001; // [m]
    float focusDist   = pushConstants.camera.focusDistance;
    float aperture    = pushConstants.camera.aperture;

    vec3 imagePlaneCenter = camPos + camDir * focalLength;
    vec3 imagePlanePoint = imagePlaneCenter
    + camRight * sensorOffset.x
    + camUp    * sensorOffset.y;
    
    // Lens sampling: offset ray origin based on lens sample
    float apertureRadius = (focalLength / aperture) * 0.5; // [m]
    vec2 lensSample = (vec2(rand(seed.x), rand(seed.y)) - 0.5) * 2.0 * apertureRadius;
    vec3 rayOrigin = camPos + camRight * lensSample.x + camUp * lensSample.y;

    // Focus point and final ray direction
    vec3 focusPoint = camPos + normalize(imagePlanePoint - camPos) * focusDist;
    vec3 rayDirection = normalize(focusPoint - rayOrigin);

    payload.color = vec3(0.0);
    payload.normal = vec3(0.0);
    payload.position = vec3(0.0);
    payload.throughput = vec3(1.0);
    payload.nextDirection = rayDirection;
    payload.done = false;

    payload.rngState = seed.x;
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