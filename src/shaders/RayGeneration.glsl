#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "SharedStructs.h"
#include "Common.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D outputImage;
layout(binding = 5, set = 0) buffer Materials { Material materials[]; };
layout(binding = 6, set = 0) buffer PointLights { PointLight pointLights[]; };

layout(push_constant) uniform PushConstants {
    PushData pushData;
    CameraData camera;
} pushConstants;

layout(location = 0) rayPayloadEXT PrimaryRayPayload payload;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadowPayload;

void main() {
    ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);
    vec3 color = vec3(0.0);

    if (bool(pushConstants.pushData.isPathtracing)) {

        uvec2 randBase = pcg2d(pixelCoord * (pushConstants.pushData.frame + 1));
        uint seedX = randBase.x;
        uint seedY = randBase.y;
        vec2 jitter = vec2(rand(seedX), rand(seedY));
        vec2 screenPos = vec2(pixelCoord) + jitter;
        vec2 uv = screenPos / vec2(screenSize);
        uv.y = 1.0 - uv.y;

        vec3 rayOrigin = pushConstants.camera.position;
        vec3 rayDirection = normalize(
            pushConstants.camera.direction +
            pushConstants.camera.horizontal * (uv.x - 0.5) +
            pushConstants.camera.vertical * (uv.y - 0.5)
        );

        payload.color = vec3(0.0);
        payload.normal = vec3(0.0);
        payload.position = vec3(0.0);
        payload.done = false;
        payload.throughput = vec3(1.0);
        for (uint depth = 0; depth < 4; ++depth) {
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 1, 0, 1, rayOrigin, 0.001, rayDirection, 10000.0, 0);

            color += payload.throughput * payload.color;

            if (payload.done)
                break;

            rayOrigin = payload.position;
            rayDirection = sampleDirection(rand(seedX), rand(seedY), payload.normal);
        }

    } else {
        vec2 uv = (vec2(pixelCoord) + vec2(0.5)) / vec2(screenSize);
        uv.y = 1.0 - uv.y;

        vec3 rayOrigin = pushConstants.camera.position;
        vec3 rayDirection = normalize(
            pushConstants.camera.direction +
            pushConstants.camera.horizontal * (uv.x - 0.5) +
            pushConstants.camera.vertical * (uv.y - 0.5)
        );

        payload.color = vec3(0.0);
        payload.normal = vec3(0.0);
        payload.position = vec3(0.0);
        payload.done = false;
        payload.throughput = vec3(1.0);

        traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayOrigin, 0.001, rayDirection, 10000.0, 0);

        color += payload.color;
        vec3 position = payload.position;
        vec3 normal = normalize(payload.normal);

        bool isShadowed = false;
        shadowPayload.hit = false;
        for (int i = 0; i < pointLights.length(); ++i) {
            PointLight light = pointLights[i];
            vec3 lightVec = light.position - position;
            float dist = length(lightVec);
            vec3 lightDir = normalize(lightVec);

            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 2, 0, 2, position, 0.01, lightDir, dist, 1);

            if (shadowPayload.hit) {
                isShadowed = true;
                break;
            }
        }

        if (isShadowed)
            color *= 0.2;
    }

    vec4 prevColor = imageLoad(outputImage, pixelCoord);
    vec4 newColor = vec4(color, 1.0);

    if (!bool(pushConstants.camera.isMoving))
        newColor = (newColor + prevColor * float(pushConstants.pushData.frame)) / float(pushConstants.pushData.frame + 1);

    imageStore(outputImage, pixelCoord, newColor);
}