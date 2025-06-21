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

    vec2 sensorOffset = vec2(uv.x - 0.5, uv.y - 0.5);

    vec3 imagePlanePoint = pushConstants.camera.position
    + pushConstants.camera.direction * (pushConstants.camera.focalLength * 0.001)
    + pushConstants.camera.horizontal * sensorOffset.x
    + pushConstants.camera.vertical * sensorOffset.y;

    float focalLengthMeters = pushConstants.camera.focalLength * 0.001;
    float apertureDiameter = focalLengthMeters / pushConstants.camera.aperture;
    float apertureRadius = apertureDiameter * 0.5;

    vec2 lensSample = (vec2(rand(seed.x), rand(seed.y)) - 0.5) * 2.0;
    lensSample *= apertureRadius;

    vec3 right = normalize(pushConstants.camera.horizontal);
    vec3 up = normalize(pushConstants.camera.vertical);

    vec3 rayOrigin = pushConstants.camera.position + right * lensSample.x + up * lensSample.y;
    float focusDistance = pushConstants.camera.focusDistance;
    vec3 focusPoint = pushConstants.camera.position + normalize(imagePlanePoint - pushConstants.camera.position) * focusDistance;
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

        vec3 position = payload.position;
        vec3 normal = payload.normal;
        vec3 lighting = vec3(0.0);

        for (int i = 0; i < pointLights.length(); ++i) {
            PointLight light = pointLights[i];

            if (light.intensity <= 0.0)
                continue;

            vec3 lightVec = light.position - position;
            float dist = length(lightVec);
            vec3 lightDir = normalize(lightVec);

            // Trace shadow ray only if light intensity > 0
            shadowPayload.hit = false;
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 2, 0, 2, position, 0.01, lightDir, dist, 1);

            if (!shadowPayload.hit) {
                float attenuation = light.intensity / max(0.01, dist - light.radius + 0.01);
                float NdotL = max(dot(normal, lightDir), 0.0);
                lighting += NdotL * attenuation * light.color;
            }
        }

        color = payload.color * lighting;
    }
    else {
        int maxDepth = 24;
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