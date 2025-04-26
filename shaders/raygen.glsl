#version 460
#pragma shader_stage(raygen)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"


layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D outputImage;

layout(push_constant) uniform PushConstants {
    int frame;
    int _pad0, _pad1, _pad2; // Pads frame to 16 bytes total

    vec4 origin;
    vec4 lowerLeft;
    vec4 horizontal;
    vec4 vertical;
    vec4 rightVector;
    vec4 upVector;
    vec4 viewVector;
    vec4 paddedLensRadius;
};


layout(location = 0) rayPayloadEXT HitPayload payload;

void createCoordinateSystem(in vec3 N, out vec3 T, out vec3 B)
{
    if(abs(N.x) > abs(N.y))
        T = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
    else
        T = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
    B = cross(N, T);
}

vec3 sampleHemisphere(float rand1, float rand2)
{
    vec3 dir;
    dir.x = cos(2 * M_PI * rand2) * sqrt(1 - rand1 * rand1);
    dir.y = sin(2 * M_PI * rand2) * sqrt(1 - rand1 * rand1);
    dir.z = rand1;
    return dir;
}

vec3 sampleDirection(float rand1, float rand2, vec3 normal)
{
    vec3 tangent;
    vec3 bitangent;
    createCoordinateSystem(normal, tangent, bitangent);
    vec3 dir = sampleHemisphere(rand1, rand2);
    return dir.x * tangent + dir.y * bitangent + dir.z * normal;
}
void main()
{
    int maxSamples = 32;
    vec3 color = vec3(0.0);

    ivec2 pixelCoord = ivec2(gl_LaunchIDEXT.xy);
    ivec2 screenSize = ivec2(gl_LaunchSizeEXT.xy);

    for (uint sampleNum = 0; sampleNum < maxSamples; sampleNum++) {

        // Random seed
        uvec2 randBase = pcg2d(pixelCoord * int(sampleNum + maxSamples * frame + 1));
        uint seedX = randBase.x;
        uint seedY = randBase.y;

        // Jittered screen position
        vec2 jitter = vec2(rand(seedX), rand(seedY));
        vec2 screenPos = vec2(pixelCoord) + jitter;

        vec2 uv = screenPos / vec2(screenSize);
        uv.y = 1.0 - uv.y; // Vulkan Y flip

        vec3 rayOrigin = origin.xyz;

        // Compute ray direction directly in world space
        vec3 rayTarget = lowerLeft.xyz
                         + horizontal.xyz * uv.x
                         + vertical.xyz * uv.y;

        vec3 rayDirection = normalize(rayTarget - rayOrigin);

        // Path tracing loop
        vec3 weight = vec3(1.0);
        payload.done = false;
        payload.isPathtracing = true;

        for (uint depth = 0; depth < 6; depth++) {

            traceRayEXT(
                topLevelAS,
                gl_RayFlagsOpaqueEXT,
                0xff,
                0,
                0,
                0,
                rayOrigin,
                0.001,
                rayDirection,
                10000.0,
                0
            );

            color += weight * payload.emission;

            if (payload.done)
                break;

            rayOrigin = payload.position;
            rayDirection = sampleDirection(rand(seedX), rand(seedY), payload.normal);

            float pdf = 1.0 / (2.0 * M_PI);
            weight *= payload.brdf * max(dot(rayDirection, payload.normal), 0.0) / pdf;
        }
    }

    color /= float(maxSamples);

    vec4 oldColor = imageLoad(outputImage, pixelCoord);
    vec4 newColor = vec4(color, 1.0);
    newColor = (newColor + oldColor * float(frame)) / float(frame + 1);

    imageStore(outputImage, pixelCoord, newColor);
}
