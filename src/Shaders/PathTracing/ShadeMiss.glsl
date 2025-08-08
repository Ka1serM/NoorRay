#include "Bindings.glsl"

void shadeMiss(in vec3 worldRayDirection, in PushData pushData, inout Payload payload) {
    vec3 viewDir = normalize(worldRayDirection);

    // Spherical projection to sample the HDRI
    vec2 uv;
    uv.x = atan(viewDir.z, viewDir.x) / (2.0 * PI) + 0.5;
    uv.y = 1.0 - acos(clamp(viewDir.y, -1.0, 1.0)) / PI;

    payload.color = vec3(0.0);
    if (pushData.hdriTexture != -1)
        payload.color = texture(textureSamplers[pushData.hdriTexture], uv).rgb * 5.0;

    payload.done = true;
}