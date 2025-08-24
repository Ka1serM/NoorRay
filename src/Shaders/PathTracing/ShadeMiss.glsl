#include "../Bindings.glsl"

void shadeMiss(in vec3 worldRayDirection, in EnvironmentData environmentData, inout Payload payload) {
    vec3 envColor = environmentData.color * environmentData.intensity;

    if (environmentData.textureIndex != -1) {
        float radRotation = radians(environmentData.rotation);
        float s = sin(radRotation);
        float c = cos(radRotation);
        vec3 rotatedDir = worldRayDirection;
        rotatedDir.x = worldRayDirection.x * c - worldRayDirection.z * s;
        rotatedDir.z = worldRayDirection.x * s + worldRayDirection.z * c;

        vec3 viewDir = normalize(rotatedDir);
        vec2 uv;
        uv.x = atan(viewDir.z, viewDir.x) / (2.0 * PI) + 0.5;
        uv.y = 1.0 - acos(clamp(viewDir.y, -1.0, 1.0)) / PI;
        envColor *= texture(textureSamplers[environmentData.textureIndex], uv).rgb
        * exp2(environmentData.exposure);
    }

    // Always contribute environment light
    payload.attenuation = vec3(1.0);
    payload.emission    = envColor;
    payload.albedo      = envColor;
    payload.normal      = vec3(0.0);

    // If environment is invisible at primary ray → mark transparent background
    if (payload.depth == 0 && environmentData.visible == 0) {
        payload.flags |= ENV_TRANSPARENT;
    }

    // Miss always terminates the ray
    payload.flags |= RAY_TERMINATED;
}
