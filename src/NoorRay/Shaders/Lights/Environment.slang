import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Sampling;

bool hasEnvironmentTexture(SceneRecord scene) { return scene.environment.x != 0; }

//! Linear sRGB radiance seen along a world direction.
float3 environmentRadiance(SceneRecord scene, float3 direction) {
    const float strength = scene.environmentColor.w;
    if (!hasEnvironmentTexture(scene))
        return scene.environmentColor.xyz * strength;
    TextureHandle texture;
    texture.value = scene.environment.x;
    SamplerHandle sampler;
    sampler.value = scene.environment.y;
    const float2 uv = environmentUv(scene, direction);
    return max(texture.SampleLevel(sampler, uv, 0.0).xyz, 0.0) * strength;
}

//! What a camera ray that misses the scene sees: the environment, or black when it is hidden from the camera.
float3 backgroundRadiance(SceneRecord scene, float3 direction) { return scene.sceneCenter.w != 0.0 ? environmentRadiance(scene, direction) : float3(0.0); }

bool hasEnvironmentTable(SceneRecord scene) { return scene.environment.z != 0 && scene.environment.w != 0; }

//! The solid angle density of sampleEnvironment, without the environment selection probability.
float environmentPdf(SceneRecord scene, float3 direction) {
    if (!hasEnvironmentTable(scene))
        return invFourPi;
    const float2 uv = environmentUv(scene, direction);
    const uint width = scene.environment.z;
    const uint height = scene.environment.w;
    const uint x = min(uint(uv.x * float(width)), width - 1);
    const uint y = min(uint(uv.y * float(height)), height - 1);
    const float sinTheta = sin(pi * (float(y) + 0.5) / float(height));
    if (sinTheta <= 0.0)
        return 0.0;
    const float discrete = scene.environmentAlias[y * width + x].pdf;
    return discrete * float(width * height) / (2.0 * pi * pi * sinTheta);
}

float3 environmentDirection(SceneRecord scene, float2 uv) { return rotateY(equirectToDirection(uv), scene.environmentParams.x); }
float2 environmentUv(SceneRecord scene, float3 direction) { return directionToEquirect(rotateY(direction, -scene.environmentParams.x)); }

float3 sampleEnvironment(SceneRecord scene, float3 u, out float pdf) {
    if (!hasEnvironmentTable(scene)) {
        pdf = invFourPi;
        return sampleUniformSphere(u.xy);
    }
    const uint width = scene.environment.z;
    const uint height = scene.environment.w;
    float discrete;
    const uint pixel = sampleAlias(scene.environmentAlias, width * height, u.z, discrete);
    const uint x = pixel % width;
    const uint y = pixel / width;
    const float2 uv = float2((float(x) + u.x) / float(width), (float(y) + u.y) / float(height));
    const float sinTheta = sin(pi * (float(y) + 0.5) / float(height));
    pdf = sinTheta > 0.0 ? discrete * float(width * height) / (2.0 * pi * pi * sinTheta) : 0.0;
    return environmentDirection(scene, uv);
}
