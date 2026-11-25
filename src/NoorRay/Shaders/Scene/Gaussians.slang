import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Random;
__exported import Scene.GaussianKernel;

struct Gaussian {
    float3 mean;
    float opacity;
    float4 rotation;
    float3 scale;
};

Gaussian loadGaussian(GaussianCloudRecord cloud, uint index, float4 positionOpacity) {
    Gaussian gaussian;
    gaussian.mean = positionOpacity.xyz;
    gaussian.opacity = positionOpacity.w;
    gaussian.rotation = cloud.rotations[index];
    gaussian.scale = max(cloud.scales[index].xyz, float3(1e-8, 1e-8, 1e-8));
    return gaussian;
}

float3 toGaussianSpace(Gaussian gaussian, float3 v) { return rotateByInverse(gaussian.rotation, v) / gaussian.scale; }

//! The same Gaussian on the same ray gets the same number in every candidate test, so acceptance does not depend on traversal order.
float gaussianRandom(uint raySeed, uint cloud, uint gaussian) { return float(hashCombine(hashCombine(raySeed, cloud), gaussian) >> 8) * (1.0 / 16777216.0); }

struct GaussianCandidate {
    bool accepted;
    float t;
    uint gaussian;
};

//! Stochastic sorting-free acceptance: the candidate wins with probability alpha at its peak inside [tMin, tMax]. The ray is in the space of the
//! instance, where the Gaussian is the unit ball of its cutoff radius, so only the opacity loads; scaled by that radius, it is in the unit normal space.
GaussianCandidate testGaussian(SceneRecord scene, RayCandidate hit, float tMin, float tMax, uint raySeed) {
    const GaussianCloudRecord cloud = scene.gaussianClouds[hit.instanceId];
    GaussianCandidate candidate;
    candidate.gaussian = hit.instanceIndex - cloud.firstInstance;
    candidate.accepted = false;
    candidate.t = 0.0;
    const float opacity = cloud.positionOpacity[candidate.gaussian].w;
    const float random = gaussianRandom(raySeed, hit.instanceId, candidate.gaussian);
    if (random >= opacity)
        return candidate;
    const float cutoff = sqrt(gaussianCutoffSquared(opacity));
    const GaussianPeak peak = findGaussianPeak(opacity, hit.objectRayOrigin * cutoff, hit.objectRayDirection * cutoff);
    if (!(peak.t > tMin) || peak.t >= tMax)
        return candidate;
    candidate.t = peak.t;
    candidate.accepted = random < peak.alpha && peak.alpha >= gaussianMinAlpha;
    return candidate;
}

float shHalf(GaussianCloudRecord cloud, uint index) {
    const uint word = cloud.sh[index >> 1];
    return f16tof32((index & 1) != 0 ? word >> 16 : word & 0xffffu);
}

float3 shCoefficient(GaussianCloudRecord cloud, uint gaussian, uint coefficient) {
    const uint base = gaussian * cloud.shStride + coefficient * 3;
    return float3(shHalf(cloud, base), shHalf(cloud, base + 1), shHalf(cloud, base + 2));
}

//! The SH sum of the 3DGS convention, before decodeGaussianColor; direction points from the camera to the Gaussian, in cloud space.
float3 evaluateGaussianSh(GaussianCloudRecord cloud, uint gaussian, float3 direction) {
    const float basis[16] = gaussianShBasis(safeNormalize(direction));
    const uint count = (cloud.shDegree + 1) * (cloud.shDegree + 1);
    float3 sum = float3(0.0, 0.0, 0.0);
    [ForceUnroll]
    for (uint coefficient = 0; coefficient < 16; ++coefficient)
        if (coefficient < count)
            sum += basis[coefficient] * shCoefficient(cloud, gaussian, coefficient);
    return sum;
}

struct GaussianSurface {
    float3 position;
    float3 normal;
    float3 color;
    uint gaussian;
};

//! The shading point of an accepted Gaussian: world position at the peak, the ellipsoid gradient as normal (facing the ray), SH colour along the ray.
GaussianSurface loadGaussianSurface(SceneRecord scene, uint cloudIndex, uint gaussianIndex, float3 worldOrigin, float3 worldDirection, float t) {
    const GaussianCloudRecord cloud = scene.gaussianClouds[cloudIndex];
    const Gaussian gaussian = loadGaussian(cloud, gaussianIndex, cloud.positionOpacity[gaussianIndex]);
    const float3 objectOrigin = transformPoint(cloud.worldToObject, worldOrigin);
    const float3 objectDirection = transformVector(cloud.worldToObject, worldDirection);
    const float3 local = toGaussianSpace(gaussian, objectOrigin + objectDirection * t - gaussian.mean);
    const float3 objectNormal = rotateQuaternion(gaussian.rotation, local / gaussian.scale);
    GaussianSurface surface;
    surface.gaussian = gaussianIndex;
    surface.position = worldOrigin + worldDirection * t;
    float3 normal = safeNormalize(transformNormal(cloud.worldToObject, objectNormal));
    if (dot(normal, normal) == 0.0)
        normal = -worldDirection;
    surface.normal = dot(normal, worldDirection) > 0.0 ? -normal : normal;
    surface.color = decodeGaussianColor(evaluateGaussianSh(cloud, gaussianIndex, objectDirection));
    return surface;
}
