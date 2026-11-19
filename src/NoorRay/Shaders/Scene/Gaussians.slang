import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Random;

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

float3 toGaussianSpace(Gaussian gaussian, float3 v) {
    const float4 inverse = float4(-gaussian.rotation.xyz, gaussian.rotation.w);
    return rotateQuaternion(inverse, v) / gaussian.scale;
}

//! The squared cutoff radius in standard deviations: the response ends at the alpha 1/255, as the proxy instances are scaled.
float gaussianCutoffSquared(float opacity) { return 2.0 * log(max(opacity * 255.0, 1.0)); }

//! The same Gaussian on the same ray gets the same number in every candidate test, so acceptance does not depend on traversal order.
float gaussianRandom(uint raySeed, uint cloud, uint gaussian) { return float(hashCombine(hashCombine(raySeed, cloud), gaussian) >> 8) * (1.0 / 16777216.0); }

struct GaussianCandidate {
    bool accepted;
    float t;
    uint gaussian;
};

//! Stochastic sorting-free acceptance: the candidate wins with probability alpha at its peak inside [tMin, tMax]. The ray is in the space of the
//! instance, where the Gaussian is the unit ball scaled by its cutoff, so only the opacity loads.
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
    const float3 o = hit.objectRayOrigin;
    const float3 d = hit.objectRayDirection;
    const float t = -dot(o, d) / dot(d, d);
    if (!(t > tMin) || t >= tMax)
        return candidate;
    const float3 closest = o + d * t;
    const float alpha = min(opacity * exp(-0.5 * gaussianCutoffSquared(opacity) * dot(closest, closest)), 0.999);
    candidate.t = t;
    candidate.accepted = random < alpha;
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

//! 3DGS convention: direction points from the camera to the Gaussian, in cloud space; the result is offset by 0.5 and clamped at zero.
float3 evaluateGaussianSh(GaussianCloudRecord cloud, uint gaussian, float3 direction) {
    const float3 d = safeNormalize(direction);
    const float x = d.x;
    const float y = d.y;
    const float z = d.z;
    float3 result = 0.28209479177387814 * shCoefficient(cloud, gaussian, 0);
    const uint degree = cloud.shDegree;
    if (degree >= 1) {
        const float c1 = 0.4886025119029199;
        result += -c1 * y * shCoefficient(cloud, gaussian, 1) + c1 * z * shCoefficient(cloud, gaussian, 2) - c1 * x * shCoefficient(cloud, gaussian, 3);
    }
    if (degree >= 2) {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        result += 1.0925484305920792 * x * y * shCoefficient(cloud, gaussian, 4) - 1.0925484305920792 * y * z * shCoefficient(cloud, gaussian, 5) +
                  0.31539156525252005 * (2.0 * zz - xx - yy) * shCoefficient(cloud, gaussian, 6) -
                  1.0925484305920792 * x * z * shCoefficient(cloud, gaussian, 7) + 0.5462742152960396 * (xx - yy) * shCoefficient(cloud, gaussian, 8);
        if (degree >= 3)
            result += -0.5900435899266435 * y * (3.0 * xx - yy) * shCoefficient(cloud, gaussian, 9) +
                      2.890611442640554 * x * y * z * shCoefficient(cloud, gaussian, 10) -
                      0.4570457994644658 * y * (4.0 * zz - xx - yy) * shCoefficient(cloud, gaussian, 11) +
                      0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoefficient(cloud, gaussian, 12) -
                      0.4570457994644658 * x * (4.0 * zz - xx - yy) * shCoefficient(cloud, gaussian, 13) +
                      1.445305721320277 * z * (xx - yy) * shCoefficient(cloud, gaussian, 14) -
                      0.5900435899266435 * x * (xx - 3.0 * yy) * shCoefficient(cloud, gaussian, 15);
    }
    return max(result + 0.5, float3(0.0, 0.0, 0.0));
}

//! 3DGS trains the SH on sRGB-encoded images, so its output is display-encoded and becomes linear radiance here.
float3 srgbToLinear(float3 encoded) {
    const float3 low = encoded / 12.92;
    const float3 high = pow((encoded + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
    return float3(encoded.x <= 0.04045 ? low.x : high.x, encoded.y <= 0.04045 ? low.y : high.y, encoded.z <= 0.04045 ? low.z : high.z);
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
    surface.color = srgbToLinear(evaluateGaussianSh(cloud, gaussianIndex, objectDirection));
    return surface;
}
