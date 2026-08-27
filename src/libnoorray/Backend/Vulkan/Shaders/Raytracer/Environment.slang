#pragma once

// Environment lighting for the Vulkan path-tracing path.
//
// A direct port of Scene/Resources/Environment.h. The mapping, radiance and
// PDF must agree with the CPU implementation, so the polynomial constants and
// branch structure are carried over verbatim. Textures are reached through
// descriptor-heap indices instead of opaque texture objects.

static const float EnvironmentPi = 3.14159265358979323846;
static const uint EnvironmentNoTexture = 0xFFFFFFFFu;

static const int EnvironmentMappingEquirectangular = 0;
static const int EnvironmentMappingEqualArea = 1;

// Uploaded once per frame by the host; see VulkanEnvironmentSnapshot.
struct EnvironmentRecord
{
    float3 color;
    float rotationSin;
    float rotationCos;
    float visibleExposureScale;
    float lightingExposureScale;
    float importanceWeight;
    uint texture;      // sampled-image heap index, or EnvironmentNoTexture
    uint cdfTexture;   // sampled-image heap index, or EnvironmentNoTexture
    int cdfWidth;
    int cdfHeight;
    int mapping;
    float3x3 environmentFromWorld;
};

struct EnvironmentSample
{
    float3 direction;
    float pdf;
};

float2 equalAreaSphereToSquare(float3 direction)
{
    const float3 d = normalize(direction);
    const float x = abs(d.x), y = abs(d.y), z = abs(d.z);
    const float r = sqrt(max(1.0 - z, 0.0));
    const float a = max(x, y);
    const float b = a == 0.0 ? 0.0 : min(x, y) / a;
    float phi = 0.4067585662467884896e-5
        + b * (0.6362265452740161349
        + b * (0.6157201789828021349e-2
        + b * (-0.2473337332812689442
        + b * (0.8817706647753162947e-1
        + b * (0.4190388180291657359e-1
        + b * -0.2513909723434835093e-1)))));
    if (x < y) phi = 1.0 - phi;
    float v = phi * r;
    float u = r - v;
    if (d.z < 0.0)
    {
        const float oldU = u;
        u = 1.0 - v;
        v = 1.0 - oldU;
    }
    u = d.x < 0.0 ? -abs(u) : abs(u);
    v = d.y < 0.0 ? -abs(v) : abs(v);
    return float2(0.5 * (u + 1.0), 0.5 * (v + 1.0));
}

float3 equalAreaSquareToSphere(float2 point)
{
    const float u = 2.0 * point.x - 1.0;
    const float v = 2.0 * point.y - 1.0;
    const float up = abs(u), vp = abs(v);
    const float signedDistance = 1.0 - (up + vp);
    const float r = 1.0 - abs(signedDistance);
    const float phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * EnvironmentPi * 0.25;
    const float z = signedDistance < 0.0 ? -(1.0 - r * r) : (1.0 - r * r);
    const float radial = r * sqrt(max(2.0 - r * r, 0.0));
    const float cosPhi = u < 0.0 ? -abs(cos(phi)) : abs(cos(phi));
    const float sinPhi = v < 0.0 ? -abs(sin(phi)) : abs(sin(phi));
    return float3(cosPhi * radial, sinPhi * radial, z);
}

float2 environmentUv(EnvironmentRecord environment, float3 direction)
{
    if (environment.mapping == EnvironmentMappingEqualArea)
    {
        float3 local = normalize(mul(environment.environmentFromWorld, direction));
        local = float3(
            environment.rotationCos * local.x - environment.rotationSin * local.y,
            environment.rotationSin * local.x + environment.rotationCos * local.y,
            local.z);
        return equalAreaSphereToSquare(local);
    }
    const float3 local = normalize(mul(environment.environmentFromWorld, direction));
    const float rotatedX = environment.rotationCos * local.x
        - environment.rotationSin * local.z;
    const float rotatedZ = environment.rotationSin * local.x
        + environment.rotationCos * local.z;
    return float2(
        0.5 + atan2(rotatedZ, rotatedX) / (2.0 * EnvironmentPi),
        acos(clamp(local.y, -1.0, 1.0)) / EnvironmentPi);
}

float4 environmentTexel(uint descriptorIndex, float2 uv)
{
    Texture2D<float4> image = DescriptorHandle<Texture2D<float4>>(
        uint2(descriptorIndex, 0));
    uint width = 0u;
    uint height = 0u;
    image.GetDimensions(width, height);
    if (width == 0u || height == 0u)
        return float4(0.0);
    const float2 wrapped = float2(uv.x - floor(uv.x), saturate(uv.y));
    const int2 pixel = int2(min(uint2(max(wrapped * float2(width, height), 0.0)),
        uint2(width - 1u, height - 1u)));
    return image.Load(int3(pixel, 0));
}

float3 environmentRadianceRgb(EnvironmentRecord environment, float3 direction,
    bool cameraRay)
{
    float3 rgb = environment.color;
    if (environment.texture != EnvironmentNoTexture)
        rgb *= environmentTexel(environment.texture,
            environmentUv(environment, direction)).rgb;
    return rgb * (cameraRay ? environment.visibleExposureScale
                            : environment.lightingExposureScale);
}

// Solid-angle PDF of environmentSampleDirection() returning `direction`.
float environmentPdf(EnvironmentRecord environment, float3 direction)
{
    if (environment.cdfTexture == EnvironmentNoTexture
        || environment.cdfWidth <= 0 || environment.cdfHeight <= 0)
        return 1.0 / (4.0 * EnvironmentPi);
    const float2 uv = environmentUv(environment, direction);
    // computeCdf() stores the final solid-angle PDF in alpha, so sampling and
    // MIS use the exact same discretized distribution.
    return max(environmentTexel(environment.cdfTexture, uv).w, 0.0);
}

float4 environmentCdfTexel(EnvironmentRecord environment, int x, int y)
{
    Texture2D<float4> cdf = DescriptorHandle<Texture2D<float4>>(
        uint2(environment.cdfTexture, 0));
    return cdf.Load(int3(x, y, 0));
}

EnvironmentSample sampleEnvironmentDirection(EnvironmentRecord environment,
    inout Rng rng)
{
    EnvironmentSample result;
    if (environment.cdfTexture == EnvironmentNoTexture
        || environment.cdfWidth <= 0 || environment.cdfHeight <= 0)
    {
        const float z = 1.0 - 2.0 * rng.nextFloat();
        const float phi = 2.0 * EnvironmentPi * rng.nextFloat();
        const float radius = sqrt(max(1.0 - z * z, 0.0));
        result.direction = float3(radius * cos(phi), z, radius * sin(phi));
        result.pdf = 1.0 / (4.0 * EnvironmentPi);
        return result;
    }

    const float marginalSample = rng.nextFloat();
    int low = 0;
    int high = environment.cdfHeight - 1;
    [loop]
    while (low < high)
    {
        const int middle = (low + high) / 2;
        if (environmentCdfTexel(environment, 0, middle).y < marginalSample)
            low = middle + 1;
        else
            high = middle;
    }
    const int y = low;
    const float conditionalSample = rng.nextFloat();
    low = 0;
    high = environment.cdfWidth - 1;
    [loop]
    while (low < high)
    {
        const int middle = (low + high) / 2;
        if (environmentCdfTexel(environment, middle, y).x < conditionalSample)
            low = middle + 1;
        else
            high = middle;
    }
    const int x = low;
    float3 local;
    if (environment.mapping == EnvironmentMappingEqualArea)
    {
        local = equalAreaSquareToSphere(float2(
            (float(x) + rng.nextFloat()) / float(environment.cdfWidth),
            (float(y) + rng.nextFloat()) / float(environment.cdfHeight)));
        local = float3(environment.rotationCos * local.x
                + environment.rotationSin * local.y,
            -environment.rotationSin * local.x
                + environment.rotationCos * local.y, local.z);
    }
    else
    {
        const float phi = 2.0 * EnvironmentPi
            * ((float(x) + rng.nextFloat()) / float(environment.cdfWidth) - 0.5);
        const float theta0 = EnvironmentPi * float(y) / float(environment.cdfHeight);
        const float theta1 = EnvironmentPi * float(y + 1)
            / float(environment.cdfHeight);
        const float cosineTheta = cos(theta0) + rng.nextFloat()
            * (cos(theta1) - cos(theta0));
        const float sineTheta = sqrt(max(1.0 - cosineTheta * cosineTheta, 0.0));
        const float3 rotated = float3(sineTheta * cos(phi), cosineTheta,
            sineTheta * sin(phi));
        local = float3(environment.rotationCos * rotated.x
                + environment.rotationSin * rotated.z,
            rotated.y, -environment.rotationSin * rotated.x
                + environment.rotationCos * rotated.z);
    }
    result.direction = normalize(mul(transpose(environment.environmentFromWorld), local));
    result.pdf = max(environmentCdfTexel(environment, x, y).w, 0.0);
    return result;
}
