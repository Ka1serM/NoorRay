import Scene.GaussianKernel;

static const uint shCount = 16;

struct RayGaussian : IDifferentiable {
    float3 mean;
    float4 rotation;
    float3 logScale;
    float logitOpacity;
    float3 sh[shCount];
}

struct RaySample : IDifferentiable {
    float alpha;
    float3 color;
}

//! rotation is the unnormalized (w, x, y, z) quaternion of 3DGS.
[Differentiable]
GaussianPeak findRayPeak(float3 mean, float4 rotation, float3 logScale, float logitOpacity, no_diff float3 origin, no_diff float3 direction) {
    const float4 unit = rotation / length(rotation);
    const float opacity = 1.0 / (1.0 + exp(-logitOpacity));
    return findGaussianPeak(mean, float4(unit.y, unit.z, unit.w, unit.x), exp(logScale), opacity, origin, direction);
}

[Differentiable]
RaySample sampleRayGaussian(RayGaussian gaussian, no_diff float3 origin, no_diff float3 direction, no_diff uint degree) {
    const float basis[shCount] = no_diff gaussianShBasis(direction);
    const uint used = (degree + 1) * (degree + 1);
    float3 shSum = float3(0.0);
    [ForceUnroll]
    for (uint coefficient = 0; coefficient < shCount; ++coefficient)
        if (coefficient < used)
            shSum = shSum + basis[coefficient] * gaussian.sh[coefficient];
    RaySample sample;
    sample.alpha = findRayPeak(gaussian.mean, gaussian.rotation, gaussian.logScale, gaussian.logitOpacity, origin, direction).alpha;
    sample.color = decodeGaussianColor(shSum);
    return sample;
}

[Differentiable]
float linearToSrgb(float linear) { return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055; }

[Differentiable]
float3 linearToSrgb(float3 linear) { return float3(linearToSrgb(linear.x), linearToSrgb(linear.y), linearToSrgb(linear.z)); }
