#include "Splatting.h"

typealias Arguments = Splatting::Arguments;

static const float maxAlpha = 0.99;
static const float alphaThreshold = 1.0 / 255.0;
static const float transmittanceThreshold = 1e-4;
static const float minOneMinusAlpha = 1e-6;
static const float gaussianExtend = 3.33;
static const uint projectedGradientStride = 9;

struct Camera {
    float3x3 rotation;
    float3 translation;
    float4 intrinsics;
    float2 size;
};

struct Footprint : IDifferentiable {
    float2 mean;
    float3 conic;
    float determinant;
    float3 covariance;
};

[Differentiable]
float3x3 getRotation(float4 quaternion) {
    const float4 q = quaternion / length(quaternion);
    const float w = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

[Differentiable]
Footprint project(float3 mean, float4 quaternion, float3 scale, no_diff Camera camera, no_diff float blur) {
    const float3x3 rotation = getRotation(quaternion);
    const float3x3 m = float3x3(rotation[0] * scale, rotation[1] * scale, rotation[2] * scale);
    const float3x3 covariance = mul(m, transpose(m));
    const float3 t = mul(camera.rotation, mean) + camera.translation;
    const float3x3 cameraCovariance = mul(mul(camera.rotation, covariance), transpose(camera.rotation));
    const float4 k = camera.intrinsics;
    const float2 tanFov = 0.5 * camera.size / k.xy;
    const float2 limitPositive = (camera.size - k.zw) / k.xy + 0.3 * tanFov;
    const float2 limitNegative = k.zw / k.xy + 0.3 * tanFov;
    const float rz = 1.0 / t.z;
    const float2 clamped = t.z * min(limitPositive, max(-limitNegative, t.xy * rz));
    const float3 jx = float3(k.x * rz, 0.0, -k.x * clamped.x * rz * rz);
    const float3 jy = float3(0.0, k.y * rz, -k.y * clamped.y * rz * rz);
    const float3 cx = mul(cameraCovariance, jx);
    const float3 cy = mul(cameraCovariance, jy);
    const float3 covariance2d = float3(dot(jx, cx) + blur, dot(jx, cy), dot(jy, cy) + blur);
    Footprint footprint;
    footprint.determinant = covariance2d.x * covariance2d.z - covariance2d.y * covariance2d.y;
    footprint.conic = float3(covariance2d.z, -covariance2d.y, covariance2d.x) / footprint.determinant;
    footprint.mean = k.xy * t.xy * rz + k.zw;
    footprint.covariance = covariance2d;
    return footprint;
}

[Differentiable]
float getAlpha(float2 mean, float3 conic, float opacity, no_diff float2 pixel) {
    const float2 delta = mean - pixel;
    const float sigma = 0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
    return min(maxAlpha, opacity * exp(-sigma));
}

float getSigma(float2 mean, float3 conic, float2 pixel) {
    const float2 delta = mean - pixel;
    return 0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
}

float3 load3(Ptr<float> values, uint index) { return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]); }

void store3(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

float4 loadQuaternion(Ptr<float> values, uint index) {
    return float4(values[index * 4], values[index * 4 + 1], values[index * 4 + 2], values[index * 4 + 3]);
}

Camera getCamera(Arguments arguments) {
    Camera camera;
    camera.rotation = float3x3(arguments.view[0].xyz, arguments.view[1].xyz, arguments.view[2].xyz);
    camera.translation = float3(arguments.view[0].w, arguments.view[1].w, arguments.view[2].w);
    camera.intrinsics = arguments.intrinsics;
    camera.size = float2(arguments.width, arguments.height);
    return camera;
}

bool isVisible(Arguments arguments, Footprint footprint, float depth, float opacity) {
    if (depth < arguments.nearPlane || depth > arguments.farPlane || footprint.determinant <= 0.0 || opacity < alphaThreshold)
        return false;
    const float extend = min(gaussianExtend, sqrt(2.0 * log(opacity / alphaThreshold)));
    const float2 radius = ceil(extend * sqrt(footprint.covariance.xz));
    if (all(radius <= 0.0))
        return false;
    const float2 size = float2(arguments.width, arguments.height);
    return all(footprint.mean + radius > 0.0) && all(footprint.mean - radius < size);
}

void addProjected(Arguments arguments, uint gaussian, uint slot, float value) {
    atomicAdd(arguments.projectedGradient + gaussian * projectedGradientStride + slot, value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void projectGaussians(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.count)
        return;
    [ForceUnroll]
    for (uint slot = 0; slot < projectedGradientStride; ++slot)
        atomicStore(arguments.projectedGradient + id.x * projectedGradientStride + slot, 0.0);
    const Camera camera = getCamera(arguments);
    const float3 mean = load3(arguments.parameters.means, id.x);
    const Footprint footprint = project(mean, loadQuaternion(arguments.parameters.quaternions, id.x), load3(arguments.parameters.scales, id.x), camera,
                                        arguments.blur);
    const float depth = dot(camera.rotation[2], mean) + camera.translation.z;
    const float opacity = arguments.parameters.opacities[id.x];
    const bool visible = isVisible(arguments, footprint, depth, opacity);
    arguments.projected[id.x * 2] = float4(footprint.mean, visible ? depth : 3.402823466e+38, opacity);
    arguments.projected[id.x * 2 + 1] = float4(footprint.conic, visible ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sortGaussians(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.count)
        return;
    const float depth = arguments.projected[id.x * 2].z;
    uint rank = 0;
    for (uint other = 0; other < arguments.count; ++other) {
        const float otherDepth = arguments.projected[other * 2].z;
        if (otherDepth < depth || (otherDepth == depth && other < id.x))
            ++rank;
    }
    arguments.order[rank] = id.x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void composite(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const uint pixel = id.y * arguments.width + id.x;
    const float2 center = float2(id.xy) + 0.5;
    float3 color = float3(0.0);
    float transmittance = 1.0;
    uint contributors = 0;
    for (uint rank = 0; rank < arguments.count; ++rank) {
        const uint gaussian = arguments.order[rank];
        const float4 first = arguments.projected[gaussian * 2];
        const float4 second = arguments.projected[gaussian * 2 + 1];
        if (second.w == 0.0)
            break;
        const float sigma = getSigma(first.xy, second.xyz, center);
        const float alpha = min(maxAlpha, first.w * exp(-sigma));
        if (sigma < 0.0 || alpha < alphaThreshold)
            continue;
        const float next = transmittance * (1.0 - alpha);
        if (next <= transmittanceThreshold)
            break;
        color += load3(arguments.parameters.colors, gaussian) * alpha * transmittance;
        transmittance = next;
        contributors = rank + 1;
    }
    store3(arguments.image, pixel, color + transmittance * arguments.background.xyz);
    arguments.transmittance[pixel] = transmittance;
    arguments.contributors[pixel] = contributors;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compositeBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const bool inside = all(id.xy < uint2(arguments.width, arguments.height));
    const uint pixel = inside ? id.y * arguments.width + id.x : 0;
    const float2 center = float2(id.xy) + 0.5;
    const float3 gradient = inside ? load3(arguments.imageGradient, pixel) : float3(0.0);
    const float finalTransmittance = arguments.transmittance[pixel];
    const float backgroundWeight = dot(arguments.background.xyz, gradient);
    const uint contributors = inside ? arguments.contributors[pixel] : 0;
    float transmittance = finalTransmittance;
    float3 behind = float3(0.0);
    for (uint rank = WaveActiveMax(contributors); rank > 0; --rank) {
        const uint gaussian = arguments.order[rank - 1];
        const float4 first = arguments.projected[gaussian * 2];
        const float4 second = arguments.projected[gaussian * 2 + 1];
        const float sigma = getSigma(first.xy, second.xyz, center);
        const float alpha = min(maxAlpha, first.w * exp(-sigma));
        float values[projectedGradientStride] = {};
        if (rank <= contributors && sigma >= 0.0 && alpha >= alphaThreshold) {
            const float inverse = 1.0 / max(minOneMinusAlpha, 1.0 - alpha);
            transmittance *= inverse;
            const float weight = alpha * transmittance;
            const float3 color = load3(arguments.parameters.colors, gaussian);
            const float alphaGradient = dot(color * transmittance - behind * inverse, gradient) - finalTransmittance * inverse * backgroundWeight;
            var mean = diffPair(first.xy);
            var conic = diffPair(second.xyz);
            var opacity = diffPair(first.w);
            bwd_diff(getAlpha)(mean, conic, opacity, center, alphaGradient);
            values = {mean.d.x, mean.d.y, conic.d.x, conic.d.y, conic.d.z, opacity.d, weight * gradient.x, weight * gradient.y, weight * gradient.z};
            behind += color * weight;
        }
        [ForceUnroll]
        for (uint slot = 0; slot < projectedGradientStride; ++slot) {
            const float sum = WaveActiveSum(values[slot]);
            if (WaveIsFirstLane() && sum != 0.0)
                addProjected(arguments, gaussian, slot, sum);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void projectBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.count)
        return;
    float values[projectedGradientStride];
    [ForceUnroll]
    for (uint slot = 0; slot < projectedGradientStride; ++slot)
        values[slot] = atomicLoad(arguments.projectedGradient + id.x * projectedGradientStride + slot);
    Footprint.Differential footprint = {};
    footprint.mean = float2(values[0], values[1]);
    footprint.conic = float3(values[2], values[3], values[4]);
    var mean = diffPair(load3(arguments.parameters.means, id.x));
    var quaternion = diffPair(loadQuaternion(arguments.parameters.quaternions, id.x));
    var scale = diffPair(load3(arguments.parameters.scales, id.x));
    bwd_diff(project)(mean, quaternion, scale, getCamera(arguments), arguments.blur, footprint);
    const Splatting::Parameters gradients = arguments.gradients;
    store3(gradients.means, id.x, mean.d);
    gradients.quaternions[id.x * 4] = quaternion.d.x;
    gradients.quaternions[id.x * 4 + 1] = quaternion.d.y;
    gradients.quaternions[id.x * 4 + 2] = quaternion.d.z;
    gradients.quaternions[id.x * 4 + 3] = quaternion.d.w;
    store3(gradients.scales, id.x, scale.d);
    gradients.opacities[id.x] = values[5];
    store3(gradients.colors, id.x, float3(values[6], values[7], values[8]));
}
