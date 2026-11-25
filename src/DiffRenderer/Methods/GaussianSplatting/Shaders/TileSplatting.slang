#include "TileSplatting.h"
import GaussianProjection;

typealias Arguments = TileSplatting::Arguments;
typealias Parameters = TileSplatting::Parameters;

static const uint tileSize = 16;
static const uint groupSize = tileSize * tileSize;
static const uint gradientStride = 9;
static const uint restStride = (shCount - 1) * 3;
static const float nearPlane = 0.2;
static const float maxAlpha = 0.99;
static const float alphaThreshold = 1.0 / 255.0;
static const float transmittanceThreshold = 0.0001;

groupshared uint sharedIds[groupSize];
groupshared float2 sharedMeans[groupSize];
groupshared float4 sharedConics[groupSize];
groupshared float3 sharedColors[groupSize];
groupshared uint doneCount;
groupshared uint tileContributors;

ProjectionView getView(Arguments arguments) {
    const TileSplatting::View source = arguments.view;
    ProjectionView view;
    view.rotation = float3x3(source.worldToCamera[0].xyz, source.worldToCamera[1].xyz, source.worldToCamera[2].xyz);
    view.translation = float3(source.worldToCamera[0].w, source.worldToCamera[1].w, source.worldToCamera[2].w);
    view.intrinsics = source.intrinsics;
    view.position = source.position.xyz;
    view.tanFov = 0.5 * float2(source.width, source.height) / source.intrinsics.xy;
    return view;
}

float3 load3(Ptr<float> values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

void store3(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

GaussianInput loadGaussian(Parameters parameters, uint index) {
    GaussianInput gaussian;
    gaussian.mean = load3(parameters.means, index);
    gaussian.logScale = load3(parameters.scales, index);
    gaussian.rotation = float4(parameters.rotations[index * 4], parameters.rotations[index * 4 + 1], parameters.rotations[index * 4 + 2],
                               parameters.rotations[index * 4 + 3]);
    gaussian.logitOpacity = parameters.opacities[index];
    gaussian.sh[0] = load3(parameters.featuresDc, index);
    [ForceUnroll]
    for (uint coefficient = 1; coefficient < shCount; ++coefficient)
        gaussian.sh[coefficient] = load3(parameters.featuresRest, index * (shCount - 1) + coefficient - 1);
    return gaussian;
}

void storeGaussian(Parameters parameters, uint index, GaussianInput gaussian) {
    store3(parameters.means, index, gaussian.mean);
    store3(parameters.scales, index, gaussian.logScale);
    [ForceUnroll]
    for (uint component = 0; component < 4; ++component)
        parameters.rotations[index * 4 + component] = gaussian.rotation[component];
    parameters.opacities[index] = gaussian.logitOpacity;
    store3(parameters.featuresDc, index, gaussian.sh[0]);
    [ForceUnroll]
    for (uint coefficient = 1; coefficient < shCount; ++coefficient)
        store3(parameters.featuresRest, index * (shCount - 1) + coefficient - 1, gaussian.sh[coefficient]);
}

uint4 getRect(float2 mean, float radius, uint2 tiles) {
    const int2 low = int2((mean - radius) / float(tileSize));
    const int2 high = int2((mean + radius + float(tileSize - 1)) / float(tileSize));
    return uint4(min(tiles, uint2(max(low, int2(0)))), min(tiles, uint2(max(high, int2(0)))));
}

uint2 getTiles(Arguments arguments) {
    return uint2(arguments.view.tilesX, arguments.view.tilesY);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void preprocess(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const ProjectionView view = getView(arguments);
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        [ForceUnroll]
        for (uint slot = 0; slot < gradientStride; ++slot)
            atomicStore(arguments.splatGradients + index * gradientStride + slot, 0.0);
        arguments.tilesTouched[index] = 0;
        arguments.projected[index] = float4(0.0);
        const GaussianInput gaussian = loadGaussian(arguments.parameters, index);
        const float depth = dot(view.rotation[2], gaussian.mean) + view.translation.z;
        if (depth <= nearPlane)
            continue;
        const float3 covariance = getCovariance2d(gaussian.mean, gaussian.logScale, gaussian.rotation, view);
        const float determinant = covariance.x * covariance.z - covariance.y * covariance.y;
        if (determinant == 0.0)
            continue;
        const float middle = 0.5 * (covariance.x + covariance.z);
        const float spread = sqrt(max(0.1, middle * middle - determinant));
        const float radius = ceil(3.0 * sqrt(max(middle + spread, middle - spread)));
        const float2 mean = getMean2d(gaussian.mean, view);
        const uint4 rect = getRect(mean, radius, getTiles(arguments));
        const uint area = (rect.z - rect.x) * (rect.w - rect.y);
        if (area == 0)
            continue;
        const Splat splat = getSplat(gaussian, view, arguments.shDegree);
        arguments.projected[index] = float4(mean, depth, radius);
        arguments.conics[index] = float4(splat.conic, splat.opacity);
        store3(arguments.colors, index, splat.color);
        arguments.tilesTouched[index] = area;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void countInstances() {
    const Arguments arguments = rootArguments<Arguments>();
    const uint last = arguments.count - 1;
    const uint total = arguments.offsets[last] + arguments.tilesTouched[last];
    arguments.instanceTotal[0] = total;
    arguments.instanceCount[0] = min(total, arguments.instanceCapacity);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void duplicateKeys(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        if (arguments.tilesTouched[index] == 0)
            continue;
        const float4 projected = arguments.projected[index];
        const uint4 rect = getRect(projected.xy, projected.w, getTiles(arguments));
        const uint depthBits = asuint(projected.z);
        uint offset = arguments.offsets[index];
        for (uint y = rect.y; y < rect.w; ++y)
            for (uint x = rect.x; x < rect.z && offset < arguments.instanceCapacity; ++x) {
                arguments.keys[offset * 2] = depthBits;
                arguments.keys[offset * 2 + 1] = y * arguments.view.tilesX + x;
                arguments.values[offset] = index;
                ++offset;
            }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearRanges(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.view.tilesX * arguments.view.tilesY)
        return;
    arguments.ranges[id.x * 2] = 0;
    arguments.ranges[id.x * 2 + 1] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void identifyRanges(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const uint count = arguments.instanceCount[0];
    for (uint index = id.x; index < count; index += arguments.threads) {
        const uint tile = arguments.keys[index * 2 + 1];
        if (index == 0) {
            arguments.ranges[tile * 2] = 0;
        } else {
            const uint previous = arguments.keys[index * 2 - 1];
            if (tile != previous) {
                arguments.ranges[previous * 2 + 1] = index;
                arguments.ranges[tile * 2] = index;
            }
        }
        if (index == count - 1)
            arguments.ranges[tile * 2 + 1] = count;
    }
}

bool countDone(bool done, uint thread) {
    if (thread == 0)
        doneCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (done)
        InterlockedAdd(doneCount, 1);
    GroupMemoryBarrierWithGroupSync();
    return doneCount == groupSize;
}

float getPower(float2 delta, float4 conic) {
    return -0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blend(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint thread : SV_GroupIndex) {
    const Arguments arguments = rootArguments<Arguments>();
    const uint2 pixel = group.xy * tileSize + local.xy;
    const bool inside = all(pixel < uint2(arguments.view.width, arguments.view.height));
    const uint tile = group.y * arguments.view.tilesX + group.x;
    const uint2 range = uint2(arguments.ranges[tile * 2], arguments.ranges[tile * 2 + 1]);
    const float2 position = float2(pixel);
    const uint rounds = (range.y - range.x + groupSize - 1) / groupSize;
    int toDo = int(range.y - range.x);
    bool done = !inside;
    float transmittance = 1.0;
    uint contributor = 0;
    uint lastContributor = 0;
    float3 color = float3(0.0);
    for (uint round = 0; round < rounds; ++round) {
        if (countDone(done, thread))
            break;
        const uint progress = round * groupSize + thread;
        if (range.x + progress < range.y) {
            const uint gaussian = arguments.values[range.x + progress];
            sharedIds[thread] = gaussian;
            sharedMeans[thread] = arguments.projected[gaussian].xy;
            sharedConics[thread] = arguments.conics[gaussian];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint entry = 0; !done && entry < uint(min(int(groupSize), toDo)); ++entry) {
            ++contributor;
            const float4 conic = sharedConics[entry];
            const float power = getPower(sharedMeans[entry] - position, conic);
            if (power > 0.0)
                continue;
            const float alpha = min(maxAlpha, conic.w * exp(power));
            if (alpha < alphaThreshold)
                continue;
            const float next = transmittance * (1.0 - alpha);
            if (next < transmittanceThreshold) {
                done = true;
                continue;
            }
            color += load3(arguments.colors, sharedIds[entry]) * alpha * transmittance;
            transmittance = next;
            lastContributor = contributor;
        }
        toDo -= int(groupSize);
    }
    if (!inside)
        return;
    const uint index = pixel.y * arguments.view.width + pixel.x;
    arguments.finalTransmittance[index] = transmittance;
    arguments.lastContributors[index] = lastContributor;
    store3(arguments.image, index, color + transmittance * arguments.view.background.xyz);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendBackward(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint thread : SV_GroupIndex) {
    const Arguments arguments = rootArguments<Arguments>();
    const uint2 pixel = group.xy * tileSize + local.xy;
    const bool inside = all(pixel < uint2(arguments.view.width, arguments.view.height));
    const uint index = inside ? pixel.y * arguments.view.width + pixel.x : 0;
    const uint tile = group.y * arguments.view.tilesX + group.x;
    const float finalTransmittance = inside ? arguments.finalTransmittance[index] : 0.0;
    const uint lastContributor = inside ? arguments.lastContributors[index] : 0;
    if (thread == 0)
        tileContributors = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint waveContributors = WaveActiveMax(lastContributor);
    if (WaveIsFirstLane())
        InterlockedMax(tileContributors, waveContributors);
    GroupMemoryBarrierWithGroupSync();
    //! The Gaussians behind the last contributor of every pixel of the tile have no gradient, so the tile starts after them.
    const uint2 range = uint2(arguments.ranges[tile * 2], arguments.ranges[tile * 2] + tileContributors);
    const float2 position = float2(pixel);
    const uint rounds = (range.y - range.x + groupSize - 1) / groupSize;
    int toDo = int(range.y - range.x);
    const float3 pixelGradient = inside ? load3(arguments.imageGradient, index) : float3(0.0);
    const float backgroundGradient = dot(arguments.view.background.xyz, pixelGradient);
    float transmittance = finalTransmittance;
    uint contributor = uint(toDo);
    float3 accumulated = float3(0.0);
    float lastAlpha = 0.0;
    float3 lastColor = float3(0.0);
    for (uint round = 0; round < rounds; ++round) {
        GroupMemoryBarrierWithGroupSync();
        const uint progress = round * groupSize + thread;
        if (range.x + progress < range.y) {
            const uint gaussian = arguments.values[range.y - progress - 1];
            sharedIds[thread] = gaussian;
            sharedMeans[thread] = arguments.projected[gaussian].xy;
            sharedConics[thread] = arguments.conics[gaussian];
            sharedColors[thread] = load3(arguments.colors, gaussian);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint entries = uint(min(int(groupSize), toDo));
        for (uint entry = 0; entry < entries; ++entry) {
            --contributor;
            float gradients[gradientStride] = {};
            bool contributes = false;
            const float4 conic = sharedConics[entry];
            const float2 delta = sharedMeans[entry] - position;
            const float power = getPower(delta, conic);
            const float gaussianValue = exp(power);
            const float alpha = min(maxAlpha, conic.w * gaussianValue);
            if (contributor < lastContributor && power <= 0.0 && alpha >= alphaThreshold) {
                contributes = true;
                transmittance = transmittance / (1.0 - alpha);
                const float3 color = sharedColors[entry];
                accumulated = lastAlpha * lastColor + (1.0 - lastAlpha) * accumulated;
                lastColor = color;
                float alphaGradient = dot(color - accumulated, pixelGradient) * transmittance;
                const float3 colorGradient = alpha * transmittance * pixelGradient;
                lastAlpha = alpha;
                alphaGradient += -finalTransmittance / (1.0 - alpha) * backgroundGradient;
                const float valueGradient = conic.w * alphaGradient;
                const float gx = gaussianValue * delta.x;
                const float gy = gaussianValue * delta.y;
                gradients[0] = valueGradient * (-gx * conic.x - gy * conic.y);
                gradients[1] = valueGradient * (-gy * conic.z - gx * conic.y);
                gradients[2] = -0.5 * gx * delta.x * valueGradient;
                gradients[3] = -gx * delta.y * valueGradient;
                gradients[4] = -0.5 * gy * delta.y * valueGradient;
                gradients[5] = gaussianValue * alphaGradient;
                gradients[6] = colorGradient.x;
                gradients[7] = colorGradient.y;
                gradients[8] = colorGradient.z;
            }
            if (!WaveActiveAnyTrue(contributes))
                continue;
            const uint gaussian = sharedIds[entry];
            [ForceUnroll]
            for (uint slot = 0; slot < gradientStride; ++slot) {
                const float sum = WaveActiveSum(gradients[slot]);
                if (WaveIsFirstLane())
                    atomicAdd(arguments.splatGradients + gaussian * gradientStride + slot, sum);
            }
        }
        toDo -= int(groupSize);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void preprocessBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const ProjectionView view = getView(arguments);
    const float2 ndcScale = 0.5 * float2(arguments.view.width, arguments.view.height);
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        if (arguments.tilesTouched[index] == 0) {
            storeGaussian(arguments.gradients, index, (GaussianInput)0);
            continue;
        }
        float values[gradientStride];
        [ForceUnroll]
        for (uint slot = 0; slot < gradientStride; ++slot)
            values[slot] = atomicLoad(arguments.splatGradients + index * gradientStride + slot);
        Splat.Differential splatGradient;
        splatGradient.mean = float2(values[0], values[1]);
        splatGradient.conic = float3(values[2], values[3], values[4]);
        splatGradient.opacity = values[5];
        splatGradient.color = float3(values[6], values[7], values[8]);
        var gaussian = diffPair(loadGaussian(arguments.parameters, index));
        bwd_diff(getSplat)(gaussian, view, arguments.shDegree, splatGradient);
        GaussianInput gradient;
        gradient.mean = gaussian.d.mean;
        gradient.logScale = gaussian.d.logScale;
        gradient.rotation = gaussian.d.rotation;
        gradient.logitOpacity = gaussian.d.logitOpacity;
        gradient.sh = gaussian.d.sh;
        storeGaussian(arguments.gradients, index, gradient);
        if (arguments.accumulate == 0)
            continue;
        arguments.gradientSums[index] += length(splatGradient.mean * ndcScale);
        arguments.gradientCounts[index] += 1.0;
    }
}
