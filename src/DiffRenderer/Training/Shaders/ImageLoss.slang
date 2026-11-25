#include "ImageLoss.h"

typealias Arguments = ImageLoss::Arguments;

static const uint radius = 5;
static const float window[11] = { 0.001028380, 0.007598758, 0.036000772, 0.109360690, 0.213005538, 0.266011725,
                                  0.213005538, 0.109360690, 0.036000772, 0.007598758, 0.001028380 };
static const float c1 = 0.01 * 0.01;
static const float c2 = 0.03 * 0.03;
static const float l1Weight = 0.8;
static const float ssimWeight = 0.2;
static const uint groupWidth = 16;
static const uint groupSize = groupWidth * groupWidth;
static const uint momentCount = 5;
static const uint partialCount = 3;

groupshared float groupSums[groupSize];

float loadImage(Arguments arguments, uint2 pixel, uint channel) {
    return arguments.image[(pixel.y * arguments.width + pixel.x) * 3 + channel];
}

float loadTarget(Arguments arguments, uint2 pixel, uint channel) {
    return float((arguments.target[pixel.y * arguments.width + pixel.x] >> (channel * 8)) & 0xff) / 255.0;
}

uint getPlane(Arguments arguments, uint index, uint channel) {
    return (index * 3 + channel) * arguments.width * arguments.height;
}

bool isInside(Arguments arguments, uint2 pixel) {
    return all(pixel < uint2(arguments.width, arguments.height));
}

void addToLoss(Arguments arguments, uint3 group, uint thread, float loss) {
    groupSums[thread] = loss;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = groupSize / 2; stride > 0; stride /= 2) {
        if (thread < stride)
            groupSums[thread] += groupSums[thread + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0)
        arguments.groupLosses[group.y * ((arguments.width + groupWidth - 1) / groupWidth) + group.x] = groupSums[0];
}

float getSign(float value) {
    return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filterMomentRows(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint index = id.y * arguments.width + id.x;
    for (uint channel = 0; channel < 3; ++channel) {
        float sums[momentCount] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        for (uint tap = 0; tap < 2 * radius + 1; ++tap) {
            const int x = int(id.x) + int(tap) - int(radius);
            if (x < 0 || x >= int(arguments.width))
                continue;
            const float value = loadImage(arguments, uint2(x, id.y), channel);
            const float target = loadTarget(arguments, uint2(x, id.y), channel);
            const float weight = window[tap];
            sums[0] += weight * value;
            sums[1] += weight * target;
            sums[2] += weight * value * value;
            sums[3] += weight * target * target;
            sums[4] += weight * value * target;
        }
        for (uint moment = 0; moment < momentCount; ++moment)
            arguments.rows[getPlane(arguments, moment, channel) + index] = sums[moment];
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filterMomentColumns(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const Arguments arguments = rootArguments<Arguments>();
    float loss = 0.0;
    if (isInside(arguments, id.xy)) {
        const uint index = id.y * arguments.width + id.x;
        for (uint channel = 0; channel < 3; ++channel) {
            float sums[momentCount] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (uint tap = 0; tap < 2 * radius + 1; ++tap) {
                const int y = int(id.y) + int(tap) - int(radius);
                if (y < 0 || y >= int(arguments.height))
                    continue;
                for (uint moment = 0; moment < momentCount; ++moment)
                    sums[moment] += window[tap] * arguments.rows[getPlane(arguments, moment, channel) + uint(y) * arguments.width + id.x];
            }
            const float mean = sums[0];
            const float targetMean = sums[1];
            const float variance = sums[2] - mean * mean;
            const float targetVariance = sums[3] - targetMean * targetMean;
            const float covariance = sums[4] - mean * targetMean;
            const float a = 2.0 * mean * targetMean + c1;
            const float b = 2.0 * covariance + c2;
            const float c = mean * mean + targetMean * targetMean + c1;
            const float d = variance + targetVariance + c2;
            const float ssim = a * b / (c * d);
            const float byMean = 2.0 * targetMean * b / (c * d) - ssim * 2.0 * mean / c;
            const float byVariance = -ssim / d;
            const float byCovariance = 2.0 * a / (c * d);
            const float weight = -ssimWeight * arguments.scale;
            arguments.partials[getPlane(arguments, 0, channel) + index] = weight * (byMean - 2.0 * mean * byVariance - targetMean * byCovariance);
            arguments.partials[getPlane(arguments, 1, channel) + index] = weight * byVariance;
            arguments.partials[getPlane(arguments, 2, channel) + index] = weight * byCovariance;
            const float difference = loadImage(arguments, id.xy, channel) - loadTarget(arguments, id.xy, channel);
            loss += arguments.scale * (l1Weight * abs(difference) + ssimWeight * (1.0 - ssim));
        }
    }
    addToLoss(arguments, group, thread, loss);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filterPartialRows(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint index = id.y * arguments.width + id.x;
    for (uint channel = 0; channel < 3; ++channel)
        for (uint partial = 0; partial < partialCount; ++partial) {
            const uint plane = getPlane(arguments, partial, channel);
            float sum = 0.0;
            for (uint tap = 0; tap < 2 * radius + 1; ++tap) {
                const int x = int(id.x) + int(tap) - int(radius);
                if (x >= 0 && x < int(arguments.width))
                    sum += window[tap] * arguments.partials[plane + id.y * arguments.width + uint(x)];
            }
            arguments.rows[plane + index] = sum;
        }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filterPartialColumns(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint index = id.y * arguments.width + id.x;
    for (uint channel = 0; channel < 3; ++channel) {
        float sums[partialCount] = { 0.0, 0.0, 0.0 };
        for (uint tap = 0; tap < 2 * radius + 1; ++tap) {
            const int y = int(id.y) + int(tap) - int(radius);
            if (y < 0 || y >= int(arguments.height))
                continue;
            for (uint partial = 0; partial < partialCount; ++partial)
                sums[partial] += window[tap] * arguments.rows[getPlane(arguments, partial, channel) + uint(y) * arguments.width + id.x];
        }
        const float value = loadImage(arguments, id.xy, channel);
        const float target = loadTarget(arguments, id.xy, channel);
        const float l1 = l1Weight * arguments.scale * getSign(value - target);
        arguments.gradient[index * 3 + channel] = l1 + sums[0] + 2.0 * value * sums[1] + target * sums[2];
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void squaredError(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const Arguments arguments = rootArguments<Arguments>();
    float loss = 0.0;
    if (isInside(arguments, id.xy)) {
        const uint index = id.y * arguments.width + id.x;
        for (uint channel = 0; channel < 3; ++channel) {
            const float difference = loadImage(arguments, id.xy, channel) - loadTarget(arguments, id.xy, channel);
            arguments.gradient[index * 3 + channel] = 2.0 * arguments.scale * difference;
            loss += arguments.scale * difference * difference;
        }
    }
    addToLoss(arguments, group, thread, loss);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sumLosses(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    float sum = 0.0;
    for (uint index = id.x; index < arguments.groupCount; index += groupSize)
        sum += arguments.groupLosses[index];
    groupSums[id.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = groupSize / 2; stride > 0; stride /= 2) {
        if (id.x < stride)
            groupSums[id.x] += groupSums[id.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (id.x == 0)
        arguments.value[0] = arguments.value[0] + groupSums[0];
}
