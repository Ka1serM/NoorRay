#include "TileSplatting.h"
import GaussianProjection;

typealias DensifyArguments = TileSplatting::DensifyArguments;

static const uint noSource = 0xffffffff;
static const uint keep = 1;
static const uint clone = 2;
static const uint split = 4;
static const float splitDivisor = 1.6;

float getMaxScale(DensifyArguments arguments, uint index) {
    const Ptr<float> scales = arguments.parameters.scales;
    return exp(max(scales[index * 3], max(scales[index * 3 + 1], scales[index * 3 + 2])));
}

uint classify(DensifyArguments arguments, uint index) {
    const float count = arguments.gradientCounts[index];
    const float gradient = count > 0.0 ? arguments.gradientSums[index] / count : 0.0;
    const float scale = getMaxScale(arguments, index);
    const float opacity = 1.0 / (1.0 + exp(-arguments.parameters.opacities[index]));
    const bool selected = gradient >= arguments.gradientThreshold;
    const bool large = scale > arguments.splitScale;
    const bool transparent = opacity < arguments.pruneOpacity;
    const bool pruned = transparent || scale > arguments.pruneScale;
    const bool childPruned = transparent || scale / splitDivisor > arguments.pruneScale;
    uint kinds = 0;
    if (!(selected && large) && !pruned)
        kinds |= keep;
    if (selected && !large && !pruned)
        kinds |= clone;
    if (selected && large && !childPruned)
        kinds |= split;
    return kinds;
}

uint getOutputCount(uint kinds) {
    return ((kinds & keep) != 0 ? 1 : 0) + ((kinds & clone) != 0 ? 1 : 0) + ((kinds & split) != 0 ? 2 : 0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void countOutputs(uint3 id : SV_DispatchThreadID) {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads)
        arguments.outputs[index] = getOutputCount(classify(arguments, index));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void totalOutputs() {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    const uint last = arguments.count - 1;
    arguments.newCount[0] = arguments.offsets[last] + arguments.outputs[last];
}

void writeSource(DensifyArguments arguments, inout uint offset, uint index, bool child, bool fresh) {
    arguments.parameterSources[offset] = index;
    arguments.momentSources[offset] = fresh ? noSource : index;
    arguments.children[offset] = child ? 1 : 0;
    ++offset;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void writeSources(uint3 id : SV_DispatchThreadID) {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        const uint kinds = classify(arguments, index);
        uint offset = arguments.offsets[index];
        if ((kinds & keep) != 0)
            writeSource(arguments, offset, index, false, false);
        if ((kinds & clone) != 0)
            writeSource(arguments, offset, index, false, true);
        if ((kinds & split) == 0)
            continue;
        writeSource(arguments, offset, index, true, true);
        writeSource(arguments, offset, index, true, true);
    }
}

uint hash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return value;
}

float getUniform(uint seed, uint index, uint draw) {
    const uint bits = hash(seed ^ hash(index * 8 + draw));
    return (float(bits >> 8) + 0.5) / 16777216.0;
}

float getNormal(uint seed, uint index, uint axis) {
    const float radius = sqrt(-2.0 * log(getUniform(seed, index, axis * 2)));
    return radius * cos(6.283185307179586 * getUniform(seed, index, axis * 2 + 1));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splitChildren(uint3 id : SV_DispatchThreadID) {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    const TileSplatting::Parameters parameters = arguments.parameters;
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        if (arguments.children[index] == 0)
            continue;
        const float3 scale = exp(float3(parameters.scales[index * 3], parameters.scales[index * 3 + 1], parameters.scales[index * 3 + 2]));
        const float3 sample = scale * float3(getNormal(arguments.seed, index, 0), getNormal(arguments.seed, index, 1), getNormal(arguments.seed, index, 2));
        const float4 quaternion = float4(parameters.rotations[index * 4], parameters.rotations[index * 4 + 1], parameters.rotations[index * 4 + 2],
                                         parameters.rotations[index * 4 + 3]);
        const float3 offset = mul(getRotationMatrix(quaternion), sample);
        const float3 childScale = log(scale / splitDivisor);
        [ForceUnroll]
        for (uint axis = 0; axis < 3; ++axis) {
            parameters.means[index * 3 + axis] += offset[axis];
            parameters.scales[index * 3 + axis] = childScale[axis];
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearFloats(uint3 id : SV_DispatchThreadID) {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads)
        arguments.destination[index] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void resetOpacity(uint3 id : SV_DispatchThreadID) {
    const DensifyArguments arguments = rootArguments<DensifyArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        arguments.parameters.opacities[index] = min(arguments.parameters.opacities[index], arguments.resetOpacity);
        arguments.firstMoments[index] = 0.0;
        arguments.secondMoments[index] = 0.0;
    }
}
