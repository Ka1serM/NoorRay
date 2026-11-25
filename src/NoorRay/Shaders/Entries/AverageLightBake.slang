import NoorRHI;
import SceneRecords;
import Common.Math;
#include "NoorRay/Shaders/Materials/GraphBakeRecords.h"

groupshared float4 averageSums[lightAverageGroupSize];

//! One group sums the texture in a fixed order, so a bake of the same texels gives the same mean.
[shader("compute")]
[numthreads(lightAverageGroupSize, 1, 1)]
void averageLightBake(uint3 lane : SV_GroupThreadID) {
    const LightAverageArguments arguments = rootArguments<LightAverageArguments>();
    const uint resolution = arguments.resolution;
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    for (uint index = lane.x; index < resolution * resolution; index += lightAverageGroupSize) {
        const float2 uv = (float2(index % resolution, index / resolution) + 0.5) / float(resolution) - 0.5;
        if (arguments.kind == graphBakeSpot && dot(uv, uv) > 0.25)
            continue;
        if (arguments.kind == graphBakeAreaDirection && uv.y >= 0.0)
            continue;
        sum += float4(unpackRgb9e5(arguments.texels[index]), 1.0);
    }
    averageSums[lane.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = lightAverageGroupSize / 2; stride > 0; stride >>= 1) {
        if (lane.x < stride)
            averageSums[lane.x] += averageSums[lane.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane.x == 0)
        arguments.average[0] = averageSums[0].w > 0.0 ? float4(averageSums[0].xyz / averageSums[0].w, 1.0) : float4(0.0, 0.0, 0.0, 1.0);
}
