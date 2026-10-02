#include "../../Shared/SplineMesh.h"

[[vk::push_constant]] ConstantBuffer<SplineMeshRoot> root;

static const uint SplineMeshGroupSize = 64u;

// Unreal's FPackedNormal: four SNORM8 components, x in the low byte.
float4 unpackSnorm8x4(uint packed)
{
    const int4 values = int4(int(packed << 24u) >> 24, int(packed << 16u) >> 24,
                             int(packed << 8u) >> 24, int(packed) >> 24);
    return max(float4(values) / 127.0, -1.0);
}

uint packSnorm8x4(float4 value)
{
    const uint4 values = uint4(int4(round(clamp(value, -1.0, 1.0) * 127.0))) & 0xffu;
    return values.x | values.y << 8u | values.z << 16u | values.w << 24u;
}

// A cross-section of the spline: its center, its frame and its X/Y scale.
struct SplineMeshSlice
{
    float3 position;
    float3 direction;
    float3 xAxis;
    float3 yAxis;
    float2 scale;
};

// Vertices past either end follow the curve's polynomial; nothing clamps them.
SplineMeshSlice splineMeshSlice(const SplineMesh spline, const float distance)
{
    const float a = distance;
    const float a2 = a * a;
    const float a3 = a2 * a;
    const float3 position = (2.0 * a3 - 3.0 * a2 + 1.0) * spline.startPosition
        + (a3 - 2.0 * a2 + a) * spline.startTangent + (a3 - a2) * spline.endTangent
        + (-2.0 * a3 + 3.0 * a2) * spline.endPosition;
    const float3 c = 6.0 * spline.startPosition + 3.0 * spline.startTangent + 3.0 * spline.endTangent
        - 6.0 * spline.endPosition;
    const float3 d = -6.0 * spline.startPosition - 4.0 * spline.startTangent - 2.0 * spline.endTangent
        + 6.0 * spline.endPosition;

    SplineMeshSlice slice;
    slice.direction = normalize(c * a2 + d * a + spline.startTangent);
    const float hermite = spline.smoothInterpRollScale != 0u ? smoothstep(0.0, 1.0, a) : a;
    const float roll = lerp(spline.startRoll, spline.endRoll, hermite);
    const float2 offset = lerp(spline.startOffset, spline.endOffset, hermite);
    slice.scale = lerp(spline.startScale, spline.endScale, hermite);

    const float3 baseX = normalize(cross(spline.upDirection, slice.direction));
    const float3 baseY = cross(slice.direction, baseX);
    const float sinRoll = sin(roll);
    const float cosRoll = cos(roll);
    slice.position = position + offset.x * baseX + offset.y * baseY;
    slice.xAxis = cosRoll * baseX - sinRoll * baseY;
    slice.yAxis = cosRoll * baseY + sinRoll * baseX;
    return slice;
}

[shader("compute")]
[numthreads(SplineMeshGroupSize, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const SplineMeshArguments arguments = *root.arguments;
    const uint vertex = threadId.x;
    if (vertex >= arguments.vertexCount)
        return;
    const SplineMesh spline = arguments.spline;
    const uint forward = spline.forwardAxis;
    const uint sideX = (forward + 1u) % 3u;
    const uint sideY = (forward + 2u) % 3u;

    const float3 local = arguments.positions[vertex];
    const SplineMeshSlice slice = splineMeshSlice(spline,
        local[forward] * spline.distanceScale + spline.distanceOffset);
    arguments.positions[vertex] = slice.position + local[sideX] * slice.scale.x * slice.xAxis
        + local[sideY] * slice.scale.y * slice.yAxis;

    // SplineMeshCalcSliceRot: the frame, with an axis flipped where its scale
    // is negative.
    const float3 xAxis = (slice.scale.x >= 0.0 ? 1.0 : -1.0) * slice.xAxis;
    const float3 yAxis = (slice.scale.y >= 0.0 ? 1.0 : -1.0) * slice.yAxis;
    const uint2 frame = arguments.tangents[vertex];
    const float4 tangentX = unpackSnorm8x4(frame.x);
    const float4 tangentZ = unpackSnorm8x4(frame.y);
    const float3 tangent = tangentX[forward] * slice.direction + tangentX[sideX] * xAxis + tangentX[sideY] * yAxis;
    const float3 normal = tangentZ[forward] * slice.direction + tangentZ[sideX] * xAxis + tangentZ[sideY] * yAxis;
    arguments.tangents[vertex] = uint2(packSnorm8x4(float4(tangent, tangentX.w)),
                                       packSnorm8x4(float4(normal, tangentZ.w)));
}
