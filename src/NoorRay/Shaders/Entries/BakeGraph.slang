import NoorRHI;
import SceneRecords;
import Common.Math;
import Materials.Graph;
#include "NoorRay/Shaders/Materials/GraphBakeRecords.h"

float3 bakeToWorld(GraphBakeRecord record, float4 local) {
    return float3(dot(record.objectToWorld[0], local), dot(record.objectToWorld[1], local), dot(record.objectToWorld[2], local));
}

//! The unit local axis i in world space, as the light record has it.
float3 bakeAxis(GraphBakeRecord record, uint i) {
    return normalize(float3(record.objectToWorld[0][i], record.objectToWorld[1][i], record.objectToWorld[2][i]));
}

//! Row i of the inverse of the affine record.objectToWorld; the columns of the inverse 3x3 are the cross products of its rows over the determinant.
float4 bakeToObject(GraphBakeRecord record, uint i) {
    const float3 r0 = record.objectToWorld[0].xyz;
    const float3 r1 = record.objectToWorld[1].xyz;
    const float3 r2 = record.objectToWorld[2].xyz;
    const float3 row = float3(cross(r1, r2)[i], cross(r2, r0)[i], cross(r0, r1)[i]) / dot(r0, cross(r1, r2));
    const float3 translation = float3(record.objectToWorld[0].w, record.objectToWorld[1].w, record.objectToWorld[2].w);
    return float4(row, -dot(row, translation));
}

//! A light or volume texel as Cycles shades it: UV from the light UV mapping, Generated in light space, the volume bounds from 0 to 1.
ShadingContext bakeContext(GraphBakeRecord record, float3 cell) {
    float3 local = float3(0.0, 0.0, 0.0);
    float3 normal = float3(0.0, 0.0, 1.0);
    float3 generated = local;
    float3 leaving = float3(0.0, 0.0, 0.0);
    switch (record.kind) {
    case graphBakeSpot: {
        //! The spot UV is 0.5 + local.xy * w / local.z, about the spot direction local +z, which is the light's -Z.
        const float2 planar = (cell.xy - 0.5) / record.shape.w;
        normal = normalize(float3(planar, -1.0));
        break;
    }
    case graphBakeArea:
        local = float3((cell.xy - 0.5) * 2.0 * record.shape.xy, 0.0);
        generated = local;
        normal = float3(0.0, 0.0, -1.0);
        break;
    case graphBakeSphere: {
        //! The inverse of Cycles' map_to_sphere: u = 0.5 - atan2(x, y) / 2pi, v = 1 - acos(z) / pi.
        const float phi = (0.5 - cell.x) * 2.0 * pi;
        const float theta = (1.0 - cell.y) * pi;
        normal = float3(sin(theta) * sin(phi), sin(theta) * cos(phi), cos(theta));
        local = normal * record.shape.z;
        generated = local;
        break;
    }
    case graphBakeAreaDirection: {
        const float phi = (0.5 - cell.x) * 2.0 * pi;
        const float theta = (1.0 - cell.y) * pi;
        leaving = float3(sin(theta) * sin(phi), sin(theta) * cos(phi), cos(theta));
        normal = float3(0.0, 0.0, -1.0);
        break;
    }
    case graphBakeWorld:
    case graphBakeWorldCamera:
        //! Cycles shades the background at the unit direction, with the normal and the incoming direction against the ray.
        local = equirectToDirection(cell.xy);
        local = float3(local.x, -local.z, local.y);
        generated = local;
        normal = -local;
        break;
    default:
        local = lerp(record.shape.xyz, record.boundsMax.xyz, cell);
        generated = cell;
        break;
    }
    const float3 position = bakeToWorld(record, float4(local, 1.0));
    const float3 worldNormal = normalize(bakeToWorld(record, float4(normal, 0.0)));
    ShadingContext context;
    context.position = float4(position, 1.0);
    context.normal = float4(worldNormal, 0.0);
    context.geometricNormal = float4(worldNormal, 0.0);
    context.tangent = float4(normalize(bakeToWorld(record, float4(1.0, 0.0, 0.0, 0.0))), 1.0);
    const bool directional = record.kind == graphBakeAreaDirection;
    const float3 leavingWorld = bakeAxis(record, 0) * leaving.x + bakeAxis(record, 1) * leaving.y + bakeAxis(record, 2) * leaving.z;
    context.incoming = float4(directional ? normalize(leavingWorld) : worldNormal, 1.0);
    context.uv01 = float4(cell.xy, cell.xy);
    context.uv23 = float4(cell.xy, cell.xy);
    context.color = float4(1.0, 1.0, 1.0, 1.0);
    context.uvGradients = float4(1.0 / float(record.resolution.x), 0.0, 0.0, 1.0 / float(record.resolution.y));
    context.dPdx = float4(0.0, 0.0, 0.0, 0.0);
    context.dPdy = float4(0.0, 0.0, 0.0, 0.0);
    context.objectPosition = float4(local, 0.0);
    context.generated = float4(generated, 0.0);
    context.objectLocation = float4(record.objectToWorld[0].w, record.objectToWorld[1].w, record.objectToWorld[2].w, 0.0);
    context.objectDx = float4(0.0, 0.0, 0.0, 0.0);
    context.objectDy = float4(0.0, 0.0, 0.0, 0.0);
    context.generatedScale = float4(1.0, 1.0, 1.0, 0.0);
    context.worldToObject0 = bakeToObject(record, 0);
    context.worldToObject1 = bakeToObject(record, 1);
    context.worldToObject2 = bakeToObject(record, 2);
    context.objectToWorld0 = record.objectToWorld[0];
    context.objectToWorld1 = record.objectToWorld[1];
    context.objectToWorld2 = record.objectToWorld[2];
    setCameraRay(context);
    if (record.kind == graphBakeWorld) {
        context.rayType = float4(0.0, 0.0, 1.0, 0.0);
        context.rayDepth = float4(1.0, 1.0, 0.0, 0.0);
    }
    return context;
}

groupshared uint bakeMaxima[graphBakeGroupSize];
groupshared uint bakeCounts[graphBakeGroupSize];

uint3 bakeTexel(uint3 resolution, uint index) {
    return uint3(index % resolution.x, (index / resolution.x) % resolution.y, index / (resolution.x * resolution.y));
}

float bakeDensity(GraphBakeRecord record, float3 cell) {
    const MaterialGraphOutput output = evaluateMaterialGraph(bakeContext(record, cell), graphMaterial(record.parameters, record.textures));
    return max(output.emission.x, 0.0);
}

float3 bakeEmission(GraphBakeRecord record, uint index) {
    const float3 cell = (float3(bakeTexel(record.resolution.xyz, index)) + 0.5) / float3(record.resolution.xyz);
    const MaterialGraphOutput output = evaluateMaterialGraph(bakeContext(record, cell), graphMaterial(record.parameters, record.textures));
    return max(output.emission.xyz, float3(0.0, 0.0, 0.0));
}

//! Brick sample index in x-fastest order.
uint3 brickCorner(uint sample) {
    return uint3(sample % volumeBrickSamples, (sample / volumeBrickSamples) % volumeBrickSamples, sample / (volumeBrickSamples * volumeBrickSamples));
}

//! The position in the bounds from 0 to 1 of brick sample s of a cell.
float3 brickSample(GraphBakeRecord record, uint3 cell, uint3 s) {
    return float3(cell * volumeBrickSize + s) / float3(record.resolution.xyz * volumeBrickSize);
}

//! A cell whose samples differ by less than this part of its maximum is one constant; a ramp that clamps the noise gives exact constants.
static const float constantCellTolerance = 1e-3;

//! One thread classifies one cell and writes its half range; the group adds its brick count and raises the largest density with one atomic each.
void bakeCells(GraphBakeRecord record, uint index, uint lane) {
    const uint3 resolution = record.resolution.xyz;
    uint largest = 0;
    uint bricks = 0;
    if (index < resolution.x * resolution.y * resolution.z) {
        const uint3 cell = bakeTexel(resolution, index);
        float low = floatMax;
        float high = 0.0;
        [loop]
        for (uint sample = 0; sample < record.resolution.w; ++sample) {
            const float density = bakeDensity(record, brickSample(record, cell, brickCorner(sample)));
            low = min(low, density);
            high = max(high, density);
        }
        const bool constant = high - low <= constantCellTolerance * high;
        if (constant) {
            low = 0.5 * (low + high);
            high = low;
        }
        const uint packed = f32tof16(low) | f32tof16(high) << 16;
        record.output[2 * index] = packed;
        record.output[2 * index + 1] = constant ? invalidIndex : 0u;
        largest = asuint(f16tof32(packed >> 16));
        bricks = constant ? 0u : 1u;
    }
    bakeMaxima[lane] = largest;
    bakeCounts[lane] = bricks;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = graphBakeGroupSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            bakeMaxima[lane] = max(bakeMaxima[lane], bakeMaxima[lane + stride]);
            bakeCounts[lane] += bakeCounts[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0)
        return;
    atomicMax(record.maximum, bakeMaxima[0]);
    atomicAdd(record.maximum + 1, bakeCounts[0]);
}

//! One thread fills the brick of one cell with its samples, 8 bits each between the half minimum and maximum of the cell.
void bakeBricks(GraphBakeRecord record, uint index) {
    const uint3 resolution = record.resolution.xyz;
    if (index >= resolution.x * resolution.y * resolution.z || record.output[2 * index + 1] == invalidIndex)
        return;
    const uint3 cell = bakeTexel(resolution, index);
    const uint packed = record.output[2 * index];
    const float low = f16tof32(packed & 0xffffu);
    const float span = max(f16tof32(packed >> 16) - low, 1e-20);
    const uint slot = atomicAdd(record.maximum + 2, 1u);
    record.output[2 * index + 1] = slot;
    const Ptr<uint> brick = record.bricks + slot * volumeBrickWords;
    uint word = 0;
    [loop]
    for (uint sample = 0; sample < record.resolution.w; ++sample) {
        const float density = bakeDensity(record, brickSample(record, cell, brickCorner(sample)));
        const uint level = uint(clamp(round((density - low) / span * 255.0), 0.0, 255.0));
        word |= level << (8 * (sample & 3u));
        if ((sample & 3u) == 3u || sample + 1 == record.resolution.w) {
            brick[sample >> 2] = word;
            word = 0;
        }
    }
}

//! GraphBakes compiles this kernel once per graph, with the code of the graph as the body of evaluateMaterialGraph.
[shader("compute")]
[numthreads(graphBakeGroupSize, 1, 1)]
void bakeGraph(uint3 id : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID) {
    const GraphBakeRecord record = rootArguments<GraphBakeArguments>().record[0];
    const uint index = id.y * graphBakeRowGroups * graphBakeGroupSize + id.x;
    if (record.kind == graphBakeVolumeCells) {
        bakeCells(record, index, lane.x);
        return;
    }
    if (record.kind == graphBakeVolumeBricks) {
        bakeBricks(record, index);
        return;
    }
    const uint3 resolution = record.resolution.xyz;
    if (index >= resolution.x * resolution.y * resolution.z)
        return;
    const float3 emission = bakeEmission(record, index);
    record.output[index] = packRgb9e5(emission);
}
