#include "GridEditing.h"
#include "Voxels.slang"

static const uint brickSize = GridLayout::brickSize;
static const float variationEpsilon = 1e-9;

groupshared uint brickSums[brickSize];

[shader("compute")]
[numthreads(256, 1, 1)]
void fill(uint3 id : SV_DispatchThreadID) {
    const GridEditing::FillArguments arguments = rootArguments<GridEditing::FillArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads)
        arguments.values[index] = arguments.value;
}

void addVariation(AtomicPtr<float> gradients, uint stride, uint offset, int self, int3 neighbours, float3 differences, float weight) {
    const float scale = weight / sqrt(dot(differences, differences) + variationEpsilon);
    atomicAdd(gradients + uint(self) * stride + offset, -scale * (differences.x + differences.y + differences.z));
    for (uint axis = 0; axis < 3; ++axis)
        if (neighbours[axis] >= 0 && differences[axis] != 0.0)
            atomicAdd(gradients + uint(neighbours[axis]) * stride + offset, scale * differences[axis]);
}

uint getHash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return value;
}

float loadOr(Ptr<float> values, int link, float fallback) {
    return link < 0 ? fallback : values[uint(link)];
}

[shader("compute")]
[numthreads(8, 8, 8)]
void totalVariation(uint3 id : SV_DispatchThreadID) {
    const GridEditing::VariationArguments arguments = rootArguments<GridEditing::VariationArguments>();
    const Grid grid = arguments.grid;
    if (any(id >= grid.resolution))
        return;
    const int self = loadLink(grid, int3(id));
    if (self < 0 || getHash(uint(self) ^ arguments.seed) >= arguments.fraction * 4294967295.0)
        return;
    const int3 neighbours = int3(loadLink(grid, int3(id) + int3(1, 0, 0)), loadLink(grid, int3(id) + int3(0, 1, 0)), loadLink(grid, int3(id) + int3(0, 0, 1)));
    const float density = grid.density[uint(self)];
    const float3 densities = float3(loadOr(grid.density, neighbours.x, density), loadOr(grid.density, neighbours.y, density), loadOr(grid.density, neighbours.z, density));
    addVariation(arguments.densityGradients, 1, 0, self, neighbours, densities - density, arguments.densityWeight);
    for (uint index = 0; index < shCount; ++index) {
        const float value = grid.sh[uint(self) * shCount + index];
        float3 values = float3(value);
        for (uint axis = 0; axis < 3; ++axis)
            if (neighbours[axis] >= 0)
                values[axis] = grid.sh[uint(neighbours[axis]) * shCount + index];
        addVariation(arguments.shGradients, shCount, index, self, neighbours, values - value, arguments.shWeight);
    }
}

[shader("compute")]
[numthreads(8, 8, 8)]
void markKept(uint3 id : SV_DispatchThreadID) {
    const GridEditing::KeepArguments arguments = rootArguments<GridEditing::KeepArguments>();
    if (any(id >= arguments.resolution))
        return;
    const uint index = getVoxelIndex(arguments.resolution, int3(id));
    const int link = arguments.links[index];
    arguments.keep[index] = 0;
    if (link < 0)
        return;
    if (asfloat(arguments.maxWeights[uint(link)]) >= arguments.threshold)
        arguments.keep[index] = 1;
}

[shader("compute")]
[numthreads(8, 8, 8)]
void dilate(uint3 id : SV_DispatchThreadID) {
    const GridEditing::DilateArguments arguments = rootArguments<GridEditing::DilateArguments>();
    if (any(id >= arguments.resolution))
        return;
    int3 direction = int3(0);
    direction[arguments.axis] = 1;
    uint kept = 0;
    for (int offset = -int(arguments.radius); offset <= int(arguments.radius) && kept == 0; ++offset) {
        const int3 voxel = int3(id) + direction * offset;
        if (isInside(arguments.resolution, voxel))
            kept = arguments.source[getVoxelIndex(arguments.resolution, voxel)];
    }
    arguments.destination[getVoxelIndex(arguments.resolution, int3(id))] = kept;
}

float3 toOldVoxel(uint3 voxel, uint resolution, uint oldResolution) {
    return (float3(voxel) + 0.5) * float(oldResolution) / float(resolution) - 0.5;
}

bool isKept(GridEditing::LinkArguments arguments, uint3 voxel) {
    if (isNull(arguments.keep))
        return true;
    const int3 base = int3(floor(toOldVoxel(voxel, arguments.resolution, arguments.oldResolution)));
    for (uint corner = 0; corner < 8; ++corner) {
        const int3 old = base + getCornerOffset(corner);
        if (isInside(arguments.oldResolution, old) && arguments.keep[getVoxelIndex(arguments.oldResolution, old)] != 0)
            return true;
    }
    return false;
}

uint3 getBrickVoxel(uint3 brick, uint thread) {
    return brick * brickWidth + uint3(thread & 7, (thread >> 3) & 7, thread >> 6);
}

uint getBrickIndex(GridEditing::LinkArguments arguments, uint3 brick) {
    const uint bricks = arguments.resolution / brickWidth;
    return (brick.z * bricks + brick.y) * bricks + brick.x;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void countBricks(uint3 brick : SV_GroupID, uint thread : SV_GroupIndex) {
    const GridEditing::LinkArguments arguments = rootArguments<GridEditing::LinkArguments>();
    brickSums[thread] = isKept(arguments, getBrickVoxel(brick, thread)) ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = brickSize / 2; stride > 0; stride /= 2) {
        if (thread < stride)
            brickSums[thread] += brickSums[thread + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0)
        arguments.brickCounts[getBrickIndex(arguments, brick)] = brickSums[0];
}

[shader("compute")]
[numthreads(512, 1, 1)]
void assignLinks(uint3 brick : SV_GroupID, uint thread : SV_GroupIndex) {
    const GridEditing::LinkArguments arguments = rootArguments<GridEditing::LinkArguments>();
    const uint3 voxel = getBrickVoxel(brick, thread);
    const bool kept = isKept(arguments, voxel);
    brickSums[thread] = kept ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1; stride < brickSize; stride *= 2) {
        const uint previous = thread >= stride ? brickSums[thread - stride] : 0;
        GroupMemoryBarrierWithGroupSync();
        brickSums[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint link = arguments.brickOffsets[getBrickIndex(arguments, brick)] + brickSums[thread] - 1;
    arguments.links[getVoxelIndex(arguments.resolution, int3(voxel))] = kept ? int(link) : -1;
}

[shader("compute")]
[numthreads(8, 8, 8)]
void resample(uint3 id : SV_DispatchThreadID) {
    const GridEditing::ResampleArguments arguments = rootArguments<GridEditing::ResampleArguments>();
    const Grid destination = arguments.destination;
    if (any(id >= destination.resolution))
        return;
    const int link = loadLink(destination, int3(id));
    if (link < 0)
        return;
    const Ptr<float> sh = destination.sh + uint(link) * shCount;
    if (isNull(arguments.source.links)) {
        destination.density[uint(link)] = arguments.initialDensity;
        for (uint index = 0; index < shCount; ++index)
            sh[index] = 0.0;
        return;
    }
    const Corners corners = getCorners(arguments.source, toOldVoxel(id, destination.resolution, arguments.source.resolution));
    destination.density[uint(link)] = corners.getDensity(arguments.source);
    float values[shCount];
    corners.getSh(arguments.source, values);
    for (uint index = 0; index < shCount; ++index)
        sh[index] = values[index];
}

[shader("compute")]
[numthreads(8, 8, 8)]
void markOccupancy(uint3 id : SV_DispatchThreadID) {
    const Grid grid = rootArguments<GridEditing::OccupancyArguments>().grid;
    if (any(id >= grid.resolution) || loadLink(grid, int3(id)) < 0)
        return;
    for (uint corner = 0; corner < 8; ++corner) {
        const uint3 brick = (id + uint3(getCornerOffset(corner))) / brickWidth;
        grid.occupancy[(brick.z * grid.occupancyResolution + brick.y) * grid.occupancyResolution + brick.x] = 1;
    }
}
