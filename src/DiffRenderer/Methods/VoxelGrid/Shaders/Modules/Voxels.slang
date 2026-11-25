#include "GridLayout.h"

typealias Grid = GridLayout::Grid;

static const uint shBasisCount = GridLayout::shBasisCount;
static const uint shCount = GridLayout::shCount;
static const uint brickWidth = GridLayout::brickWidth;

int3 getCornerOffset(uint corner) {
    return int3(corner & 1, (corner >> 1) & 1, corner >> 2);
}

bool isInside(uint resolution, int3 voxel) {
    return all(voxel >= 0) && all(voxel < int(resolution));
}

uint getVoxelIndex(uint resolution, int3 voxel) {
    return (uint(voxel.z) * resolution + uint(voxel.y)) * resolution + uint(voxel.x);
}

int loadLink(Grid grid, int3 voxel) {
    if (!isInside(grid.resolution, voxel))
        return -1;
    return grid.links[getVoxelIndex(grid.resolution, voxel)];
}

float3 toVoxel(uint resolution, float3 position) {
    return (position + 1.0) * 0.5 * float(resolution) - 0.5;
}

bool isOccupied(Grid grid, int3 base) {
    const int3 cell = base + 1;
    if (any(cell < 0) || any(cell > int(grid.resolution)))
        return false;
    const uint3 brick = uint3(cell) / brickWidth;
    return grid.occupancy[(brick.z * grid.occupancyResolution + brick.y) * grid.occupancyResolution + brick.x] != 0;
}

struct Corners {
    int links[8];
    float weights[8];
    bool empty;

    float getDensity(Grid grid) {
        float density = 0.0;
        for (uint corner = 0; corner < 8; ++corner)
            if (links[corner] >= 0)
                density += weights[corner] * grid.density[uint(links[corner])];
        return density;
    }

    void getSh(Grid grid, out float sh[shCount]) {
        for (uint index = 0; index < shCount; ++index)
            sh[index] = 0.0;
        for (uint corner = 0; corner < 8; ++corner) {
            if (links[corner] < 0)
                continue;
            const Ptr<float> values = grid.sh + uint(links[corner]) * shCount;
            for (uint index = 0; index < shCount; ++index)
                sh[index] += weights[corner] * values[index];
        }
    }
};

Corners getCorners(Grid grid, float3 voxel) {
    const float3 base = floor(voxel);
    const float3 fraction = voxel - base;
    Corners corners;
    corners.empty = true;
    for (uint corner = 0; corner < 8; ++corner) {
        const int3 offset = getCornerOffset(corner);
        const float3 weight = select(offset != 0, fraction, 1.0 - fraction);
        corners.links[corner] = loadLink(grid, int3(base) + offset);
        corners.weights[corner] = weight.x * weight.y * weight.z;
        corners.empty = corners.empty && corners.links[corner] < 0;
    }
    return corners;
}

void getShBasis(float3 direction, out float basis[shBasisCount]) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    basis[0] = 0.28209479177387814;
    basis[1] = -0.4886025119029199 * y;
    basis[2] = 0.4886025119029199 * z;
    basis[3] = -0.4886025119029199 * x;
    basis[4] = 1.0925484305920792 * x * y;
    basis[5] = -1.0925484305920792 * y * z;
    basis[6] = 0.31539156525252005 * (2.0 * z * z - x * x - y * y);
    basis[7] = -1.0925484305920792 * x * z;
    basis[8] = 0.5462742152960396 * (x * x - y * y);
}

float3 getRawColor(float sh[shCount], float basis[shBasisCount]) {
    float3 color = float3(0.5);
    for (uint index = 0; index < shBasisCount; ++index)
        color += basis[index] * float3(sh[index], sh[shBasisCount + index], sh[2 * shBasisCount + index]);
    return color;
}
