#include "SurfaceGrid.slang"

float3 getNodeWorld(Grid grid, int3 node) {
    return grid.origin.xyz + float3(node) * grid.origin.w;
}

float loadNode(Arguments arguments, int3 node) {
    return arguments.sdf[getNodeIndex(arguments.grid, clamp(node, int3(0), int3(arguments.grid.size.xyz) - 1))];
}

bool isInterface(Arguments arguments, int3 node, float value) {
    [ForceUnroll]
    for (uint neighbour = 0; neighbour < 6; ++neighbour) {
        int3 offset = int3(0);
        offset[neighbour / 2] = (neighbour & 1) != 0 ? 1 : -1;
        if (isNode(arguments.grid, node + offset) && ((loadNode(arguments, node + offset) > 0.0) != (value > 0.0)))
            return true;
    }
    return false;
}

//! A node next to a sign change seeds the closest point of the zero level set by one Newton step along the central-difference gradient.
void seedNode(Arguments arguments, uint index) {
    const Grid grid = arguments.grid;
    const int3 node = getNodePosition(grid, index);
    const float value = arguments.sdf[index];
    if (!isInterface(arguments, node, value)) {
        arguments.nextSeeds[index] = float4(0.0);
        return;
    }
    float3 slope;
    [ForceUnroll]
    for (uint axis = 0; axis < 3; ++axis) {
        int3 offset = int3(0);
        offset[axis] = 1;
        slope[axis] = (loadNode(arguments, node + offset) - loadNode(arguments, node - offset)) / (2.0 * grid.origin.w);
    }
    const float squared = dot(slope, slope);
    const float3 position = getNodeWorld(grid, node);
    const float3 closest = squared > 1e-12 ? position - value * slope / squared : position;
    arguments.nextSeeds[index] = float4(closest, 1.0);
}

//! One pass of the 3D jump flooding of Rong and Tan 2006 over the 26 neighbours at the jump distance.
void floodNode(Arguments arguments, uint index) {
    const Grid grid = arguments.grid;
    const int3 node = getNodePosition(grid, index);
    const float3 position = getNodeWorld(grid, node);
    float4 best = arguments.seeds[index];
    float bestDistance = best.w != 0.0 ? length(best.xyz - position) : 1e30;
    const int jump = int(arguments.jump);
    for (uint neighbour = 0; neighbour < 27; ++neighbour) {
        const int3 other = node + (int3(int(neighbour % 3), int((neighbour / 3) % 3), int(neighbour / 9)) - 1) * jump;
        if (neighbour == 13 || !isNode(grid, other))
            continue;
        const float4 seed = arguments.seeds[getNodeIndex(grid, other)];
        if (seed.w == 0.0)
            continue;
        const float distance = length(seed.xyz - position);
        if (distance >= bestDistance)
            continue;
        best = seed;
        bestDistance = distance;
    }
    arguments.nextSeeds[index] = best;
}

//! The nodes next to the surface keep their values, so that the zero level set stays where the optimizer put it; the signs never change.
void applyDistance(Arguments arguments, uint index) {
    const Grid grid = arguments.grid;
    const int3 node = getNodePosition(grid, index);
    const float value = arguments.sdf[index];
    const float4 seed = arguments.seeds[index];
    if (seed.w == 0.0 || isInterface(arguments, node, value))
        return;
    const float distance = max(length(seed.xyz - getNodeWorld(grid, node)), 1e-6);
    arguments.sdf[index] = value > 0.0 ? distance : -distance;
}
