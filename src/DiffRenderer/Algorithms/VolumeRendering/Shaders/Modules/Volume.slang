struct Grid {
    Ptr<float> values;
    AtomicPtr<float> gradients;
    uint size;
};

bool isInside(Grid grid, int3 voxel) {
    return all(voxel >= 0) && all(voxel < int(grid.size));
}

uint getIndex(Grid grid, int3 voxel) {
    return (uint(voxel.z) * grid.size + uint(voxel.y)) * grid.size + uint(voxel.x);
}

[Differentiable]
[BackwardDerivative(loadVoxelBackward)]
float loadVoxel(no_diff Grid grid, no_diff int3 voxel) {
    if (!isInside(grid, voxel))
        return 0.0;
    return grid.values[getIndex(grid, voxel)];
}

void loadVoxelBackward(Grid grid, int3 voxel, float gradient) {
    if (isNull(grid.gradients) || !isInside(grid, voxel))
        return;
    atomicAdd(grid.gradients + getIndex(grid, voxel), gradient);
}

[Differentiable]
float lerpEdge(float a, float b, float t) {
    return a + (b - a) * t;
}

[Differentiable]
float sampleGrid(no_diff Grid grid, float3 coordinate) {
    const float3 base = no_diff floor(detach(coordinate));
    const float3 fraction = coordinate - base;
    const int3 voxel = int3(base);
    const float x00 = lerpEdge(loadVoxel(grid, voxel), loadVoxel(grid, voxel + int3(1, 0, 0)), fraction.x);
    const float x10 = lerpEdge(loadVoxel(grid, voxel + int3(0, 1, 0)), loadVoxel(grid, voxel + int3(1, 1, 0)), fraction.x);
    const float x01 = lerpEdge(loadVoxel(grid, voxel + int3(0, 0, 1)), loadVoxel(grid, voxel + int3(1, 0, 1)), fraction.x);
    const float x11 = lerpEdge(loadVoxel(grid, voxel + int3(0, 1, 1)), loadVoxel(grid, voxel + int3(1, 1, 1)), fraction.x);
    return lerpEdge(lerpEdge(x00, x10, fraction.y), lerpEdge(x01, x11, fraction.y), fraction.z);
}

[Differentiable]
float3 toVoxel(float3 world, no_diff uint size) {
    return (world + 0.5) * float(size) - 0.5;
}
