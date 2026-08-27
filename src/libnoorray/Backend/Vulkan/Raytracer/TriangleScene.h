#pragma once

#include <gpu/gpu.hpp>

// Small scene used by the raytracer smoke path. Its geometry and acceleration
// structures use the same gpu::Device path as the full native scene.
class TriangleScene
{
public:
    explicit TriangleScene(gpu::Device& device);
    ~TriangleScene() = default;

    TriangleScene(const TriangleScene&) = delete;
    TriangleScene& operator=(const TriangleScene&) = delete;

    gpu::AccelerationStructureHandle topLevelHandle() const { return topLevel.handle(); }

private:
    gpu::Device& device;
    gpu::Buffer<gpu::float3> vertexBuffer;
    gpu::Buffer<std::uint32_t> indexBuffer;
    gpu::AccelerationStructure bottomLevel;
    gpu::AccelerationStructure topLevel;
};
