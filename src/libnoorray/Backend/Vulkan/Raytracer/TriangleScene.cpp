#include "TriangleScene.h"

#include <array>
TriangleScene::TriangleScene(gpu::Device& device)
    : device(device)
{
    const std::array<gpu::float3, 3> vertices{
        // Camera space looks down -Z, matching PerspectiveCamera and the
        // Vulkan raytracer shader's ray convention.
        gpu::float3{-1.0f, -1.0f, -2.0f},
        gpu::float3{ 1.0f, -1.0f, -2.0f},
        gpu::float3{ 0.0f,  1.0f, -2.0f},
    };
    const std::array<std::uint32_t, 3> indices{0, 1, 2};
    vertexBuffer = device.buffer<gpu::float3>(vertices.size());
    indexBuffer = device.buffer<std::uint32_t>(indices.size());
    device.upload(vertexBuffer, std::span<const gpu::float3>(vertices));
    device.upload(indexBuffer, std::span<const std::uint32_t>(indices));
    const gpu::TriangleGeometry triangle{vertexBuffer.ptr(), indexBuffer.ptr(), 1};
    bottomLevel = device.build_blas(std::span<const gpu::TriangleGeometry>(&triangle, 1));
    gpu::float4x4 transform{};
    transform.values[0][0] = 1.0f;
    transform.values[1][1] = 1.0f;
    transform.values[2][2] = 1.0f;
    transform.values[3][3] = 1.0f;
    const gpu::Instance instance{bottomLevel, transform};
    topLevel = device.build_tlas(std::span<const gpu::Instance>(&instance, 1));
}
