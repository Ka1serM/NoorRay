#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "Backend/CUDA/Unique/AsyncDeviceBuffer.h"

class Blas
{
public:
    Blas() = default;
    ~Blas() noexcept = default;

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) noexcept;
    Blas& operator=(Blas&& other) noexcept;

    // These material arguments are retained for the mesh-build API, but are
    // not routed through OptiX SBT records. Geometry.h reads Face's
    // materialIndex and selects the corresponding global material directly.
    void build(
        OptixDeviceContext context,
        cudaStream_t stream,
        const void* vertices,
        uint32_t vertexCount,
        uint32_t vertexStride,
        const uint32_t* indices,
        uint32_t triangleCount,
        const int* materialIndices = nullptr,
        uint32_t materialIndexStride = 0,
        uint32_t materialSlotCount = 1);
    // Refit (update) an existing BLAS after vertex data changes but topology
    // stays the same. Material selection is independent of the BLAS/SBT.
    void refit(
        OptixDeviceContext context,
        cudaStream_t stream,
        const void* vertices,
        uint32_t vertexCount,
        uint32_t vertexStride,
        const uint32_t* indices,
        uint32_t triangleCount,
        const int* materialIndices = nullptr,
        uint32_t materialIndexStride = 0,
        uint32_t materialSlotCount = 1);
    void reset() noexcept;

    OptixTraversableHandle getTraversable() const { return handle; }
    bool isValid() const { return handle != 0; }

private:
    OptixTraversableHandle handle{};
    nr::cuda::UniqueAsyncDeviceBuffer buffer;
    uint32_t vertexCount_{};
    uint32_t triangleCount_{};
    uint32_t vertexStride_{};
    uint32_t outputSizeInBytes_{};
};
