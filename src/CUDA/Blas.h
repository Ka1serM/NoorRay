#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

class Blas
{
public:
    Blas() = default;
    ~Blas() noexcept;

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    Blas(Blas&& other) noexcept;
    Blas& operator=(Blas&& other) noexcept;

    void build(
        OptixDeviceContext context,
        cudaStream_t stream,
        const void* vertices,
        uint32_t vertexCount,
        uint32_t vertexStride,
        const uint32_t* indices,
        uint32_t triangleCount);
    void destroy(cudaStream_t stream) noexcept;

    OptixTraversableHandle getTraversable() const { return handle; }

private:
    void destroy() noexcept;

    OptixTraversableHandle handle{};
    CUdeviceptr buffer{};
};
