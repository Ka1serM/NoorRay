#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

class GaussianProxyBlas
{
public:
    GaussianProxyBlas() = default;
    ~GaussianProxyBlas() noexcept;

    GaussianProxyBlas(const GaussianProxyBlas&) = delete;
    GaussianProxyBlas& operator=(const GaussianProxyBlas&) = delete;
    GaussianProxyBlas(GaussianProxyBlas&& other) noexcept;
    GaussianProxyBlas& operator=(GaussianProxyBlas&& other) noexcept;

    void build(OptixDeviceContext context, cudaStream_t stream);
    void destroy(cudaStream_t stream) noexcept;

    OptixTraversableHandle getTraversable() const { return handle; }
    bool isValid() const { return handle != 0; }

private:
    void destroy() noexcept;

    OptixTraversableHandle handle{};
    CUdeviceptr buffer{};
};
