#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "CUDA/Unique/AsyncDeviceBuffer.h"

inline constexpr float GaussianCutoffSigma = 3.0f;

enum class GaussianProxyType : int
{
    Icosphere,
    Octahedron,
    Icosahedron,
    IcosphereLevel2,
};

class GaussianProxyBlas
{
public:
    GaussianProxyBlas() = default;
    ~GaussianProxyBlas() noexcept;

    GaussianProxyBlas(const GaussianProxyBlas&) = delete;
    GaussianProxyBlas& operator=(const GaussianProxyBlas&) = delete;
    GaussianProxyBlas(GaussianProxyBlas&& other) noexcept;
    GaussianProxyBlas& operator=(GaussianProxyBlas&& other) noexcept;

    void build(OptixDeviceContext context, cudaStream_t stream,
        GaussianProxyType type, float cutoffSigma);
    void reset() noexcept;

    OptixTraversableHandle getTraversable() const { return handle; }
    bool isValid() const { return handle != 0; }

private:
    OptixTraversableHandle handle{};
    nr::cuda::UniqueAsyncDeviceBuffer buffer;
};
