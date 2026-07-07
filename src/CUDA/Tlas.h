#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "CUDA/GaussianProxyBlas.h"
#include "CUDA/rstd/Vector.h"

#include "Scene/GpuInstance.h"

class Scene;

struct AccelInstanceInput
{
    std::array<float, 12> transform{};
    OptixTraversableHandle blasHandle{};
    uint32_t instanceId{};
};

class Tlas
{
public:
    Tlas() = default;
    ~Tlas() = default;

    Tlas(const Tlas&) = delete;
    Tlas& operator=(const Tlas&) = delete;

    void build(
        OptixDeviceContext context,
        cudaStream_t stream,
        const Scene& scene,
        nr::rstd::vector<GpuInstance>& instancesOut);
    void destroy(cudaStream_t stream) noexcept;

    OptixTraversableHandle getTraversable() const { return tlasHandle; }

private:
    std::vector<AccelInstanceInput> buildInstanceInputs(
        const Scene& scene,
        nr::rstd::vector<GpuInstance>& instancesOut) const;

    void buildInternal(
        OptixDeviceContext context,
        cudaStream_t stream,
        const std::vector<AccelInstanceInput>& instances,
        OptixBuildOperation operation);

    void buildGaussianInstances(
        OptixDeviceContext context,
        cudaStream_t stream,
        const Scene& scene,
        std::vector<AccelInstanceInput>& inputs);

    CUdeviceptr instanceBuffer{};
    CUdeviceptr tlasBuffer{};
    size_t tlasBufferSize{};
    uint32_t instanceCount{};
    OptixTraversableHandle tlasHandle{};

    GaussianProxyBlas* proxyBlas{};
    GaussianProxyType lastProxyType{GaussianProxyType::Tetrahedron};
    float lastCutoffSigma{3.0f};
    uint32_t meshInstanceCount{};
};
