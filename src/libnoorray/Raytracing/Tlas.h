#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "CUDA/Unique/AsyncDeviceBuffer.h"
#include "CUDA/Unique/ManagedBuffer.h"
#include "CUDA/rstd/Vector.h"
#include "Raytracing/GaussianProxyBlas.h"

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
    ~Tlas() noexcept = default;

    Tlas(const Tlas&) = delete;
    Tlas& operator=(const Tlas&) = delete;

    void build(
        OptixDeviceContext context,
        cudaStream_t stream,
        const Scene& scene,
        nr::rstd::vector<GpuInstance>& instancesOut);
    void reset() noexcept;

    OptixTraversableHandle getTraversable() const { return tlasHandle; }

private:
    // Fast path for pure transform edits: writes only the changed instances
    // directly into the managed instance buffer and does a cheap refit.
    // Returns false (nothing done) if a full rebuild is required instead.
    bool tryPartialUpdate(
        OptixDeviceContext context,
        cudaStream_t stream,
        const Scene& scene,
        nr::rstd::vector<GpuInstance>& instancesOut);

    void updateMeshInstanceInPlace(
        const Scene& scene,
        nr::rstd::vector<GpuInstance>& instancesOut,
        uint32_t index);

    OptixInstance* instancePtr() const { return reinterpret_cast<OptixInstance*>(instanceBuffer.devicePtr()); }

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

    nr::cuda::UniqueManagedBuffer instanceBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer tlasBuffer;
    size_t tlasBufferSize{};
    uint32_t instanceCount{};
    OptixTraversableHandle tlasHandle{};

    std::unique_ptr<GaussianProxyBlas> proxyBlas;
    GaussianProxyType lastProxyType{GaussianProxyType::TriangularBipyramid};
    float lastCutoffSigma{3.0f};
    uint32_t meshInstanceCount{};
};
