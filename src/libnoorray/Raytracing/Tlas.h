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
#include "Mesh/GaussianCutoff.h"
#include "Raytracing/GaussianProxyBlas.h"
#include "Raytracing/Types.h"

#include "Scene/GpuInstance.h"

class Scene;
class MeshInstance;
class GaussianInstance;

struct AccelInstanceInput
{
    std::array<float, 12> transform{};
    OptixTraversableHandle blasHandle{};
    uint32_t instanceId{};
    uint32_t sbtOffset{};
    uint8_t visibilityMask{FullVisibility};
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
        const std::vector<std::shared_ptr<MeshInstance>>& meshInstances,
        nr::rstd::vector<GpuInstance>& instancesOut,
        uint32_t index);
    void updateGaussianInstanceInPlace(
        const std::vector<std::shared_ptr<GaussianInstance>>& gaussianInstances,
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

    struct GaussianInstanceAccel
    {
        uint64_t objectId{};
        uint32_t gaussianCount{};
        uint32_t globalOffset{};
        nr::cuda::UniqueManagedBuffer instanceBuffer;
        nr::cuda::UniqueAsyncDeviceBuffer accelBuffer;
        OptixTraversableHandle handle{};
    };

    void buildGaussianInstanceAccel(
        OptixDeviceContext context,
        cudaStream_t stream,
        GaussianInstanceAccel& destination,
        const GaussianInstance& instance,
        uint32_t globalOffset);

    nr::cuda::UniqueManagedBuffer instanceBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer tlasBuffer;
    size_t tlasBufferSize{};
    uint32_t instanceCount{};
    OptixTraversableHandle tlasHandle{};

    std::unique_ptr<GaussianProxyBlas> proxyBlas;
    GaussianProxyType lastProxyType{GaussianProxyType::Octahedron};
    float lastCutoffSigma{GaussianCutoffSigma};
    uint32_t meshInstanceCount{};
    uint32_t gaussianInstanceCount{};
    std::vector<GaussianInstanceAccel> gaussianInstanceAccels;
};
