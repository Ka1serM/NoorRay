#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "Backend/CUDA/Unique/AsyncDeviceBuffer.h"
#include "Backend/CUDA/Unique/ManagedBuffer.h"
#include "Backend/CUDA/rstd/Vector.h"
#include "Backend/OptiX/Acceleration/GaussianProxyBlas.h"
#include "Backend/OptiX/ABI/Types.h"

#include "Scene/GpuInstance.h"
#include "Scene/Handle.h"

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
        nr::rstd::vector<GpuInstance>& instancesOut,
        OptixTraversableHandle lightBlas = 0,
        uint32_t lightPrimitiveCount = 0,
        OptixTraversableHandle meshLightBlas = 0,
        uint32_t meshLightPrimitiveCount = 0);
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
        nr::rstd::vector<GpuInstance>& instancesOut,
        OptixTraversableHandle lightBlas,
        uint32_t lightPrimitiveCount,
        OptixTraversableHandle meshLightBlas,
        uint32_t meshLightPrimitiveCount);

    void updateMeshInstanceInPlace(
        const Scene& scene,
        const std::vector<std::shared_ptr<MeshInstance>>& meshInstances,
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

    struct GaussianInstanceAccel
    {
        SceneObjectHandle objectHandle;
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
    OptixTraversableHandle lightBlas{};
    uint32_t lightPrimitiveCount{};
    OptixTraversableHandle meshLightBlas{};
    uint32_t meshLightPrimitiveCount{};
    std::vector<GaussianInstanceAccel> gaussianInstanceAccels;
};
