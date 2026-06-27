#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

struct AccelMeshInput
{
    CUdeviceptr vertices{};
    uint32_t vertexCount{};
    uint32_t vertexStride{};
    CUdeviceptr indices{};
    uint32_t triangleCount{};
};

struct AccelInstanceInput
{
    std::array<float, 12> transform{};
    uint32_t meshIndex{};
    uint32_t instanceId{};
};

class AccelBuilder
{
public:
    AccelBuilder() = default;
    ~AccelBuilder();

    AccelBuilder(const AccelBuilder&) = delete;
    AccelBuilder& operator=(const AccelBuilder&) = delete;

    void initialize(OptixDeviceContext context, cudaStream_t stream);
    void rebuild(
        const std::vector<AccelMeshInput>& meshes,
        const std::vector<AccelInstanceInput>& instances);
    void updateInstances(const std::vector<AccelInstanceInput>& instances);
    void destroy() noexcept;

    OptixTraversableHandle getTraversable() const { return tlasHandle; }

private:
    OptixDeviceContext context{};
    cudaStream_t stream{};
    std::vector<CUdeviceptr> blasBuffers;
    std::vector<OptixTraversableHandle> blasHandles;
    CUdeviceptr instanceBuffer{};
    CUdeviceptr tlasBuffer{};
    size_t tlasBufferSize{};
    uint32_t instanceCount{};
    OptixTraversableHandle tlasHandle{};

    void buildTlas(const std::vector<AccelInstanceInput>& instances, OptixBuildOperation operation);
};
