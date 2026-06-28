#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

struct AccelInstanceInput
{
    std::array<float, 12> transform{};
    OptixTraversableHandle blasHandle{};
    uint32_t instanceId{};
};

class TlasBuilder
{
public:
    TlasBuilder() = default;
    ~TlasBuilder();

    TlasBuilder(const TlasBuilder&) = delete;
    TlasBuilder& operator=(const TlasBuilder&) = delete;

    void build(
        OptixDeviceContext context,
        cudaStream_t stream,
        const std::vector<AccelInstanceInput>& instances);
    void update(
        OptixDeviceContext context,
        cudaStream_t stream,
        const std::vector<AccelInstanceInput>& instances);
    void destroy(cudaStream_t stream) noexcept;

    OptixTraversableHandle getTraversable() const { return tlasHandle; }

private:
    void buildInternal(
        OptixDeviceContext context,
        cudaStream_t stream,
        const std::vector<AccelInstanceInput>& instances,
        OptixBuildOperation operation);

    CUdeviceptr instanceBuffer{};
    CUdeviceptr tlasBuffer{};
    size_t tlasBufferSize{};
    uint32_t instanceCount{};
    OptixTraversableHandle tlasHandle{};
};
