#pragma once

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

class TriangleBlas
{
public:
    TriangleBlas() = default;
    ~TriangleBlas();

    TriangleBlas(const TriangleBlas&) = delete;
    TriangleBlas& operator=(const TriangleBlas&) = delete;

    void build(OptixDeviceContext context, cudaStream_t stream, const std::vector<AccelMeshInput>& meshes);
    void destroy(cudaStream_t stream) noexcept;

    const std::vector<OptixTraversableHandle>& getHandles() const { return handles; }

private:
    std::vector<CUdeviceptr> buffers;
    std::vector<OptixTraversableHandle> handles;
};
