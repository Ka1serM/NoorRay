#pragma once

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include "Backend/CUDA/Unique/AsyncDeviceBuffer.h"

// A hardware-traversed custom-primitive light GAS. Each primitive is a
// conservative AABB for one light category; the OptiX intersection program
// performs the exact intersection. Analytic and mesh-light GASes are placed in
// separate TLAS instances so regular path rays can avoid duplicate mesh hits.
class AnalyticLightBlas
{
public:
    AnalyticLightBlas() = default;
    ~AnalyticLightBlas() noexcept = default;

    AnalyticLightBlas(const AnalyticLightBlas&) = delete;
    AnalyticLightBlas& operator=(const AnalyticLightBlas&) = delete;

    void build(OptixDeviceContext context, cudaStream_t stream,
        const std::vector<OptixAabb>& aabbs);
    void reset() noexcept;

    OptixTraversableHandle getTraversable() const { return handle; }
    bool isValid() const { return handle != 0; }
    uint32_t primitiveCount() const { return primitiveCount_; }

private:
    OptixTraversableHandle handle{};
    nr::cuda::UniqueAsyncDeviceBuffer buffer;
    uint32_t primitiveCount_{};
};
