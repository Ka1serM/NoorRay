#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"

class CudaTexture
{
public:
    CudaTexture() = default;

    CudaTexture(cudaArray_t array, cudaTextureObject_t object) noexcept
        : array_(array), object_(object) {}

    ~CudaTexture() noexcept { destroy(); }

    CudaTexture(const CudaTexture&) = delete;
    CudaTexture& operator=(const CudaTexture&) = delete;

    CudaTexture(CudaTexture&& other) noexcept
        : array_(other.array_), object_(other.object_)
    {
        other.array_ = nullptr;
        other.object_ = 0;
    }

    CudaTexture& operator=(CudaTexture&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            array_ = other.array_;
            object_ = other.object_;
            other.array_ = nullptr;
            other.object_ = 0;
        }
        return *this;
    }

    CudaTexture(const float* pixels, int width, int height, cudaStream_t stream);

    void destroy() noexcept
    {
        if (object_ != 0)
            cudaDestroyTextureObject(object_);
        if (array_ != nullptr)
            cudaFreeArray(array_);
        array_ = nullptr;
        object_ = 0;
    }

    cudaArray_t getArray() const noexcept { return array_; }
    cudaTextureObject_t getObject() const noexcept { return object_; }

#if defined(NR_GPU_CODE)
    NR_GPU glm::vec4 sample(const glm::vec2 uv) const
    {
        const float4 value = tex2D<float4>(object_, uv.x, uv.y);
        return {value.x, value.y, value.z, value.w};
    }
#endif

private:
    cudaArray_t array_{};
    cudaTextureObject_t object_{};
};
