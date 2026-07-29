#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"

namespace nr::cuda
{

class UniqueTexture
{
public:
    UniqueTexture() = default;

    UniqueTexture(cudaArray_t array, cudaTextureObject_t object) noexcept
        : array_(array), object_(object) {}

    ~UniqueTexture() noexcept { reset(); }

    UniqueTexture(const UniqueTexture&) = delete;
    UniqueTexture& operator=(const UniqueTexture&) = delete;

    UniqueTexture(UniqueTexture&& other) noexcept
        : array_(other.array_), object_(other.object_), nearestObject_(other.nearestObject_)
    {
        other.array_ = nullptr;
        other.object_ = 0;
        other.nearestObject_ = 0;
    }

    UniqueTexture& operator=(UniqueTexture&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            array_ = other.array_;
            object_ = other.object_;
            nearestObject_ = other.nearestObject_;
            other.array_ = nullptr;
            other.object_ = 0;
            other.nearestObject_ = 0;
        }
        return *this;
    }

    UniqueTexture(const float* pixels, int width, int height, cudaStream_t stream);

    static UniqueTexture uploadFloat4(
        const float* pixels,
        int width,
        int height,
        cudaStream_t stream,
        cudaTextureAddressMode addressMode = cudaAddressModeWrap,
        cudaTextureFilterMode filterMode = cudaFilterModeLinear,
        int normalizedCoords = 1);
    static UniqueTexture uploadNormalizedUInt8x4(
        const uint8_t* pixels,
        int width,
        int height,
        cudaStream_t stream,
        bool srgb);
    static UniqueTexture uploadHalf4(
        const uint16_t* pixels,
        int width,
        int height,
        cudaStream_t stream);
    static UniqueTexture uploadNormalizedUInt16Lut2D(
        const uint16_t* data,
        int width,
        int height,
        cudaStream_t stream);
    static UniqueTexture uploadNormalizedUInt16Lut3D(
        const uint16_t* data,
        int width,
        int height,
        int depth,
        cudaStream_t stream);
    static UniqueTexture uploadFloat4Lut3D(
        const float4* data,
        int width,
        int height,
        int depth,
        cudaStream_t stream);

    void reset() noexcept
    {
        if (object_ != 0)
            cudaDestroyTextureObject(object_);
        if (nearestObject_ != 0)
            cudaDestroyTextureObject(nearestObject_);
        if (array_ != nullptr)
            cudaFreeArray(array_);
        array_ = nullptr;
        object_ = 0;
        nearestObject_ = 0;
    }

    NR_CPU_GPU cudaArray_t getArray() const noexcept { return array_; }
    NR_CPU_GPU cudaTextureObject_t getObject() const noexcept { return object_; }

#if defined(NR_GPU_CODE)
    NR_GPU glm::vec4 sample(const glm::vec2 uv, const bool closest = false) const
    {
        const float4 value = tex2D<float4>(closest ? nearestObject_ : object_, uv.x, uv.y);
        return {value.x, value.y, value.z, value.w};
    }
#endif

private:
    cudaArray_t array_{};
    cudaTextureObject_t object_{};
    cudaTextureObject_t nearestObject_{};
};

}
