#include "CUDA/Unique/Texture.h"

#include "CUDA/Checks.h"

nr::cuda::UniqueTexture::UniqueTexture(const float* pixels, int width, int height, cudaStream_t stream)
    : UniqueTexture(uploadFloat4(pixels, width, height, stream))
{
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadFloat4(
    const float* pixels,
    const int width,
    const int height,
    const cudaStream_t stream,
    const cudaTextureAddressMode addressMode,
    const cudaTextureFilterMode filterMode,
    const int normalizedCoords)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    NR_GPU_CHECK(cudaMallocArray(&result.array_, &format, width, height));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        result.array_, 0, 0, pixels, width * sizeof(float4),
        width * sizeof(float4), height, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = addressMode;
    description.addressMode[1] = addressMode;
    description.filterMode = filterMode;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = normalizedCoords;
    description.mipmapFilterMode = filterMode;

    NR_GPU_CHECK(cudaCreateTextureObject(&result.object_, &resource, &description, nullptr));
    cudaTextureDesc nearest = description;
    nearest.filterMode = cudaFilterModePoint;
    nearest.mipmapFilterMode = cudaFilterModePoint;
    NR_GPU_CHECK(cudaCreateTextureObject(&result.nearestObject_, &resource, &nearest, nullptr));
    return result;
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadNormalizedUInt8x4(
    const uint8_t* pixels,
    const int width,
    const int height,
    const cudaStream_t stream,
    const bool srgb)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    NR_GPU_CHECK(cudaMallocArray(&result.array_, &format, width, height));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        result.array_, 0, 0, pixels, width * sizeof(uchar4),
        width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeWrap;
    description.addressMode[1] = cudaAddressModeWrap;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeNormalizedFloat;
    description.sRGB = srgb ? 1 : 0;
    description.normalizedCoords = 1;
    description.mipmapFilterMode = cudaFilterModeLinear;

    NR_GPU_CHECK(cudaCreateTextureObject(
        &result.object_, &resource, &description, nullptr));
    cudaTextureDesc nearest = description;
    nearest.filterMode = cudaFilterModePoint;
    nearest.mipmapFilterMode = cudaFilterModePoint;
    NR_GPU_CHECK(cudaCreateTextureObject(&result.nearestObject_, &resource, &nearest, nullptr));
    return result;
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadHalf4(
    const uint16_t* pixels,
    const int width,
    const int height,
    const cudaStream_t stream)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDescHalf4();
    NR_GPU_CHECK(cudaMallocArray(&result.array_, &format, width, height));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        result.array_, 0, 0, pixels, width * 4 * sizeof(uint16_t),
        width * 4 * sizeof(uint16_t), height, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeWrap;
    description.addressMode[1] = cudaAddressModeWrap;
    description.filterMode = cudaFilterModeLinear;
    // Half channels are converted to float by tex2D<float4> while preserving
    // their full HDR range; normalized integer read mode is not applicable.
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 1;
    description.mipmapFilterMode = cudaFilterModeLinear;

    NR_GPU_CHECK(cudaCreateTextureObject(
        &result.object_, &resource, &description, nullptr));
    cudaTextureDesc nearest = description;
    nearest.filterMode = cudaFilterModePoint;
    nearest.mipmapFilterMode = cudaFilterModePoint;
    NR_GPU_CHECK(cudaCreateTextureObject(&result.nearestObject_, &resource, &nearest, nullptr));
    return result;
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
    const uint16_t* data,
    const int width,
    const int height,
    const cudaStream_t stream)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uint16_t>();
    NR_GPU_CHECK(cudaMallocArray(&result.array_, &format, width, height));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        result.array_, 0, 0, data, width * sizeof(uint16_t),
        width * sizeof(uint16_t), height, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeNormalizedFloat;
    description.normalizedCoords = 0;

    NR_GPU_CHECK(cudaCreateTextureObject(&result.object_, &resource, &description, nullptr));
    return result;
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut3D(
    const uint16_t* data,
    const int width,
    const int height,
    const int depth,
    const cudaStream_t stream)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uint16_t>();
    const cudaExtent extent = make_cudaExtent(width, height, depth);
    NR_GPU_CHECK(cudaMalloc3DArray(&result.array_, &format, extent));

    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<uint16_t*>(data), width * sizeof(uint16_t), width, height);
    copy.dstArray = result.array_;
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    NR_GPU_CHECK(cudaMemcpy3DAsync(&copy, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.addressMode[2] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeNormalizedFloat;
    description.normalizedCoords = 0;

    NR_GPU_CHECK(cudaCreateTextureObject(&result.object_, &resource, &description, nullptr));
    return result;
}

nr::cuda::UniqueTexture nr::cuda::UniqueTexture::uploadFloat4Lut3D(
    const float4* data,
    const int width,
    const int height,
    const int depth,
    const cudaStream_t stream)
{
    UniqueTexture result;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    const cudaExtent extent = make_cudaExtent(width, height, depth);
    NR_GPU_CHECK(cudaMalloc3DArray(&result.array_, &format, extent));

    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<float4*>(data), width * sizeof(float4), width, height);
    copy.dstArray = result.array_;
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    NR_GPU_CHECK(cudaMemcpy3DAsync(&copy, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = result.array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.addressMode[2] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;

    NR_GPU_CHECK(cudaCreateTextureObject(&result.object_, &resource, &description, nullptr));
    return result;
}
