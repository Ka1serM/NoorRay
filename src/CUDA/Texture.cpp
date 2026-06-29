#include "CUDA/Texture.h"

#include "CUDA/Checks.h"

CudaTexture::CudaTexture(const float* pixels, int width, int height, cudaStream_t stream)
{

    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    NR_GPU_CHECK(cudaMallocArray(&array_, &format, width, height));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        array_, 0, 0, pixels, width * sizeof(float4),
        width * sizeof(float4), height, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeWrap;
    description.addressMode[1] = cudaAddressModeWrap;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 1;
    description.mipmapFilterMode = cudaFilterModeLinear;

    NR_GPU_CHECK(cudaCreateTextureObject(&object_, &resource, &description, nullptr));
}
