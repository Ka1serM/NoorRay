#include "Mesh/OpenPbrEnergy.h"

#include "CUDA/Checks.h"

namespace
{
// Uploads a 32^3 uint16 table into a 3D CUDA array and wraps it in a
// linear-filtered, clamp-to-edge, normalized-float texture object.
cudaTextureObject_t upload3D(cudaArray_t& array, const uint16_t* data, const cudaStream_t stream)
{
    const int size = nr::openpbr::EnergyTableSize;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uint16_t>();
    const cudaExtent extent = make_cudaExtent(size, size, size);
    NR_GPU_CHECK(cudaMalloc3DArray(&array, &format, extent));

    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<uint16_t*>(data), size * sizeof(uint16_t), size, size);
    copy.dstArray = array;
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    NR_GPU_CHECK(cudaMemcpy3DAsync(&copy, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.addressMode[2] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeNormalizedFloat;
    description.normalizedCoords = 0;

    cudaTextureObject_t texture{};
    NR_GPU_CHECK(cudaCreateTextureObject(&texture, &resource, &description, nullptr));
    return texture;
}

// Same as upload3D but for a 32x32 uint16 table.
cudaTextureObject_t upload2D(cudaArray_t& array, const uint16_t* data, const cudaStream_t stream)
{
    const int size = nr::openpbr::EnergyTableSize;
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uint16_t>();
    NR_GPU_CHECK(cudaMallocArray(&array, &format, size, size));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        array, 0, 0, data, size * sizeof(uint16_t),
        size * sizeof(uint16_t), size, cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeNormalizedFloat;
    description.normalizedCoords = 0;

    cudaTextureObject_t texture{};
    NR_GPU_CHECK(cudaCreateTextureObject(&texture, &resource, &description, nullptr));
    return texture;
}
}

namespace nr::openpbr
{
void uploadEnergyLuts(EnergyLutStorage& storage, EnergyLutTextures& textures, const cudaStream_t stream)
{
    textures.opaqueEnergy = upload3D(
        storage.opaqueEnergyArray, OpaqueDielectricEnergyComplement, stream);
    textures.opaqueAverageEnergy = upload2D(
        storage.opaqueAverageEnergyArray, OpaqueDielectricAverageEnergyComplement, stream);
    textures.idealEnergy = upload3D(
        storage.idealEnergyArray, IdealDielectricEnergyComplement, stream);
    textures.idealAverageEnergy = upload2D(
        storage.idealAverageEnergyArray, IdealDielectricAverageEnergyComplement, stream);
    textures.idealReflectionRatio = upload2D(
        storage.idealReflectionRatioArray, IdealDielectricReflectionRatio, stream);
}

void destroyEnergyLuts(EnergyLutStorage& storage, EnergyLutTextures& textures) noexcept
{
    if (textures.opaqueEnergy) cudaDestroyTextureObject(textures.opaqueEnergy);
    if (textures.opaqueAverageEnergy) cudaDestroyTextureObject(textures.opaqueAverageEnergy);
    if (textures.idealEnergy) cudaDestroyTextureObject(textures.idealEnergy);
    if (textures.idealAverageEnergy) cudaDestroyTextureObject(textures.idealAverageEnergy);
    if (textures.idealReflectionRatio) cudaDestroyTextureObject(textures.idealReflectionRatio);
    textures = {};

    if (storage.opaqueEnergyArray) cudaFreeArray(storage.opaqueEnergyArray);
    if (storage.opaqueAverageEnergyArray) cudaFreeArray(storage.opaqueAverageEnergyArray);
    if (storage.idealEnergyArray) cudaFreeArray(storage.idealEnergyArray);
    if (storage.idealAverageEnergyArray) cudaFreeArray(storage.idealAverageEnergyArray);
    if (storage.idealReflectionRatioArray) cudaFreeArray(storage.idealReflectionRatioArray);
    storage = {};
}
}
