#include "Shading/OpenPbrEnergy.h"

namespace nr::openpbr
{
void uploadEnergyLuts(EnergyLutStorage& storage, EnergyLutTextures& textures, const cudaStream_t stream)
{
    storage.reset();
    constexpr int size = EnergyTableSize;
    storage.opaqueEnergy = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut3D(
        OpaqueDielectricEnergyComplement, size, size, size, stream);
    storage.opaqueAverageEnergy = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        OpaqueDielectricAverageEnergyComplement, size, size, stream);
    storage.idealEnergy = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut3D(
        IdealDielectricEnergyComplement, size, size, size, stream);
    storage.idealAverageEnergy = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        IdealDielectricAverageEnergyComplement, size, size, stream);
    storage.idealReflectionRatio = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        IdealDielectricReflectionRatio, size, size, stream);
    textures = getEnergyLutTextures(storage);
}

EnergyLutTextures getEnergyLutTextures(const EnergyLutStorage& storage) noexcept
{
    return {
        storage.opaqueEnergy.getObject(),
        storage.opaqueAverageEnergy.getObject(),
        storage.idealEnergy.getObject(),
        storage.idealAverageEnergy.getObject(),
        storage.idealReflectionRatio.getObject(),
    };
}
}
