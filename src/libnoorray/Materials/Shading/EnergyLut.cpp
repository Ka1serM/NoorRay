#include "Materials/Shading/EnergyLut.h"

nr::shading::energy_lut::Textures
nr::shading::energy_lut::Storage::upload(const cudaStream_t stream)
{
    ggxDirectional_ = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        data::GgxDirectionalAlbedo, GgxCosineSize, GgxRoughnessSize, stream);
    ggxAverage_ = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        data::GgxAverageAlbedo, GgxRoughnessSize, 1, stream);
    dielectricDirectional_ =
        nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut3D(
            data::DielectricDirectionalAlbedo,
            DielectricCosineSize, DielectricRoughnessSize,
            DielectricF0Size, stream);
    dielectricAverage_ =
        nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
            data::DielectricAverageAlbedo,
            DielectricRoughnessSize, DielectricF0Size, stream);
    glassDirectional_ =
        nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut3D(
            data::GlassDirectionalAlbedo,
            GlassCosineSize, GlassRoughnessSize, GlassIorSize, stream);
    glassAverage_ = nr::cuda::UniqueTexture::uploadNormalizedUInt16Lut2D(
        data::GlassAverageAlbedo,
        GlassRoughnessSize, GlassIorSize, stream);
    return textures();
}

nr::shading::energy_lut::Textures
nr::shading::energy_lut::Storage::textures() const noexcept
{
    return {
        ggxDirectional_.getObject(),
        ggxAverage_.getObject(),
        dielectricDirectional_.getObject(),
        dielectricAverage_.getObject(),
        glassDirectional_.getObject(),
        glassAverage_.getObject()};
}
