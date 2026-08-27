#include "Materials/Shading/EnergyLut.h"

nr::shading::energy_lut::Textures
nr::shading::energy_lut::Storage::upload() noexcept
{
    return textures();
}

nr::shading::energy_lut::Textures
nr::shading::energy_lut::Storage::textures() const noexcept
{
    return {
        data::GgxDirectionalAlbedo,
        data::GgxAverageAlbedo,
        data::DielectricDirectionalAlbedo,
        data::DielectricAverageAlbedo,
        data::GlassDirectionalAlbedo,
        data::GlassAverageAlbedo};
}
