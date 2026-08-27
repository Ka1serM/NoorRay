#include "ShadingTables.h"

#include "Materials/Shading/EnergyLut/EnergyLutConfig.h"
#include "Materials/Shading/Spectrum.h"

extern const int sRGBToSpectrumTable_Res;
extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

namespace nr::vulkan
{
namespace
{
// The same generated initializers are shared, so the two
// backends cannot drift.
constexpr std::uint16_t GgxDirectionalAlbedo[
    nr::shading::energy_lut::GgxCosineSize
    * nr::shading::energy_lut::GgxRoughnessSize] = {
#include "Materials/Shading/EnergyLut/Generated/ggx_directional_albedo.h"
};
constexpr std::uint16_t GgxAverageAlbedo[
    nr::shading::energy_lut::GgxRoughnessSize] = {
#include "Materials/Shading/EnergyLut/Generated/ggx_average_albedo.h"
};
constexpr std::uint16_t DielectricDirectionalAlbedo[
    nr::shading::energy_lut::DielectricCosineSize
    * nr::shading::energy_lut::DielectricRoughnessSize
    * nr::shading::energy_lut::DielectricF0Size] = {
#include "Materials/Shading/EnergyLut/Generated/dielectric_directional_albedo.h"
};
constexpr std::uint16_t DielectricAverageAlbedo[
    nr::shading::energy_lut::DielectricRoughnessSize
    * nr::shading::energy_lut::DielectricF0Size] = {
#include "Materials/Shading/EnergyLut/Generated/dielectric_average_albedo.h"
};
constexpr std::uint16_t GlassDirectionalAlbedo[
    nr::shading::energy_lut::GlassCosineSize
    * nr::shading::energy_lut::GlassRoughnessSize
    * nr::shading::energy_lut::GlassIorSize] = {
#include "Materials/Shading/EnergyLut/Generated/glass_directional_albedo.h"
};
constexpr std::uint16_t GlassAverageAlbedo[
    nr::shading::energy_lut::GlassRoughnessSize
    * nr::shading::energy_lut::GlassIorSize] = {
#include "Materials/Shading/EnergyLut/Generated/glass_average_albedo.h"
};

template <std::size_t N>
void append(std::vector<std::uint16_t>& out, const std::uint16_t (&values)[N])
{
    out.insert(out.end(), values, values + N);
}
}

std::vector<std::uint16_t> packEnergyLutTables()
{
    std::vector<std::uint16_t> packed;
    packed.reserve(std::size(GgxDirectionalAlbedo) + std::size(GgxAverageAlbedo)
        + std::size(DielectricDirectionalAlbedo) + std::size(DielectricAverageAlbedo)
        + std::size(GlassDirectionalAlbedo) + std::size(GlassAverageAlbedo));
    append(packed, GgxDirectionalAlbedo);
    append(packed, GgxAverageAlbedo);
    append(packed, DielectricDirectionalAlbedo);
    append(packed, DielectricAverageAlbedo);
    append(packed, GlassDirectionalAlbedo);
    append(packed, GlassAverageAlbedo);
    // The shader reads two unorm16 elements per 32-bit word.
    if (packed.size() % 2u != 0u)
        packed.push_back(0u);
    return packed;
}

std::vector<float> packSpectralTables()
{
    std::vector<float> packed;
    constexpr std::size_t rgbScaleCount = 64u;
    constexpr std::size_t rgbCoefficientCount = 3u * 64u * 64u * 64u * 3u;
    packed.reserve(3u * NrCIESamples + NrD65Samples
        + rgbScaleCount + rgbCoefficientCount);
    packed.insert(packed.end(), NrCIE_X, NrCIE_X + NrCIESamples);
    packed.insert(packed.end(), NrCIE_Y, NrCIE_Y + NrCIESamples);
    packed.insert(packed.end(), NrCIE_Z, NrCIE_Z + NrCIESamples);
    packed.insert(packed.end(), NrD65, NrD65 + NrD65Samples);
    packed.insert(packed.end(), sRGBToSpectrumTable_Scale,
        sRGBToSpectrumTable_Scale + rgbScaleCount);
    const float* coefficients = &sRGBToSpectrumTable_Data[0][0][0][0][0];
    packed.insert(packed.end(), coefficients,
        coefficients + rgbCoefficientCount);
    return packed;
}

} // namespace nr::vulkan
