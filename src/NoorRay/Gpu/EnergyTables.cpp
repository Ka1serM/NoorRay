#include "NoorRay/Gpu/EnergyTables.h"

#include <array>
#include <span>

namespace NoorRay::Gpu {

namespace {

constexpr std::uint16_t ggxDirectional[] = {
#include "NoorRay/Gpu/EnergyLut/GgxDirectionalAlbedo.inc"
};
constexpr std::uint16_t ggxAverage[] = {
#include "NoorRay/Gpu/EnergyLut/GgxAverageAlbedo.inc"
};
constexpr std::uint16_t dielectricDirectional[] = {
#include "NoorRay/Gpu/EnergyLut/DielectricDirectionalAlbedo.inc"
};
constexpr std::uint16_t dielectricAverage[] = {
#include "NoorRay/Gpu/EnergyLut/DielectricAverageAlbedo.inc"
};
constexpr std::uint16_t glassDirectional[] = {
#include "NoorRay/Gpu/EnergyLut/GlassDirectionalAlbedo.inc"
};
constexpr std::uint16_t glassAverage[] = {
#include "NoorRay/Gpu/EnergyLut/GlassAverageAlbedo.inc"
};

static_assert(std::size(ggxDirectional) == 32 * 32 && std::size(ggxAverage) == 32);
static_assert(std::size(dielectricDirectional) == 16 * 16 * 16 && std::size(dielectricAverage) == 16 * 16);
static_assert(std::size(glassDirectional) == 16 * 16 * 32 && std::size(glassAverage) == 16 * 32);

}

std::vector<std::uint16_t> EnergyTables::buildValues() {
    std::vector<std::uint16_t> packed;
    for (const std::span<const std::uint16_t> table :
         std::array<std::span<const std::uint16_t>, 6>{ggxDirectional, ggxAverage, dielectricDirectional, dielectricAverage, glassDirectional, glassAverage})
        packed.insert(packed.end(), table.begin(), table.end());
    return packed;
}

std::expected<void, Core::Error> EnergyTables::open(NoorRHI::Device& device) {
    const std::vector<std::uint16_t> packed = buildValues();
    std::vector<std::uint32_t> words((packed.size() + 1) / 2, 0);
    for (std::size_t i = 0; i < packed.size(); ++i)
        words[i / 2] |= std::uint32_t{packed[i]} << (16 * (i % 2));
    if (auto opened = values.open(device, words.size()); !opened)
        return Core::fail(std::move(opened.error()));
    return Core::check(values.upload(words));
}

}
