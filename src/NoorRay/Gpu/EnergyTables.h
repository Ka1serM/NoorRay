#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Gpu {

//! The Kulla-Conty tables of Bsdf/EnergyLut.slang as two unorm16 per word: GGX, dielectric and glass, each directional then average.
class EnergyTables {
public:
    static std::vector<std::uint16_t> buildValues();
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);
    NoorRHI::Ptr<std::uint32_t> getValues() const { return values.getPtr(); }

private:
    NoorRHI::Buffer<std::uint32_t> values;
};

}
