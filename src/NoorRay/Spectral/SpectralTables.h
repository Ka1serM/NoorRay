#pragma once

#include <expected>
#include <span>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Shaders/Spectral/SpectralRecords.h"

namespace NoorRay::Spectral {

//! The basis spectra and the film response, both in linear sRGB with a Bradford adaptation from illuminant E to D65, so that the constant spectrum 1 is white.
class SpectralTables {
public:
    static Core::float3 basis(float lambda);
    static Core::float3 film(float lambda);
    static std::vector<Shaders::SpectralEntry> buildEntries();
    static std::vector<Shaders::ConductorSample> buildConductors();

    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);

    NoorRHI::Ptr<Shaders::SpectralEntry> getEntries() const { return entries.getPtr(); }
    NoorRHI::Ptr<Shaders::ConductorSample> getConductors() const { return conductors.getPtr(); }

private:
    NoorRHI::Buffer<Shaders::SpectralEntry> entries;
    NoorRHI::Buffer<Shaders::ConductorSample> conductors;
};

}
