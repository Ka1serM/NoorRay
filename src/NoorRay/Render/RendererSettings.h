#pragma once

#include <cstdint>

#include "NoorRay/Render/Raster/ToneMap.h"
#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"

namespace NoorRay::Render {

enum class PrimaryVisibility : std::uint8_t { Raster, Traced };
enum class Shading : std::uint8_t { Raster, PathTraced };
enum class Reuse : std::uint8_t { Off, Di, DiPt };
enum class Denoiser : std::uint8_t { Off, Relax };

struct RendererSettings {
    PrimaryVisibility primaryVisibility = PrimaryVisibility::Traced;
    Shading shading = Shading::PathTraced;
    Reuse reuse = Reuse::DiPt;
    Denoiser denoiser = Denoiser::Off;
    bool accumulate = true;
    std::uint32_t maxSamples = 0;
    std::uint32_t temporalMCap = 20;
    WavefrontSettings wavefront;
    DisplaySettings display;

    static RendererSettings viewport();
    static RendererSettings pathTraced();

    RendererSettings validated(bool cameraNeedsTracedPrimary) const;
    WavefrontSettings getWavefrontSettings() const;
    //! The display and the sample limit do not change the accumulated image.
    bool changesImage(const RendererSettings& previous) const;
    bool operator==(const RendererSettings&) const = default;
};

}
