#pragma once

#include <cstdint>

#include "NoorRay/Render/Raster/ToneMap.h"
#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"

namespace NoorRay::Render {

enum class PrimaryVisibility : std::uint8_t { Raster, Traced };
enum class Shading : std::uint8_t { Raster, PathTraced };
//! OpenDLSS-RR denoises and upscales in one pass, so it takes the place of the upscaler too; a build without its plugin has no OpenDLSS-RR.
enum class Denoiser : std::uint8_t { Off, Nrd, OpenDlssRr };
enum class Upscaler : std::uint8_t { Off, Fsr };

struct RendererSettings {
    PrimaryVisibility primaryVisibility = PrimaryVisibility::Traced;
    Shading shading = Shading::PathTraced;
    //! Accumulation never resamples paths: its frames would share samples and lose the Sobol sequence, so path tracing converges faster.
    Reuse reuse = Reuse::Off;
    Denoiser denoiser = Denoiser::Nrd;
    //! FSR is temporal, so it runs only without accumulation and with temporal on.
    Upscaler upscaler = Upscaler::Off;
    //! The render resolution over the display resolution when FSR or DLSS-RR upscales.
    float renderScale = 0.5f;
    float sharpness = 0.2f;
    //! With accumulation, the denoised frame blends into the accumulated mean with weight samples / (samples + this).
    float denoiseBlend = 16.0f;
    bool accumulate = true;
    //! Without accumulation, a view that stays still for a moment switches to the accumulation and converges; only final renders do this.
    bool convergesWhenStill = false;
    //! False keeps each frame apart: no temporal ReSTIR, no denoiser history and no FSR, so nothing smears.
    bool temporal = true;
    //! The SHaRC radiance cache ends ReSTIR paths early in frames that are not accumulated; it is biased, so accumulation never uses it.
    bool radianceCache = true;
    std::uint32_t maxSamples = 0;
    std::uint32_t temporalMCap = 20;
    WavefrontSettings wavefront;
    DisplaySettings display;

    static RendererSettings viewport();
    static RendererSettings pathTraced();
    //! Path tracing at a lower resolution with RELAX and FSR, for interaction.
    static RendererSettings realtime();
    //! Path tracing at a lower resolution with OpenDLSS-RR.
    static RendererSettings reconstructed();

    RendererSettings validated(bool cameraNeedsTracedPrimary, bool panoramic = false) const;
    WavefrontSettings getWavefrontSettings() const;
    //! FSR or DLSS-RR renders at the render scale and resolves to the display size.
    bool upscales() const { return upscaler == Upscaler::Fsr || denoiser == Denoiser::OpenDlssRr; }
    //! The display and the sample limit do not change the accumulated image.
    bool changesImage(const RendererSettings& previous) const;
    bool operator==(const RendererSettings&) const = default;
};

}
