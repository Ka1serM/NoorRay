#include "NoorRay/Render/RendererSettings.h"

#include <algorithm>

namespace NoorRay::Render {

RendererSettings RendererSettings::viewport() {
    RendererSettings settings;
    settings.primaryVisibility = PrimaryVisibility::Raster;
    settings.shading = Shading::Raster;
    settings.reuse = Reuse::Off;
    settings.denoiser = Denoiser::Off;
    settings.accumulate = false;
    return settings;
}

RendererSettings RendererSettings::pathTraced() {
    return {};
}

RendererSettings RendererSettings::realtime() {
    RendererSettings settings;
    settings.upscaler = Upscaler::Fsr;
    settings.accumulate = false;
    return settings;
}

RendererSettings RendererSettings::reconstructed() {
    RendererSettings settings;
    settings.denoiser = Denoiser::OpenDlssRr;
    settings.accumulate = false;
    return settings;
}

RendererSettings RendererSettings::validated(const bool cameraNeedsTracedPrimary, const bool panoramic) const {
    RendererSettings result = *this;
    //! RELAX, FSR and DLSS-RR reproject through a perspective or orthographic matrix.
    if (panoramic) {
        result.denoiser = Denoiser::Off;
        result.upscaler = Upscaler::Off;
    }
    if (result.shading == Shading::Raster) {
        result.primaryVisibility = PrimaryVisibility::Raster;
        result.reuse = Reuse::Off;
        result.denoiser = Denoiser::Off;
        result.upscaler = Upscaler::Off;
    }
    if (result.accumulate || !result.temporal)
        result.upscaler = Upscaler::Off;
    if (result.denoiser == Denoiser::OpenDlssRr && (result.accumulate || !result.temporal))
        result.denoiser = Denoiser::Off;
    if (result.denoiser == Denoiser::OpenDlssRr)
        result.upscaler = Upscaler::Off;
    if (result.accumulate)
        result.reuse = Reuse::Off;
    result.renderScale = std::clamp(result.renderScale, 0.25f, 1.0f);
    if (cameraNeedsTracedPrimary || result.reuse != Reuse::Off)
        result.primaryVisibility = PrimaryVisibility::Traced;
    return result;
}

WavefrontSettings RendererSettings::getWavefrontSettings() const {
    WavefrontSettings result = wavefront;
    result.accumulate = accumulate;
    result.reuse = reuse;
    result.reusesHistory = temporal;
    result.radianceCache = radianceCache && !accumulate;
    result.rasterShading = shading == Shading::Raster;
    result.splatsGaussians = result.rasterShading && primaryVisibility == PrimaryVisibility::Raster;
    result.splitsLighting = denoiser != Denoiser::Off && !result.rasterShading;
    result.approximatesSpecular = denoiser == Denoiser::OpenDlssRr;
    return result;
}

bool RendererSettings::changesImage(const RendererSettings& previous) const {
    RendererSettings compared = previous;
    compared.display = display;
    compared.maxSamples = maxSamples;
    compared.sharpness = sharpness;
    compared.denoiseBlend = denoiseBlend;
    return !(compared == *this);
}

}
