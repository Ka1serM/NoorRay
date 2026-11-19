#include "NoorRay/Render/RendererSettings.h"

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

RendererSettings RendererSettings::validated(const bool cameraNeedsTracedPrimary) const {
    RendererSettings result = *this;
    if (result.shading == Shading::Raster) {
        result.primaryVisibility = PrimaryVisibility::Raster;
        result.reuse = Reuse::Off;
        result.denoiser = Denoiser::Off;
    }
    if (cameraNeedsTracedPrimary || result.reuse != Reuse::Off)
        result.primaryVisibility = PrimaryVisibility::Traced;
    return result;
}

WavefrontSettings RendererSettings::getWavefrontSettings() const {
    WavefrontSettings result = wavefront;
    result.accumulate = accumulate;
    result.resamplesPaths = reuse != Reuse::Off;
    result.rasterShading = shading == Shading::Raster;
    result.splatsGaussians = result.rasterShading && primaryVisibility == PrimaryVisibility::Raster;
    return result;
}

bool RendererSettings::changesImage(const RendererSettings& previous) const {
    RendererSettings compared = previous;
    compared.display = display;
    compared.maxSamples = maxSamples;
    return !(compared == *this);
}

}
