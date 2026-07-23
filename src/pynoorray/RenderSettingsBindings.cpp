#include "Bindings.h"

#include "Scene/RenderSettings.h"

namespace nb = nanobind;

void bindRenderSettings(nb::module_& module)
{
    nb::enum_<BufferVisualization>(module, "BufferVisualization")
        .value("BEAUTY", BufferVisualization::Beauty)
        .value("ALBEDO", BufferVisualization::Albedo)
        .value("NORMAL", BufferVisualization::Normal)
        .value("CRYPTOMATTE", BufferVisualization::Cryptomatte)
        .value("POSITION", BufferVisualization::Position);

    nb::enum_<GaussianShadingMode>(module, "GaussianShadingMode")
        .value("GLOBAL_ILLUMINATION", GaussianShadingMode::GlobalIllumination)
        .value("DIRECT_COLOR", GaussianShadingMode::DirectColor);

    nb::class_<RenderSettings>(module, "RenderSettings")
        .def(nb::init<>())
        .def_rw("samples", &RenderSettings::samples)
        .def_rw("max_samples", &RenderSettings::maxSamples)
        .def_rw("noise_limit_enabled", &RenderSettings::noiseLimitEnabled)
        .def_rw("noise_level", &RenderSettings::noiseLevel)
        .def_rw("aov_enabled", &RenderSettings::aovEnabled)
        .def_rw("optix_denoiser_enabled", &RenderSettings::optixDenoiserEnabled)
        .def_rw("optix_denoiser_min_samples", &RenderSettings::optixDenoiserMinSamples)
        .def_rw("max_bounces", &RenderSettings::maxBounces)
        .def_rw("russian_roulette_start_bounce", &RenderSettings::russianRouletteStartBounce)
        .def_rw("tonemapping_enabled", &RenderSettings::tonemappingEnabled)
        .def_rw("transparent_background", &RenderSettings::transparentBackground)
        .def_rw("gaussian_cutoff_sigma", &RenderSettings::gaussianCutoffSigma)
        .def_rw("gaussian_shading_mode", &RenderSettings::gaussianShadingMode)
        .def_rw("buffer_visualization", &RenderSettings::bufferVisualization);
}
