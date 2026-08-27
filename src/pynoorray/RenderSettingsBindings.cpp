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
        .value("POSITION", BufferVisualization::Position)
        .value("PROXY_OVERDRAW", BufferVisualization::ProxyOverdraw);

    nb::enum_<GaussianShadingMode>(module, "GaussianShadingMode")
        .value("GLOBAL_ILLUMINATION", GaussianShadingMode::GlobalIllumination)
        .value("DIRECT_COLOR", GaussianShadingMode::DirectColor);

    nb::enum_<GaussianProxyType>(module, "GaussianProxyType")
        .value("ICOSPHERE", GaussianProxyType::Icosphere)
        .value("OCTAHEDRON", GaussianProxyType::Octahedron)
        .value("ICOSAHEDRON", GaussianProxyType::Icosahedron)
        .value("ICOSPHERE_LEVEL2", GaussianProxyType::IcosphereLevel2);

    nb::enum_<SphericalHarmonicsOrder>(module, "SphericalHarmonicsOrder")
        .value("DEGREE_0", SphericalHarmonicsOrder::Degree0)
        .value("DEGREE_1", SphericalHarmonicsOrder::Degree1)
        .value("DEGREE_2", SphericalHarmonicsOrder::Degree2)
        .value("DEGREE_3", SphericalHarmonicsOrder::Degree3);

    nb::class_<RenderSettings>(module, "RenderSettings")
        .def(nb::init<>())
        .def_rw("samples", &RenderSettings::samples)
        .def_rw("max_samples", &RenderSettings::maxSamples)
        .def_rw("aov_enabled", &RenderSettings::aovEnabled)
        .def_rw("max_bounces", &RenderSettings::maxBounces)
        .def_rw("indirect_light_clamp", &RenderSettings::indirectLightClamp)
        .def_rw("tonemapping_enabled", &RenderSettings::tonemappingEnabled)
        .def_rw("transparent_background", &RenderSettings::transparentBackground)
        .def_rw("camera_exposure", &RenderSettings::cameraExposure)
        .def_rw("gaussian_cutoff_sigma", &RenderSettings::gaussianCutoffSigma)
        .def_rw("gaussian_proxy_type", &RenderSettings::gaussianProxyType)
        .def_rw("gaussian_shading_mode", &RenderSettings::gaussianShadingMode)
        .def_rw("gaussian_render_spherical_harmonics",
            &RenderSettings::gaussianRenderSphericalHarmonics)
        .def_rw("gaussian_proxy_overdraw_visualization",
            &RenderSettings::gaussianProxyOverdrawVisualization)
        .def_rw("gaussian_proxy_overdraw_max",
            &RenderSettings::gaussianProxyOverdrawMax)
        .def_rw("buffer_visualization", &RenderSettings::bufferVisualization);
}
