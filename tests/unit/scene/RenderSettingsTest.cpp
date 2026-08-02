#include <catch2/catch_test_macros.hpp>

#include "Scene/RenderSettings.h"

TEST_CASE("Denoised view does not enable the OptiX denoiser", "[render-settings][denoiser]")
{
    RenderSettings settings;
    settings.optixDenoiserEnabled = false;
    settings.bufferVisualization = BufferVisualization::Denoised;

    CHECK_FALSE(runsOptixDenoiser(settings));
}

TEST_CASE("OptiX checkbox is the sole denoiser enable switch",
    "[render-settings][denoiser]")
{
    RenderSettings settings;
    settings.optixDenoiserEnabled = true;
    settings.bufferVisualization = BufferVisualization::Beauty;

    CHECK(runsOptixDenoiser(settings));
    CHECK(settings.bufferVisualization == BufferVisualization::Beauty);
}

TEST_CASE("Proxy overdraw suppresses the OptiX denoiser",
    "[render-settings][denoiser]")
{
    RenderSettings settings;
    settings.optixDenoiserEnabled = true;
    settings.gaussianProxyOverdrawVisualization = true;
    settings.bufferVisualization = BufferVisualization::ProxyOverdraw;

    CHECK_FALSE(runsOptixDenoiser(settings));
    CHECK(rendersProxyOverdraw(settings));
}

TEST_CASE("Proxy Overdraw view does not enable the diagnostic",
    "[render-settings][proxy-overdraw]")
{
    RenderSettings settings;
    settings.gaussianProxyOverdrawVisualization = false;
    settings.bufferVisualization = BufferVisualization::ProxyOverdraw;

    CHECK_FALSE(rendersProxyOverdraw(settings));
}
