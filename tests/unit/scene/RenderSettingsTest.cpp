#include <catch2/catch_test_macros.hpp>
#include "Shared/RenderSettings.h"

TEST_CASE("indirect light clamp defaults to Blender's value", "[render-settings][clamp]")
{
    RenderSettings settings;

    CHECK(settings.indirectLightClamp == 10.0f);
}

TEST_CASE("Proxy Overdraw view does not enable the diagnostic",
    "[render-settings][proxy-overdraw]")
{
    RenderSettings settings;
    settings.gaussianProxyOverdrawVisualization = false;
    settings.bufferVisualization = BufferVisualization::ProxyOverdraw;

    CHECK_FALSE(rendersProxyOverdraw(settings));
}
