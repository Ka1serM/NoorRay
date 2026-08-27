#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "Rendering/IndirectLightClamp.h"
#include "Scene/RenderSettings.h"

TEST_CASE("indirect light clamp defaults to Blender's value", "[render-settings][clamp]")
{
    RenderSettings settings;

    CHECK(settings.indirectLightClamp == 10.0f);
}

TEST_CASE("indirect light clamp scales each indirect contribution", "[render-settings][clamp]")
{
    SampledSpectrum contribution;
    contribution[0] = 6.0f;
    contribution[1] = -2.0f;
    contribution[2] = 4.0f;
    contribution[3] = 0.0f;

    const SampledSpectrum clamped = nr::lighting::clampIndirectLightContribution(
        contribution, 2.0f, 1u);

    CHECK(clamped[0] == Catch::Approx(3.0f));
    CHECK(clamped[1] == Catch::Approx(-1.0f));
    CHECK(clamped[2] == Catch::Approx(2.0f));
    CHECK(clamped[3] == Catch::Approx(0.0f));
}

TEST_CASE("zero indirect light clamp disables clamping", "[render-settings][clamp]")
{
    SampledSpectrum contribution(100.0f);

    const SampledSpectrum unclamped = nr::lighting::clampIndirectLightContribution(
        contribution, 0.0f, 2u);

    for (int i = 0; i < NrSpectrumSamples; ++i)
        CHECK(unclamped[i] == Catch::Approx(contribution[i]));
}

TEST_CASE("indirect light clamp leaves camera contributions unchanged", "[render-settings][clamp]")
{
    SampledSpectrum contribution(100.0f);

    const SampledSpectrum direct = nr::lighting::clampIndirectLightContribution(
        contribution, 1.0f, 0u);

    for (int i = 0; i < NrSpectrumSamples; ++i)
        CHECK(direct[i] == Catch::Approx(contribution[i]));
}

TEST_CASE("Proxy Overdraw view does not enable the diagnostic",
    "[render-settings][proxy-overdraw]")
{
    RenderSettings settings;
    settings.gaussianProxyOverdrawVisualization = false;
    settings.bufferVisualization = BufferVisualization::ProxyOverdraw;

    CHECK_FALSE(rendersProxyOverdraw(settings));
}
