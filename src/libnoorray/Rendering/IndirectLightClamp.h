#pragma once

#include "Materials/Shading/Spectrum.h"

namespace nr::lighting
{

// Blender/Cycles expresses the clamp in RGB units while the renderer stores
// each contribution as a sampled spectrum.  Cycles converts the user value to
// the sum-of-components space by multiplying it by three before applying the
// proportional clamp.
inline constexpr float BlenderRgbComponentCount = 3.0f;

NR_CPU_GPU inline SampledSpectrum clampIndirectLightContribution(
    SampledSpectrum contribution, const float clamp, const uint32_t depth)
{
    // A camera contribution is direct light.  As in Cycles, a value of zero
    // disables clamping; treating invalid/negative values as disabled also
    // keeps the GPU path safe if a setting is edited outside the UI.
    if (depth == 0u || !(clamp > 0.0f))
        return contribution;

    const float limit = clamp * BlenderRgbComponentCount;
    float sum = 0.0f;
    for (int i = 0; i < NrSpectrumSamples; ++i)
        sum += fabsf(contribution[i]);
    if (sum > limit)
        contribution *= limit / sum;
    return contribution;
}

} // namespace nr::lighting
