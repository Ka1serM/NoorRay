#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Each material addresses its own allocations from zero, so there are no
// offsets into a scene-wide arena.
// How rays treat a material's surfaces, as Unreal's material flags decide it.
// The defaults (no flags) are a two-sided surface that casts shadows.
// An Unreal "Is Sky" material: seen by camera rays only; later bounces find
// it black, since the sky light carries its light.
static const uint MaterialFlagSky = 0x1u;
// One-sided: camera rays skip its back faces and its back emits nothing.
static const uint MaterialFlagOneSided = 0x2u;
// Transparent to shadow rays (Unreal's Cast Ray Traced Shadows off).
static const uint MaterialFlagNoShadows = 0x4u;

struct Material
{
    // The parameter block the realtime renderer's hit shaders of this
    // material read (MaterialInterface.slang). Which shaders those are is
    // chosen by the shader binding table, not by this record.
    GpuPtr(uint) shaderParameters;
    // MaterialFlag* bits.
    uint flags;
    uint padding;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
