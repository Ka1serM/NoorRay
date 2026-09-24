#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Each material addresses its own allocations from zero, so there are no
// offsets into a scene-wide arena.
struct Material
{
    // The parameter block the realtime renderer's hit shaders of this
    // material read (MaterialInterface.slang). Which shaders those are is
    // chosen by the shader binding table, not by this record.
    GpuPtr(uint) shaderParameters;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
