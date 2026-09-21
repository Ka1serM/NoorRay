#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

// Each material addresses its own allocations from zero, so there are no
// offsets into a scene-wide arena.
struct Material
{
    GpuPtr(uint) bytecode;
    // Resource-descriptor-heap index per texture slot.
    GpuPtr(uint) textures;
    // Word count of `bytecode`; bounds the interpreter loop.
    uint bytecodeLength;
    uint stackSize;
    uint shadowOpaque;
    // Shader-binding-table index of the realtime renderer's callable shader
    // for this material, compiled from its MaterialX document; ~0u while it
    // has none.
    uint shader;
    // The parameter block that shader reads (MaterialInterface.slang).
    GpuPtr(uint) shaderParameters;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
