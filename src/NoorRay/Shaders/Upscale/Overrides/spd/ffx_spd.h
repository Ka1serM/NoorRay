//! Shadows the SPD include of the FSR pyramids: WGSL needs a uniform counter load before the last workgroup's barriers, and only the SPD call sees it.
#define SpdGetAtomicCounter() workgroupUniformLoad(spdCounter)
#include "FidelityFX/gpu/spd/ffx_spd.h"
#undef SpdGetAtomicCounter
