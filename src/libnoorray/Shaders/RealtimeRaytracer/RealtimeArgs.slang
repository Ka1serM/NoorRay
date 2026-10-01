#pragma once

#include "../Raytracer/RaytracingAbi.slang"
#include "../../Shared/RealtimeArgs.h"

// Every realtime pass is launched with a RealtimeRoot pointing at the frame's
// RealtimeArgs; each stage reads its own block of it through this pointer.
RealtimeArgs* realtimeArgs()
{
    return ((RealtimeRoot*)root.arguments)->args;
}

// The largest finite value of the 16-bit float targets. Radiance beyond it,
// such as a sun in an HDRI, is clamped rather than stored as infinity.
static const float RealtimeHalfMax = 65504.0;
