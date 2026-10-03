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

// Realtime radiance is stored pre-exposed, so the half-float targets, the
// denoiser and the upscaler see display-referred values in dark and bright
// scenes alike.
float realtimeExposureScale()
{
    return exp2(realtimeArgs()->frame.camera.exposure);
}
