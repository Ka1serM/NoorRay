#pragma once

#include "../Raytracer/RaytracingAbi.slang"
#include "../../Shared/RealtimeArgs.h"

// Every realtime pass is launched with a RealtimeRoot pointing at the frame's
// RealtimeArgs; each stage reads its own block of it through this pointer.
RealtimeArgs* realtimeArgs()
{
    return ((RealtimeRoot*)root.arguments)->args;
}
