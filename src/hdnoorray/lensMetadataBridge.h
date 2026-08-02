#pragma once

#include "api.h"

// Narrow ABI used by the Blender add-on to approximate a physical lens in
// Blender's raster viewport. Returns a negative value when the lens cannot be
// loaded. The render delegate remains authoritative for optical ray tracing.
extern "C" HDNOORRAY_API float HdNoorRayLensFocalLengthMm(
    const char* lensPath, const char* glassCatalogPaths, int skipZeroThicknessSurfaces);
