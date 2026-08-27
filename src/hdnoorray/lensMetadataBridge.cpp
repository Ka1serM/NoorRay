#include "lensMetadataBridge.h"

extern "C" float HdNoorRayLensFocalLengthMm(const char* lensPath,
    const char* glassCatalogPaths, const int skipZeroThicknessSurfaces)
{
    // The former implementation linked against OpenLensIO/libross.  That
    // dependency was only used by Blender's viewport helper and made the
    // render delegate impossible to distribute with libnoorray alone.  The
    // Vulkan renderer reads its camera data directly from Hydra, so report
    // that no optional focal-length hint is available instead.
    (void)lensPath;
    (void)glassCatalogPaths;
    (void)skipZeroThicknessSurfaces;
    return -1.0f;
}
