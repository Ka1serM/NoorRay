#include "lensMetadataBridge.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "openlensfileio/glasscatalogs/glasscatalog/GlassCatalogLibrary.h"

extern "C" float HdNoorRayLensFocalLengthMm(const char* lensPath,
    const char* glassCatalogPaths, const int skipZeroThicknessSurfaces)
{
    if (lensPath == nullptr || *lensPath == '\0')
        return -1.0f;
    try {
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths != nullptr
            ? glassCatalogPaths : "";
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        const ross::CameraLens lens = ross::CameraLensSystemReader::readCameraLens(
            lensPath, catalogs,
            ross::ReadOptions{1.0f, skipZeroThicknessSurfaces != 0});
        const float focalLengthMm = lens.metadata.focalLength * 10.0f;
        return std::isfinite(focalLengthMm) && focalLengthMm > 0.0f
            ? focalLengthMm : -1.0f;
    } catch (...) {
        return -1.0f;
    }
}
