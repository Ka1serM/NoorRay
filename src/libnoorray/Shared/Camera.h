#pragma once

#include "Types.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

struct Camera
{
    float cameraToWorld[16];
    uint projection;
    float sensorWidthMm;
    float sensorHeightMm;
    float focalLengthMm;
    float focusDistanceCm;
    float apertureDiameterMm;
    uint sensorOrigin;
    float exposure;
    uint reserved[2];
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
