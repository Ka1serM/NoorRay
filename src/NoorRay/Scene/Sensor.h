#pragma once

#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

//! Size in metres. The offset moves the sensor in its plane (shift lens), in metres.
struct Sensor {
    float width = 0.036f;
    float height = 0.024f;
    Core::float2 offset;

    float getAspect() const { return width / height; }
};

}
