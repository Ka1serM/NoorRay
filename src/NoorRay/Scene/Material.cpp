#include "NoorRay/Scene/Material.h"

#include <cmath>

namespace NoorRay::Scene {

UvTransform UvTransform::fromTopLeft(const Core::float2 offset, const float rotation, const Core::float2 scale) {
    const float c = std::cos(rotation);
    const float s = std::sin(rotation);
    //! Top-left matrix: [c*sx, s*sy, ox; -s*sx, c*sy, oy]. Conjugated by the flip v -> 1 - v.
    const float a = c * scale.x;
    const float b = s * scale.y;
    const float d = -s * scale.x;
    const float e = c * scale.y;
    UvTransform transform;
    transform.row0 = {a, -b, b + offset.x};
    transform.row1 = {-d, e, 1.0f - e - offset.y};
    return transform;
}

}
