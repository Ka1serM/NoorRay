#pragma once

#include "NoorRay/Core/Math.h"
#include "NoorRay/Schema/NoorRayScene_generated.h"

namespace NoorRay::Format {

inline Core::float3 toFloat3(const Schema::Float3& value) {
    return {value.x(), value.y(), value.z()};
}

inline Schema::Float3 toSchema(const Core::float3 value) {
    return {value.x, value.y, value.z};
}

inline Core::Transform toTransform(const Schema::Transform& value) {
    Core::Transform transform;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 4; ++column)
            transform.rows[row][column] = value.rows()->Get(row * 4 + column);
    return transform;
}

inline Schema::Transform toSchema(const Core::Transform& transform) {
    return Schema::Transform(flatbuffers::span<const float, 12>(&transform.rows[0][0], 12));
}

inline Core::Bounds toBounds(const Schema::Bounds* value) {
    if (!value)
        return {};
    return {toFloat3(value->min()), toFloat3(value->max())};
}

inline Schema::Bounds toSchema(const Core::Bounds& bounds) {
    return {toSchema(bounds.min), toSchema(bounds.max)};
}

}
