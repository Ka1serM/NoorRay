#pragma once

#include <cmath>

namespace DiffRenderer::Methods {

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vector3 operator+(const Vector3 other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vector3 operator-(const Vector3 other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vector3 operator*(const float scale) const { return {x * scale, y * scale, z * scale}; }
    Vector3& operator+=(const Vector3 other) { return *this = *this + other; }
    float dot(const Vector3 other) const { return x * other.x + y * other.y + z * other.z; }
    Vector3 cross(const Vector3 other) const { return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x}; }
    float getLength() const { return std::sqrt(dot(*this)); }
    Vector3 getNormalized() const {
        const float length = getLength();
        return length > 0.0f ? *this * (1.0f / length) : Vector3{};
    }
};

}
