#pragma once

#include <cmath>

#include "GPU/Annotations.h"
#include "Kernels/Types.h"

NR_CPU_GPU inline glm::vec3 makeVec3(const float value) { return glm::vec3(value, value, value); }
NR_CPU_GPU inline glm::vec3 add(const glm::vec3 a, const glm::vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
NR_CPU_GPU inline glm::vec3 subtract(const glm::vec3 a, const glm::vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
NR_CPU_GPU inline glm::vec3 multiply(const glm::vec3 a, const glm::vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
NR_CPU_GPU inline glm::vec3 multiply(const glm::vec3 a, const float b) { return {a.x * b, a.y * b, a.z * b}; }
NR_CPU_GPU inline glm::vec3 divide(const glm::vec3 a, const float b) { return multiply(a, 1.0f / b); }
NR_CPU_GPU inline float dot3(const glm::vec3 a, const glm::vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
NR_CPU_GPU inline glm::vec3 cross3(const glm::vec3 a, const glm::vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
NR_CPU_GPU inline float length3(const glm::vec3 value) { return sqrtf(dot3(value, value)); }
NR_CPU_GPU inline glm::vec3 normalize3(const glm::vec3 value)
{
    const float length = length3(value);
    return length > 1e-20f ? divide(value, length) : glm::vec3(0.0f, 0.0f, 1.0f);
}
NR_CPU_GPU inline glm::vec3 reflect3(const glm::vec3 direction, const glm::vec3 normal)
{
    return subtract(direction, multiply(normal, 2.0f * dot3(direction, normal)));
}
NR_CPU_GPU inline glm::vec3 clamp3(const glm::vec3 value, const float low, const float high)
{
    return {fminf(fmaxf(value.x, low), high), fminf(fmaxf(value.y, low), high), fminf(fmaxf(value.z, low), high)};
}
NR_CPU_GPU inline glm::vec3 lerp3(const glm::vec3 a, const glm::vec3 b, const float t)
{
    return add(multiply(a, 1.0f - t), multiply(b, t));
}
NR_CPU_GPU inline float maxComponent(const glm::vec3 value) { return fmaxf(value.x, fmaxf(value.y, value.z)); }
NR_CPU_GPU inline glm::vec3 transformPoint(const float matrix[12], const glm::vec3 point)
{
    return {
        matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3],
        matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7],
        matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11]};
}
NR_CPU_GPU inline glm::vec3 transformVector(const float matrix[12], const glm::vec3 vector)
{
    return {
        matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z,
        matrix[4] * vector.x + matrix[5] * vector.y + matrix[6] * vector.z,
        matrix[8] * vector.x + matrix[9] * vector.y + matrix[10] * vector.z};
}
NR_CPU_GPU inline glm::vec3 transformNormal(const float matrix[9], const glm::vec3 normal)
{
    return normalize3({
        matrix[0] * normal.x + matrix[1] * normal.y + matrix[2] * normal.z,
        matrix[3] * normal.x + matrix[4] * normal.y + matrix[5] * normal.z,
        matrix[6] * normal.x + matrix[7] * normal.y + matrix[8] * normal.z});
}
