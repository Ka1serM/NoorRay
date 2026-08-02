#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"

enum class SphericalHarmonicsOrder : uint32_t
{
    Degree0 = 0,
    Degree1 = 1,
    Degree2 = 2,
    Degree3 = 3,
};

inline constexpr uint32_t MaxSphericalHarmonicsCoefficientCount = 16;
inline constexpr uint32_t SphericalHarmonicsChannelCount = 3;
inline constexpr float SphericalHarmonicsC0 = 0.28209479177387814f;

struct SphericalHarmonicsCoefficients
{
    std::array<__half, MaxSphericalHarmonicsCoefficientCount
        * SphericalHarmonicsChannelCount> values{};
    uint32_t count{};

    NR_CPU_GPU glm::vec3 get(const uint32_t index) const
    {
        return {
            __half2float(values[index * SphericalHarmonicsChannelCount + 0]),
            __half2float(values[index * SphericalHarmonicsChannelCount + 1]),
            __half2float(values[index * SphericalHarmonicsChannelCount + 2]),
        };
    }

    NR_CPU_GPU void set(const uint32_t index, const glm::vec3 value)
    {
        values[index * SphericalHarmonicsChannelCount + 0] = __float2half(value.x);
        values[index * SphericalHarmonicsChannelCount + 1] = __float2half(value.y);
        values[index * SphericalHarmonicsChannelCount + 2] = __float2half(value.z);
    }
};

static_assert(sizeof(SphericalHarmonicsCoefficients) == 100);

NR_CPU_GPU constexpr uint32_t sphericalHarmonicsCoefficientCount(
    const SphericalHarmonicsOrder order)
{
    const uint32_t degree = static_cast<uint32_t>(order);
    return (degree + 1) * (degree + 1);
}

constexpr SphericalHarmonicsOrder clampSphericalHarmonicsOrder(const int degree)
{
    return degree <= 0 ? SphericalHarmonicsOrder::Degree0
        : degree == 1 ? SphericalHarmonicsOrder::Degree1
        : degree == 2 ? SphericalHarmonicsOrder::Degree2
                      : SphericalHarmonicsOrder::Degree3;
}

NR_GPU inline glm::vec3 evaluateSphericalHarmonics(
    const __half* coefficients,
    const SphericalHarmonicsOrder order,
    const glm::vec3 inputDirection)
{
    constexpr float C1 = 0.4886025119029199f;
    constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f,
        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
    constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f,
        -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
        1.445305721320277f, -0.5900435899266435f};
    const auto coefficient = [coefficients](const uint32_t index) {
        return glm::vec3(
            __half2float(coefficients[index * SphericalHarmonicsChannelCount + 0]),
            __half2float(coefficients[index * SphericalHarmonicsChannelCount + 1]),
            __half2float(coefficients[index * SphericalHarmonicsChannelCount + 2]));
    };
    const glm::vec3 direction = glm::normalize(inputDirection);
    const float x = direction.x, y = direction.y, z = direction.z;
    glm::vec3 result = SphericalHarmonicsC0 * coefficient(0);
    if (order >= SphericalHarmonicsOrder::Degree1)
        result += -C1 * y * coefficient(1) + C1 * z * coefficient(2)
            - C1 * x * coefficient(3);
    if (order >= SphericalHarmonicsOrder::Degree2)
    {
        result += C2[0] * x * y * coefficient(4)
            + C2[1] * y * z * coefficient(5)
            + C2[2] * (2.0f * z * z - x * x - y * y) * coefficient(6)
            + C2[3] * x * z * coefficient(7)
            + C2[4] * (x * x - y * y) * coefficient(8);
    }
    if (order >= SphericalHarmonicsOrder::Degree3)
    {
        result += C3[0] * y * (3.0f * x * x - y * y) * coefficient(9)
            + C3[1] * x * y * z * coefficient(10)
            + C3[2] * y * (4.0f * z * z - x * x - y * y) * coefficient(11)
            + C3[3] * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y)
                * coefficient(12)
            + C3[4] * x * (4.0f * z * z - x * x - y * y) * coefficient(13)
            + C3[5] * z * (x * x - y * y) * coefficient(14)
            + C3[6] * x * (x * x - 3.0f * y * y) * coefficient(15);
    }
    return result;
}

// Decode one Gaussian's view-dependent SH color. Beauty and AOV passes share
// this primitive and select the order appropriate to the pass.
NR_GPU inline glm::vec3 gaussianAlbedoRgb(
    const __half* allCoefficients,
    const uint32_t coefficientCount,
    const uint32_t gaussianId,
    const SphericalHarmonicsOrder order,
    const glm::vec3 viewDirection)
{
    const __half* coefficients = allCoefficients
        + static_cast<std::size_t>(gaussianId)
            * coefficientCount * SphericalHarmonicsChannelCount;
    return glm::vec3(0.5f) + evaluateSphericalHarmonics(
        coefficients, order, viewDirection);
}
