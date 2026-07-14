#pragma once

#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Scene/SphericalHarmonicsOrder.h"

NR_GPU inline glm::vec3 evaluateGaussianSphericalHarmonics(
    const glm::vec3* coefficients,
    const SphericalHarmonicsOrder order,
    const glm::vec3 direction)
{
    constexpr float C0 = 0.28209479177387814f;
    constexpr float C1 = 0.4886025119029199f;
    constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f,
        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
    constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f,
        -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
        1.445305721320277f, -0.5900435899266435f};
    const float x = direction.x, y = direction.y, z = direction.z;
    glm::vec3 result = C0 * coefficients[0];
    if (order >= SphericalHarmonicsOrder::Degree1)
        result += -C1 * y * coefficients[1] + C1 * z * coefficients[2]
            - C1 * x * coefficients[3];
    if (order >= SphericalHarmonicsOrder::Degree2)
    {
        result += C2[0] * x * y * coefficients[4]
            + C2[1] * y * z * coefficients[5]
            + C2[2] * (2.0f * z * z - x * x - y * y) * coefficients[6]
            + C2[3] * x * z * coefficients[7]
            + C2[4] * (x * x - y * y) * coefficients[8];
    }
    if (order >= SphericalHarmonicsOrder::Degree3)
    {
        result += C3[0] * y * (3.0f * x * x - y * y) * coefficients[9]
            + C3[1] * x * y * z * coefficients[10]
            + C3[2] * y * (4.0f * z * z - x * x - y * y) * coefficients[11]
            + C3[3] * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y)
                * coefficients[12]
            + C3[4] * x * (4.0f * z * z - x * x - y * y) * coefficients[13]
            + C3[5] * z * (x * x - y * y) * coefficients[14]
            + C3[6] * x * (x * x - 3.0f * y * y) * coefficients[15];
    }
    return result;
}
