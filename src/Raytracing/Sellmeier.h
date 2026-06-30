#pragma once

#include <cmath>

#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

// Standard visible Fraunhofer lines used by optical-glass catalogs.
static constexpr float FraunhoferRedNm   = 656.2725f; // C line, H
static constexpr float FraunhoferGreenNm = 546.0740f; // e line, Hg
static constexpr float FraunhoferBlueNm  = 486.1327f; // F line, H

struct SellmeierCoefficients
{
    // n^2(lambda) - 1 = sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in um.
    glm::vec3 b{1.25f, 0.0f, 0.0f};
    glm::vec3 c{0.0f, 0.01f, 100.0f};
};

NR_CPU_GPU inline float sellmeierIor(
    const SellmeierCoefficients& coefficients, const float wavelengthNm)
{
    const float wavelengthUm = wavelengthNm * 0.001f;
    const float lambda2 = wavelengthUm * wavelengthUm;
    float n2 = 1.0f;
    for (int i = 0; i < 3; ++i)
    {
        const float denominator = lambda2 - coefficients.c[i];
        if (fabsf(denominator) > 1e-8f)
            n2 += coefficients.b[i] * lambda2 / denominator;
    }
    return sqrtf(fmaxf(n2, 1.0f));
}

inline SellmeierCoefficients fitSellmeierFromFraunhofer(const glm::vec3 iorRgb)
{
    // Three samples do not determine six independent Sellmeier parameters.
    // Fix one constant, one UV, and one IR pole and solve the three linear
    // oscillator strengths. This interpolates all three entered IORs exactly.
    SellmeierCoefficients result;
    const float wavelengths[3] = {
        FraunhoferRedNm * 0.001f,
        FraunhoferGreenNm * 0.001f,
        FraunhoferBlueNm * 0.001f,
    };
    float matrix[3][4]{};
    for (int row = 0; row < 3; ++row)
    {
        const float lambda2 = wavelengths[row] * wavelengths[row];
        for (int column = 0; column < 3; ++column)
            matrix[row][column] = lambda2 / (lambda2 - result.c[column]);
        const float ior = fmaxf(iorRgb[row], 1.0f);
        matrix[row][3] = ior * ior - 1.0f;
    }

    for (int column = 0; column < 3; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row)
            if (fabsf(matrix[row][column]) > fabsf(matrix[pivot][column]))
                pivot = row;
        for (int entry = column; entry < 4; ++entry)
        {
            const float temporary = matrix[column][entry];
            matrix[column][entry] = matrix[pivot][entry];
            matrix[pivot][entry] = temporary;
        }
        const float divisor = matrix[column][column];
        if (fabsf(divisor) < 1e-12f)
            return result;
        for (int entry = column; entry < 4; ++entry)
            matrix[column][entry] /= divisor;
        for (int row = 0; row < 3; ++row)
        {
            if (row == column)
                continue;
            const float factor = matrix[row][column];
            for (int entry = column; entry < 4; ++entry)
                matrix[row][entry] -= factor * matrix[column][entry];
        }
    }
    result.b = glm::vec3(matrix[0][3], matrix[1][3], matrix[2][3]);
    return result;
}

NR_CPU_GPU inline glm::vec3 sellmeierFraunhoferIors(
    const SellmeierCoefficients& coefficients)
{
    return glm::vec3(
        sellmeierIor(coefficients, FraunhoferRedNm),
        sellmeierIor(coefficients, FraunhoferGreenNm),
        sellmeierIor(coefficients, FraunhoferBlueNm));
}

inline SellmeierCoefficients constantIorSellmeier(const float ior)
{
    SellmeierCoefficients result;
    const float clamped = fmaxf(ior, 1.0f);
    result.b = glm::vec3(clamped * clamped - 1.0f, 0.0f, 0.0f);
    return result;
}
