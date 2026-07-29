/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Port of intern/cycles/kernel/svm/math.h. MaterialX scalar math categories
 * select an operation in this shared Cycles-sized instruction.
 */

#pragma once

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "CUDA/Annotations.h"
#include "SVM/SvmTypes.h"

namespace nr::svm::detail
{

NR_GPU inline float svmMath(const MathOp operation, const float a, const float b,
    const float /*c*/)
{
    switch (operation) {
    case MathOp::Add: return a + b;
    case MathOp::Subtract: return a - b;
    case MathOp::Multiply: return a * b;
    case MathOp::Divide: return b != 0.0f ? a / b : 0.0f;
    // MaterialX mx_mod follows GLSL's floor-based modulo (unlike C fmod),
    // which keeps periodic procedural nodes correct for negative inputs.
    case MathOp::Modulo: return b != 0.0f ? a - b * floorf(a / b) : 0.0f;
    case MathOp::Power: return powf(a, b);
    case MathOp::Absval: return fabsf(a);
    case MathOp::Floor: return floorf(a);
    case MathOp::Ceil: return ceilf(a);
    case MathOp::Round: return roundf(a);
    case MathOp::Sign: return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f);
    case MathOp::Min: return fminf(a, b);
    case MathOp::Max: return fmaxf(a, b);
    case MathOp::Sin: return sinf(a);
    case MathOp::Cos: return cosf(a);
    case MathOp::Tan: return tanf(a);
    case MathOp::Asin: return asinf(glm::clamp(a, -1.0f, 1.0f));
    case MathOp::Acos: return acosf(glm::clamp(a, -1.0f, 1.0f));
    case MathOp::Atan2: return atan2f(a, b);
    case MathOp::Ln: return a > 0.0f ? logf(a) : 0.0f;
    case MathOp::Exp: return expf(a);
    case MathOp::Sqrt: return a > 0.0f ? sqrtf(a) : 0.0f;
    case MathOp::InverseSqrt: return a > 0.0f ? rsqrtf(a) : 0.0f;
    case MathOp::Fract: return a - floorf(a);
    case MathOp::GreaterThan: return a > b ? 1.0f : 0.0f;
    case MathOp::GreaterEqual: return a >= b ? 1.0f : 0.0f;
    case MathOp::Equal: return a == b ? 1.0f : 0.0f;
    case MathOp::And: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
    case MathOp::Or: return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
    case MathOp::Not: return a != 0.0f ? 0.0f : 1.0f;
    }
    return 0.0f;
}

NR_GPU inline glm::vec3 svmVectorMath(const VectorMathOp operation, const glm::vec3 a,
    const glm::vec3 b, const float parameter = 0.0f)
{
    switch (operation) {
    case VectorMathOp::Add: return a + b;
    case VectorMathOp::Subtract: return a - b;
    case VectorMathOp::Multiply: return a * b;
    case VectorMathOp::Divide:
        return {b.x != 0.0f ? a.x / b.x : 0.0f, b.y != 0.0f ? a.y / b.y : 0.0f,
            b.z != 0.0f ? a.z / b.z : 0.0f};
    case VectorMathOp::CrossProduct: return glm::cross(a, b);
    case VectorMathOp::DotProduct: return glm::vec3(glm::dot(a, b));
    case VectorMathOp::Normalize:
        return glm::dot(a, a) > 0.0f ? glm::normalize(a) : glm::vec3(0.0f);
    case VectorMathOp::Magnitude: return glm::vec3(glm::length(a));
    case VectorMathOp::Distance: return glm::vec3(glm::distance(a, b));
    case VectorMathOp::Reflect: return glm::reflect(a, b);
    case VectorMathOp::Refract: return glm::refract(a, b, parameter);
    }
    return glm::vec3(0.0f);
}

} // namespace nr::svm::detail
