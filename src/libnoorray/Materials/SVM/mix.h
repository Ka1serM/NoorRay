/* SPDX-License-Identifier: Apache-2.0
 *
 * MaterialX blend evaluation for the NoorRay SVM.
 */
#pragma once

#include <glm/common.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Materials/SVM/SvmTypes.h"

namespace nr::svm::detail
{
NR_GPU inline glm::vec3 materialXMixColor(const MaterialXBlendType type, const float t,
    const glm::vec3 a, const glm::vec3 b)
{
    const float tm = 1.0f - t;
    switch (type) {
    case MaterialXBlendType::Blend: return glm::mix(a, b, t);
    case MaterialXBlendType::Add: return glm::mix(a, a + b, t);
    case MaterialXBlendType::Subtract: return glm::mix(a, a - b, t);
    case MaterialXBlendType::Difference: return glm::mix(a, glm::abs(a - b), t);
    case MaterialXBlendType::Screen:
        return glm::vec3(1.0f) - (glm::vec3(tm) + t * (glm::vec3(1.0f) - b))
            * (glm::vec3(1.0f) - a);
    case MaterialXBlendType::Overlay: {
        glm::vec3 out = a;
        for (int i = 0; i < 3; ++i)
            out[i] = out[i] < 0.5f ? out[i] * (tm + 2.0f * t * b[i])
                : 1.0f - (tm + 2.0f * t * (1.0f - b[i])) * (1.0f - out[i]);
        return out;
    }
    case MaterialXBlendType::Dodge: {
        glm::vec3 out = a;
        for (int i = 0; i < 3; ++i)
            if (out[i] != 0.0f) {
                const float divisor = 1.0f - t * b[i];
                out[i] = divisor <= 0.0f ? 1.0f : fminf(out[i] / divisor, 1.0f);
            }
        return out;
    }
    case MaterialXBlendType::Burn: {
        glm::vec3 out = a;
        for (int i = 0; i < 3; ++i) {
            const float divisor = tm + t * b[i];
            out[i] = divisor <= 0.0f ? 0.0f : glm::clamp(1.0f - (1.0f - out[i]) / divisor, 0.0f, 1.0f);
        }
        return out;
    }
    default: return glm::mix(a, b, t);
    }
}
} // namespace nr::svm::detail
