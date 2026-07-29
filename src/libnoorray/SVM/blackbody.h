/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0
 *
 * MaterialX blackbody port.  This follows the same one-node SVM shape as
 * Cycles' blackbody.h; its body is the vendored mx_blackbody.glsl reference
 * implementation so the MaterialX node retains MaterialX color semantics.
 */

#pragma once

#include <glm/common.hpp>

#include "CUDA/Annotations.h"

namespace nr::svm::detail
{

NR_GPU inline glm::vec3 svmBlackbody(const float inputTemperature)
{
    const float temperature = glm::clamp(inputTemperature, 1667.0f, 25000.0f);
    const float t = 1000.0f / temperature;
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float xc = temperature < 4000.0f
        ? -0.2661239f * t3 - 0.2343580f * t2 + 0.8776956f * t + 0.179910f
        : -3.0258469f * t3 + 2.1070379f * t2 + 0.2226347f * t + 0.240390f;
    const float xc2 = xc * xc;
    const float xc3 = xc2 * xc;
    const float yc = temperature < 2222.0f
        ? -1.1063814f * xc3 - 1.34811020f * xc2 + 2.18555832f * xc - 0.20219683f
        : temperature < 4000.0f
        ? -0.9549476f * xc3 - 1.37418593f * xc2 + 2.09137015f * xc - 0.16748867f
        : 3.0817580f * xc3 - 5.87338670f * xc2 + 3.75112997f * xc - 0.37001483f;
    if (yc <= 0.0f)
        return glm::vec3(1.0f);
    const glm::vec3 xyz{xc / yc, 1.0f, (1.0f - xc - yc) / yc};
    return glm::max(glm::vec3(
        3.2406f * xyz.x - 0.9689f * xyz.y + 0.0557f * xyz.z,
        -1.5372f * xyz.x + 1.8758f * xyz.y - 0.2040f * xyz.z,
        -0.4986f * xyz.x + 0.0415f * xyz.y + 1.0570f * xyz.z), glm::vec3(0.0f));
}

} // namespace nr::svm::detail
