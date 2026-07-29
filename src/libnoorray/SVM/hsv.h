/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Port of intern/cycles/kernel/svm/hsv.h and util/color.h. MaterialX
 * `hsvadjust` maps directly to this Cycles NODE_HSV-sized operation.
 */

#pragma once

#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

namespace nr::svm::detail
{

NR_GPU inline glm::vec3 svmRgbToHsv(const glm::vec3 rgb)
{
    const float cmax = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    const float cmin = fminf(rgb.x, fminf(rgb.y, rgb.z));
    const float delta = cmax - cmin;
    float h = 0.0f;
    const float s = cmax != 0.0f ? delta / cmax : 0.0f;
    if (s != 0.0f) {
        const glm::vec3 c = (glm::vec3(cmax) - rgb) / delta;
        h = rgb.x == cmax ? c.z - c.y : rgb.y == cmax ? 2.0f + c.x - c.z
                                                               : 4.0f + c.y - c.x;
        h /= 6.0f;
        if (h < 0.0f)
            h += 1.0f;
    }
    return {h, s, cmax};
}

NR_GPU inline glm::vec3 svmHsvToRgb(glm::vec3 hsv)
{
    if (hsv.y == 0.0f)
        return glm::vec3(hsv.z);
    if (hsv.x == 1.0f)
        hsv.x = 0.0f;
    const float h = hsv.x * 6.0f;
    const int i = static_cast<int>(floorf(h));
    const float f = h - static_cast<float>(i);
    const float p = hsv.z * (1.0f - hsv.y);
    const float q = hsv.z * (1.0f - hsv.y * f);
    const float t = hsv.z * (1.0f - hsv.y * (1.0f - f));
    switch (i) {
    case 0: return {hsv.z, t, p}; case 1: return {q, hsv.z, p};
    case 2: return {p, hsv.z, t}; case 3: return {p, q, hsv.z};
    case 4: return {t, p, hsv.z}; default: return {hsv.z, p, q};
    }
}

} // namespace nr::svm::detail
