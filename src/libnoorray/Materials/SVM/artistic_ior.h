/* SPDX-License-Identifier: Apache-2.0
 *
 * MaterialX value-node evaluation. The structure keeps math beside its
 * opcode, while SvmEval.h only loads
 * operands and stores results.  Formula is mx_artistic_ior.glsl from the
 * MaterialX pbrlib standard library.
 */

#pragma once

#include <glm/common.hpp>
#include <glm/exponential.hpp>

#include "Backend/Host/Platform.h"

namespace nr::svm::detail
{

NR_GPU inline void svmArtisticIor(const glm::vec3 reflectivity,
    const glm::vec3 edgeColor, glm::vec3& ior, glm::vec3& extinction)
{
    const glm::vec3 r = glm::clamp(reflectivity, glm::vec3(0.0f), glm::vec3(0.99f));
    const glm::vec3 rSqrt = glm::sqrt(r);
    const glm::vec3 nMin = (glm::vec3(1.0f) - r) / (glm::vec3(1.0f) + r);
    const glm::vec3 nMax = (glm::vec3(1.0f) + rSqrt) / (glm::vec3(1.0f) - rSqrt);
    ior = glm::mix(nMax, nMin, edgeColor);

    const glm::vec3 np1 = ior + glm::vec3(1.0f);
    const glm::vec3 nm1 = ior - glm::vec3(1.0f);
    glm::vec3 k2 = (np1 * np1 * r - nm1 * nm1) / (glm::vec3(1.0f) - r);
    k2 = glm::max(k2, glm::vec3(0.0f));
    extinction = glm::sqrt(k2);
}

} // namespace nr::svm::detail
