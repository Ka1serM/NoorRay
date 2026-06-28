#pragma once

#include <cstdint>

#if defined(NR_GPU_CODE)
#include <cuda.h>
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using glm::ivec2;
using glm::mat4;
using glm::quat;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::angleAxis;
using glm::cross;
using glm::dot;
using glm::inverse;
using glm::length;
using glm::mat4_cast;
using glm::normalize;
using glm::perspective;
using glm::quat_cast;
using glm::radians;
using glm::transpose;

static constexpr int MaxRealisticLensElements = 64;
static constexpr int MaxRealisticExitPupilBounds = 64;

struct LightGpu
{
    vec3 position{};
    int type{};
    vec3 direction{};
    int _pad0{};
    vec3 color{};
    float intensity{};
    float range{};
    float innerConeAngle{};
    float outerConeAngle{};
    float sourceRadius{};
    float softSourceRadius{};
    float sourceLength{};
    float sourceWidth{};
    float sourceHeight{};
    float lightSourceAngle{};
    float lightSourceSoftAngle{};
    float lightFalloffExponent{};
    int useInverseSquaredFalloff{};
    int _pad1{};
    int _pad2{};
    int _pad3{};
    int _pad4{};
};
