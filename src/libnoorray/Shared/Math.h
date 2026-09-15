#pragma once

// Names shared by the host ABI and Slang. C++ uses GLM's tightly packed
// vector types; Slang already provides the corresponding built-in types.
#ifdef __cplusplus
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using float4x4 = glm::mat4;
using half = std::uint16_t;
using uint = std::uint32_t;
#endif
