#pragma once

#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/trigonometric.hpp>

// NoorRay uses OpenPBR's own official per-language interop layer
// (external/openpbr-bsdf/interop/openpbr_interop_cuda.h on CUDA, auto-selected
// by interop/openpbr_interop.h via __CUDACC__ detection — OPENPBR_USE_CUSTOM_INTEROP
// is intentionally left at its default of 0).
//
// The CUDA interop's vec2/vec3/vec4 aliases point at CUDA's plain float2/float3/float4
// structs, which lack the constructors/operators OpenPBR's code relies on (see the
// vendored README's "CUDA Backend: Vector Types" known limitation). This header
// supplies GLM-backed replacements instead, per the documented escape hatch:
// set OPENPBR_USE_CUSTOM_VEC_TYPES=1 and provide compatible vector types/math
// functions before including openpbr.h. Everything else (OPENPBR_INLINE_FUNCTION,
// OPENPBR_OUT/CONST_REF, mix(), saturate(), assertions, etc.) comes from the
// vendor's own openpbr_interop_cuda.h unmodified.
#define OPENPBR_USE_CUSTOM_VEC_TYPES 1

// Force the CUDA backend explicitly rather than relying on __CUDACC__
// auto-detection (per README: "Any backend can also be forced explicitly by
// setting the corresponding OPENPBR_LANGUAGE_TARGET_* flag").
#define OPENPBR_LANGUAGE_TARGET_CUDA 1

// GLM vector types as GLSL-compatible names
using glm::vec2;
using glm::vec3;
using glm::vec4;

// Math functions
using glm::abs;
using glm::acos;
using glm::asin;
using glm::atan;
using glm::ceil;
using glm::clamp;
using glm::cos;
using glm::exp;
using glm::floor;
using glm::fract;
using glm::log;
using glm::max;
using glm::min;
using glm::mix;
using glm::mod;
using glm::pow;
using glm::sin;
using glm::smoothstep;
using glm::sqrt;
using glm::tan;

// Geometric functions
using glm::cross;
using glm::dot;
using glm::length;
using glm::normalize;
using glm::reflect;
using glm::refract;
using glm::faceforward;

// Vector comparison (returns bool for GLM, which matches OpenPBR usage)
using glm::equal;
using glm::greaterThan;
using glm::greaterThanEqual;
using glm::lessThan;
using glm::lessThanEqual;
using glm::notEqual;

// Boolean reduction
using glm::all;
using glm::any;

// Everything else (OPENPBR_INLINE_FUNCTION, OPENPBR_OUT/CONST_REF,
// OPENPBR_UINT32/16, mix()/equal()/notEqual(), swizzle helpers,
// OPENPBR_MAKE_STRUCT_*, saturate(), assertions, and the specialization
// constant hook) comes from the vendor's own
// external/openpbr-bsdf/interop/openpbr_interop_cuda.h, included next by
// openpbr.h now that OPENPBR_LANGUAGE_TARGET_CUDA is set above.
