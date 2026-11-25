//! MathLib includes x86intrin.h, which Emscripten lacks: its SSE headers emulate the intrinsics with WebAssembly SIMD.
#pragma once
#include <immintrin.h>
