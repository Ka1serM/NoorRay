#pragma once

#include "Math.h"

// The records use Slang's spelling directly. C++ gets only the aliases needed
// for vector types from Math.h. Device pointers are the one exception: the
// host stores their GPU address, while Slang stores a typed pointer.
#ifdef __cplusplus
#include <cstdint>
#define GpuPtr(type) uint64_t
#else
#define GpuPtr(type) type*
#endif
