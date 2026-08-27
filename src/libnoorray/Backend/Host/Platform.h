#pragma once

// Backend-neutral host/shader annotations. Slang supplies its own attributes
// when compiling device code; ordinary C++ builds intentionally expand these
// to nothing and do not include a graphics API header.
#define NR_CPU_GPU
#define NR_GPU
#define NR_GPU_KERNEL
