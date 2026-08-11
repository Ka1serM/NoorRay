#pragma once

#include <cuda.h>

// Runtime access to the CUDA driver API.
//
// libcuda.so.1 ships with the NVIDIA driver and is not redistributable, so it is
// present only on machines that have that driver installed. Linking it normally
// records a DT_NEEDED entry, which makes it a *load-time* requirement: the
// executable then refuses to start at all on an AMD or Intel machine, before any
// code can run to report the situation gracefully.
//
// Resolving the entry points through dlopen instead makes it a runtime-optional
// dependency. Nothing extra ships beside the executable - this removes a
// dependency rather than adding one - and a machine without the driver simply
// reports raytracing as unavailable.
namespace nr::cuda {

// True when libcuda.so.1 was found and the required entry points resolved.
// Cheap after the first call; the library is loaded at most once.
bool driverAvailable();

// cuCtxGetCurrent, or CUDA_ERROR_NOT_INITIALIZED when the driver is absent.
CUresult ctxGetCurrent(CUcontext* context);

}  // namespace nr::cuda
