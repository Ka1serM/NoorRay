#include "Backend/CUDA/CudaDriverApi.h"


#include <dlfcn.h>

#include "Log.h"

namespace nr::cuda {

namespace {

using CuCtxGetCurrentFn = CUresult (*)(CUcontext*);

struct DriverApi
{
    void* library{};
    CuCtxGetCurrentFn ctxGetCurrent{};
    bool available{};
};

const DriverApi& driver()
{
    // Loaded once, on first use, and never unloaded: the CUDA runtime holds its
    // own reference to the same library for the lifetime of the process.
    static DriverApi api = [] {
        DriverApi loaded{};
        // The SONAME, not the "libcuda.so" development symlink - that one comes
        // from the CUDA toolkit rather than the driver and is usually absent on
        // an end-user machine.
        loaded.library = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (loaded.library == nullptr)
        {
            LOG_WARN("libcuda.so.1 not found (" << dlerror()
                << "); this machine has no NVIDIA driver, so raytracing is unavailable.");
            return loaded;
        }
        loaded.ctxGetCurrent =
            reinterpret_cast<CuCtxGetCurrentFn>(dlsym(loaded.library, "cuCtxGetCurrent"));
        loaded.available = loaded.ctxGetCurrent != nullptr;
        if (!loaded.available)
            LOG_WARN("libcuda.so.1 is missing cuCtxGetCurrent; raytracing is unavailable.");
        return loaded;
    }();
    return api;
}

}  // namespace

bool driverAvailable()
{
    return driver().available;
}

CUresult ctxGetCurrent(CUcontext* context)
{
    const DriverApi& api = driver();
    if (!api.available)
        return CUDA_ERROR_NOT_INITIALIZED;
    return api.ctxGetCurrent(context);
}

}  // namespace nr::cuda
