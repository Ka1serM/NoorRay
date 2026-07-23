#include <cuda_fp16.h>
#include <optix_device.h>

#include "Raytracing/Path/PathIntegrator.h"

// OptiX modules cannot link against CUDA device code from libross. Compile
// the small optics routines used by camera ray generation into this module.
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupil.cpp"
#include "libross/imaging/cameralens/CameraLens.cpp"
#include "libross/imaging/cameralens/LensSurface.cpp"
#include "libross/imaging/cameralens/raytracing/sequential/SequentialRaytracer.cpp"
#include "libross/imaging/cameralens/raytracing/sequential/FromFilmToWorldRaytracer.cpp"

extern "C"
{
__constant__ KernelParams params;
}

extern "C" __global__ void __raygen__pathTrace()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    const uint32_t x = launchIndex.x;
    const uint32_t y = launchIndex.y;
    const uint32_t pixel = y * params.frame.width + x;
    PathIntegrator(params).renderSample(pixel, x, y);
}

#include "Kernels/Path/Aov.h"
#include "Kernels/Path/GaussianHit.h"
#include "Kernels/Path/GaussianProxyOverdraw.h"
