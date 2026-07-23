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
    const uint32_t pixel = NR_GPU_OPTIX_LAUNCH_ID;
    if (pixel < params.frame.width * params.frame.height)
        PathIntegrator(params).renderSample(pixel);
}

#include "Kernels/Path/Aov.h"
#include "Kernels/Path/GaussianHit.h"
#include "Kernels/Path/GaussianProxyOverdraw.h"
