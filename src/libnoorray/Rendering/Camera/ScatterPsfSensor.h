#pragma once

#include "Rendering/Camera/Sensor.h"
#include "Backend/CUDA/rstd/UniquePtr.h"

#include <cuda_runtime_api.h>

#include "libross/imaging/interpolatedpsfgrid/InterpolatedPsfGrid.h"
#include "libross/foundation/parallel/GpuParallelFor.h"

#include <memory>
#include <string>

class ScatterPsfSensor : public Sensor::Type<ScatterPsfSensor, RectangularSensor> {
public:
    nr::rstd::unique_ptr<ross::InterpolatedPsfGrid> psfGrid;

    ScatterPsfSensor();
    explicit ScatterPsfSensor(const Sensor& other);
    std::string psfGridPath;
    std::string psfLoadStatus;
    std::unique_ptr<pfd::open_file> psfGridDialog;

    ~ScatterPsfSensor();
    bool renderUi(Sensor& owner);
    void freePsfGrid();
    void loadPsfGrid();
    uint32_t psfBinCount() const;

    NR_CPU_GPU glm::vec4 resolvePixel(uint32_t pixel, uint32_t width, const glm::vec4* accumulation) const
    {
        if (accumulation == nullptr)
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

        const glm::vec4 p = accumulation[pixel];
        glm::vec3 rgb(0.0f);
        if (p.w != 0.0f) {
            rgb.x = p.x / p.w;
            rgb.y = p.y / p.w;
            rgb.z = p.z / p.w;
        }
        (void)width;
        return glm::vec4(rgb, 1.0f);
    }

    template <typename WritePixel>
    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float sampleWeight, const SensorSampleContext& ctx,
        const WritePixel&) const
    {
        if (!psfGrid || ctx.accumulation == nullptr)
            return;
        glm::vec3 rgb = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        const uint32_t x = pixel % ctx.width;
        const uint32_t y = pixel / ctx.width;
        auto addSplat = [=](const ross::Vector2i& dest, float psfWeight) {
            glm::vec4& out = ctx.accumulation[static_cast<uint32_t>(dest.y) * ctx.width
                + static_cast<uint32_t>(dest.x)];
            sensorAtomicAdd(&out.x, rgb.x * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.y, rgb.y * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.z, rgb.z * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.w, sampleWeight * psfWeight);
        };
        psfGrid->splatPsfForPixel(ross::Vector2i(static_cast<int>(x), static_cast<int>(y)),
            wl[0] * 0.001f, addSplat);
    }
};

#if defined(__CUDACC__) && defined(NR_BUILD_SCATTER_PSF_RESOLVE)

cudaError_t launchScatterPsfResolveKernel(
    const ScatterPsfSensor* sensor, const glm::vec4* accumulation,
    cudaSurfaceObject_t output, const uint32_t width, const uint32_t height,
    const cudaStream_t stream)
{
    const cudaError_t syncResult = cudaStreamSynchronize(stream);
    if (syncResult != cudaSuccess)
        return syncResult;

    ross::parallelFor2dGpu(width, height,
        [=] ROSS_GPU(ross::Index2d index) {
            const uint32_t pixel = static_cast<uint32_t>(index.y) * width
                + static_cast<uint32_t>(index.x);
            const glm::vec4 value = sensor->resolvePixel(pixel, width, accumulation);
            surf2Dwrite(make_float4(value.x, value.y, value.z, value.w),
                output, index.x * sizeof(float4), index.y);
        });
    return cudaGetLastError();
}

#else

cudaError_t launchScatterPsfResolveKernel(
    const ScatterPsfSensor*, const glm::vec4*, cudaSurfaceObject_t,
    uint32_t, uint32_t, cudaStream_t);

#endif
