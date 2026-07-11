#pragma once

#include "Camera/Sensor.h"

#if !defined(NR_OPTIX_PTX_BUILD)
#include "libross/imaging/interpolatedpsfgrid/InterpolatedPsfGrid.h"
#endif

#ifndef NR_GPU_CODE
#include <memory>
#include <string>
#include "portable-file-dialogs.h"
#endif

class ScatterPsfSensor : public RectangularSensor {
public:
    ross::InterpolatedPsfGrid* psfGrid{};

#ifndef NR_GPU_CODE
    std::string psfGridPath;
    std::string psfLoadStatus;
    std::unique_ptr<pfd::open_file> psfGridDialog;

    ~ScatterPsfSensor();
    bool renderUi(Sensor& owner);
    void freePsfGrid();
    void loadPsfGrid();
    uint32_t psfBinCount() const;
#endif

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

    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float sampleWeight, const SensorSampleContext& ctx) const
    {
#if defined(NR_OPTIX_PTX_BUILD)
        (void)pixel; (void)L; (void)wl; (void)sampleWeight; (void)ctx;
#else
        if (psfGrid == nullptr || ctx.accumulation == nullptr)
            return;
        glm::vec3 rgb = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        const uint32_t x = pixel % ctx.width;
        const uint32_t y = pixel / ctx.width;
        auto addSplat = [=](const ross::Vector2i& dest, float psfWeight) {
            if (dest.x < 0 || dest.y < 0 ||
                dest.x >= static_cast<int>(ctx.width) || dest.y >= static_cast<int>(ctx.height))
                return;
            glm::vec4& out = ctx.accumulation[static_cast<uint32_t>(dest.y) * ctx.width
                + static_cast<uint32_t>(dest.x)];
            sensorAtomicAdd(&out.x, rgb.x * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.y, rgb.y * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.z, rgb.z * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.w, sampleWeight * psfWeight);
        };
        psfGrid->splatPsfForPixel(ross::Vector2i(static_cast<int>(x), static_cast<int>(y)),
            wl[0] * 0.001f, addSplat);
#endif
    }
};
