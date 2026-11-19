#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/CameraBuffers.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/Raster/GaussianSplats.h"
#include "NoorRay/Render/Raster/ToneMap.h"
#include "NoorRay/Render/Raster/VisibilityBuffer.h"
#include "NoorRay/Render/RendererSettings.h"
#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"
#include "NoorRay/Scene/Camera.h"

namespace NoorRay::Render {

//! The one renderer: the settings choose the passes of a frame, and every frame ends in the sRGB display image.
class Renderer {
public:
    //! Without a shared cache the renderer compiles its own pipelines; renderers on one device can share one.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height,
                                                        std::shared_ptr<PipelineCache> pipelines = nullptr);
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void setSettings(const RendererSettings& changed);
    void restartAccumulation() { restartPending = true; }

    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Scene::Camera& camera);

    const RendererSettings& getSettings() const { return settings; }
    const std::shared_ptr<PipelineCache>& getPipelines() const { return pipelines; }
    const NoorRHI::Image& getDisplay() const { return display; }
    const Targets& getTargets() const { return pathTracer.getTargets(); }
    std::uint32_t getWidth() const { return display.getWidth(); }
    std::uint32_t getHeight() const { return display.getHeight(); }
    std::uint32_t getSampleCount() const { return pathTracer.getAccumulatedFrames(); }
    //! True when accumulation reached the sample limit; later frames only refresh the display.
    bool isComplete() const;

private:
    NoorRHI::Device* device = nullptr;
    RendererSettings settings;
    bool restartPending = true;
    WavefrontPathTracer pathTracer;
    VisibilityBuffer visibility;
    GaussianSplats splats;
    //! The splat image of the last traced frame goes over the film, also after accumulation completes.
    bool splatted = false;
    ToneMap toneMap;
    Gpu::CameraBuffers camera;
    std::shared_ptr<PipelineCache> pipelines;
    NoorRHI::Image display;
    std::uint32_t frameIndex = 0;
};

}
