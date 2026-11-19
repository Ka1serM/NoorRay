#include "NoorRay/Render/Renderer.h"

namespace NoorRay::Render {

namespace {

constexpr Scene::Change restartingChanges = Scene::lightingChanges | Scene::Change::Camera;

float halton(std::uint32_t index, const std::uint32_t base) {
    float result = 0.0f;
    float fraction = 1.0f / static_cast<float>(base);
    for (++index; index > 0; index /= base, fraction /= static_cast<float>(base))
        result += fraction * static_cast<float>(index % base);
    return result;
}

//! The raster sample of a frame: the pixel centre without accumulation, else a Halton (2, 3) point, the same estimator as a jittered traced primary ray.
NoorRHI::float2 rasterJitter(const bool accumulate, const std::uint32_t frame) {
    if (!accumulate)
        return {0.5f, 0.5f};
    return {halton(frame, 2), halton(frame, 3)};
}

}

std::expected<void, Core::Error> Renderer::open(NoorRHI::Device& openedDevice, const std::uint32_t width, const std::uint32_t height,
                                                std::shared_ptr<PipelineCache> sharedPipelines) {
    device = &openedDevice;
    pipelines = sharedPipelines ? std::move(sharedPipelines) : std::make_shared<PipelineCache>();
    if (auto opened = pathTracer.open(*device, width, height, pipelines); !opened)
        return opened;
    return resize(width, height);
}

std::expected<void, Core::Error> Renderer::resize(const std::uint32_t width, const std::uint32_t height) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the renderer is not open");
    if (display && display.getWidth() == width && display.getHeight() == height)
        return {};
    if (auto resized = pathTracer.resize(width, height); !resized)
        return resized;
    if (auto resized = visibility.resize(*device, width, height); !resized)
        return resized;
    if (auto resized = splats.resize(*device, width, height); !resized)
        return resized;
    const NoorRHI::ImageDesc desc{
        .width = width,
        .height = height,
        .format = ToneMap::displayFormat,
        .usage = NoorRHI::ImageUsage::Sampled | NoorRHI::ImageUsage::ColorAttachment,
        .name = "NoorRay display",
    };
    restartPending = true;
    return Core::check(display.open(*device, desc));
}

void Renderer::setSettings(const RendererSettings& changed) {
    if (changed.changesImage(settings))
        restartPending = true;
    settings = changed;
}

bool Renderer::isComplete() const {
    return settings.accumulate && settings.maxSamples != 0 && pathTracer.getAccumulatedFrames() >= settings.maxSamples;
}

std::expected<void, Core::Error> Renderer::record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Scene::Camera& sceneCamera) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the renderer is not open");
    const RendererSettings frame = settings.validated(sceneCamera.hasDepthOfField() || !sceneCamera.isRasterizable());
    if ((scene.getLastChange() & restartingChanges) != Scene::Change::None)
        restartPending = true;
    if (restartPending) {
        pathTracer.restartAccumulation();
        restartPending = false;
    }
    if (auto updated = camera.update(*device, sceneCamera, display.getWidth(), display.getHeight()); !updated)
        return updated;
    if (!isComplete()) {
        const bool rgb = frame.shading == Shading::Raster;
        Shaders::RasterRecord raster{.rgb = rgb ? 1u : 0u};
        const WavefrontSettings wavefront = frame.getWavefrontSettings();
        splatted = wavefront.splatsGaussians && scene.getGaussianCount() != 0;
        if (frame.primaryVisibility == PrimaryVisibility::Raster) {
            const NoorRHI::float2 jitter = rasterJitter(frame.accumulate, frameIndex);
            if (auto drawn = visibility.record(*device, list, scene, camera.getRecord(), jitter); !drawn)
                return drawn;
            raster = visibility.getRecord(scene, jitter, rgb);
            if (splatted)
                if (auto drawn = splats.record(*device, list, *pipelines, scene, camera.getRecord(), visibility.getDepth(), jitter); !drawn)
                    return drawn;
        }
        if (auto traced = pathTracer.record(list, scene, camera.getRecord(), wavefront, raster); !traced)
            return traced;
        ++frameIndex;
    }
    return toneMap.record(*device, list, pathTracer.getTargets().getOutput(), display, frame.display, splatted ? &splats.getImage() : nullptr);
}

}
