#include "NoorRay/Render/Renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <utility>

#include "NoorRay/Shaders/Denoise/DenoiseRecords.h"
#include "NoorRay/Shaders/Display/DisplayRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

constexpr Scene::Change restartingChanges = Scene::lightingChanges | Scene::Change::Camera;
//! A realtime view converges after this long without a change, so that the redraws between the events of a slow camera move stay realtime.
constexpr std::chrono::milliseconds stillSettle{200};

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

std::uint32_t scaled(const std::uint32_t size, const float scale) {
    return std::max(static_cast<std::uint32_t>(std::lround(size * scale)), 1u);
}

//! Composite keeps the mean of the denoised frames and blend runs only when this is positive.
float getHistoryBlend(const RendererSettings& settings) {
    if (!settings.accumulate)
        return 0.0f;
    return settings.denoiseBlend;
}
}

std::expected<void, Core::Error> Renderer::open(NoorRHI::Device& openedDevice, const std::uint32_t width, const std::uint32_t height,
                                                std::shared_ptr<PipelineCache> sharedPipelines) {
    device = &openedDevice;
    pipelines = sharedPipelines ? std::move(sharedPipelines) : std::make_shared<PipelineCache>();
    if (auto opened = pathTracer.open(*device, pipelines); !opened)
        return opened;
    if (auto opened = Core::check(previousCamera.open(*device, 1)); !opened)
        return opened;
    return resize(width, height);
}

std::expected<void, Core::Error> Renderer::resize(const std::uint32_t width, const std::uint32_t height) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the renderer is not open");
    if (display && display.getWidth() == width && display.getHeight() == height)
        return {};
    const NoorRHI::ImageDesc desc{
        .width = width,
        .height = height,
        .format = ToneMap::displayFormat,
        .usage = NoorRHI::ImageUsage::Sampled | NoorRHI::ImageUsage::ColorAttachment,
        .name = "NoorRay display",
    };
    restartPending = true;
    historyPending = true;
    //! The film follows at the next record, at the render size of its settings, so a scaled render never opens a film of the display size.
    return Core::check(display.open(*device, desc));
}

std::expected<void, Core::Error> Renderer::resizeFilm(const RendererSettings& frame, const bool fog) {
    const bool upscales = frame.upscales();
    const std::uint32_t width = upscales ? scaled(display.getWidth(), frame.renderScale) : display.getWidth();
    const std::uint32_t height = upscales ? scaled(display.getHeight(), frame.renderScale) : display.getHeight();
    if (getRenderWidth() != width || getRenderHeight() != height) {
        restartPending = true;
        historyPending = true;
    }
    if (auto resized = pathTracer.resize(width, height); !resized)
        return resized;
    if (auto resized = visibility.resize(*device, width, height); !resized)
        return resized;
    if (auto resized = splats.resize(*device, width, height); !resized)
        return resized;
    if (frame.denoiser == Denoiser::Nrd && !denoisesFilm(frame))
        if (auto resized = relax.resize(*device, width, height, fog); !resized)
            return resized;
    if (denoisesFilm(frame) && relax.getWidth() != 0)
        relax = {};
    if (frame.upscaler == Upscaler::Fsr)
        if (auto resized = fsr.resize(*device, width, height, display.getWidth(), display.getHeight()); !resized)
            return resized;
    if (frame.denoiser == Denoiser::OpenDlssRr)
        if (auto resized = dlssRr.resize(*device, width, height, display.getWidth(), display.getHeight()); !resized) {
            dlssRrFailed = true;
            return resized;
        }
    return {};
}

const NoorRHI::Image* Renderer::getResolved(const RendererSettings& frame) const {
    if (frame.upscaler == Upscaler::Fsr)
        return &fsr.getOutput();
    if (frame.denoiser == Denoiser::OpenDlssRr)
        return &dlssRr.getOutput();
    return nullptr;
}

void Renderer::setSettings(const RendererSettings& changed) {
    if (changed.denoiser != settings.denoiser)
        dlssRrFailed = false;
    if (changed.changesImage(settings)) {
        restartPending = true;
        historyPending = true;
    }
    settings = changed;
}

std::string Renderer::getCompileNotice() const {
    switch (pathTracer.getCompilingReuse()) {
    case Reuse::Off:
        return {};
    case Reuse::Di:
        return "Compiling ReSTIR DI, rendering without ReSTIR until it is ready";
    case Reuse::Pt:
        return "Compiling ReSTIR PT, rendering without ReSTIR until it is ready";
    }
    return {};
}

bool Renderer::isComplete() const {
    return !restartPending && settings.maxSamples != 0 && getSampleCount() >= settings.maxSamples;
}

void Renderer::setProfiler(GpuProfiler* timing) {
    profiler = timing;
    pathTracer.setProfiler(timing);
}

std::expected<void, Core::Error> Renderer::recordUpscaledCopy(NoorRHI::CommandList& list, NoorRHI::Buffer<NoorRHI::float4>& destination) {
    const NoorRHI::Image* resolved = getResolved(frameSettings);
    if (!resolved || !*resolved || destination.getSize() < std::size_t{getWidth()} * getHeight())
        return Core::fail(Core::ErrorCode::InvalidState, "there is no resolved image of the display size");
    auto pipeline = pipelines->get(*device, *Shaders::readImage[0]);
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    const Targets& targets = pathTracer.getTargets();
    const Shaders::ReadImageArguments arguments{.image = resolved->getSampledHandle(),
                                                .width = getWidth(),
                                                .height = getHeight(),
                                                .destination = destination.getPtr(),
                                                .film = targets.getOutput().getPtr(),
                                                .filmWidth = targets.getWidth(),
                                                .filmHeight = targets.getHeight()};
    list.dispatch(**pipeline, {getWidth(), getHeight(), 1}, arguments);
    return {};
}

std::expected<void, Core::Error> Renderer::recordSharedCopy(NoorRHI::CommandList& list, const NoorRHI::Image& resolved) {
    auto pipeline = pipelines->get(*device, *Shaders::writeImage[0]);
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    const Targets& targets = pathTracer.getTargets();
    const Shaders::WriteImageArguments arguments{
        .image = resolved.getSampledHandle(),
        .width = getWidth(),
        .height = getHeight(),
        .destination = sharedOutput->getStorageHandle(),
        .film = targets.getOutput().getPtr(),
        .filmWidth = targets.getWidth(),
        .filmHeight = targets.getHeight(),
    };
    list.dispatch(**pipeline, {getWidth(), getHeight(), 1}, arguments);
    return {};
}

std::expected<void, Core::Error> Renderer::recordDenoise(NoorRHI::CommandList& list, const NoorRHI::Ptr<Shaders::WavefrontFrame> frame,
                                                         const RendererSettings& settings, const DenoisePass pass) {
    const bool film = denoisesFilm(settings);
    const bool denoises = settings.denoiser == Denoiser::Nrd && relax.getWidth() != 0;
    const bool fog = denoises && relax.hasFog();
    const bool upscales = settings.upscaler == Upscaler::Fsr;
    const bool reconstructs = settings.denoiser == Denoiser::OpenDlssRr;
    const auto guide = [reconstructs](const NoorRHI::Image& image) { return reconstructs ? image.getStorageHandle() : NoorRHI::StorageTextureHandle{}; };
    const NoorRHI::StorageTextureHandle color = upscales ? fsr.getColor().getStorageHandle() : guide(dlssRr.getColor());
    const Shaders::DenoiseArguments arguments{
        .frame = frame,
        .previousCamera = previousCamera.getPtr(),
        .filmSums = film ? pathTracer.getTargets().getFilmSums().getPtr() : NoorRHI::Ptr<float>{},
        .diffuse = denoises ? relax.getDiffuse().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .specular = denoises ? relax.getSpecular().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .normalRoughness = denoises ? relax.getNormalRoughness().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .viewZ = denoises ? relax.getViewZ().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .motion = denoises ? relax.getMotion().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .fog = fog ? relax.getFog().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .fogNormalRoughness = fog ? relax.getFogNormalRoughness().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .fogViewZ = fog ? relax.getFogViewZ().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .fogMotion = fog ? relax.getFogMotion().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .diffuseSh = denoises ? relax.getDiffuseSh().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .specularSh = denoises ? relax.getSpecularSh().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .upscaleDepth = upscales ? fsr.getDepth().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .upscaleMotion = upscales ? fsr.getMotion().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .color = color,
        .diffuseAlbedo = guide(dlssRr.getDiffuseAlbedo()),
        .specularAlbedo = guide(dlssRr.getSpecularAlbedo()),
        .normals = guide(dlssRr.getNormals()),
        .roughness = guide(dlssRr.getRoughness()),
        .depth = guide(dlssRr.getDepth()),
        .reconstructMotion = guide(dlssRr.getMotion()),
        .specularHitDistance = guide(dlssRr.getSpecularHitDistance()),
        .denoisedDiffuse = denoises ? relax.getDenoisedDiffuse().getSampledHandle() : NoorRHI::TextureHandle{},
        .denoisedSpecular = denoises ? relax.getDenoisedSpecular().getSampledHandle() : NoorRHI::TextureHandle{},
        .denoisedDiffuseSh = denoises ? relax.getDenoisedDiffuseSh().getSampledHandle() : NoorRHI::TextureHandle{},
        .denoisedSpecularSh = denoises ? relax.getDenoisedSpecularSh().getSampledHandle() : NoorRHI::TextureHandle{},
        .denoisedFog = fog ? relax.getDenoisedFog().getSampledHandle() : NoorRHI::TextureHandle{},
        .denoisedMean = denoises ? relax.getDenoisedMean().getStorageHandle() : NoorRHI::StorageTextureHandle{},
        .flags = (denoises ? Shaders::denoiseFlagRelax : 0u) | (fog ? Shaders::denoiseFlagFog : 0u) | (upscales ? Shaders::denoiseFlagUpscale : 0u)
                 | (reconstructs ? Shaders::denoiseFlagReconstruct : 0u) | (film ? Shaders::denoiseFlagFilm : 0u),
        .historyBlend = film ? 0.0f : getHistoryBlend(settings),
    };
    static constexpr std::array names{"denoisePrepare", "denoiseGather", "denoiseComposite", "denoiseBlend"};
    const std::array entries{Shaders::denoisePrepare[0], Shaders::denoiseGather[0], Shaders::denoiseComposite[0], Shaders::denoiseBlend[0]};
    const GpuScope scope(profiler, list, names[std::to_underlying(pass)]);
    auto pipeline = pipelines->get(*device, *entries[std::to_underlying(pass)]);
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    list.dispatch(**pipeline, {getRenderWidth(), getRenderHeight(), 1}, arguments);
    return {};
}

std::expected<void, Core::Error> Renderer::recordLightingClear(NoorRHI::CommandList& list) {
    auto pipeline = pipelines->get(*device, *Shaders::denoiseClear[0]);
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    const Targets& targets = pathTracer.getTargets();
    const Shaders::DenoiseClearArguments arguments{.lighting = targets.getLighting().getPtr(), .width = targets.getWidth(), .height = targets.getHeight()};
    list.dispatch(**pipeline, {targets.getWidth(), targets.getHeight(), 1}, arguments);
    return {};
}

std::expected<void, Core::Error> Renderer::recordFilmPasses(NoorRHI::CommandList& list, const NoorRHI::Ptr<Shaders::WavefrontFrame> frame) {
    if (auto resized = relax.resize(*device, getRenderWidth(), getRenderHeight(), volumes); !resized)
        return resized;
    const Shaders::CameraRecord& record = camera.getRecord();
    list.upload(previousCamera, std::span<const Shaders::CameraRecord>(&record, 1));
    list.pushLabel("Denoise film");
    auto denoised = recordDenoise(list, frame, frameSettings, DenoisePass::Prepare);
    if (denoised) {
        const GpuScope scope(profiler, list, "relax");
        denoised = relax.record(*device, list, *pipelines, DenoiseView::fromCamera(record, {0.0f, 0.0f}), true, RelaxInput::Film);
    }
    if (denoised)
        denoised = recordDenoise(list, frame, frameSettings, DenoisePass::Composite);
    list.popLabel();
    //! The images of RELAX go back to the device after the frame, so the accumulation of the next render does not hold them.
    relax = {};
    filmDenoised = denoised.has_value();
    return denoised;
}

std::expected<void, Core::Error> Renderer::recordFilmDenoise(NoorRHI::CommandList& list) {
    if (!device || filmDenoised || !denoisesFilm(frameSettings) || pathTracer.getAccumulatedFrames() == 0)
        return {};
    if (auto denoised = recordFilmPasses(list, pathTracer.getFrame()); !denoised)
        return denoised;
    const GpuScope scope(profiler, list, "toneMap");
    return toneMap.record(*device, list, pathTracer.getTargets().getOutput(), display, frameSettings.display, splatted ? &splats.getImage() : nullptr,
                          getResolved(frameSettings));
}

std::expected<void, Core::Error> Renderer::recordPost(NoorRHI::CommandList& list, const NoorRHI::Ptr<Shaders::WavefrontFrame> frame,
                                                      const RendererSettings& settings, const NoorRHI::float2 jitter) {
    const Shaders::CameraRecord& record = camera.getRecord();
    list.pushLabel("Denoise");
    if (settings.denoiser == Denoiser::Nrd) {
        const DenoiseView view = DenoiseView::fromCamera(record, {jitter.x - 0.5f, jitter.y - 0.5f});
        const GpuScope scope(profiler, list, "relax");
        if (auto denoised = relax.record(*device, list, *pipelines, view, historyPending || !settings.temporal, RelaxInput::Frame); !denoised) {
            list.popLabel();
            return denoised;
        }
    }
    auto composited = recordDenoise(list, frame, settings, DenoisePass::Composite);
    //! Composite leaves the colour to blend when it accumulates the mean of the denoised frames.
    if (composited && settings.denoiser == Denoiser::Nrd && getHistoryBlend(settings) > 0.0f)
        composited = recordDenoise(list, frame, settings, DenoisePass::Blend);
    list.popLabel();
    if (!composited || !settings.upscales())
        return composited;
    const auto now = std::chrono::steady_clock::now();
    const float frameTime = lastFrame == std::chrono::steady_clock::time_point{}
                                ? 16.6f
                                : std::clamp(std::chrono::duration<float, std::milli>(now - lastFrame).count(), 1.0f, 1000.0f);
    lastFrame = now;
    if (settings.denoiser == Denoiser::OpenDlssRr) {
        const DlssRrFrame reconstruct{
            .jitter = jitter,
            .view = DenoiseView::fromCamera(record, {jitter.x - 0.5f, jitter.y - 0.5f}),
            .frameTime = frameTime,
            .reset = historyPending,
        };
        const GpuScope scope(profiler, list, "dlssRr");
        std::expected<void, Core::Error> recorded = dlssRr.record(*device, list, reconstruct);
        if (!recorded)
            dlssRrFailed = true;
        return recorded;
    }
    const float focal = std::max(record.projection.x, 1e-6f);
    const UpscaleFrame upscale{
        .jitter = jitter,
        .near = std::max(record.clip.x, 1e-4f),
        .verticalFov = 2.0f * std::atan(0.5f * record.sensor.y / focal),
        .frameTime = frameTime,
        .sharpness = settings.sharpness,
        .reset = historyPending,
    };
    const GpuScope scope(profiler, list, "fsr");
    return fsr.record(*device, list, *pipelines, upscale);
}

std::expected<void, Core::Error> Renderer::record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Scene::Camera& sceneCamera) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the renderer is not open");
    if (auto set = pipelines->setGraphCodes(*device, scene.getMaterials().getGraphCodes(), scene.getMaterials().getAlphaGraphCodes()); !set)
        return set;
    pipelines->setVolumeGraphCode(scene.getVolumes().getGraphCode());
    //! Only changes from outside count here; the film resize of a switch to or from the accumulation restarts it too.
    const bool changed = restartPending || (!accumulatesChanges && (scene.getLastChange() & restartingChanges) != Scene::Change::None);
    const auto now = std::chrono::steady_clock::now();
    if (changed || lastChange == std::chrono::steady_clock::time_point{})
        lastChange = now;
    const bool converges = !settings.accumulate && settings.convergesWhenStill && settings.shading == Shading::PathTraced && now - lastChange >= stillSettle;
    //! The camera does not move while the view converges, so the ReSTIR reservoirs and the FSR history of the last realtime frame still fit after it.
    if (std::exchange(converging, converges) != converges)
        restartPending = true;
    RendererSettings wanted = settings;
    if (wanted.denoiser == Denoiser::OpenDlssRr && dlssRrFailed)
        wanted.denoiser = Denoiser::Nrd;
    if (converging) {
        wanted.accumulate = true;
        //! DLSS-RR does not accumulate, so RELAX denoises the converging film.
        if (wanted.denoiser == Denoiser::OpenDlssRr)
            wanted.denoiser = Denoiser::Nrd;
    }
    const RendererSettings frame = wanted.validated(sceneCamera.hasDepthOfField() || !sceneCamera.isRasterizable(), sceneCamera.isPanoramic());
    frameSettings = frame;
    volumes = scene.getContent().hasVolumes;
    if (auto resized = resizeFilm(frame, scene.getContent().hasVolumes); !resized)
        return resized;
    if (changed)
        restartPending = true;
    if (restartPending) {
        pathTracer.restartAccumulation();
        realtimeFrames = 0;
        restartPending = false;
    }
    if (auto updated = camera.update(*device, sceneCamera, getRenderWidth(), getRenderHeight()); !updated)
        return updated;
    const bool denoises = frame.denoiser != Denoiser::Off;
    const bool resolves = frame.upscales();
    sharedWritten = false;
    if (sharedOutput && (sharedOutput->getWidth() != getWidth() || sharedOutput->getHeight() != getHeight()))
        return Core::fail(Core::ErrorCode::InvalidArgument, "the shared output is not at display size");
    if (!isComplete()) {
        const bool rgb = frame.shading == Shading::Raster;
        Shaders::RasterRecord raster{.rgb = rgb ? 1u : 0u};
        const WavefrontSettings wavefront = frame.getWavefrontSettings();
        splatted = wavefront.splatsGaussians && scene.getGaussianCount() != 0;
        //! The denoiser and the upscaler need one known sample position per frame: the FSR or DLSS-RR sequence, else the raster sequence.
        const NoorRHI::float2 jitter = frame.upscaler == Upscaler::Fsr          ? fsr.getJitter(frameIndex)
                                       : frame.denoiser == Denoiser::OpenDlssRr ? dlssRr.getJitter(frameIndex)
                                                                                : rasterJitter(frame.accumulate, frameIndex);
        if (frame.primaryVisibility == PrimaryVisibility::Raster) {
            if (auto drawn = visibility.record(*device, list, scene, camera.getRecord(), jitter); !drawn)
                return drawn;
            raster = visibility.getRecord(scene, jitter, rgb);
            if (splatted)
                if (auto drawn = splats.record(*device, list, *pipelines, scene, camera.getRecord(), visibility.getDepth(), jitter); !drawn)
                    return drawn;
        }
        FilmHooks hooks;
        const bool film = denoisesFilm(frame);
        if (film) {
            if (auto opened = pathTracer.openFilmSums(); !opened)
                return opened;
            if (pathTracer.getAccumulatedFrames() == 0)
                if (auto cleared = recordLightingClear(list); !cleared)
                    return cleared;
            hooks.before = [&](NoorRHI::CommandList& hooked, const NoorRHI::Ptr<Shaders::WavefrontFrame> traced) {
                return recordDenoise(hooked, traced, frame, DenoisePass::Gather);
            };
            if (frame.maxSamples != 0 && pathTracer.getAccumulatedFrames() + 1 >= frame.maxSamples)
                hooks.after = [&](NoorRHI::CommandList& hooked, const NoorRHI::Ptr<Shaders::WavefrontFrame> traced) {
                    return recordFilmPasses(hooked, traced);
                };
        } else if (denoises || resolves) {
            const Shaders::CameraRecord& previous = hasLastCamera ? lastCamera : camera.getRecord();
            list.upload(previousCamera, std::span<const Shaders::CameraRecord>(&previous, 1));
            hooks.jitter = jitter;
            hooks.before = [&](NoorRHI::CommandList& hooked, const NoorRHI::Ptr<Shaders::WavefrontFrame> traced) {
                return recordDenoise(hooked, traced, frame, DenoisePass::Prepare);
            };
            hooks.after = [&](NoorRHI::CommandList& hooked, const NoorRHI::Ptr<Shaders::WavefrontFrame> traced) {
                return recordPost(hooked, traced, frame, jitter);
            };
        }
        //! Without a resolve the film is at display size, so its passes write the shared output directly.
        pathTracer.setSharedOutput(sharedOutput && !resolves ? sharedOutput->getStorageHandle() : NoorRHI::StorageTextureHandle{});
        filmDenoised = false;
        if (auto traced = pathTracer.record(list, scene, camera.getRecord(), wavefront, raster, hooks); !traced)
            return traced;
        if (sharedOutput && resolves)
            if (auto copied = recordSharedCopy(list, *getResolved(frame)); !copied)
                return copied;
        sharedWritten = sharedOutput != nullptr;
        lastCamera = camera.getRecord();
        hasLastCamera = true;
        if (denoises || resolves)
            historyPending = false;
        ++frameIndex;
        ++realtimeFrames;
    }
    const GpuScope scope(profiler, list, "toneMap");
    return toneMap.record(*device, list, pathTracer.getTargets().getOutput(), display, frame.display, splatted ? &splats.getImage() : nullptr,
                          getResolved(frame));
}

}
