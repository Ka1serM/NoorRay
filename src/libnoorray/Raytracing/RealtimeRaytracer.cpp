#include "RealtimeRaytracer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>

#include <glm/geometric.hpp>

#include "Logging/Log.h"
#include "Realtime/ShaderLoading.h"
#include "Shared/RealtimeArgs.h"

namespace
{
constexpr const char* lightingSpv = "RealtimeRaytracer/RealtimeRaytracer.spv";
constexpr const char* layeredLightingSpv = "RealtimeRaytracer/RealtimeRaytracerLayered.spv";
constexpr const char* gBufferSpv = "RealtimeRaytracer/RealtimeGBuffer.spv";
constexpr const char* layeredGBufferSpv = "RealtimeRaytracer/RealtimeGBufferLayered.spv";
constexpr const char* missSpv = "RealtimeRaytracer/RealtimeMiss.spv";
constexpr const char* shadowMissSpv = "RealtimeRaytracer/RealtimeShadowMiss.spv";
constexpr const char* defaultMaterialHitSpv = "RealtimeRaytracer/DefaultMaterialHit.spv";
constexpr const char* compositeSpv = "RealtimeRaytracer/RealtimeComposite.spv";
constexpr const char* layeredCompositeSpv = "RealtimeRaytracer/RealtimeCompositeLayered.spv";
constexpr const char* outputAovsSpv = "RealtimeRaytracer/RealtimeOutputAovs.spv";
constexpr const char* pickSpv = "RealtimeRaytracer/RealtimePick.spv";

// Matches RealtimeComposite.slang and RealtimeOutputAovs.slang.
constexpr uint32_t CompositeGroupSize = 8u;
constexpr float NearPlane = 1.0f;

// Every realtime stage agrees on this; the surface payload is the largest.
constexpr noorrhi::RayTracingInterface TraceInterface{sizeof(nr::graphics::RealtimeHitPayload)};

// Hit groups of the linked pipeline, counted library by library. The pass
// library's come first; each material then adds MaterialGroupCount.
constexpr uint32_t DefaultMaterialGroup = 0u;
constexpr uint32_t DefaultFilteredShadowGroup = 1u;
// No shaders: shadow rays pass opaque materials' records without any work,
// and Gaussian records, which realtime rays never reach, need some group.
constexpr uint32_t EmptyGroup = 2u;
constexpr uint32_t FirstMaterialGroup = 3u;
enum MaterialGroup : uint32_t { TransparentSurface, OpaqueSurface, Shadow, MaterialGroupCount };
constexpr uint32_t NoMaterialGroups = ~0u;

// Runs work on its own thread and sets `done` as it finishes, before `onDone`
// wakes the host.
template <class Work>
auto runInBackground(std::atomic<bool>& done, const std::function<void()>& onDone, Work work)
{
    return std::async(std::launch::async, [&done, &onDone, work = std::move(work)]() mutable {
        const auto finished = [&done, &onDone] {
            done = true;
            if (onDone)
                onDone();
        };
        try {
            auto result = work();
            finished();
            return result;
        } catch (...) {
            // get() rethrows it on the thread that links.
            finished();
            throw;
        }
    });
}

noorrhi::RayTracingLibrary buildPassLibrary(noorrhi::Device& device,
    std::vector<noorrhi::Shader> raygens)
{
    return device.ray_tracing_library({std::move(raygens),
        {loadShader(device, missSpv), loadShader(device, shadowMissSpv)},
        {loadShader(device, defaultMaterialHitSpv), noorrhi::Shader{}, noorrhi::Shader{}},
        {loadShader(device, defaultMaterialHitSpv, "anyHit"),
            loadShader(device, defaultMaterialHitSpv, "shadowAnyHit"), noorrhi::Shader{}},
        {}}, TraceInterface);
}

long long millisecondsSince(const std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
}

std::array<float, 16> multiplyColumnMajor(const std::array<float, 16>& a,
    const std::array<float, 16>& b)
{
    std::array<float, 16> result{};
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                result[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    return result;
}

// NRD matrices are column-major with column vectors, in a left-handed view
// space: x right, y up, z forward (view Z grows away from the camera).
std::array<float, 16> worldToViewMatrix(const nr::graphics::Camera& camera)
{
    const float* m = camera.cameraToWorld;
    const glm::vec3 position{m[3], m[7], m[11]};
    const glm::vec3 right = glm::normalize(glm::vec3{m[0], m[4], m[8]});
    const glm::vec3 up = glm::normalize(glm::vec3{m[1], m[5], m[9]});
    const glm::vec3 forward = -glm::normalize(glm::vec3{m[2], m[6], m[10]});
    const glm::vec3 rows[3] = {right, up, forward};
    std::array<float, 16> matrix{};
    for (int row = 0; row < 3; ++row) {
        matrix[0 * 4 + row] = rows[row].x;
        matrix[1 * 4 + row] = rows[row].y;
        matrix[2 * 4 + row] = rows[row].z;
        matrix[3 * 4 + row] = -glm::dot(rows[row], position);
    }
    matrix[15] = 1.0f;
    return matrix;
}

// Mirrors generateCameraRay(): film u runs left to right and image row 0 is
// the bottom of the view unless the sensor origin is upper-left. Clip Y is up
// with image row 0 at clip Y = +1, as NRD samples its inputs.
std::array<float, 16> viewToClipMatrix(const nr::graphics::Camera& camera)
{
    const float sensorWidth = std::max(camera.sensorWidthMm, 0.001f);
    const float sensorHeight = std::max(camera.sensorHeightMm, 0.001f);
    const float focalLength = std::max(camera.focalLengthMm, 0.001f);
    const float ySign = camera.sensorOrigin != 0u ? 1.0f : -1.0f;
    std::array<float, 16> matrix{};
    // Perspective with an infinite far plane.
    matrix[0] = 2.0f * focalLength / sensorWidth;
    matrix[5] = ySign * 2.0f * focalLength / sensorHeight;
    matrix[10] = 1.0f;
    matrix[11] = 1.0f;
    matrix[14] = -NearPlane;
    return matrix;
}
}

RealtimeRaytracer::RealtimeRaytracer(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory,
    std::function<void()> onMaterialShadersCompiled)
    : common(device, width, height, exportColorMemory,
        RaytracerResources::FullOutputAovs::Omitted, true)
    , lightingRaygen(loadShader(device, lightingSpv))
    , layeredLightingRaygen(loadShader(device, layeredLightingSpv))
    , materialSampler(device.sampler({}))
    , gBufferRaygen(loadShader(device, gBufferSpv))
    , layeredGBufferRaygen(loadShader(device, layeredGBufferSpv))
    , restir(device)
    , denoiser(device)
    , upscaler(device)
    , compositePipeline(device.compute(loadShader(device, compositeSpv)))
    , layeredCompositePipeline(device.compute(loadShader(device, layeredCompositeSpv)))
    , outputAovsPipeline(device.compute(loadShader(device, outputAovsSpv)))
    , pickPipeline(device.compute(loadShader(device, pickSpv)))
    , pickPosition(device.buffer<noorrhi::float4>(1))
    , args(std::make_unique<nr::graphics::RealtimeArgs>())
    , layerArgs(std::make_unique<nr::graphics::RealtimeArgs>())
    , nextMaterialGroup(FirstMaterialGroup)
    , onMaterialShadersCompiled(std::move(onMaterialShadersCompiled))
{
    RaytracerResources::Callbacks callbacks;
    callbacks.imageAllocationChanged = [this] { onImageAllocationChanged(); };
    callbacks.renderSettingsApplied = [this](const RenderSettings& settings) {
        onRenderSettingsApplied(settings);
    };
    callbacks.lightsUploaded = [this] { onLightsUploaded(); };
    callbacks.materialShadersChanged = [this](const std::span<const MaterialHitShaders> shaders) {
        onMaterialShadersChanged(shaders);
    };
    callbacks.hitRecordsChanged = [this](const std::span<const HitRecord> records) {
        onHitRecordsChanged(records);
    };
    common.setCallbacks(std::move(callbacks));
    const auto started = std::chrono::steady_clock::now();
    std::vector<noorrhi::Shader> raygens{lightingRaygen, layeredLightingRaygen,
        gBufferRaygen, layeredGBufferRaygen};
    std::ranges::copy(restir.raygens(), std::back_inserter(raygens));
    passLibrary = buildPassLibrary(device, std::move(raygens));
    NR_LOG_INFO("Compiled the ray-tracing pass library in " << millisecondsSince(started) << " ms");
    linkTracePipeline();
    ensureResources();
}

RealtimeRaytracer::~RealtimeRaytracer() = default;

Extent RealtimeRaytracer::outputExtent() const
{
    return {common.width(), common.height()};
}

Extent RealtimeRaytracer::renderExtent() const
{
    return upscaler.renderExtent(outputExtent());
}

Extent RealtimeRaytracer::lightingExtent() const
{
    return lightingExtent(renderExtent());
}

Extent RealtimeRaytracer::lightingExtent(const Extent render) const
{
    if (lightingResolution == LightingResolution::ThreeQuarter)
        return {render.width / 4u * 3u + divideRoundingUp(render.width % 4u * 3u, 4u),
            render.height / 4u * 3u + divideRoundingUp(render.height % 4u * 3u, 4u)};
    const uint32_t scale = static_cast<uint32_t>(lightingScale(lightingResolution));
    return {divideRoundingUp(render.width, scale), divideRoundingUp(render.height, scale)};
}

RealtimeRaytracer::ResourceLayout RealtimeRaytracer::requiredLayout() const
{
    // The render targets are allocated at the render size the upscaler ratio
    // asks for in the output allocation; a new ratio replaces them.
    const Extent output{imageWidth(), imageHeight()};
    const Extent render = upscaler.renderExtent(output);
    ResourceLayout layout;
    layout.targets.render = render;
    layout.targets.lighting = lightingExtent(render);
    layout.targets.sharedGuides = lightingScale(lightingResolution) == 1.0f;
    layout.targets.layers = transparentMaterials;
    layout.targets.sphericalHarmonics = denoiserMode != DenoiserMode::Off;
    layout.targets.upscaled = upscaler.mode() != UpscalerMode::Off;
    layout.output = output;
    layout.denoiser = denoiserMode;
    return layout;
}

uint32_t RealtimeRaytracer::traceWidth() const
{
    return renderExtent().width;
}

uint32_t RealtimeRaytracer::traceHeight() const
{
    return renderExtent().height;
}

namespace
{
uint32_t outputToRenderCoordinate(const uint32_t outputPixel, const uint32_t outputSize,
    const uint32_t renderSize, const float jitter)
{
    if (!outputSize || !renderSize) return 0u;
    const float source = (static_cast<float>(outputPixel) + 0.5f)
        * static_cast<float>(renderSize) / static_cast<float>(outputSize) - 0.5f - jitter;
    return static_cast<uint32_t>(std::clamp(std::lround(source), 0l,
        static_cast<long>(renderSize - 1u)));
}
}

std::uint32_t RealtimeRaytracer::readCryptomatteAtOutput(const uint32_t x, const uint32_t y)
{
    // The image on screen is the latest frame's, whose rectangles `args` holds.
    const Extent output{args->view.outputWidth, args->view.outputHeight};
    const Extent render{args->frame.width, args->frame.height};
    if (x >= output.width || y >= output.height || !render.width || !render.height) return ~0u;
    const uint32_t sx = outputToRenderCoordinate(x, output.width, render.width, previousJitter[0]);
    const uint32_t sy = outputToRenderCoordinate(y, output.height, render.height, previousJitter[1]);
    std::uint32_t id = ~0u;
    targets->cryptomatteImage().download_region(sx, sy, 1, 1,
        std::as_writable_bytes(std::span(&id, 1)));
    return id;
}

noorrhi::float4 RealtimeRaytracer::readPositionAtOutput(const uint32_t x, const uint32_t y)
{
    const Extent output{args->view.outputWidth, args->view.outputHeight};
    const Extent render{args->frame.width, args->frame.height};
    noorrhi::float4 position{};
    if (x >= output.width || y >= output.height || !render.width || !render.height) return position;
    const uint32_t sx = outputToRenderCoordinate(x, output.width, render.width, previousJitter[0]);
    const uint32_t sy = outputToRenderCoordinate(y, output.height, render.height, previousJitter[1]);
    const noorrhi::StagedArguments staged = common.device().stage(*args);
    common.device().label("Pick Position", [&] {
        pickPipeline.launch({1, 1, 1},
            nr::graphics::RealtimePickRoot{staged.address(), pickPosition.ptr().address, sx, sy});
    });
    pickPosition.download(std::span(&position, 1));
    return position;
}

void RealtimeRaytracer::onImageAllocationChanged()
{
    ensureResources();
}

void RealtimeRaytracer::onLightsUploaded()
{
    restir.uploadLights(common.pointLightRecords(), common.spotLightRecords(),
        common.rectLightRecords(), common.directionalLightRecords());
}

void RealtimeRaytracer::onRenderSettingsApplied(const RenderSettings& settings)
{
    lightingResolution = settings.lightingResolution;
    denoiserMode = settings.denoiserMode;
    upscaler.setMode(settings.upscalerMode);
}

void RealtimeRaytracer::onMaterialShadersChanged(const std::span<const MaterialHitShaders> shaders)
{
    materialShaders.assign(shaders.begin(), shaders.end());
    compilePendingMaterialShaders();
}

void RealtimeRaytracer::compilePendingMaterialShaders()
{
    // The driver compiles one library at a time across every core, so one
    // library per publication costs the least.
    if (!compilingMaterialLibraries.empty() || requestedMaterialShaderCount >= materialShaders.size())
        return;
    const std::size_t begin = requestedMaterialShaderCount;
    const std::size_t end = materialShaders.size();

    noorrhi::RayTracingPipelineDesc desc;
    // Every imported material is new code; caching it only grows pipeline.cache.
    desc.use_pipeline_cache = false;
    for (std::size_t i = begin; i < end; ++i) {
        const MaterialHitShaders& material = materialShaders[i];
        desc.closest_hit.insert(desc.closest_hit.end(),
            {material.closestHit, material.closestHit, noorrhi::Shader{}});
        desc.any_hit.insert(desc.any_hit.end(),
            {material.anyHit, noorrhi::Shader{}, material.shadowAnyHit});
    }
    compilingMaterialLibraries.emplace_back();
    MaterialLibraryBatch& batch = compilingMaterialLibraries.back();
    batch.shaderCount = end - begin;
    batch.library = runInBackground(batch.done, onMaterialShadersCompiled,
        [this, desc = std::move(desc), count = batch.shaderCount]() mutable {
            const auto started = std::chrono::steady_clock::now();
            noorrhi::RayTracingLibrary library = common.device().ray_tracing_library(desc, TraceInterface);
            NR_LOG_INFO("Compiled a ray-tracing library of " << count << " material shaders in "
                << millisecondsSince(started) << " ms");
            return library;
        });
    requestedMaterialShaderCount = end;
}

void RealtimeRaytracer::addCompiledBatch(MaterialLibraryBatch& batch)
{
    const std::size_t first = compiledMaterialShaderCount;
    try {
        noorrhi::RayTracingLibrary library = batch.library.get();
        if (library) {
            materialLibraries.push_back(std::move(library));
            for (std::size_t i = 0; i < batch.shaderCount; ++i) {
                materialGroupBases.push_back(nextMaterialGroup);
                nextMaterialGroup += MaterialGroupCount;
            }
        } else {
            materialGroupBases.insert(materialGroupBases.end(), batch.shaderCount, NoMaterialGroups);
        }
    } catch (const std::exception& error) {
        NR_LOG_WARN("Material shader batch rejected; using default material: " << error.what());
        materialGroupBases.insert(materialGroupBases.end(), batch.shaderCount, NoMaterialGroups);
    }
    compiledMaterialShaderCount = first + batch.shaderCount;
}

void RealtimeRaytracer::startPipelineLink()
{
    requestedLibraryCount = materialLibraries.size();
    const std::size_t shaderCount = compiledMaterialShaderCount;
    std::vector<noorrhi::RayTracingLibrary> libraries{passLibrary};
    libraries.insert(libraries.end(), materialLibraries.begin(), materialLibraries.end());
    pipelineLink.emplace();
    PipelineLink& link = *pipelineLink;
    link.shaderCount = shaderCount;
    link.pipeline = runInBackground(link.done, onMaterialShadersCompiled,
        [this, libraries = std::move(libraries)]() mutable {
            const auto started = std::chrono::steady_clock::now();
            try {
                noorrhi::RayTracingPipeline pipeline = common.device().ray_tracing(
                    std::span<const noorrhi::RayTracingLibrary>(libraries), {});
                NR_LOG_INFO("Linked " << libraries.size() << " ray-tracing libraries in "
                    << millisecondsSince(started) << " ms");
                return pipeline;
            } catch (const std::exception& error) {
                NR_LOG_WARN("The driver rejected linking the material shaders; they render "
                    "with the default material: " << error.what());
                return noorrhi::RayTracingPipeline{};
            }
        });
}

bool RealtimeRaytracer::adoptLinkedPipeline()
{
    if (!pipelineLink || !pipelineLink->done)
        return false;
    noorrhi::RayTracingPipeline pipeline = pipelineLink->pipeline.get();
    const std::size_t shaderCount = pipelineLink->shaderCount;
    pipelineLink.reset();
    if (!pipeline)
        return false;
    linkedMaterialShaderCount = shaderCount;
    assignHitGroups();
    tracePipeline = common.device().ray_tracing(pipeline, hitGroups);
    return true;
}

bool RealtimeRaytracer::linkCompiledMaterialShaders()
{
    while (!compilingMaterialLibraries.empty() && compilingMaterialLibraries.front().done) {
        addCompiledBatch(compilingMaterialLibraries.front());
        compilingMaterialLibraries.pop_front();
    }
    compilePendingMaterialShaders();
    const bool adopted = adoptLinkedPipeline();
    if (!pipelineLink && compilingMaterialLibraries.empty()
        && materialLibraries.size() > requestedLibraryCount)
        startPipelineLink();
    return adopted;
}

MaterialShaderStage RealtimeRaytracer::materialShaderStage() const
{
    if (pipelineLink)
        return MaterialShaderStage::LinkingPipeline;
    if (!compilingMaterialLibraries.empty() || requestedMaterialShaderCount < materialShaders.size())
        return MaterialShaderStage::CompilingLibraries;
    return MaterialShaderStage::Idle;
}

void RealtimeRaytracer::waitForMaterialShaders()
{
    while (!compilingMaterialLibraries.empty() || pipelineLink) {
        if (!compilingMaterialLibraries.empty())
            compilingMaterialLibraries.front().library.wait();
        else
            pipelineLink->pipeline.wait();
        linkCompiledMaterialShaders();
    }
}

void RealtimeRaytracer::onHitRecordsChanged(const std::span<const HitRecord> records)
{
    hitRecords.assign(records.begin(), records.end());
    assignHitGroups();
    const auto started = std::chrono::steady_clock::now();
    linkTracePipeline();
    NR_LOG_INFO("Linked the ray-tracing pipeline for " << records.size()
        << " hit records in " << millisecondsSince(started) << " ms");
}

void RealtimeRaytracer::assignHitGroups()
{
    hitGroups.clear();
    transparentMaterials = false;
    for (const HitRecord& record : hitRecords) {
        const bool compiled = record.kind == HitRecord::Kind::Section
            && record.materialShaders < linkedMaterialShaderCount
            && record.materialShaders < materialGroupBases.size()
            && materialGroupBases[record.materialShaders] != NoMaterialGroups;
        const uint32_t materialGroups = compiled ? materialGroupBases[record.materialShaders] : 0u;
        transparentMaterials |= compiled && record.transparent;
        if (record.rayType == nr::graphics::RealtimeRayTypeSurface)
            hitGroups.push_back(!compiled ? (record.kind == HitRecord::Kind::Section
                    ? DefaultMaterialGroup : EmptyGroup)
                : materialGroups + ((record.transparent || record.needsFacingTest)
                    ? TransparentSurface : OpaqueSurface));
        else
            hitGroups.push_back(compiled && (record.transparent || record.shadowFiltered
                    || record.needsFacingTest) ? materialGroups + Shadow
                : !compiled && record.kind == HitRecord::Kind::Section
                    && (record.shadowFiltered || record.needsFacingTest)
                    ? DefaultFilteredShadowGroup : EmptyGroup);
    }
}

void RealtimeRaytracer::linkTracePipeline()
{
    if (tracePipeline) {
        tracePipeline = common.device().ray_tracing(tracePipeline, hitGroups);
        return;
    }
    tracePipeline = common.device().ray_tracing(
        std::span<const noorrhi::RayTracingLibrary>(&passLibrary, 1), hitGroups);
}

bool RealtimeRaytracer::prepareFrameResources()
{
    const bool replaced = common.prepareFrameResources();
    ensureResources();
    return replaced;
}

void RealtimeRaytracer::render(const uint32_t frameIndex, const uint32_t sampleIndex)
{
    if (common.width() > common.imageWidth() || common.height() > common.imageHeight())
        prepareFrameResources();
    else
        ensureResources();
    common.dispatch(frameIndex, sampleIndex, [this] { renderImpl(); });
}

void RealtimeRaytracer::restartTemporalHistory()
{
    hasHistory = false;
}

void RealtimeRaytracer::ensureResources()
{
    if (common.data.maxBounces > 0u)
        restir.enableIndirect();
    // Viewport resizes inside the base class's image allocation only move
    // the rectangles; the settings and that allocation decide the layout.
    const ResourceLayout required = requiredLayout();
    if (allocatedLayout == required)
        return;
    NR_LOG_INFO("DIAG realtime resources reallocated: render " << required.targets.render.width << "x" << required.targets.render.height << " layers=" << required.targets.layers << " output " << required.output.width << "x" << required.output.height << " denoiser=" << int(required.denoiser) << " upscaler=" << int(upscaler.mode()) << " lightingRes=" << int(lightingResolution)); // DIAG
    common.device().synchronize();
    targets.reset();
    targets.emplace(common.device(), required.targets);
    restir.resize(required.targets.lighting, required.targets.layers);
    denoiser.configure({required.targets.lighting, required.denoiser,
        required.targets.sphericalHarmonics, required.targets.layers});
    upscaler.resize(required.targets.render, required.output);
    allocatedLayout = required;
    hasHistory = false;
}

FrameContext RealtimeRaytracer::beginFrame()
{
    FrameContext frame;
    frame.output = outputExtent();
    frame.render = renderExtent();
    frame.lighting = lightingExtent();
    frame.lightingScale = lightingScale(lightingResolution);
    // A viewport resize keeps history: every temporal stage reprojects from
    // the previous frame's rectangle into this one's.
    // An exposure change keeps history: FSR rescales its own, and the
    // denoiser and ReSTIR converge to the new scale within a few frames.
    frame.resetHistory = !hasHistory;
    frame.exposureScale = std::exp2(common.data.camera.exposure);
    frame.previousLighting = frame.resetHistory ? frame.lighting : previousLighting;
    const auto now = std::chrono::steady_clock::now();
    frame.frameTimeMilliseconds = previousFrameStart.time_since_epoch().count() == 0 ? 16.7f
        : std::chrono::duration<float, std::milli>(now - previousFrameStart).count();
    previousFrameStart = now;

    const nr::graphics::Camera& camera = common.data.camera;
    frame.worldToView = worldToViewMatrix(camera);
    frame.viewToClip = viewToClipMatrix(camera);
    frame.jitter = upscaler.nextJitter(frame.render, frame.output, frame.resetHistory);
    frame.previousWorldToView = frame.resetHistory ? frame.worldToView : previousWorldToView;
    frame.previousViewToClip = frame.resetHistory ? frame.viewToClip : previousViewToClip;
    frame.previousJitter = frame.resetHistory ? frame.jitter : previousJitter;
    frame.nearPlane = NearPlane;
    frame.verticalFieldOfView = 2.0f * std::atan(std::max(camera.sensorHeightMm, 0.001f)
        / (2.0f * std::max(camera.focalLengthMm, 0.001f)));

    const float* cameraToWorld = camera.cameraToWorld;
    const std::array<float, 3> cameraPosition{
        cameraToWorld[3], cameraToWorld[7], cameraToWorld[11]};
    const std::array<float, 3> cameraBefore =
        frame.resetHistory ? cameraPosition : previousCameraPosition;
    const std::array<float, 3> cameraBeforeBefore =
        frame.resetHistory ? cameraPosition : previousPreviousCameraPosition;
    const glm::vec3 forward = -glm::normalize(glm::vec3{
        cameraToWorld[2], cameraToWorld[6], cameraToWorld[10]});

    nr::graphics::RealtimeView& view = args->view;
    const std::array<float, 16> worldToClip =
        multiplyColumnMajor(frame.viewToClip, frame.worldToView);
    const std::array<float, 16> previousWorldToClip =
        multiplyColumnMajor(frame.previousViewToClip, frame.previousWorldToView);
    std::copy(worldToClip.begin(), worldToClip.end(), view.worldToClip);
    std::copy(previousWorldToClip.begin(), previousWorldToClip.end(), view.previousWorldToClip);
    view.cameraPosition = {cameraPosition[0], cameraPosition[1], cameraPosition[2]};
    view.previousCameraPosition = {cameraBefore[0], cameraBefore[1], cameraBefore[2]};
    view.previousPreviousCameraPosition = {cameraBeforeBefore[0],
        cameraBeforeBefore[1], cameraBeforeBefore[2]};
    view.cameraForward = forward;
    view.nearPlane = NearPlane;
    // Use the host's monotonically advancing frame index for fresh per-frame
    // noise. The separate sample index controls the beauty average.
    view.frameIndex = common.data.frameIndex;
    view.outputWidth = frame.output.width;
    view.outputHeight = frame.output.height;
    view.jitter = {frame.jitter[0], frame.jitter[1]};
    view.lightingScale = frame.lightingScale;
    view.lightingWidth = frame.lighting.width;
    view.lightingHeight = frame.lighting.height;
    view.surfaceStride = targets->layout().lighting.width;
    view.previousLightingWidth = frame.previousLighting.width;
    view.previousLightingHeight = frame.previousLighting.height;

    hasHistory = true;
    previousLighting = frame.lighting;
    previousWorldToView = frame.worldToView;
    previousViewToClip = frame.viewToClip;
    previousJitter = frame.jitter;
    previousPreviousCameraPosition = cameraBefore;
    previousCameraPosition = cameraPosition;
    return frame;
}

void RealtimeRaytracer::renderImpl()
{
    ensureResources();
    const FrameContext frame = beginFrame();
    nr::graphics::RealtimeArgs& frameArgs = *args;
    // The realtime passes see the render resolution; everything outside this
    // renderer keeps seeing the output size in `data`.
    frameArgs.frame = common.data;
    frameArgs.frame.width = frame.render.width;
    frameArgs.frame.height = frame.render.height;
    frameArgs.frame.materialSampler = materialSampler.handle().value;
    frameArgs.frame.textureLodBias = upscaler.textureLodBias(frame.render, frame.output);
    frameArgs.frame.nearPlane = NearPlane;
    // Realtime always traces pinhole perspective rays, regardless of the
    // projection type authored on the shared camera object.
    frameArgs.frame.camera.projection = 0u;
    frameArgs.targets = targets->handles();
    frameArgs.targets.color = upscaler.compositeTarget(*targets, outputTexture());
    frameArgs.denoiser = denoiser.args(*targets);
    restir.prepare(frame, frameArgs);

    noorrhi::Device& device = common.device();
    const noorrhi::DispatchSize renderGroups{
        divideRoundingUp(frame.render.width, CompositeGroupSize),
        divideRoundingUp(frame.render.height, CompositeGroupSize), 1};
    const noorrhi::DispatchSize outputGroups{
        divideRoundingUp(frame.output.width, CompositeGroupSize),
        divideRoundingUp(frame.output.height, CompositeGroupSize), 1};

    // Every pass reads the same few kilobytes of arguments: stage them once
    // and hand each launch only their address.
    const noorrhi::StagedArguments staged = device.stage(frameArgs);
    const nr::graphics::RealtimeRoot root{staged.address()};
    device.label("ReSTIR Presample", [&] { restir.presample(frameArgs, root); });
    if (frame.lightingScale != 1.0f)
        device.label("G-Buffer", [&] {
            tracePipeline.trace(transparentMaterials ? layeredGBufferRaygen : gBufferRaygen,
                {frame.render.width, frame.render.height, 1}, root);
        });
    device.label("Lighting Surfaces", [&] {
        tracePipeline.trace(transparentMaterials ? layeredLightingRaygen : lightingRaygen,
            {frame.lighting.width, frame.lighting.height, 1}, root);
    });

    device.label("ReSTIR Resample", [&] {
        if (transparentMaterials)
        {
            nr::graphics::RealtimeArgs& layerFrameArgs = *layerArgs;
            layerFrameArgs = frameArgs;
            layerFrameArgs.lighting = restir.layerLighting(frameArgs.lighting);
            layerFrameArgs.targets.diffuse = frameArgs.targets.layerDiffuse;
            layerFrameArgs.targets.specular = frameArgs.targets.layerSpecular;
            layerFrameArgs.denoiser.sphericalHarmonics = 0u;
            const noorrhi::StagedArguments layerStaged = device.stage(layerFrameArgs);
            const std::array<nr::graphics::RealtimeRoot, 2> surfaceSets{root,
                nr::graphics::RealtimeRoot{layerStaged.address()}};
            restir.resample(frameArgs, surfaceSets, frame.lighting, tracePipeline);
        }
        else
        {
            const std::array<nr::graphics::RealtimeRoot, 1> surfaceSets{root};
            restir.resample(frameArgs, surfaceSets, frame.lighting, tracePipeline);
        }
    });
    device.barrier(noorrhi::Stage::RayTracing, noorrhi::Stage::Compute);
    device.label("Denoise", [&] { denoiser.record(frame, *targets); });
    device.label("Composite", [&] {
        (transparentMaterials ? layeredCompositePipeline : compositePipeline).launch(renderGroups, root);
    });
    device.label("Upscale", [&] {
        upscaler.record(frame, *targets, outputImageHandle(), {imageWidth(), imageHeight()});
    });
    // Resolve coverage, output AOVs and the beauty average after upscaling. The
    // alpha resolve reads the render-resolution composite and restores
    // coverage in the final image.
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    device.label("Output AOVs", [&] { outputAovsPipeline.launch(outputGroups, root); });
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
}
