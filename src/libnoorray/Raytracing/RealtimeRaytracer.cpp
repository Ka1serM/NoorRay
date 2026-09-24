#include "RealtimeRaytracer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include <glm/geometric.hpp>

#include "Logging/Log.h"
#include "Realtime/ShaderLoading.h"
#include "Shared/RealtimeArgs.h"

namespace
{
alignas(uint32_t) constexpr unsigned char raygenSpv[] = {
    #embed "RealtimeRaytracer/RealtimeRaytracer.spv"
};
alignas(uint32_t) constexpr unsigned char missSpv[] = {
    #embed "RealtimeRaytracer/RealtimeMiss.spv"
};
alignas(uint32_t) constexpr unsigned char shadowMissSpv[] = {
    #embed "RealtimeRaytracer/RealtimeShadowMiss.spv"
};
alignas(uint32_t) constexpr unsigned char defaultMaterialHitSpv[] = {
    #embed "RealtimeRaytracer/DefaultMaterialHit.spv"
};
alignas(uint32_t) constexpr unsigned char compositeSpv[] = {
    #embed "RealtimeRaytracer/RealtimeComposite.spv"
};
alignas(uint32_t) constexpr unsigned char outputAovsSpv[] = {
    #embed "RealtimeRaytracer/RealtimeOutputAovs.spv"
};

// Matches RealtimeComposite.slang and RealtimeOutputAovs.slang.
constexpr uint32_t CompositeGroupSize = 8u;
constexpr float NearPlane = 0.01f;
// Far plane of the orthographic projection: beyond anything the denoiser
// treats as geometry.
constexpr float OrthographicFarPlane = 4.0e5f;

// Every realtime stage agrees on this; the surface payload is the largest.
constexpr noorrhi::RayTracingInterface TraceInterface{sizeof(nr::graphics::RealtimeHitPayload)};

// Hit groups of the linked pipeline, counted library by library. The pass
// library's come first; each material then adds MaterialGroupCount.
constexpr uint32_t DefaultMaterialGroup = 0u;
// No shaders: shadow rays pass opaque materials' records without any work,
// and Gaussian records, which realtime rays never reach, need some group.
constexpr uint32_t EmptyGroup = 1u;
constexpr uint32_t FirstMaterialGroup = 2u;
enum MaterialGroup : uint32_t { TransparentSurface, OpaqueSurface, Shadow, MaterialGroupCount };

noorrhi::RayTracingLibrary buildPassLibrary(noorrhi::Device& device,
    std::vector<noorrhi::Shader> raygens)
{
    return device.ray_tracing_library({std::move(raygens),
        {loadShader(device, missSpv), loadShader(device, shadowMissSpv)},
        {loadShader(device, defaultMaterialHitSpv), noorrhi::Shader{}}, {}, {}}, TraceInterface);
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
    if (camera.projection == 1u) {
        // Orthographic, with generateCameraRay()'s film height.
        const float height = 10.0f * 21.0f / focalLength;
        const float width = height * sensorWidth / sensorHeight;
        matrix[0] = 2.0f / width;
        matrix[5] = ySign * 2.0f / height;
        matrix[10] = 1.0f / OrthographicFarPlane;
        matrix[15] = 1.0f;
        return matrix;
    }
    // Perspective with an infinite far plane. Fisheye and sequential lenses
    // are approximated by the pinhole of the same film and focal length.
    matrix[0] = 2.0f * focalLength / sensorWidth;
    matrix[5] = ySign * 2.0f * focalLength / sensorHeight;
    matrix[10] = 1.0f;
    matrix[11] = 1.0f;
    matrix[14] = -NearPlane;
    return matrix;
}
}

RealtimeRaytracer::RealtimeRaytracer(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory)
    : Raytracer(device, width, height, exportColorMemory)
    , imageRaygen(loadShader(device, raygenSpv))
    , radianceCache(device)
    , restir(device)
    , denoiser(device)
    , upscaler(device)
    , accumulator(device)
    , targets(device, upscaler.renderExtent(outputExtent()))
    , compositePipeline(device.compute(loadShader(device, compositeSpv)))
    , outputAovsPipeline(device.compute(loadShader(device, outputAovsSpv)))
    , args(std::make_unique<nr::graphics::RealtimeArgs>())
{
    const auto started = std::chrono::steady_clock::now();
    std::vector<noorrhi::Shader> raygens{imageRaygen, radianceCache.raygen()};
    std::ranges::copy(restir.raygens(), std::back_inserter(raygens));
    passLibrary = buildPassLibrary(device, std::move(raygens));
    device.save_pipeline_cache();
    NR_LOG_INFO("Compiled the ray-tracing pass library in " << millisecondsSince(started) << " ms");
    linkTracePipeline();
    upscaler.resize({imageWidth(), imageHeight()});
    resizeRenderResolution();
}

RealtimeRaytracer::~RealtimeRaytracer() = default;

Extent RealtimeRaytracer::outputExtent() const
{
    return {logicalRenderWidth(), logicalRenderHeight()};
}

Extent RealtimeRaytracer::renderExtent() const
{
    return upscaler.renderExtent(outputExtent());
}

uint32_t RealtimeRaytracer::traceWidth() const
{
    return renderExtent().width;
}

uint32_t RealtimeRaytracer::traceHeight() const
{
    return renderExtent().height;
}

void RealtimeRaytracer::onImageAllocationChanged()
{
    // The base class has synchronized the device before reallocating.
    upscaler.resize({imageWidth(), imageHeight()});
    resizeRenderResolution();
}

void RealtimeRaytracer::onLightsUploaded()
{
    restir.uploadLights(pointLightRecords(), spotLightRecords(), rectLightRecords(),
        directionalLightRecords());
}

void RealtimeRaytracer::onRenderSettingsApplied(const RenderSettings& settings)
{
    radianceCache.setMode(settings.radianceCacheMode);
    restir.setMode(settings.realtimeLighting);
    denoiser.setMode(settings.denoiserMode);
    upscaler.setMode(settings.upscalerMode);
}

void RealtimeRaytracer::onMaterialShadersChanged(const std::span<const MaterialHitShaders> shaders)
{
    // The list only grows, so the shaders past the last batch are exactly
    // the new ones. Each gets its MaterialGroup hit groups, in that order.
    const auto started = std::chrono::steady_clock::now();
    noorrhi::RayTracingPipelineDesc desc;
    for (const MaterialHitShaders& material : shaders.subspan(materialShaderCount)) {
        desc.closest_hit.insert(desc.closest_hit.end(),
            {material.closestHit, material.closestHit, noorrhi::Shader{}});
        desc.any_hit.insert(desc.any_hit.end(),
            {material.anyHit, noorrhi::Shader{}, material.shadowAnyHit});
    }
    materialLibraries.push_back(renderDevice().ray_tracing_library(desc, TraceInterface));
    NR_LOG_INFO("Compiled " << shaders.size() - materialShaderCount
        << " material shader(s) in " << millisecondsSince(started) << " ms");
    materialShaderCount = shaders.size();
    renderDevice().save_pipeline_cache();
}

void RealtimeRaytracer::onHitRecordsChanged(const std::span<const HitRecord> records)
{
    hitGroups.clear();
    for (const HitRecord& record : records) {
        const uint32_t materialGroups = FirstMaterialGroup
            + record.materialShaders * MaterialGroupCount;
        const bool compiled = record.kind == HitRecord::Kind::Section
            && record.materialShaders != ~0u;
        if (record.rayType == nr::graphics::RealtimeRayTypeSurface)
            hitGroups.push_back(!compiled ? (record.kind == HitRecord::Kind::Section
                    ? DefaultMaterialGroup : EmptyGroup)
                : materialGroups + (record.transparent ? TransparentSurface : OpaqueSurface));
        else
            hitGroups.push_back(compiled && record.transparent ? materialGroups + Shadow
                : EmptyGroup);
    }
    const auto started = std::chrono::steady_clock::now();
    linkTracePipeline();
    NR_LOG_INFO("Linked the ray-tracing pipeline for " << records.size()
        << " hit records in " << millisecondsSince(started) << " ms");
}

void RealtimeRaytracer::linkTracePipeline()
{
    std::vector<noorrhi::RayTracingLibrary> libraries{passLibrary};
    libraries.insert(libraries.end(), materialLibraries.begin(), materialLibraries.end());
    tracePipeline = renderDevice().ray_tracing(
        std::span<const noorrhi::RayTracingLibrary>(libraries), hitGroups);
}

void RealtimeRaytracer::prepareFrameResources()
{
    ensureRenderResolution();
}

void RealtimeRaytracer::ensureRenderResolution()
{
    // The render resolution moves with the viewport and with the upscaler
    // mode, and only a viewport change reallocates the base class's images.
    if (targets.extent() == renderExtent())
        return;
    renderDevice().synchronize();
    resizeRenderResolution();
}

void RealtimeRaytracer::resizeRenderResolution()
{
    const Extent render = renderExtent();
    targets = RenderTargets(renderDevice(), render);
    restir.resize(render);
    denoiser.resize(render);
}

FrameContext RealtimeRaytracer::beginFrame()
{
    FrameContext frame;
    frame.output = outputExtent();
    frame.render = targets.extent();
    const HistoryKey key{frame.render, frame.output, radianceCache.mode(), restir.mode(),
        denoiser.mode(), upscaler.mode()};
    frame.resetHistory = !hasHistory || key != historyKey;

    const nr::graphics::Camera& camera = data.camera;
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
    view.cameraForward = forward;
    view.nearPlane = NearPlane;
    view.frameIndex = frameIndex++;
    view.outputWidth = frame.output.width;
    view.outputHeight = frame.output.height;
    view.jitter = {frame.jitter[0], frame.jitter[1]};

    historyKey = key;
    hasHistory = true;
    previousWorldToView = frame.worldToView;
    previousViewToClip = frame.viewToClip;
    previousJitter = frame.jitter;
    previousCameraPosition = cameraPosition;
    return frame;
}

void RealtimeRaytracer::renderImpl()
{
    ensureRenderResolution();
    const FrameContext frame = beginFrame();
    nr::graphics::RealtimeArgs& frameArgs = *args;
    // The realtime passes see the render resolution; everything outside this
    // renderer keeps seeing the output size in `data`.
    frameArgs.frame = data;
    frameArgs.frame.width = frame.render.width;
    frameArgs.frame.height = frame.render.height;
    frameArgs.targets = targets.handles();
    frameArgs.targets.color = upscaler.compositeTarget(targets, outputTexture());
    frameArgs.radianceCache = radianceCache.args(frame);
    frameArgs.denoiser = denoiser.args(targets);
    restir.prepare(frame, frameArgs);

    noorrhi::Device& device = renderDevice();
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

    restir.presample(frameArgs, root);
    radianceCache.record(frameArgs, root, tracePipeline);
    tracePipeline.trace(imageRaygen, {frame.render.width, frame.render.height, 1}, root);
    device.barrier(noorrhi::Stage::RayTracing, noorrhi::Stage::Compute);
    restir.resample(frameArgs, root, frame.render, tracePipeline);
    denoiser.record(frame, targets);
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    compositePipeline.launch(renderGroups, root);
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    upscaler.record(frame, targets, outputImageHandle(), {imageWidth(), imageHeight()});
    device.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    outputAovsPipeline.launch(outputGroups, root);
    accumulator.record(frameArgs, root);
}
