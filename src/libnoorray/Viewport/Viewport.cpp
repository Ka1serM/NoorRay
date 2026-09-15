#include "Viewport.h"

#include <algorithm>
#include <cstddef>
#include <span>

#include "Logging/Log.h"
#include "Scene/Scene.h"
#include "Scene/LightInstance.h"

namespace
{
alignas(uint32_t) constexpr unsigned char noorRayViewportSpv[] = {
    #embed "Viewport/Viewport.spv"
};
constexpr std::size_t noorRayViewportSpvLength = sizeof(noorRayViewportSpv);

alignas(uint32_t) constexpr unsigned char noorRayViewportBillboardsSpv[] = {
    #embed "Viewport/ViewportBillboards.spv"
};
constexpr std::size_t noorRayViewportBillboardsSpvLength = sizeof(noorRayViewportBillboardsSpv);

constexpr uint32_t ViewportGroupSize = 16;

std::span<const std::byte> shader_bytes(const unsigned char* data, const std::size_t size)
{
    return {reinterpret_cast<const std::byte*>(data), size};
}

}

Viewport::Viewport(noorrhi::Device& gpu_device, const uint32_t width, const uint32_t height,
                   const ViewportInputs& inputs, const noorrhi::ImageFormat outputImageFormat,
                   const bool exportOutputMemory)
: gpuDevice(gpu_device), inputs(inputs), exportOutputMemory_(exportOutputMemory)
{
    createOutputImage(width, height, outputImageFormat);
    shader = gpuDevice.create_shader(
        shader_bytes(noorRayViewportSpv, noorRayViewportSpvLength));
    pipeline = gpuDevice.compute(shader);
    createBillboardPipeline();
    reserveBillboards(1);
}

Viewport::~Viewport()
{
    NR_LOG_INFO("Destroying Viewport");
}

void Viewport::createOutputImage(const uint32_t width, const uint32_t height,
                                 const noorrhi::ImageFormat format)
{
    outputImage = gpuDevice.image<std::byte>(width, height,
        noorrhi::ImageUsage::Sampled | noorrhi::ImageUsage::Storage
            | noorrhi::ImageUsage::ColorAttachment
            | (exportOutputMemory_ ? noorrhi::ImageUsage::ExternalMemory
                                   : noorrhi::ImageUsage{}),
        format);
    outputFormat_ = format;
}

void Viewport::createBillboardPipeline()
{
    const auto bytes = shader_bytes(noorRayViewportBillboardsSpv,
        noorRayViewportBillboardsSpvLength);
    billboardVertexShader = gpuDevice.create_shader(bytes, "vertMain");
    billboardFragmentShader = gpuDevice.create_shader(bytes, "fragMain");
    noorrhi::GraphicsState state{};
    state.cull = noorrhi::CullMode::None;
    state.depth_test = false;
    state.depth_write = false;
    state.blend.enabled = true;
    billboardPipeline = gpuDevice.graphics({billboardVertexShader,
        billboardFragmentShader, state, outputFormat_});
}

void Viewport::reserveBillboards(const uint32_t capacity)
{
    if (capacity <= billboardCapacity)
        return;

    // The buffer may still be referenced by an in-flight command buffer from a
    // previous frame; growth is rare (only when the light count exceeds the
    // current capacity), so waiting here is cheap insurance.
    if (billboardCapacity > 0)
        gpuDevice.synchronize();
    billboardBuffer = gpuDevice.buffer<std::byte>(
        static_cast<std::size_t>(capacity) * sizeof(ViewportBillboard));
    billboardCapacity = capacity;
    billboardEntry = billboardBuffer.ptr();
}

void Viewport::updateBillboards(const Scene& scene)
{
    if (observedLightRevision == scene.getLightRevision())
        return;

    const uint32_t lightCount = scene.getPointLightCount()
        + scene.getSpotLightCount()
        + scene.getRectLightCount()
        + scene.getDirectionalLightCount();
    reserveBillboards(std::max(1u, lightCount));
    billboardData.clear();
    for (const auto& obj : scene.getSceneObjects())
    {
        if (const auto* light = dynamic_cast<LightInstance*>(obj.get()))
        {
            billboardData.push_back(ViewportBillboard{
                glm::vec4(light->getWorldTransform().getPosition(),
                    static_cast<float>(light->lightType)),
                glm::vec4(light->getColor(), 1.0f)});
        }
    }
    billboardCount = static_cast<uint32_t>(billboardData.size());
    if (billboardCount > 0)
        billboardBuffer.upload(std::as_bytes(
            std::span<const ViewportBillboard>(billboardData.data(), billboardCount)));
    observedLightRevision = scene.getLightRevision();
}

void Viewport::drawBillboards(const glm::mat4& viewProjection)
{
    // The path-traced image uses bottom-left row order and is flipped once by
    // ImGui. Draw overlays into that same raw orientation so their projected
    // position receives the identical presentation flip. clear=false keeps the
    // composite pass's output that this draws on top of.
    const nr::graphics::ViewportBillboardPushConstants arguments{
        // Slang emits this root-argument matrix with row-major storage. GLM
        // stores matrices column-major, so transpose once at the ABI boundary
        // to preserve the same mathematical matrix in the shader.
        glm::transpose(viewProjection),
        billboardEntry.address,
        glm::vec2(static_cast<float>(outputImage.width()),
                  static_cast<float>(outputImage.height())),
        ViewportBillboardPixelRadius,
        0u};
    gpuDevice.render({.color = outputImage.handle(), .clear = false, .flip_y = false},
        [this, &arguments] {
            billboardPipeline.draw_instanced(6, billboardCount, arguments);
        });
}

void Viewport::dispatch(
    const uint32_t selectedCryptomatteId,
    const glm::mat4& viewProjection,
    const float exposure,
    const int bufferVisualization,
    const int gaussianOverdrawMax,
    const bool tonemappingEnabled,
    const bool showBillboards)
{
    // Before the first resize, or with AOVs switched off, some inputs do not
    // exist yet; skip until a later call supplies the full complement.
    if (!inputs || !outputImage)
        return;

    const nr::graphics::ViewportCompositePushConstants arguments{
        inputs.color.value,
        outputImage.storage_handle().value,
        inputs.crypto.value,
        inputs.albedo.value,
        inputs.normal.value,
        inputs.position.value,
        inputs.overdraw.address,
        selectedCryptomatteId, exposure, bufferVisualization, tonemappingEnabled ? 1 : 0,
        static_cast<uint32_t>(std::max(gaussianOverdrawMax, 1)),
        0u};
    const uint32_t groupCountX =
        (outputImage.width() + ViewportGroupSize - 1) / ViewportGroupSize;
    const uint32_t groupCountY =
        (outputImage.height() + ViewportGroupSize - 1) / ViewportGroupSize;
    pipeline.launch({groupCountX, groupCountY, 1}, arguments);

    if (showBillboards && billboardCount > 0)
        drawBillboards(viewProjection);
}

void Viewport::resize(const uint32_t width, const uint32_t height,
                      const ViewportInputs& newInputs,
                      const noorrhi::ImageFormat outputImageFormat)
{
    if (width == 0 || height == 0)
        return;

    const bool outputChanged = outputImage.width() != width
        || outputImage.height() != height
        || outputFormat_ != outputImageFormat;
    if (outputChanged)
    {
        // Only replacing the viewport resources requires waiting for work
        // that may still reference the old image or format-specific pipeline.
        gpuDevice.synchronize();
        const noorrhi::ImageFormat previousFormat = outputFormat_;
        createOutputImage(width, height, outputImageFormat);
        // The billboard pipeline bakes in its color-attachment format, so it
        // only has to be rebuilt when that format actually changes.
        if (outputImageFormat != previousFormat)
            createBillboardPipeline();
    }
    inputs = newInputs;
}

std::vector<noorrhi::float4> Viewport::readOutput() const
{
    if (outputFormat_ != noorrhi::ImageFormat::Rgba32Float)
        throw noorrhi::Error(noorrhi::ErrorCode::InvalidArgument,
            "Viewport::readOutput requires an RGBA32F output texture");
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(outputWidth())
        * outputHeight());
    outputImage.download(std::as_writable_bytes(std::span(result)));
    return result;
}
