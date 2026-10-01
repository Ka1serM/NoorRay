#include "Realtime/FfxPlatform.h"
#include "Realtime/Upscaler.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <noorrhi/interop.hpp>

#include "Logging/Log.h"

namespace
{
std::mutex resourceCreateMutex;
std::unordered_map<FfxInterface*, FfxCreateResourceFunc> resourceCreateCallbacks;

void check(const FfxErrorCode result, const char* what)
{
    if (result != FFX_OK)
        throw std::runtime_error(std::string("FSR: ") + what + " failed ("
            + std::to_string(result) + ")");
}

FfxErrorCode createResourceWithRgba8LumaHistory(
    FfxInterface* backendInterface,
    const FfxCreateResourceDescription* description,
    const FfxUInt32 effectContextId,
    FfxResourceInternal* outResource)
{
    FfxCreateResourceFunc createResource = nullptr;
    {
        const std::lock_guard lock(resourceCreateMutex);
        const auto callback = resourceCreateCallbacks.find(backendInterface);
        if (callback != resourceCreateCallbacks.end())
            createResource = callback->second;
    }
    if (!createResource || !description)
        return FFX_ERROR_INVALID_POINTER;

    const std::wstring_view name = description->name ? description->name : L"";
    if (name == L"FSR3UPSCALER_LumaHistory1"
        || name == L"FSR3UPSCALER_LumaHistory2")
    {
        // Stock FidelityFX GLSL declares this storage image as rgba8. Match
        // its allocation to avoid the Vulkan storage image format mismatch
        // and reduce luma-history storage and bandwidth to 8 bits per channel.
        FfxCreateResourceDescription rgba8Description = *description;
        rgba8Description.resourceDescription.format = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
        return createResource(backendInterface, &rgba8Description, effectContextId, outResource);
    }

    return createResource(backendInterface, description, effectContextId, outResource);
}

// The device's persistent pipeline cache. The backend creates its pipelines
// without a cache, which costs seconds whenever a context is recreated.
VkPipelineCache pipelineCache = VK_NULL_HANDLE;
PFN_vkCreateComputePipelines createComputePipelines = nullptr;

VkResult VKAPI_CALL createComputePipelinesCached(VkDevice device, VkPipelineCache,
    const uint32_t count, const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
    return createComputePipelines(device, pipelineCache, count, infos, allocator, pipelines);
}

// The backend resolves device functions by name, including KHR aliases of
// functions that became core in Vulkan 1.1-1.3 (vkGetBufferMemoryRequirements2KHR).
// Those aliases resolve only when their extension was enabled explicitly, so
// fall back to the core name.
PFN_vkVoidFunction VKAPI_CALL deviceProcAddr(VkDevice device, const char* name)
{
    if (std::string_view(name) == "vkCreateComputePipelines") {
        createComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(
            vkGetDeviceProcAddr(device, name));
        return reinterpret_cast<PFN_vkVoidFunction>(createComputePipelinesCached);
    }
    if (const PFN_vkVoidFunction function = vkGetDeviceProcAddr(device, name))
        return function;
    const std::string_view view(name);
    if (!view.ends_with("KHR"))
        return nullptr;
    const std::string core(view.substr(0, view.size() - 3));
    return vkGetDeviceProcAddr(device, core.c_str());
}

void message(const FfxMsgType type, const wchar_t* text)
{
    std::fprintf(stderr, "[FSR %s] %ls\n",
        type == FFX_MESSAGE_TYPE_ERROR ? "error" : "warning", text ? text : L"");
}
}

// The upscaler and the shared resources it writes each take a backend
// effect context, as in the SDK's combined FSR3 component.
constexpr std::size_t BackendContextCount = FFX_FSR3UPSCALER_CONTEXT_COUNT + 1;

struct Upscaler::State
{
    VkDeviceContext deviceContext{};
    std::vector<unsigned char> scratch;
    FfxInterface backend{};
    FfxUInt32 sharedEffectContext{};
    bool sharedContextCreated{};
    std::unique_ptr<FfxFsr3UpscalerContext> context;
    // FSR 3.1 writes these for effects that follow it; the application owns them.
    FfxResourceInternal dilatedDepth{};
    FfxResourceInternal dilatedMotionVectors{};
    FfxResourceInternal reconstructedPreviousNearestDepth{};

    void destroyContext()
    {
        if (!context)
            return;
        for (const FfxResourceInternal resource : {dilatedDepth, dilatedMotionVectors,
                 reconstructedPreviousNearestDepth})
            backend.fpDestroyResource(&backend, resource, sharedEffectContext);
        ffxFsr3UpscalerContextDestroy(context.get());
        context.reset();
    }
};

Upscaler::Upscaler(noorrhi::Device& device)
    : device_(device)
    , state_(std::make_unique<State>())
{
    const noorrhi::interop::DeviceHandles handles = noorrhi::interop::device_handles(device_);
    State& state = *state_;
    state.deviceContext.vkDevice = reinterpret_cast<VkDevice>(handles.device);
    state.deviceContext.vkPhysicalDevice = reinterpret_cast<VkPhysicalDevice>(handles.physical_device);
    state.deviceContext.vkDeviceProcAddr = deviceProcAddr;
    pipelineCache = reinterpret_cast<VkPipelineCache>(handles.pipeline_cache);
    state.scratch.resize(ffxGetScratchMemorySizeVK(state.deviceContext.vkPhysicalDevice,
        BackendContextCount));
    check(ffxGetInterfaceVK(&state.backend, ffxGetDeviceVK(&state.deviceContext),
        state.scratch.data(), state.scratch.size(), BackendContextCount),
        "ffxGetInterfaceVK");
    {
        const std::lock_guard lock(resourceCreateMutex);
        resourceCreateCallbacks.emplace(&state.backend, state.backend.fpCreateResource);
    }
    state.backend.fpCreateResource = createResourceWithRgba8LumaHistory;
}

Upscaler::~Upscaler()
{
    if (!state_)
        return;
    device_.synchronize();
    state_->destroyContext();
    if (state_->sharedContextCreated)
        state_->backend.fpDestroyBackendContext(&state_->backend, state_->sharedEffectContext);
    const std::lock_guard lock(resourceCreateMutex);
    resourceCreateCallbacks.erase(&state_->backend);
}

void Upscaler::resize(const Extent maxRender, const Extent maxOutput)
{
    State& state = *state_;
    state.destroyContext();
    if (mode_ == UpscalerMode::Off)
        return;
    FfxFsr3UpscalerContextDescription description{};
    // Inverted infinite depth is what FSR recommends. Render resolution
    // follows the renderer's scale, so the context allows it to change.
    description.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE
        | FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED
        | FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE
        | FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION;
    description.maxRenderSize = {maxRender.width, maxRender.height};
    description.maxUpscaleSize = {maxOutput.width, maxOutput.height};
    description.fpMessage = message;
    description.backendInterface = state.backend;
    auto context = std::make_unique<FfxFsr3UpscalerContext>();
    check(ffxFsr3UpscalerContextCreate(context.get(), &description), "ffxFsr3UpscalerContextCreate");
    state.context = std::move(context);

    if (!state.sharedContextCreated) {
        check(state.backend.fpCreateBackendContext(&state.backend, FFX_EFFECT_SHAREDRESOURCES,
            nullptr, &state.sharedEffectContext), "fpCreateBackendContext");
        state.sharedContextCreated = true;
    }
    FfxFsr3UpscalerSharedResourceDescriptions shared{};
    check(ffxFsr3UpscalerGetSharedResourceDescriptions(state.context.get(), &shared),
        "ffxFsr3UpscalerGetSharedResourceDescriptions");
    check(state.backend.fpCreateResource(&state.backend, &shared.dilatedDepth,
        state.sharedEffectContext, &state.dilatedDepth), "fpCreateResource");
    check(state.backend.fpCreateResource(&state.backend, &shared.dilatedMotionVectors,
        state.sharedEffectContext, &state.dilatedMotionVectors), "fpCreateResource");
    check(state.backend.fpCreateResource(&state.backend, &shared.reconstructedPrevNearestDepth,
        state.sharedEffectContext, &state.reconstructedPreviousNearestDepth), "fpCreateResource");
}

Extent Upscaler::renderExtent(const Extent output) const
{
    const float scale = 1.0f / upscalerRatio(mode_);
    const auto scaled = [scale](const std::uint32_t logical) {
        return std::clamp<std::uint32_t>(static_cast<std::uint32_t>(std::lround(
            static_cast<float>(logical) * scale)), 1u, std::max(logical, 1u));
    };
    return {scaled(output.width), scaled(output.height)};
}

std::array<float, 2> Upscaler::nextJitter(const Extent render, const Extent output,
    const bool reset)
{
    if (mode_ == UpscalerMode::Off)
        return {0.0f, 0.0f};
    // The Halton sequence only integrates to a clean pixel footprint when it
    // is walked from its start, so a restarted history restarts the phase too.
    if (reset)
        jitterIndex_ = 0;
    const std::int32_t phaseCount = ffxFsr3UpscalerGetJitterPhaseCount(
        static_cast<std::int32_t>(render.width), static_cast<std::int32_t>(output.width));
    const std::uint32_t index = jitterIndex_++;
    float x = 0.0f;
    float y = 0.0f;
    ffxFsr3UpscalerGetJitterOffset(&x, &y,
        static_cast<std::int32_t>(index % static_cast<std::uint32_t>(std::max(phaseCount, 1))),
        phaseCount);
    // FSR's convention is that render pixel p carries the sample at
    // p + 0.5 - jitter (ffx_fsr3upscaler_upsample.h).
    return {-x, -y};
}

float Upscaler::textureLodBias(const Extent render, const Extent output) const
{
    if (mode_ == UpscalerMode::Off)
        return 0.0f;
    return std::log2(static_cast<float>(render.height)
        / static_cast<float>(std::max(output.height, 1u))) - 1.0f;
}

std::uint32_t Upscaler::compositeTarget(const RenderTargets& targets,
    const noorrhi::TextureHandle output) const
{
    return mode_ == UpscalerMode::Off ? output.value : targets.colorTexture().value;
}

void Upscaler::record(const FrameContext& frame, const RenderTargets& targets,
    const noorrhi::ImageHandle output, const Extent outputAllocation)
{
    if (mode_ == UpscalerMode::Off)
        return;
    State& state = *state_;
    if (!state.context)
        return;

    struct Image
    {
        noorrhi::ImageHandle handle;
        Extent extent;
        VkFormat format;
    };
    // Registered with the extent they are allocated at.
    const auto registered = [&](const Image& image, const wchar_t* name) {
        const VkImage vkImage = reinterpret_cast<VkImage>(noorrhi::interop::image(device_, image.handle));
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = image.format;
        info.extent = {image.extent.width, image.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        // NoorRHI keeps every image in GENERAL, which FSR calls UNORDERED_ACCESS.
        return ffxGetResourceVK(vkImage,
            ffxGetImageResourceDescriptionVK(vkImage, info, FFX_RESOURCE_USAGE_UAV),
            name, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    };
    const Extent allocation = targets.layout().render;
    const FfxResource color = registered({targets.color(), allocation,
        VK_FORMAT_R16G16B16A16_SFLOAT}, L"FSR3_InputColor");
    const FfxResource depth = registered({targets.depth(), allocation,
        VK_FORMAT_R32_SFLOAT}, L"FSR3_InputDepth");
    const FfxResource motion = registered({targets.motion(), allocation,
        VK_FORMAT_R16G16_SFLOAT}, L"FSR3_InputMotionVectors");
    const FfxResource upscaled = registered({output, outputAllocation,
        VK_FORMAT_R32G32B32A32_SFLOAT}, L"FSR3_Output");
    FfxResourceDescription noneDescription{};
    noneDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    const FfxResource none = ffxGetResourceVK(nullptr, noneDescription, L"None",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    // Only translucent layers write the mask.
    const FfxResource transparency = targets.layout().layers
        ? registered({targets.transparencyMask(), allocation, VK_FORMAT_R8_UNORM},
            L"FSR3_TransparencyAndComposition")
        : none;

    FfxFsr3UpscalerDispatchDescription description{};
    description.color = color;
    description.depth = depth;
    description.motionVectors = motion;
    description.exposure = none;
    description.reactive = none;
    description.transparencyAndComposition = transparency;
    description.dilatedDepth = state.backend.fpGetResource(&state.backend, state.dilatedDepth);
    description.dilatedMotionVectors = state.backend.fpGetResource(&state.backend,
        state.dilatedMotionVectors);
    description.reconstructedPrevNearestDepth = state.backend.fpGetResource(&state.backend,
        state.reconstructedPreviousNearestDepth);
    description.output = upscaled;
    // FSR's jitter convention is the negation of the renderer's.
    description.jitterOffset = {-frame.jitter[0], -frame.jitter[1]};
    // Motion is written in UV units.
    description.motionVectorScale = {static_cast<float>(frame.render.width),
        static_cast<float>(frame.render.height)};
    description.renderSize = {frame.render.width, frame.render.height};
    description.upscaleSize = {frame.output.width, frame.output.height};
    description.enableSharpening = false;
    description.frameTimeDelta = std::clamp(frame.frameTimeMilliseconds, 0.1f, 100.0f);
    description.preExposure = 1.0f;
    description.reset = frame.resetHistory;
    // With inverted depth FSR takes the planes swapped, as in AMD's sample.
    description.cameraNear = FLT_MAX;
    description.cameraFar = frame.nearPlane;
    description.cameraFovAngleVertical = frame.verticalFieldOfView;
    description.viewSpaceToMetersFactor = 1.0f;

    noorrhi::interop::record(device_, [&](const std::uintptr_t commandBuffer) {
        // The images are registered in the state NoorRHI keeps them in
        // (GENERAL, which FSR calls UNORDERED_ACCESS) and are usable as storage,
        // so the backend reads them without changing their layout.
        description.commandList = ffxGetCommandListVK(reinterpret_cast<VkCommandBuffer>(commandBuffer));
        const FfxErrorCode result = ffxFsr3UpscalerContextDispatch(state.context.get(), &description);
        if (result != FFX_OK)
            NR_LOG_ERROR("FSR: ffxFsr3UpscalerContextDispatch failed (" << result << ")");
    });
}
