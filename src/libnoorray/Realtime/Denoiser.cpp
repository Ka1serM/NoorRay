#include "Realtime/Denoiser.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

#include <NRD.h>
#include <noorrhi/interop.hpp>

namespace
{
constexpr nrd::Identifier ReblurIdentifier = 0;
constexpr nrd::Identifier RelaxIdentifier = 1;
constexpr nrd::Identifier ReblurShIdentifier = 2;
constexpr nrd::Identifier RelaxShIdentifier = 3;
constexpr nrd::Identifier ReblurLayerIdentifier = 4;
constexpr nrd::Identifier RelaxLayerIdentifier = 5;
// Weight of each frame's time in the smoothed frame time.
constexpr float FrameTimeSmoothing = 0.1f;
// View Z beyond this is background; the image pass writes twice this value.
constexpr float DenoisingRange = 100000.0f;
// Constant-buffer ring capacity in frames of dispatches. Entries are reused
// only after this many frames, by which time their commands have completed.
constexpr std::uint64_t ConstantBufferFrames = 16;
constexpr std::uint32_t DescriptorSetsPerPool = 64;

std::uint32_t divideRoundingUp(const std::uint32_t value, const std::uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

void check(const VkResult result, const char* what)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("NRD: ") + what + " failed ("
            + std::to_string(static_cast<int>(result)) + ")");
}

void check(const nrd::Result result, const char* what)
{
    if (result != nrd::Result::SUCCESS)
        throw std::runtime_error(std::string("NRD: ") + what + " failed ("
            + std::to_string(static_cast<int>(result)) + ")");
}

// Histories span each denoiser's recommended accumulation time at `fps`.
std::uint32_t historyFrames(const float accumulationTime, const float fps,
    const std::uint32_t maximum)
{
    return std::clamp(nrd::GetMaxAccumulatedFrameNum(accumulationTime, fps), 1u, maximum);
}

// The ReSTIR PT path's first-bounce hit distance goes to one lobe per pixel,
// picked by the lobes' share of the BSDF, so the other lobe's is
// reconstructed from neighbours. ReSTIR DI provides a clean direct-light hit
// distance for diffuse; the small minHitDistanceWeight gives that guide the
// authority to reject samples across a hard local-light shadow boundary
// instead of softening it.
nrd::ReblurSettings reblurSettings(const float fps)
{
    nrd::ReblurSettings settings{};
    settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    settings.minHitDistanceWeight = 0.01f;
    settings.maxAccumulatedFrameNum = historyFrames(nrd::REBLUR_DEFAULT_ACCUMULATION_TIME, fps,
        nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    return settings;
}

nrd::RelaxSettings relaxSettings(const float fps)
{
    nrd::RelaxSettings settings{};
    settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    settings.minHitDistanceWeight = 0.01f;
    settings.enableAntiFirefly = true;
    settings.diffuseMaxAccumulatedFrameNum = settings.specularMaxAccumulatedFrameNum =
        historyFrames(nrd::RELAX_DEFAULT_ACCUMULATION_TIME, fps,
            nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    return settings;
}

VkFormat vulkanFormat(const nrd::Format format)
{
    switch (format) {
    case nrd::Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
    case nrd::Format::R8_SNORM: return VK_FORMAT_R8_SNORM;
    case nrd::Format::R8_UINT: return VK_FORMAT_R8_UINT;
    case nrd::Format::R8_SINT: return VK_FORMAT_R8_SINT;
    case nrd::Format::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case nrd::Format::RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case nrd::Format::RG8_UINT: return VK_FORMAT_R8G8_UINT;
    case nrd::Format::RG8_SINT: return VK_FORMAT_R8G8_SINT;
    case nrd::Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case nrd::Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case nrd::Format::RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case nrd::Format::RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case nrd::Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case nrd::Format::R16_UNORM: return VK_FORMAT_R16_UNORM;
    case nrd::Format::R16_SNORM: return VK_FORMAT_R16_SNORM;
    case nrd::Format::R16_UINT: return VK_FORMAT_R16_UINT;
    case nrd::Format::R16_SINT: return VK_FORMAT_R16_SINT;
    case nrd::Format::R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
    case nrd::Format::RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case nrd::Format::RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case nrd::Format::RG16_UINT: return VK_FORMAT_R16G16_UINT;
    case nrd::Format::RG16_SINT: return VK_FORMAT_R16G16_SINT;
    case nrd::Format::RG16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case nrd::Format::RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case nrd::Format::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case nrd::Format::RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case nrd::Format::RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case nrd::Format::RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case nrd::Format::R32_UINT: return VK_FORMAT_R32_UINT;
    case nrd::Format::R32_SINT: return VK_FORMAT_R32_SINT;
    case nrd::Format::R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case nrd::Format::RG32_UINT: return VK_FORMAT_R32G32_UINT;
    case nrd::Format::RG32_SINT: return VK_FORMAT_R32G32_SINT;
    case nrd::Format::RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case nrd::Format::RGB32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case nrd::Format::RGB32_SINT: return VK_FORMAT_R32G32B32_SINT;
    case nrd::Format::RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case nrd::Format::RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case nrd::Format::RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
    case nrd::Format::RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case nrd::Format::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case nrd::Format::R10_G10_B10_A2_UINT: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case nrd::Format::R11_G11_B10_UFLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case nrd::Format::R9_G9_B9_E5_UFLOAT: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    default: throw std::runtime_error("NRD: unsupported texture format");
    }
}
}

struct Denoiser::Texture
{
    VkImage image{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
};

// Size-independent Vulkan objects.
struct Denoiser::Resources
{
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkPipelineCache pipelineCache{VK_NULL_HANDLE};
    std::vector<VkSampler> samplers;
    // Indexed by descriptor set number: NRD's resources space and its
    // constant-buffer-and-samplers space.
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::uint32_t resourcesSet{};
    std::uint32_t constantsSet{};
    std::uint32_t textureBinding{};
    std::uint32_t storageBinding{};
    std::uint32_t textureCount{};
    std::uint32_t storageCount{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    std::vector<VkPipeline> pipelines;
    VkBuffer constantBuffer{VK_NULL_HANDLE};
    VkDeviceMemory constantMemory{VK_NULL_HANDLE};
    void* constantMapping{};
    std::uint64_t constantViewSize{};
    std::uint64_t constantBufferSize{};
    VkDescriptorPool constantsPool{VK_NULL_HANDLE};
    VkDescriptorSet constants{VK_NULL_HANDLE};
    VkPhysicalDeviceMemoryProperties memoryProperties{};

    std::uint32_t memoryType(const std::uint32_t allowed, const VkMemoryPropertyFlags flags) const
    {
        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
            if ((allowed & (1u << i)) != 0
                && (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        throw std::runtime_error("NRD: no suitable memory type");
    }
};

Denoiser::Denoiser(noorrhi::Device& device)
    : device_(device)
    , resources_(std::make_unique<Resources>())
{
    const noorrhi::interop::DeviceHandles handles = noorrhi::interop::device_handles(device_);
    resources_->device = reinterpret_cast<VkDevice>(handles.device);
    resources_->physicalDevice = reinterpret_cast<VkPhysicalDevice>(handles.physical_device);
    resources_->pipelineCache = reinterpret_cast<VkPipelineCache>(handles.pipeline_cache);
    vkGetPhysicalDeviceMemoryProperties(resources_->physicalDevice,
        &resources_->memoryProperties);
}

Denoiser::~Denoiser()
{
    if (!instance_)
        return;
    // Recorded dispatches may still reference everything configure() made.
    device_.synchronize();
    destroyInstance();
}

std::vector<std::uint32_t> Denoiser::identifiers() const
{
    const bool relax = layout_.mode == DenoiserMode::Relax;
    std::vector<std::uint32_t> result{layout_.sphericalHarmonics
        ? (relax ? RelaxShIdentifier : ReblurShIdentifier)
        : (relax ? RelaxIdentifier : ReblurIdentifier)};
    if (layout_.layers)
        result.push_back(relax ? RelaxLayerIdentifier : ReblurLayerIdentifier);
    return result;
}

void Denoiser::configure(const DenoiserLayout& layout)
{
    destroyInstance();
    layout_ = layout;
    if (layout_.mode == DenoiserMode::Off)
        return;
    // Only the denoisers this layout runs: every one in an instance keeps
    // its own history pools, whether it runs or not.
    std::vector<nrd::DenoiserDesc> denoisers;
    for (const std::uint32_t identifier : identifiers()) {
        const bool sh = identifier == ReblurShIdentifier || identifier == RelaxShIdentifier;
        const bool relax = identifier == RelaxIdentifier || identifier == RelaxShIdentifier
            || identifier == RelaxLayerIdentifier;
        denoisers.push_back({identifier, relax
            ? (sh ? nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH : nrd::Denoiser::RELAX_DIFFUSE_SPECULAR)
            : (sh ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH : nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR)});
    }
    nrd::InstanceCreationDesc creation{};
    creation.denoisers = denoisers.data();
    creation.denoisersNum = static_cast<std::uint32_t>(denoisers.size());
    check(nrd::CreateInstance(creation, instance_), "CreateInstance");
    // A new instance starts from NRD's default settings.
    historyFrames_ = {};
    updateHistoryLength(smoothedFrameTimeMilliseconds_ == 0.0f
        ? 1000.0f / 60.0f : smoothedFrameTimeMilliseconds_);
    createPipelines();
    createPools();
}

void Denoiser::destroyInstance()
{
    if (!instance_)
        return;
    Resources& r = *resources_;
    destroyPools();
    vkDestroyDescriptorPool(r.device, r.constantsPool, nullptr);
    if (r.constantMapping)
        vkUnmapMemory(r.device, r.constantMemory);
    vkDestroyBuffer(r.device, r.constantBuffer, nullptr);
    vkFreeMemory(r.device, r.constantMemory, nullptr);
    for (VkPipeline pipeline : r.pipelines)
        vkDestroyPipeline(r.device, pipeline, nullptr);
    vkDestroyPipelineLayout(r.device, r.pipelineLayout, nullptr);
    for (VkDescriptorSetLayout layout : r.setLayouts)
        vkDestroyDescriptorSetLayout(r.device, layout, nullptr);
    for (VkSampler sampler : r.samplers)
        vkDestroySampler(r.device, sampler, nullptr);
    Resources device;
    device.device = r.device;
    device.physicalDevice = r.physicalDevice;
    device.pipelineCache = r.pipelineCache;
    device.memoryProperties = r.memoryProperties;
    r = std::move(device);
    nrd::DestroyInstance(*instance_);
    instance_ = nullptr;
    diffuseOutput_ = {};
    specularOutput_ = {};
    diffuseSh1Output_ = {};
    specularSh1Output_ = {};
    layerDiffuseOutput_ = {};
    layerSpecularOutput_ = {};
}

nr::graphics::DenoiserArgs Denoiser::args(const RenderTargets& targets) const
{
    nr::graphics::DenoiserArgs args{};
    const bool off = layout_.mode == DenoiserMode::Off;
    args.diffuse = off ? targets.handles().diffuse : diffuseOutput_.storage_handle().value;
    args.specular = off ? targets.handles().specular : specularOutput_.storage_handle().value;
    args.diffuseSh1 = diffuseSh1Output_.storage_handle().value;
    args.specularSh1 = specularSh1Output_.storage_handle().value;
    args.layerDiffuse = off ? targets.handles().layerDiffuse
        : layerDiffuseOutput_.storage_handle().value;
    args.layerSpecular = off ? targets.handles().layerSpecular
        : layerSpecularOutput_.storage_handle().value;
    args.relax = layout_.mode == DenoiserMode::Relax ? 1u : 0u;
    args.sphericalHarmonics = sphericalHarmonics() ? 1u : 0u;
    args.denoisingRange = DenoisingRange;
    const nrd::ReblurHitDistanceParameters hitDistance{};
    args.hitDistanceParameters = {hitDistance.A, hitDistance.B, hitDistance.C};
    return args;
}

void Denoiser::record(const FrameContext& frame, const RenderTargets& targets)
{
    if (layout_.mode == DenoiserMode::Off)
        return;
    nrd::CommonSettings settings{};
    std::memcpy(settings.worldToViewMatrix, frame.worldToView.data(), sizeof(float) * 16);
    std::memcpy(settings.viewToClipMatrix, frame.viewToClip.data(), sizeof(float) * 16);
    std::memcpy(settings.worldToViewMatrixPrev, frame.previousWorldToView.data(),
        sizeof(float) * 16);
    std::memcpy(settings.viewToClipMatrixPrev, frame.previousViewToClip.data(),
        sizeof(float) * 16);
    // The motion target is the UV offset to the previous frame, which is
    // NRD's 2D convention: uvPrev = uv + motion.
    settings.isMotionVectorInWorldSpace = false;
    settings.motionVectorScale[0] = 1.0f;
    settings.motionVectorScale[1] = 1.0f;
    settings.motionVectorScale[2] = 0.0f;
    // NRD's convention is sampleUv = pixelUv + cameraJitter, in pixels, and
    // it denoises lighting pixels: the render jitter over the lighting scale.
    // A scale change restarts history, so both frames share the scale.
    const float lightingScale = static_cast<float>(frame.lightingScale);
    settings.cameraJitter[0] = frame.jitter[0] / lightingScale;
    settings.cameraJitter[1] = frame.jitter[1] / lightingScale;
    settings.cameraJitterPrev[0] = frame.previousJitter[0] / lightingScale;
    settings.cameraJitterPrev[1] = frame.previousJitter[1] / lightingScale;
    // NRD's dynamic resolution: the images keep their allocation while the
    // lighting rectangle follows the viewport, and history carries across a
    // change of rectangle.
    settings.resourceSize[0] = settings.resourceSizePrev[0] =
        static_cast<std::uint16_t>(layout_.extent.width);
    settings.resourceSize[1] = settings.resourceSizePrev[1] =
        static_cast<std::uint16_t>(layout_.extent.height);
    settings.rectSize[0] = static_cast<std::uint16_t>(frame.lighting.width);
    settings.rectSize[1] = static_cast<std::uint16_t>(frame.lighting.height);
    settings.rectSizePrev[0] = static_cast<std::uint16_t>(frame.previousLighting.width);
    settings.rectSizePrev[1] = static_cast<std::uint16_t>(frame.previousLighting.height);
    settings.denoisingRange = DenoisingRange;
    settings.frameIndex = frameIndex_++;
    settings.accumulationMode = frame.resetHistory
        ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    updateHistoryLength(frame.frameTimeMilliseconds);
    dispatch(settings, targets);
}

void Denoiser::updateHistoryLength(const float frameTimeMilliseconds)
{
    smoothedFrameTimeMilliseconds_ = smoothedFrameTimeMilliseconds_ == 0.0f
        ? frameTimeMilliseconds
        : smoothedFrameTimeMilliseconds_
            + FrameTimeSmoothing * (frameTimeMilliseconds - smoothedFrameTimeMilliseconds_);
    const float fps = 1000.0f / std::max(smoothedFrameTimeMilliseconds_, 0.1f);
    const nrd::ReblurSettings reblur = reblurSettings(fps);
    const nrd::RelaxSettings relax = relaxSettings(fps);
    // NRD re-derives its constants on every settings change; only pass on
    // one that changes a history length.
    const std::array<std::uint32_t, 2> frames{reblur.maxAccumulatedFrameNum,
        relax.diffuseMaxAccumulatedFrameNum};
    if (frames == historyFrames_)
        return;
    historyFrames_ = frames;
    const bool relaxMode = layout_.mode == DenoiserMode::Relax;
    for (const std::uint32_t identifier : identifiers())
        check(nrd::SetDenoiserSettings(*instance_, identifier,
            relaxMode ? static_cast<const void*>(&relax) : &reblur), "SetDenoiserSettings");
}

void Denoiser::createPipelines()
{
    Resources& r = *resources_;
    const nrd::LibraryDesc& library = *nrd::GetLibraryDesc();
    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*instance_);
    const nrd::SPIRVBindingOffsets& offsets = library.spirvBindingOffsets;

    for (std::uint32_t i = 0; i < desc.samplersNum; ++i) {
        const VkFilter filter = desc.samplers[i] == nrd::Sampler::NEAREST_CLAMP
            ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = filter;
        info.minFilter = filter;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler{};
        check(vkCreateSampler(r.device, &info, nullptr, &sampler), "vkCreateSampler");
        r.samplers.push_back(sampler);
    }

    // NRD's HLSL register spaces become descriptor sets; registers are
    // shifted into bindings by the library's SPIR-V offsets.
    r.resourcesSet = desc.resourcesSpaceIndex;
    r.constantsSet = desc.constantBufferAndSamplersSpaceIndex;
    r.textureBinding = offsets.textureOffset + desc.resourcesBaseRegisterIndex;
    r.storageBinding = offsets.storageTextureAndBufferOffset + desc.resourcesBaseRegisterIndex;
    r.textureCount = desc.descriptorPoolDesc.perSetTexturesMaxNum;
    r.storageCount = desc.descriptorPoolDesc.perSetStorageTexturesMaxNum;
    const std::uint32_t constantBinding = offsets.constantBufferOffset
        + desc.constantBufferRegisterIndex;

    std::vector<VkDescriptorSetLayoutBinding> resourceBindings;
    for (std::uint32_t i = 0; i < r.textureCount; ++i)
        resourceBindings.push_back({r.textureBinding + i,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    for (std::uint32_t i = 0; i < r.storageCount; ++i)
        resourceBindings.push_back({r.storageBinding + i,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});

    std::vector<VkDescriptorSetLayoutBinding> constantBindings;
    constantBindings.push_back({constantBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    for (std::uint32_t i = 0; i < desc.samplersNum; ++i)
        constantBindings.push_back({offsets.samplerOffset + desc.samplersBaseRegisterIndex + i,
            VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});

    r.setLayouts.assign(std::max(r.resourcesSet, r.constantsSet) + 1, VK_NULL_HANDLE);
    const auto createLayout = [&](const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        VkDescriptorSetLayout layout{};
        check(vkCreateDescriptorSetLayout(r.device, &info, nullptr, &layout),
            "vkCreateDescriptorSetLayout");
        return layout;
    };
    r.setLayouts[r.resourcesSet] = createLayout(resourceBindings);
    r.setLayouts[r.constantsSet] = createLayout(constantBindings);
    // Any set number between the two stays empty.
    for (VkDescriptorSetLayout& layout : r.setLayouts)
        if (layout == VK_NULL_HANDLE)
            layout = createLayout({});

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = static_cast<std::uint32_t>(r.setLayouts.size());
    layoutInfo.pSetLayouts = r.setLayouts.data();
    check(vkCreatePipelineLayout(r.device, &layoutInfo, nullptr, &r.pipelineLayout),
        "vkCreatePipelineLayout");

    for (std::uint32_t i = 0; i < desc.pipelinesNum; ++i) {
        const nrd::ComputeShaderDesc& shader = desc.pipelines[i].computeShaderSPIRV;
        if (!shader.bytecode || shader.size == 0)
            throw std::runtime_error("NRD: library was built without SPIR-V shaders");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = shader.size;
        moduleInfo.pCode = static_cast<const std::uint32_t*>(shader.bytecode);
        VkShaderModule module{};
        check(vkCreateShaderModule(r.device, &moduleInfo, nullptr, &module),
            "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = desc.shaderEntryPoint;
        pipelineInfo.layout = r.pipelineLayout;
        VkPipeline pipeline{};
        const VkResult result = vkCreateComputePipelines(r.device, r.pipelineCache, 1,
            &pipelineInfo, nullptr, &pipeline);
        vkDestroyShaderModule(r.device, module, nullptr);
        check(result, "vkCreateComputePipelines");
        r.pipelines.push_back(pipeline);
    }

    // Constant buffer: a persistently mapped ring with one aligned slot per
    // dispatch, bound through a dynamic offset.
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(r.physicalDevice, &properties);
    const std::uint64_t alignment = std::max<std::uint64_t>(
        properties.limits.minUniformBufferOffsetAlignment, 1);
    r.constantViewSize = (std::max<std::uint64_t>(desc.constantBufferMaxDataSize, 1)
        + alignment - 1) / alignment * alignment;
    r.constantBufferSize = r.constantViewSize
        * std::max<std::uint32_t>(desc.descriptorPoolDesc.setsMaxNum, 1) * ConstantBufferFrames;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = r.constantBufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    check(vkCreateBuffer(r.device, &bufferInfo, nullptr, &r.constantBuffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(r.device, r.constantBuffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = r.memoryType(requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(r.device, &allocation, nullptr, &r.constantMemory), "vkAllocateMemory");
    check(vkBindBufferMemory(r.device, r.constantBuffer, r.constantMemory, 0), "vkBindBufferMemory");
    check(vkMapMemory(r.device, r.constantMemory, 0, VK_WHOLE_SIZE, 0, &r.constantMapping),
        "vkMapMemory");

    std::array<VkDescriptorPoolSize, 2> sizes{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, std::max<std::uint32_t>(desc.samplersNum, 1)},
    }};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    check(vkCreateDescriptorPool(r.device, &poolInfo, nullptr, &r.constantsPool),
        "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = r.constantsPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &r.setLayouts[r.constantsSet];
    check(vkAllocateDescriptorSets(r.device, &setInfo, &r.constants), "vkAllocateDescriptorSets");

    std::vector<VkDescriptorImageInfo> samplerInfos(desc.samplersNum);
    std::vector<VkWriteDescriptorSet> writes;
    const VkDescriptorBufferInfo constantInfo{r.constantBuffer, 0, r.constantViewSize};
    VkWriteDescriptorSet constantWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    constantWrite.dstSet = r.constants;
    constantWrite.dstBinding = constantBinding;
    constantWrite.descriptorCount = 1;
    constantWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    constantWrite.pBufferInfo = &constantInfo;
    writes.push_back(constantWrite);
    for (std::uint32_t i = 0; i < desc.samplersNum; ++i) {
        samplerInfos[i].sampler = r.samplers[i];
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = r.constants;
        write.dstBinding = offsets.samplerOffset + desc.samplersBaseRegisterIndex + i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &samplerInfos[i];
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(r.device, static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);
}

void Denoiser::destroyPools()
{
    Resources& r = *resources_;
    descriptorSets_.clear();
    for (const std::uint64_t pool : descriptorPools_)
        vkDestroyDescriptorPool(r.device, reinterpret_cast<VkDescriptorPool>(pool), nullptr);
    descriptorPools_.clear();
    for (Texture& texture : pool_) {
        vkDestroyImageView(r.device, texture.view, nullptr);
        vkDestroyImage(r.device, texture.image, nullptr);
        vkFreeMemory(r.device, texture.memory, nullptr);
    }
    pool_.clear();
}

void Denoiser::createPools()
{
    Resources& r = *resources_;
    const std::uint32_t width = layout_.extent.width;
    const std::uint32_t height = layout_.extent.height;
    const auto output = [&] {
        return device_.image<std::byte>(width, height,
            noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled,
            noorrhi::ImageFormat::Rgba16Float);
    };
    diffuseOutput_ = output();
    specularOutput_ = output();
    if (layout_.sphericalHarmonics) {
        diffuseSh1Output_ = output();
        specularSh1Output_ = output();
    }
    if (layout_.layers) {
        layerDiffuseOutput_ = output();
        layerSpecularOutput_ = output();
    }

    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*instance_);
    const std::uint32_t textureCount = desc.permanentPoolSize + desc.transientPoolSize;
    pool_.resize(textureCount);
    for (std::uint32_t i = 0; i < textureCount; ++i) {
        const nrd::TextureDesc& textureDesc = i < desc.permanentPoolSize
            ? desc.permanentPool[i] : desc.transientPool[i - desc.permanentPoolSize];
        const VkFormat format = vulkanFormat(textureDesc.format);
        Texture& texture = pool_[i];

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {divideRoundingUp(width, textureDesc.downsampleFactor),
            divideRoundingUp(height, textureDesc.downsampleFactor), 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(r.device, &imageInfo, nullptr, &texture.image), "vkCreateImage");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(r.device, texture.image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = r.memoryType(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(r.device, &allocation, nullptr, &texture.memory),
            "vkAllocateMemory");
        check(vkBindImageMemory(r.device, texture.image, texture.memory, 0),
            "vkBindImageMemory");

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = texture.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(r.device, &viewInfo, nullptr, &texture.view),
            "vkCreateImageView");
    }
    poolsNeedTransition_ = true;
    constantOffset_ = 0;
    previousConstantOffset_ = 0;
}

void Denoiser::transitionPoolsToGeneral(const std::uintptr_t commandBuffer)
{
    std::vector<VkImageMemoryBarrier> barriers;
    for (const Texture& texture : pool_) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(barrier);
    }
    if (!barriers.empty())
        vkCmdPipelineBarrier(reinterpret_cast<VkCommandBuffer>(commandBuffer),
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, nullptr, 0, nullptr, static_cast<std::uint32_t>(barriers.size()), barriers.data());
}

std::uint64_t Denoiser::descriptorSet(const std::uint16_t pipelineIndex,
    const std::vector<std::uint64_t>& views)
{
    Resources& r = *resources_;
    std::vector<std::uint64_t> key;
    key.reserve(views.size() + 1);
    key.push_back(pipelineIndex);
    key.insert(key.end(), views.begin(), views.end());
    if (const auto found = descriptorSets_.find(key); found != descriptorSets_.end())
        return found->second;

    const auto allocate = [&](VkDescriptorPool pool, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        info.descriptorPool = pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &r.setLayouts[r.resourcesSet];
        return vkAllocateDescriptorSets(r.device, &info, &set);
    };
    VkDescriptorSet set{VK_NULL_HANDLE};
    if (descriptorPools_.empty() || allocate(reinterpret_cast<VkDescriptorPool>(
            descriptorPools_.back()), set) != VK_SUCCESS) {
        std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                std::max<std::uint32_t>(r.textureCount, 1) * DescriptorSetsPerPool},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                std::max<std::uint32_t>(r.storageCount, 1) * DescriptorSetsPerPool},
        }};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        info.maxSets = DescriptorSetsPerPool;
        info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        info.pPoolSizes = sizes.data();
        VkDescriptorPool pool{};
        check(vkCreateDescriptorPool(r.device, &info, nullptr, &pool), "vkCreateDescriptorPool");
        descriptorPools_.push_back(reinterpret_cast<std::uint64_t>(pool));
        check(allocate(pool, set), "vkAllocateDescriptorSets");
    }

    // `views` holds (view, isStorage) pairs in NRD's range order.
    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*instance_);
    const nrd::PipelineDesc& pipeline = desc.pipelines[pipelineIndex];
    std::vector<VkDescriptorImageInfo> infos(views.size() / 2);
    std::vector<VkWriteDescriptorSet> writes;
    std::size_t n = 0;
    for (std::uint32_t range = 0; range < pipeline.resourceRangesNum; ++range) {
        const nrd::ResourceRangeDesc& resourceRange = pipeline.resourceRanges[range];
        const bool storage = resourceRange.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
        for (std::uint32_t i = 0; i < resourceRange.descriptorsNum; ++i, ++n) {
            infos[n].imageView = reinterpret_cast<VkImageView>(views[2 * n]);
            infos[n].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = set;
            write.dstBinding = (storage ? r.storageBinding : r.textureBinding) + i;
            write.descriptorCount = 1;
            write.descriptorType = storage
                ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.pImageInfo = &infos[n];
            writes.push_back(write);
        }
    }
    vkUpdateDescriptorSets(r.device, static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);
    const std::uint64_t handle = reinterpret_cast<std::uint64_t>(set);
    descriptorSets_.emplace(std::move(key), handle);
    return handle;
}

void Denoiser::dispatch(const nrd::CommonSettings& settings, const RenderTargets& targets)
{
    const std::vector<std::uint32_t> running = identifiers();
    const nrd::Identifier layerIdentifier = layout_.mode == DenoiserMode::Relax
        ? RelaxLayerIdentifier : ReblurLayerIdentifier;
    Resources& r = *resources_;
    check(nrd::SetCommonSettings(*instance_, settings), "SetCommonSettings");
    const nrd::DispatchDesc* dispatches = nullptr;
    std::uint32_t dispatchCount = 0;
    check(nrd::GetComputeDispatches(*instance_, running.data(),
            static_cast<std::uint32_t>(running.size()), dispatches, dispatchCount),
        "GetComputeDispatches");

    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*instance_);
    const auto view = [&](const noorrhi::ImageHandle image) {
        return static_cast<std::uint64_t>(noorrhi::interop::image_view(device_, image));
    };
    const auto inputView = [&](const nrd::ResourceType type, const bool layer) -> std::uint64_t {
        switch (type) {
        case nrd::ResourceType::IN_MV: return view(layer ? targets.layerLightingMotion() : targets.lightingMotion());
        case nrd::ResourceType::IN_NORMAL_ROUGHNESS: return view(layer ? targets.layerLightingNormalRoughness() : targets.lightingNormalRoughness());
        case nrd::ResourceType::IN_VIEWZ: return view(layer ? targets.layerLightingViewZ() : targets.lightingViewZ());
        case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: return view(layer ? targets.layerDiffuse() : targets.diffuse());
        case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST: return view(layer ? targets.layerSpecular() : targets.specular());
        case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: return view(layer ? layerDiffuseOutput_.handle() : diffuseOutput_.handle());
        case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: return view(layer ? layerSpecularOutput_.handle() : specularOutput_.handle());
        case nrd::ResourceType::IN_DIFF_SH0: return view(targets.diffuse());
        case nrd::ResourceType::IN_DIFF_SH1: return view(targets.diffuseSh1());
        case nrd::ResourceType::IN_SPEC_SH0: return view(targets.specular());
        case nrd::ResourceType::IN_SPEC_SH1: return view(targets.specularSh1());
        case nrd::ResourceType::OUT_DIFF_SH0: return view(diffuseOutput_.handle());
        case nrd::ResourceType::OUT_DIFF_SH1: return view(diffuseSh1Output_.handle());
        case nrd::ResourceType::OUT_SPEC_SH0: return view(specularOutput_.handle());
        case nrd::ResourceType::OUT_SPEC_SH1: return view(specularSh1Output_.handle());
        default:
            throw std::runtime_error(std::string("NRD: unsupported resource ")
                + nrd::GetResourceTypeString(type));
        }
    };

    struct Recorded
    {
        VkPipeline pipeline;
        VkDescriptorSet resources;
        std::uint32_t constantOffset;
        std::uint32_t groupsX;
        std::uint32_t groupsY;
    };
    std::vector<Recorded> recorded;
    recorded.reserve(dispatchCount);
    std::vector<std::uint64_t> views;
    for (std::uint32_t i = 0; i < dispatchCount; ++i) {
        const nrd::DispatchDesc& dispatch = dispatches[i];
        const bool layer = dispatch.identifier == layerIdentifier;
        views.clear();
        for (std::uint32_t j = 0; j < dispatch.resourcesNum; ++j) {
            const nrd::ResourceDesc& resource = dispatch.resources[j];
            std::uint64_t handle;
            if (resource.type == nrd::ResourceType::PERMANENT_POOL)
                handle = reinterpret_cast<std::uint64_t>(pool_.at(resource.indexInPool).view);
            else if (resource.type == nrd::ResourceType::TRANSIENT_POOL)
                handle = reinterpret_cast<std::uint64_t>(
                    pool_.at(desc.permanentPoolSize + resource.indexInPool).view);
            else
                handle = inputView(resource.type, layer);
            views.push_back(handle);
            views.push_back(resource.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE);
        }

        std::uint64_t offset = previousConstantOffset_;
        if (dispatch.constantBufferDataSize != 0 && !dispatch.constantBufferDataMatchesPreviousDispatch) {
            if (constantOffset_ + r.constantViewSize > r.constantBufferSize)
                constantOffset_ = 0;
            offset = constantOffset_;
            constantOffset_ += r.constantViewSize;
            std::memcpy(static_cast<char*>(r.constantMapping) + offset,
                dispatch.constantBufferData, dispatch.constantBufferDataSize);
            previousConstantOffset_ = offset;
        }

        recorded.push_back({r.pipelines.at(dispatch.pipelineIndex),
            reinterpret_cast<VkDescriptorSet>(descriptorSet(dispatch.pipelineIndex, views)),
            static_cast<std::uint32_t>(offset), dispatch.gridWidth, dispatch.gridHeight});
    }

    const bool transition = poolsNeedTransition_;
    poolsNeedTransition_ = false;
    noorrhi::interop::record(device_, [&](const std::uintptr_t commandBuffer) {
        const VkCommandBuffer command = reinterpret_cast<VkCommandBuffer>(commandBuffer);
        if (transition)
            transitionPoolsToGeneral(commandBuffer);

        // NRD dispatches form a strict producer/consumer chain through its
        // transient and permanent images, and a pipeline bind does not make
        // earlier shader writes visible in Vulkan. RELAX is especially
        // sensitive because its longer A-trous chain can otherwise read
        // partially written tiles, typically exposed as black blocks at a
        // screen edge. The interop recording itself is ordered against the
        // renderer's passes around it.
        VkMemoryBarrier chain{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        chain.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        chain.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        std::vector<VkDescriptorSet> sets(r.setLayouts.size(), VK_NULL_HANDLE);
        for (std::size_t i = 0; i < recorded.size(); ++i) {
            const Recorded& dispatch = recorded[i];
            if (i != 0)
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &chain, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
            sets[r.resourcesSet] = dispatch.resources;
            sets[r.constantsSet] = r.constants;
            for (std::uint32_t set = 0; set < sets.size(); ++set) {
                if (sets[set] == VK_NULL_HANDLE)
                    continue;
                const bool constants = set == r.constantsSet;
                vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                    r.pipelineLayout, set, 1, &sets[set], constants ? 1 : 0,
                    constants ? &dispatch.constantOffset : nullptr);
            }
            vkCmdDispatch(command, dispatch.groupsX, dispatch.groupsY, 1);
        }
    });
}
