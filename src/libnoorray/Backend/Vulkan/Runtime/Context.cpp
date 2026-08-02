#include "Context.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "Backend/CUDA/Checks.h"
#include "Log.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// OptiX reports per-entry-function register counts and pipeline statistics at
// callback level 4. That output is far too noisy for normal rendering, so it
// stays opt-in through NR_OPTIX_LOG_LEVEL for shader performance work.
static unsigned int optixLogLevel()
{
    static const unsigned int level = [] {
        const char* requested = std::getenv("NR_OPTIX_LOG_LEVEL");
        if (requested == nullptr)
            return 3u;
        const int parsed = std::atoi(requested);
        return static_cast<unsigned int>(std::clamp(parsed, 0, 4));
    }();
    return level;
}

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    if (level <= 2)
        LOG_ERROR("OptiX[" << tag << "] " << message);
    else if (level <= optixLogLevel())
        LOG_INFO("OptiX[" << tag << "] " << message);
}

static int selectCudaDeviceForVulkan(const vk::PhysicalDevice physicalDevice)
{
    vk::PhysicalDeviceIDProperties idProperties{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &idProperties;
    physicalDevice.getProperties2(&properties);

    int deviceCount = 0;
    NR_GPU_CHECK(cudaGetDeviceCount(&deviceCount));
    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
    {
        cudaDeviceProp cudaProperties{};
        NR_GPU_CHECK(cudaGetDeviceProperties(&cudaProperties, deviceIndex));
        if (std::memcmp(idProperties.deviceUUID, cudaProperties.uuid.bytes, VK_UUID_SIZE) == 0)
        {
            NR_GPU_CHECK(cudaSetDevice(deviceIndex));
            return deviceIndex;
        }
    }

    throw std::runtime_error("No CUDA device matches the selected Vulkan physical device UUID");
}

Context::Context()
    : Context(nullptr)
{
}

Context::Context(VulkanSurfaceProvider& provider)
    : Context(&provider)
{
}

Context::Context(VulkanSurfaceProvider* provider)
    : headless(provider == nullptr), surfaceProvider(provider)
{
    try {
        const auto vkGetInstanceProcAddr = headless
            ? &::vkGetInstanceProcAddr
            : surfaceProvider->getVulkanInstanceProcAddr();
        if (!vkGetInstanceProcAddr) {
            LOG_FATAL("Failed to get vkGetInstanceProcAddr");
            throw std::runtime_error("Failed to get vkGetInstanceProcAddr.");
        }

        // Initialize the dispatcher with the function pointer loader. This is needed to find vkCreateInstance.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        createVulkanInstance();

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

#ifdef DEBUG
        if (validationEnabled) {
        vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
        messengerInfo.setMessageSeverity(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
        );
        messengerInfo.setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
        );
        messengerInfo.pfnUserCallback = &debugUtilsMessengerCallback;
        messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
        }
#endif

        if (!headless)
            surface = vk::UniqueSurfaceKHR(
                surfaceProvider->createVulkanSurface(instance.get()), {instance.get()});
        
        pickPhysicalDevice();
        createLogicalDevice();

        // This final init call loads all device-level functions, including extensions.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());
        
        graphicsQueue = device->getQueue(graphicsFamilyIndex, 0);

        vk::CommandPoolCreateInfo commandPoolInfo;
        commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolInfo.setQueueFamilyIndex(graphicsFamilyIndex);
        commandPool = device->createCommandPoolUnique(commandPoolInfo);

        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eSampler, 64 },
            { vk::DescriptorType::eCombinedImageSampler, 10000 },
            { vk::DescriptorType::eSampledImage, 64 },
            { vk::DescriptorType::eStorageImage, 64 },
            { vk::DescriptorType::eUniformTexelBuffer, 64 },
            { vk::DescriptorType::eStorageTexelBuffer, 64 },
            { vk::DescriptorType::eUniformBuffer, 128 },
            { vk::DescriptorType::eStorageBuffer, 30128 },
            { vk::DescriptorType::eUniformBufferDynamic, 64 },
            { vk::DescriptorType::eStorageBufferDynamic, 64 },
            { vk::DescriptorType::eInputAttachment, 8 },
        };

        constexpr uint32_t maxSets = 210;
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
        poolInfo.maxSets = maxSets;
        poolInfo.setPoolSizes(poolSizes);
        descriptorPool = device->createDescriptorPoolUnique(poolInfo);

        createAllocator();

        // Initialize CUDA + OptiX
        selectCudaDeviceForVulkan(physicalDevice);
        int leastPriority = 0, greatestPriority = 0;
        NR_GPU_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        cudaStream.createWithPriority(cudaStreamNonBlocking, greatestPriority);

        CUcontext currentCtx{};
        if (cuCtxGetCurrent(&currentCtx) != CUDA_SUCCESS || currentCtx == nullptr)
            throw std::runtime_error("CUDA primary context is unavailable");

        NR_OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctxOpts{};
        ctxOpts.logCallbackFunction = optixLogCallback;
        ctxOpts.logCallbackLevel = 3;
        NR_OPTIX_CHECK(optixDeviceContextCreate(currentCtx, &ctxOpts, optixCtx.put()));

    } catch (const vk::Error& e) {
        LOG_FATAL("Vulkan Error in Context constructor: " << e.what());
        throw std::runtime_error("Vulkan initialization failed.");
    } catch (const std::exception& e) {
        LOG_FATAL("Error in Context constructor: " << e.what());
        throw;
    }
}

void Context::createAllocator() {
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device.get();
    allocatorCreateInfo.instance = instance.get();
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
            
    // Enable bufferDeviceAddress feature in VMA if it's supported and enabled
    vk::PhysicalDeviceVulkan12Features features12{};
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);
    if(features12.bufferDeviceAddress)
        allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create VMA allocator!");
    
    LOG_INFO("VMA Allocator created successfully.");
}

void Context::createVulkanInstance() {
std::vector<const char*> extensions;
if (!headless)
    extensions = surfaceProvider->getRequiredVulkanInstanceExtensions();
std::vector<const char*> layers;

// --- Create the Debug Messenger Info ---
// We will chain this to both the instance creation and create a persistent one later.
vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;

#ifdef DEBUG
for (const vk::LayerProperties& layer : vk::enumerateInstanceLayerProperties()) {
    if (std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
        validationEnabled = true;
        break;
    }
}
if (validationEnabled) {
    LOG_INFO("Enabling Vulkan validation layers.");
    layers.push_back("VK_LAYER_KHRONOS_validation");
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
} else {
    LOG_INFO("Vulkan validation layer unavailable; continuing without it.");
}

messengerInfo.setMessageSeverity(
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
);
messengerInfo.setMessageType(
    vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
);
messengerInfo.pfnUserCallback = &debugUtilsMessengerCallback;
#endif

#ifdef __APPLE__
extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

constexpr vk::ApplicationInfo appInfo("NoorRay", 1, "No Engine", 1, VK_API_VERSION_1_3);
vk::InstanceCreateInfo instanceInfo{};
instanceInfo.setPApplicationInfo(&appInfo)
    .setPEnabledLayerNames(layers)
    .setPEnabledExtensionNames(extensions);

#ifdef __APPLE__ 
instanceInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

#if DEBUG
vk::ValidationFeaturesEXT validationFeatures{};
std::vector<vk::ValidationFeatureEnableEXT> enabledFeatures = {
    // Catches shader-based out-of-bounds access and other GPU errors. CRITICAL for DeviceLostError.
    vk::ValidationFeatureEnableEXT::eGpuAssisted,
    // Reserves a binding slot for the validation layer's own use, important for bindless.
    vk::ValidationFeatureEnableEXT::eGpuAssistedReserveBindingSlot,
    // Checks for non-optimal but still valid API usage.
    vk::ValidationFeatureEnableEXT::eBestPractices,
    // Enables the use of printf() in shaders for debugging.
    vk::ValidationFeatureEnableEXT::eDebugPrintf,
    // Adds extra, more intensive checks for synchronization bugs (race conditions).
    vk::ValidationFeatureEnableEXT::eSynchronizationValidation
};
if (validationEnabled) {
    validationFeatures.setEnabledValidationFeatures(enabledFeatures);
    validationFeatures.pNext = &messengerInfo;
    instanceInfo.pNext = &validationFeatures;
}
#endif

instance = vk::createInstanceUnique(instanceInfo);
}

void Context::pickPhysicalDevice() {
    const std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        LOG_FATAL("Failed to find GPUs with Vulkan support!");
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    LOG_INFO("Available GPUs:");

    struct Candidate {
        vk::PhysicalDevice device;
        uint64_t vram = 0;
    };

    auto findBestDevice = [&](const std::vector<const char*>& requiredExts) -> std::optional<Candidate> {
        std::optional<Candidate> bestDiscrete, bestFallback;

        for (const auto& device : devices) {
            const auto props = device.getProperties();
            const auto memProps = device.getMemoryProperties();

            std::set<std::string> missing(requiredExts.begin(), requiredExts.end());
            for (const auto& ext : device.enumerateDeviceExtensionProperties())
                missing.erase(ext.extensionName);

            const bool hasAllExtensions = missing.empty();

            uint64_t vramSize = 0;
            for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
                if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                    vramSize += memProps.memoryHeaps[i].size;

            LOG_INFO(props.deviceName << " (Type: " << vk::to_string(props.deviceType)
                     << ", VRAM: " << (vramSize / (1024 * 1024)) << "MB"
                     << ", Extensions OK: " << (hasAllExtensions ? "Yes" : "No") << ")");

            if (!hasAllExtensions) {
                for (const auto& extName : missing) {
                    LOG_WARN("Missing device extension: " << extName);
                }
            }

            if (!hasAllExtensions)
                continue;

            Candidate candidate{device, vramSize};
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                if (!bestDiscrete || vramSize > bestDiscrete->vram)
                    bestDiscrete = candidate;
            } else {
                if (!bestFallback || vramSize > bestFallback->vram)
                    bestFallback = candidate;
            }
        }

        if (bestDiscrete.has_value())
            return bestDiscrete;
        return bestFallback;
    };

    // In headless mode the swapchain extension is unnecessary (we never present).
    if (headless) {
        auto& exts = RequiredDeviceExtensions;
        exts.erase(std::remove_if(exts.begin(), exts.end(),
            [](const char* s) { return std::string_view(s) == VK_KHR_SWAPCHAIN_EXTENSION_NAME; }),
            exts.end());
    }

    auto best = findBestDevice(RequiredDeviceExtensions);

    if (!best) {
        LOG_FATAL("No suitable GPU found that supports required Vulkan extensions!");
        throw std::runtime_error("No suitable GPU found!");
    }

    physicalDevice = best->device;
    LOG_INFO("Picked GPU: " << physicalDevice.getProperties().deviceName);
}

void Context::createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();
    std::optional<uint32_t> foundGraphicsFamily, foundComputeFamily, foundPresentFamily;
    // First pass: find any suitable family for each type
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& flags = queueFamilies[i].queueFlags;
        
        if (flags & vk::QueueFlagBits::eGraphics)
            foundGraphicsFamily = i;
        if (flags & vk::QueueFlagBits::eCompute)
            foundComputeFamily = i;
        if (headless || physicalDevice.getSurfaceSupportKHR(i, surface.get()))
            foundPresentFamily = i;
        
        if (foundGraphicsFamily == i && foundComputeFamily == i && foundPresentFamily == i)
            break;
    }
    // Second pass: Prioritize a dedicated compute queue
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& flags = queueFamilies[i].queueFlags;
        if ((flags & vk::QueueFlagBits::eCompute) && !(flags & vk::QueueFlagBits::eGraphics)) {
            foundComputeFamily = i;
            break;
        }
    }
    
    if (!foundGraphicsFamily.has_value() || !foundComputeFamily.has_value() || !foundPresentFamily.has_value()) {
        throw std::runtime_error("Could not find all required queue families!");
    }

    graphicsFamilyIndex = foundGraphicsFamily.value();
    
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    constexpr float queuePriority = 1.0f;
    queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags{}, graphicsFamilyIndex, 1, &queuePriority);
    
    auto enabledExtensions = RequiredDeviceExtensions;
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    vk::PhysicalDeviceVulkan12Features features12{};

    // Build the pNext chain. This creates a linked list of feature structs.
    void** nextFeature = nullptr;

    features12.sType = vk::StructureType::ePhysicalDeviceVulkan12Features;
    nextFeature = &features12.pNext;

    dynamicRenderingFeatures.sType = vk::StructureType::ePhysicalDeviceDynamicRenderingFeatures;
    *nextFeature = &dynamicRenderingFeatures;
    nextFeature = &dynamicRenderingFeatures.pNext;


    // Put the chain head into the main features2 struct to query for support
    vk::PhysicalDeviceFeatures2 features2{};
    features2.sType = vk::StructureType::ePhysicalDeviceFeatures2;
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);

    // After querying, explicitly enable the features
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;

    // Create the logical device, pass the head of the feature chain.
    vk::DeviceCreateInfo createInfo;
    createInfo.pNext = &features2;
    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledExtensionNames(enabledExtensions);
    
    device = physicalDevice.createDeviceUnique(createInfo);
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    LOG_FATAL("Failed to find suitable memory type!");
    throw std::runtime_error("Failed to find suitable memory type!");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) {
    try {
        const vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
        vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

        commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        func(*commandBuffer);
        commandBuffer->end();

        vk::UniqueFence fence = device->createFenceUnique({});
        const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
        graphicsQueue.submit(submitInfo, *fence);

        if (device->waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            LOG_ERROR("Fence wait failed during one-time submit.");
        
    } catch (const vk::Error& e) {
        LOG_ERROR("Vulkan error during one-time submit: " << e.what());
    }
}

vk::PresentModeKHR Context::chooseSwapPresentMode() const {
    const std::vector<vk::PresentModeKHR> modes =
        physicalDevice.getSurfacePresentModesKHR(surface.get());
    if (std::find(modes.begin(), modes.end(), vk::PresentModeKHR::eMailbox) != modes.end()) {
        LOG_INFO("Present Mode: Mailbox");
        return vk::PresentModeKHR::eMailbox;
    }
    if (std::find(modes.begin(), modes.end(), vk::PresentModeKHR::eImmediate) != modes.end()) {
        LOG_INFO("Present Mode: Immediate");
        return vk::PresentModeKHR::eImmediate;
    }

    LOG_INFO("Present Mode: FIFO (V-Sync fallback)");
    return vk::PresentModeKHR::eFifo;
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::debugUtilsMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    std::string severity;
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
        severity = "[VULKAN ERROR]";
    else
        severity = "[VULKAN WARN]";
    
    LOG_ERROR(severity << ": " << pCallbackData->pMessage);
    return vk::False;
}

Context::~Context() {
    LOG_INFO("Destroying Context...");
    if (cudaStream)
        cudaStreamSynchronize(cudaStream.get());
    try {
        if (device)
            device->waitIdle();
    } catch (const vk::Error& e) {
        LOG_ERROR("Vulkan error during device->waitIdle(): " << e.what());
    }
    optixCtx.reset();
    cudaStream.reset();
    if (allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

}
