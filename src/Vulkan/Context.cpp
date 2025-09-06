#include "Context.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "Log.h"

#if !defined(NDEBUG)
    constexpr bool EnableValidationLayers = true;
#else
constexpr bool EnableValidationLayers = false;
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(const int width, const int height) : windowWidth(width), windowHeight(height), dpiScale(1) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        throw std::runtime_error("Failed to initialize SDL: " + std::string(SDL_GetError()));

    if (SDL_Vulkan_LoadLibrary(nullptr) < 0)
        throw std::runtime_error("Failed to load Vulkan library via SDL: " + std::string(SDL_GetError()));

    float dpiScaleFloat = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    if (dpiScaleFloat != 0.0f) //only if this doesnt fail
    {
        dpiScale = dpiScaleFloat;
        windowWidth  = static_cast<int>(windowWidth  * dpiScale);
        windowHeight = static_cast<int>(windowHeight * dpiScale);
    }
    
    window = SDL_CreateWindow("NoorRay by Marcel K.", windowWidth, windowHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window)
        throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));

    auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    if (!vkGetInstanceProcAddr)
        throw std::runtime_error("Failed to get vkGetInstanceProcAddr: " + std::string(SDL_GetError()));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createVulkanInstance();

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    if (false) {
        // Validation messenger
        vk::DebugUtilsMessengerCreateInfoEXT validationMessengerInfo;
        validationMessengerInfo
            .setMessageSeverity(
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            )
            .setMessageType(
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            )
            .pfnUserCallback = &Context::debugUtilsMessengerCallback; // static function
        
        debugMessenger = instance->createDebugUtilsMessengerEXTUnique(validationMessengerInfo);

        // Debug Printf messenger
            vk::DebugUtilsMessengerCreateInfoEXT debugPrintfInfo{};
            debugPrintfInfo
                .setMessageSeverity(
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                )
                .setMessageType(
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
                )
                .pfnUserCallback = &Context::debugPrintfCallback;

            debugPrintfMessenger = instance->createDebugUtilsMessengerEXTUnique(debugPrintfInfo);
    }


    VkSurfaceKHR _surface;
    if (!SDL_Vulkan_CreateSurface(window, instance.get(), nullptr, &_surface))
        throw std::runtime_error("Failed to create window surface with SDL: " + std::string(SDL_GetError()));

    surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});

    pickPhysicalDevice();
    createLogicalDevice();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    // Create command pool
    vk::CommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIndex);
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

    if (rtxSupported)
        poolSizes.emplace_back(vk::DescriptorType::eAccelerationStructureKHR, 16);

    constexpr uint32_t maxSets = 210;
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |  vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    poolInfo.maxSets = maxSets;
    poolInfo.setPoolSizes(poolSizes);
    descriptorPool = device->createDescriptorPoolUnique(poolInfo);

    LOG_INFO( "Context initialized with graphics and compute command pools.");
}

void Context::createVulkanInstance() {

    unsigned int sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (!sdlExtensions)
        throw std::runtime_error("Failed to get Vulkan instance extensions from SDL: " + std::string(SDL_GetError()));

    std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
    std::vector<const char*> layers;
    
    if (EnableValidationLayers) {
        LOG_INFO("INFO: Validation layers are ENABLED.");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
         layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    else
        LOG_INFO( "INFO: Validation layers are DISABLED (Release Mode).");

    constexpr vk::ApplicationInfo appInfo("NoorRay", 1, "No Engine", 1, VK_API_VERSION_1_3);
    
    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&appInfo).setPEnabledLayerNames(layers).setPEnabledExtensionNames(extensions);

    // Enable debug printf feature if validation layers are enabled
    if (false)
    {
        vk::ValidationFeaturesEXT validationFeaturesInfo{};
        const std::vector enabledFeatures = { vk::ValidationFeatureEnableEXT::eDebugPrintf };
        validationFeaturesInfo
            .setEnabledValidationFeatureCount(static_cast<uint32_t>(enabledFeatures.size()))
            .setPEnabledValidationFeatures(enabledFeatures.data());
        instanceInfo.pNext = &validationFeaturesInfo;
    }
    
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif
    
    instance = createInstanceUnique(instanceInfo);
}

void Context::pickPhysicalDevice() {
    const std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if (devices.empty())
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");

    LOG_INFO( "Available GPUs:\n");

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

            LOG_INFO( "  - " << props.deviceName
                      << " (Type: " << vk::to_string(props.deviceType)
                      << ", VRAM: " << (vramSize / (1024 * 1024)) << "MB"
                      << ", Extensions OK: " << (hasAllExtensions ? "Yes" : "No") << ")"
                     );

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

        if (bestDiscrete)
            return bestDiscrete;

        return bestFallback;
    };

    std::vector<const char*> allExtensions = RequiredDeviceExtensions;
    allExtensions.insert(allExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());

    auto best = findBestDevice(allExtensions);
    rtxSupported = best.has_value();

    if (!best) {
        best = findBestDevice(RequiredDeviceExtensions);
        rtxSupported = false;
    }

    if (!best)
        throw std::runtime_error("No suitable GPU found that supports required Vulkan extensions!");

    physicalDevice = best->device;

    //rtxSupported = false;
    if (rtxSupported)
        LOG_INFO( "\nPicked GPU: " << physicalDevice.getProperties().deviceName << " (Ray Tracing Enabled)");
    else
        LOG_INFO( "\nPicked GPU: " << physicalDevice.getProperties().deviceName << " (Ray Tracing Not Supported)");
}

void Context::createLogicalDevice() {

    std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& flags = queueFamilies[i].queueFlags;
        bool hasGraphics = static_cast<bool>(flags & vk::QueueFlagBits::eGraphics);
        bool hasCompute  = static_cast<bool>(flags & vk::QueueFlagBits::eCompute);
        bool hasPresent  = physicalDevice.getSurfaceSupportKHR(i, surface.get());

        if (hasGraphics && hasCompute && hasPresent) {
            queueFamilyIndex = i;
            break;
        }

    }

    if (queueFamilyIndex == UINT32_MAX)
        throw std::runtime_error("Could not find a suitable queue family!");
    
    // Ray tracing extensions
    if (rtxSupported) {
        LOG_INFO( "Ray tracing extensions are supported. Enabling them.");
        RequiredDeviceExtensions.insert(RequiredDeviceExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());
    } else
        LOG_INFO( "Ray tracing extensions are not supported. Proceeding without them.");

    // Prepare device features
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = vk::StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType = vk::StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR;
    vk::PhysicalDeviceVulkan12Features features12{};
    features12.sType = vk::StructureType::ePhysicalDeviceVulkan12Features;
    features12.pNext = &rtFeatures;
    rtFeatures.pNext = &accelFeatures;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.sType = vk::StructureType::ePhysicalDeviceFeatures2;
    features2.pNext = &features12;
    physicalDevice.getFeatures2(&features2);

    auto& coreFeatures = features2.features;

    // Enable core & Vulkan 1.2 features
    coreFeatures.shaderInt64 = VK_TRUE;
    coreFeatures.samplerAnisotropy = VK_TRUE;

    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;

    // Enable ray tracing features if supported
    if (rtxSupported && rtFeatures.rayTracingPipeline && accelFeatures.accelerationStructure) {
        rtFeatures.rayTracingPipeline = VK_TRUE;
        accelFeatures.accelerationStructure = VK_TRUE;
    } else {
        rtFeatures.rayTracingPipeline = VK_FALSE;
        accelFeatures.accelerationStructure = VK_FALSE;
        if (rtxSupported)
            LOG_INFO( "Warning: Ray tracing features requested but not fully supported, disabling them.");
    }

    features12.pNext = &rtFeatures;
    rtFeatures.pNext = &accelFeatures;
    features2.pNext = &features12;

    constexpr float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Create device
    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = vk::StructureType::eDeviceCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(RequiredDeviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = RequiredDeviceExtensions.data();
    deviceCreateInfo.pNext = &features2;
    deviceCreateInfo.pEnabledFeatures = nullptr;

    device = physicalDevice.createDeviceUnique(deviceCreateInfo);

    // Retrieve queue
    queue  = device->getQueue(queueFamilyIndex, 0);
}

uint32_t Context::findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    throw std::runtime_error("Failed to find suitable memory type!");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) {

    const vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(*commandBuffer);
    commandBuffer->end();

    vk::UniqueFence fence = device->createFenceUnique({});
    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    
    (void)device->waitForFences(*fence, VK_TRUE, UINT64_MAX);
}

vk::PresentModeKHR Context::chooseSwapPresentMode() const {
    const std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eImmediate) {
            LOG_INFO( "Present Mode: Immediate (Unlocked, Tearing)");
            return mode;
        }
    }
    
    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            LOG_INFO( "Present Mode: Mailbox (Low-latency, No Tearing)");
            return mode;
        }
    }
    
    LOG_INFO( "Present Mode: FIFO (V-Sync)");
    return vk::PresentModeKHR::eFifo;
}

vk::SurfaceFormatKHR Context::chooseSwapSurfaceFormat() const {
    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface.get());

    for (const auto& availableFormat : availableFormats)
        if (availableFormat.format == vk::Format::eR8G8B8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;

    // fallback to the first available format
    if (!availableFormats.empty())
        return availableFormats[0];
    
    throw std::runtime_error("No suitable swap surface format found! Expected eR8G8B8A8Unorm with SrgbNonlinear color space.");
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::debugUtilsMessengerCallback(
    const vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
        LOG_ERROR("[Validation ERROR] " << pCallbackData->pMessage);
    else if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
        LOG_INFO("[Validation WARNING] " << pCallbackData->pMessage);
    else if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
        LOG_INFO("[Validation INFO] " << pCallbackData->pMessage);
    else
        LOG_INFO("[Validation VERBOSE] " << pCallbackData->pMessage);

    return VK_FALSE;
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::debugPrintfCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    if (pCallbackData && pCallbackData->pMessage)
        LOG_INFO( "[DebugPrintf] " << pCallbackData->pMessage);

    return VK_FALSE;
}

bool Context::queryWindowSize()
{
    int newWidth = 0, newHeight = 0;
    SDL_GetWindowSizeInPixels(window, &newWidth, &newHeight);

    if (newWidth != windowWidth || newHeight != windowHeight) {
        windowWidth = newWidth;
        windowHeight = newHeight;
        return true; // resize occurred
    }
    return false; // no change
}

Context::~Context() {
    LOG_INFO("Destroying Context...");
    if (device)
        device->waitIdle();

    SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}
