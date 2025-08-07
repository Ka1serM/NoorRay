#include "Context.h"
#include <iostream>
#include <set>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if !defined(NDEBUG)
    constexpr bool EnableValidationLayers = true;
#else
constexpr bool EnableValidationLayers = false;
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(const int width, const int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        throw std::runtime_error("Failed to initialize SDL: " + std::string(SDL_GetError()));

    if (SDL_Vulkan_LoadLibrary(nullptr) < 0)
        throw std::runtime_error("Failed to load Vulkan library via SDL: " + std::string(SDL_GetError()));

    window = SDL_CreateWindow("Vulkan Toy Path Tracer by Marcel K.", width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window)
        throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));

    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createVulkanInstance();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    if (EnableValidationLayers) {
        vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
        messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);
        messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
        messengerInfo.pfnUserCallback = &debugUtilsMessengerCallback;
        messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
    }

    VkSurfaceKHR _surface;
    if (!SDL_Vulkan_CreateSurface(window, instance.get(), nullptr, &_surface))
         throw std::runtime_error("Failed to create window surface with SDL: " + std::string(SDL_GetError()));

    surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});

    pickPhysicalDevice();
    createLogicalDevice();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    queue = device->getQueue(queueFamilyIndices.front(), 0);

    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIndices.front());
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

    if(rtxSupported)
        poolSizes.push_back({ vk::DescriptorType::eAccelerationStructureKHR, 16 });

    uint32_t maxSets = 210;
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    poolInfo.maxSets = maxSets;
    poolInfo.setPoolSizes(poolSizes);
    descriptorPool = device->createDescriptorPoolUnique(poolInfo);
}

void Context::createVulkanInstance() {
    unsigned int sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (!sdlExtensions)
        throw std::runtime_error("Failed to get Vulkan instance extensions from SDL: " + std::string(SDL_GetError()));

    std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
    std::vector<const char*> layers;

    if (EnableValidationLayers) {
        std::cout << "INFO: Validation layers are ENABLED." << std::endl;
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    } else
        std::cout << "INFO: Validation layers are DISABLED (Release Mode)." << std::endl;

    vk::ApplicationInfo appInfo("Vulkan Pathtracer", 1, "No Engine", 1, VK_API_VERSION_1_3);
    
    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&appInfo).setPEnabledLayerNames(layers).setPEnabledExtensionNames(extensions);
    instance = createInstanceUnique(instanceInfo);
}

void Context::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::cout << "Available GPUs:\n";

    struct Candidate {
        vk::PhysicalDevice device;
        uint64_t vram = 0;
    };

    auto findBestDevice = [&](const std::vector<const char*>& requiredExts) -> std::optional<Candidate> {
        std::optional<Candidate> bestDiscrete, bestFallback;

        for (const auto& device : devices) {
            const auto props = device.getProperties();
            const auto memProps = device.getMemoryProperties();

            // Check for extension support
            std::set<std::string> missing(requiredExts.begin(), requiredExts.end());
            for (const auto& ext : device.enumerateDeviceExtensionProperties())
                missing.erase(ext.extensionName);

            bool hasAllExtensions = missing.empty();

            // VRAM calculation
            uint64_t vramSize = 0;
            for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
                if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                    vramSize += memProps.memoryHeaps[i].size;

            std::cout << "  - " << props.deviceName
                      << " (Type: " << vk::to_string(props.deviceType)
                      << ", VRAM: " << (vramSize / (1024 * 1024)) << "MB"
                      << ", Extensions OK: " << (hasAllExtensions ? "Yes" : "No") << ")"
                      << std::endl;

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

    // First: try with RTX + required extensions
    std::vector<const char*> allExtensions = RequiredDeviceExtensions;
    allExtensions.insert(allExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());

    auto best = findBestDevice(allExtensions);
    rtxSupported = best.has_value();

    // Fallback: required-only extensions
    if (!best) {
        best = findBestDevice(RequiredDeviceExtensions);
        rtxSupported = false;
    }

    if (!best)
        throw std::runtime_error("No suitable GPU found that supports required Vulkan extensions!");

    physicalDevice = best->device;

    std::cout << "\nPicked GPU: " << physicalDevice.getProperties().deviceName << (rtxSupported ? " (Ray Tracing Enabled)" : " (Ray Tracing Not Supported)") << std::endl;
}

void Context::createLogicalDevice() {
    std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& flags = queueFamilies[i].queueFlags;
        bool hasGraphics = static_cast<bool>(flags & vk::QueueFlagBits::eGraphics);
        bool hasCompute = static_cast<bool>(flags & vk::QueueFlagBits::eCompute);
        bool hasPresent = physicalDevice.getSurfaceSupportKHR(i, surface.get());

        if (hasGraphics && hasCompute && hasPresent) {
            queueFamilyIndices.push_back(i);
            break;
        }
    }

    if (queueFamilyIndices.empty())
        throw std::runtime_error("Could not find a suitable queue family!");

    if (rtxSupported) {
        std::cout << "Ray tracing extensions are supported. Enabling them." << std::endl;
        RequiredDeviceExtensions.insert(RequiredDeviceExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());
    } else
        std::cout << "Ray tracing extensions are not supported. Proceeding without them." << std::endl;

    // Feature structs
    vk::PhysicalDeviceFeatures2 features2{};
    vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};

    // Enable base features
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
    
    indexingFeatures.sType = vk::StructureType::ePhysicalDeviceDescriptorIndexingFeatures;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features2.pNext = &indexingFeatures;

    bufferDeviceAddressFeatures.sType = vk::StructureType::ePhysicalDeviceBufferDeviceAddressFeatures;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
    indexingFeatures.pNext = &bufferDeviceAddressFeatures;

    void** pNextTail = &bufferDeviceAddressFeatures.pNext;

    if (rtxSupported) {
        rayTracingPipelineFeatures.sType = vk::StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR;
        rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
        *pNextTail = &rayTracingPipelineFeatures;
        pNextTail = &rayTracingPipelineFeatures.pNext;

        accelerationStructureFeatures.sType = vk::StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR;
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;
        *pNextTail = &accelerationStructureFeatures;
        pNextTail = &accelerationStructureFeatures.pNext;
    }

    // Queue setup
    constexpr float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndices.front(), 1, &queuePriority);

    // Create device
    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = vk::StructureType::eDeviceCreateInfo;
    deviceCreateInfo.pNext = &features2;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(RequiredDeviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = RequiredDeviceExtensions.data();

    device = physicalDevice.createDeviceUnique(deviceCreateInfo);
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    throw std::runtime_error("Failed to find suitable memory type!");
}

void Context::oneTimeSubmitAsync(const std::function<void(vk::CommandBuffer)>& func, vk::Fence fence) const {
    vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(*commandBuffer);
    commandBuffer->end();

    vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, fence);
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
    vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(*commandBuffer);
    commandBuffer->end();

    vk::UniqueFence fence = device->createFenceUnique({});
    vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);

    (void)device->waitForFences(*fence, VK_TRUE, UINT64_MAX);
}

vk::PresentModeKHR Context::chooseSwapPresentMode() const {
    const std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eImmediate) {
            std::cout << "Present Mode: Immediate (Unlocked, Tearing)" << std::endl;
            return mode;
        }
    }
    
    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            std::cout << "Present Mode: Mailbox (Low-latency, No Tearing)" << std::endl;
            return mode;
        }
    }
    
    std::cout << "Present Mode: FIFO (V-Sync)" << std::endl;
    return vk::PresentModeKHR::eFifo;
}

vk::SurfaceFormatKHR Context::chooseSwapSurfaceFormat() const {
    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface.get());

    for (const auto& availableFormat : availableFormats)
        if (availableFormat.format == vk::Format::eR8G8B8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;

    throw std::runtime_error("No suitable swap surface format found! Expected eR8G8B8A8Unorm with SrgbNonlinear color space.");
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::debugUtilsMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return vk::False;
}

Context::~Context() {
    std::cout << "Destroying Context..." << std::endl;
    if (device)
        device->waitIdle();

    SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}