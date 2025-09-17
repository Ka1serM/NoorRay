#include "Context.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if !defined(NDEBUG)
    constexpr bool EnableValidationLayers = false;
#else
    constexpr bool EnableValidationLayers = false;
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(const int width, const int height) : windowWidth(width), windowHeight(height) {
    try {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "[FATAL] Failed to initialize SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to initialize SDL.");
        }

        if (!SDL_Vulkan_LoadLibrary(nullptr)) {
            std::cerr << "[FATAL] Failed to load Vulkan library via SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to load Vulkan library.");
        }

        const float dpiScaleFloat = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
        if (dpiScaleFloat != 0.0f) {
            dpiScale = dpiScaleFloat;
            windowWidth  = static_cast<int>(static_cast<float>(windowWidth)  * dpiScale);
            windowHeight = static_cast<int>(static_cast<float>(windowHeight) * dpiScale);
        }
        
        window = SDL_CreateWindow("NoorRay by Marcel K.", windowWidth, windowHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!window) {
            std::cerr << "[FATAL] Failed to create SDL window: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to create SDL window.");
        }

        auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
        if (!vkGetInstanceProcAddr) {
            std::cerr << "[FATAL] Failed to get vkGetInstanceProcAddr: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to get vkGetInstanceProcAddr.");
        }

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
        if (!SDL_Vulkan_CreateSurface(window, instance.get(), nullptr, &_surface)) {
            std::cerr << "[FATAL] Failed to create window surface with SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to create window surface.");
        }
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});
        
        pickPhysicalDevice();
        createLogicalDevice();

        VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());
        
        graphicsQueue = device->getQueue(graphicsFamilyIndex, 0);
        computeQueue = device->getQueue(computeFamilyIndex, 0);
        presentQueue = device->getQueue(presentFamilyIndex, 0);

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

        if(rtxSupported)
            poolSizes.emplace_back( vk::DescriptorType::eAccelerationStructureKHR, 16 );

        constexpr uint32_t maxSets = 210;
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
        poolInfo.maxSets = maxSets;
        poolInfo.setPoolSizes(poolSizes);
        descriptorPool = device->createDescriptorPoolUnique(poolInfo);

        createAllocator();

    } catch (const vk::Error& e) {
        std::cerr << "[FATAL] Vulkan Error in Context constructor: " << e.what() << std::endl;
        throw std::runtime_error("Vulkan initialization failed.");
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Error in Context constructor: " << e.what() << std::endl;
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
    
    std::cout << "[INFO] VMA Allocator created successfully." << std::endl;
}

void Context::createVulkanInstance() {
    unsigned int sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (!sdlExtensions) {
        std::cerr << "[FATAL] Failed to get Vulkan instance extensions from SDL: " << SDL_GetError() << std::endl;
        throw std::runtime_error("Failed to get Vulkan instance extensions.");
    }

    std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
    std::vector<const char*> layers;

    if (EnableValidationLayers) {
        std::cout << "[INFO] Validation layers are ENABLED." << std::endl;
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

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

    instance = vk::createInstanceUnique(instanceInfo);
}

void Context::pickPhysicalDevice() {
    const std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        std::cerr << "[FATAL] Failed to find GPUs with Vulkan support!" << std::endl;
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::cout << "[INFO] Available GPUs:\n";

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

            std::cout << "  - " << props.deviceName
                      << " (Type: " << vk::to_string(props.deviceType)
                      << ", VRAM: " << (vramSize / (1024 * 1024)) << "MB"
                      << ", Extensions OK: " << (hasAllExtensions ? "Yes" : "No") << ")"
                      << std::endl;

            if (!hasAllExtensions) {
                for (const auto& extName : missing) {
                    std::cout << "      [MISSING] " << extName << std::endl;
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

    // Try to find a GPU supporting all extensions including RTX
    std::vector<const char*> allExtensions = RequiredDeviceExtensions;
    allExtensions.insert(allExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());

    auto best = findBestDevice(allExtensions);
    rtxSupported = best.has_value();

    if (!best) {
        // RTX not supported: fall back to GPU with just required device extensions
        best = findBestDevice(RequiredDeviceExtensions);
        rtxSupported = false; // fallback: RTX disabled
    }

    if (!best) {
        std::cerr << "[FATAL] No suitable GPU found that supports required Vulkan extensions!" << std::endl;
        throw std::runtime_error("No suitable GPU found!");
    }

    physicalDevice = best->device;
    std::cout << "\n[INFO] Picked GPU: " << physicalDevice.getProperties().deviceName
              << (rtxSupported ? " (Ray Tracing Enabled)" : " (Ray Tracing Not Supported)") << std::endl;
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
        if (physicalDevice.getSurfaceSupportKHR(i, surface.get()))
            foundPresentFamily = i;
        
        if (foundGraphicsFamily == i && foundComputeFamily == i && foundPresentFamily == i)
            break; // Stop early if we find a perfect candidate
    }

    // Second pass (optional but good for performance): Prioritize a dedicated compute queue
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& flags = queueFamilies[i].queueFlags;
        // Look for a queue family that has compute but NOT graphics
        if ((flags & vk::QueueFlagBits::eCompute) && !(flags & vk::QueueFlagBits::eGraphics)) {
            foundComputeFamily = i; // Overwrite with the dedicated async compute queue
            break;
        }
    }
    
    if (!foundGraphicsFamily.has_value() || !foundComputeFamily.has_value() || !foundPresentFamily.has_value()) {
        std::cerr << "[FATAL] Could not find all required queue families (Graphics, Compute, Present)!" << std::endl;
        throw std::runtime_error("Could not find all required queue families!");
    }

    graphicsFamilyIndex = foundGraphicsFamily.value();
    computeFamilyIndex = foundComputeFamily.value();
    presentFamilyIndex = foundPresentFamily.value();
    
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set uniqueQueueFamilies = {graphicsFamilyIndex, computeFamilyIndex, presentFamilyIndex};
    constexpr float queuePriority = 1.0f;
    for (uint32_t familyIndex : uniqueQueueFamilies)
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags{}, familyIndex, 1, &queuePriority);

    auto enabledExtensions = RequiredDeviceExtensions;
    if (rtxSupported) {
        std::cout << "[INFO] Ray tracing extensions are supported. Enabling them." << std::endl;
        enabledExtensions.insert(enabledExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());
    } else
        std::cout << "[INFO] Ray tracing extensions are not supported. Proceeding without them." << std::endl;

    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = vk::StructureType::ePhysicalDeviceDynamicRenderingFeatures;

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = vk::StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType = vk::StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR;

    vk::PhysicalDeviceVulkan12Features features12{};
    features12.sType = vk::StructureType::ePhysicalDeviceVulkan12Features;

    features12.pNext = &rtFeatures;
    rtFeatures.pNext = &accelFeatures;
    dynamicRenderingFeatures.pNext = &features12;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.sType = vk::StructureType::ePhysicalDeviceFeatures2;
    features2.pNext = &dynamicRenderingFeatures;

    physicalDevice.getFeatures2(&features2);

    auto& coreFeatures = features2.features;

    std::cout << "=== Core Features ===" << std::endl;
    std::cout << "shaderInt64: " << coreFeatures.shaderInt64 << std::endl;
    std::cout << "samplerAnisotropy: " << coreFeatures.samplerAnisotropy << std::endl;

    std::cout << "\n=== Vulkan 1.2 Features ===" << std::endl;
    std::cout << "bufferDeviceAddress: " << features12.bufferDeviceAddress << std::endl;
    std::cout << "descriptorIndexing: " << features12.descriptorIndexing << std::endl;
    std::cout << "runtimeDescriptorArray: " << features12.runtimeDescriptorArray << std::endl;
    std::cout << "descriptorBindingPartiallyBound: " << features12.descriptorBindingPartiallyBound << std::endl;
    std::cout << "descriptorBindingSampledImageUpdateAfterBind: " << features12.descriptorBindingSampledImageUpdateAfterBind << std::endl;
    std::cout << "descriptorBindingVariableDescriptorCount: " << features12.descriptorBindingVariableDescriptorCount << std::endl;

    std::cout << "\n=== Ray Tracing Features ===" << std::endl;
    std::cout << "rayTracingPipeline: " << rtFeatures.rayTracingPipeline << std::endl;
    std::cout << "accelerationStructure: " << accelFeatures.accelerationStructure << std::endl;

    if (dynamicRenderingFeatures.dynamicRendering) {
        dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
    } else {
        throw std::runtime_error("Dynamic Rendering feature is not supported on this device!");
    }

    coreFeatures.shaderInt64 = VK_TRUE;
    coreFeatures.samplerAnisotropy = VK_TRUE;

    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    
    if (rtxSupported && rtFeatures.rayTracingPipeline && accelFeatures.accelerationStructure) {
        rtFeatures.rayTracingPipeline = VK_TRUE;
        accelFeatures.accelerationStructure = VK_TRUE;
    } else {
        rtFeatures.rayTracingPipeline = VK_FALSE;
        accelFeatures.accelerationStructure = VK_FALSE;
        if (rtxSupported)
            std::cout << "[WARN] Ray tracing extensions found, but features not fully supported. Disabling." << std::endl;
    }

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.setQueueCreateInfos(queueCreateInfos);
    deviceCreateInfo.setPEnabledExtensionNames(enabledExtensions);
    deviceCreateInfo.pNext = &features2;

    device = physicalDevice.createDeviceUnique(deviceCreateInfo);
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    std::cerr << "[FATAL] Failed to find suitable memory type!" << std::endl;
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
            std::cerr << "[ERROR] Fence wait failed during one-time submit." << std::endl;
        
    } catch (const vk::Error& e) {
        std::cerr << "[ERROR] Vulkan error during one-time submit: " << e.what() << std::endl;
    }
}

vk::PresentModeKHR Context::chooseSwapPresentMode() const {
    const std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    for (const auto& mode : availablePresentModes)
        if (mode == vk::PresentModeKHR::eMailbox) {
            std::cout << "Present Mode: Mailbox (Low-latency, No Tearing)" << std::endl;
            return mode;
        }
    
    for (const auto& mode : availablePresentModes)
        if (mode == vk::PresentModeKHR::eFifo) {
            std::cout << "Present Mode: FIFO (V-Sync)" << std::endl;
            return mode;
        }
    
    std::cout << "Present Mode: Immediate (Unlocked, Tearing)" << std::endl;
    return vk::PresentModeKHR::eImmediate;
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
    else if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
        severity = "[VULKAN WARN]";
    else
        return vk::False; // Don't log info or verbose messages for brevity
    
    std::cerr << severity << ": " << pCallbackData->pMessage << std::endl;
    return vk::False;
}

void Context::queryWindowSize()
{
    SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight);
}

Context::~Context() {
    std::cout << "[INFO] Destroying Context..." << std::endl;
    try {
        if (device) {
            device->waitIdle();
        }
    } catch (const vk::Error& e) {
        std::cerr << "[ERROR] Vulkan error during device->waitIdle(): " << e.what() << std::endl;
    }

    SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}