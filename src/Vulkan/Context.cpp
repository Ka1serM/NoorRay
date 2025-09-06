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

Context::Context(const int width, const int height) : windowWidth(width), windowHeight(height), dpiScale(1) {
    try {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "[FATAL] Failed to initialize SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to initialize SDL.");
        }

        if (SDL_Vulkan_LoadLibrary(nullptr) < 0) {
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

        queue = device->getQueue(queueFamilyIndex, 0);

        vk::CommandPoolCreateInfo commandPoolInfo;
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

        if(rtxSupported)
            poolSizes.emplace_back( vk::DescriptorType::eAccelerationStructureKHR, 16 );

        constexpr uint32_t maxSets = 210;
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
        poolInfo.maxSets = maxSets;
        poolInfo.setPoolSizes(poolSizes);
        descriptorPool = device->createDescriptorPoolUnique(poolInfo);

    } catch (const vk::Error& e) {
        std::cerr << "[FATAL] Vulkan Error in Context constructor: " << e.what() << std::endl;
        throw std::runtime_error("Vulkan initialization failed.");
    } catch (const std::exception& e) {
        // Log if it's a standard exception that we threw
        std::cerr << "[FATAL] Error in Context constructor: " << e.what() << std::endl;
        throw;
    }
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
        if (!checkValidationLayerSupport())
            std::cerr << "[WARN] Validation layers requested, but not available!" << std::endl;
        else {
            std::cout << "[INFO] Validation layers are ENABLED." << std::endl;
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }

#ifdef __APPLE__
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    vk::ApplicationInfo appInfo("Vulkan Pathtracer", 1, "No Engine", 1, VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.setPApplicationInfo(&appInfo)
                .setPEnabledLayerNames(layers)
                .setPEnabledExtensionNames(extensions);

#ifdef __APPLE__ 
    instanceInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    instance = vk::createInstanceUnique(instanceInfo);
}

bool Context::checkValidationLayerSupport() {
    try {
        std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto& layerProperties : availableLayers)
            if (strcmp("VK_LAYER_KHRONOS_validation", layerProperties.layerName) == 0)
                return true;
        
    } catch (const vk::Error& e) {
        std::cerr << "[WARN] Could not enumerate instance layer properties: " << e.what() << std::endl;
    }
    return false;
}

void Context::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
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

            if (!hasAllExtensions)
                continue;

            Candidate candidate{device, vramSize};
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                if (!bestDiscrete || vramSize > bestDiscrete->vram) bestDiscrete = candidate;
            } else {
                if (!bestFallback || vramSize > bestFallback->vram) bestFallback = candidate;
            }
        }
        if (bestDiscrete.has_value())
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

    if (!best) {
        std::cerr << "[FATAL] No suitable GPU found that supports required Vulkan extensions!" << std::endl;
        throw std::runtime_error("No suitable GPU found!");
    }

    physicalDevice = best->device;
    std::cout << "\n[INFO] Picked GPU: " << physicalDevice.getProperties().deviceName << (rtxSupported ? " (Ray Tracing Enabled)" : " (Ray Tracing Not Supported)") << std::endl;
}

void Context::createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& flags = queueFamilies[i].queueFlags;
        bool hasGraphics = static_cast<bool>(flags & vk::QueueFlagBits::eGraphics);
        bool hasCompute = static_cast<bool>(flags & vk::QueueFlagBits::eCompute);
        bool hasPresent = physicalDevice.getSurfaceSupportKHR(i, surface.get());

        if (hasGraphics && hasCompute && hasPresent) {
            queueFamilyIndex = i;
            break;
        }
    }

    if (queueFamilyIndex == UINT32_MAX) {
        std::cerr << "[FATAL] Could not find a suitable queue family!" << std::endl;
        throw std::runtime_error("Could not find a suitable queue family!");
    }

    auto enabledExtensions = RequiredDeviceExtensions;
    if (rtxSupported) {
        std::cout << "[INFO] Ray tracing extensions are supported. Enabling them." << std::endl;
        enabledExtensions.insert(enabledExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());
    } else
        std::cout << "[INFO] Ray tracing extensions are not supported. Proceeding without them." << std::endl;

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
        if (rtxSupported) {
            std::cout << "[WARN] Ray tracing extensions found, but features not fully supported. Disabling." << std::endl;
        }
    }

    constexpr float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo({}, 1, &queueCreateInfo);
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
        vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
        vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

        commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        func(*commandBuffer);
        commandBuffer->end();

        vk::UniqueFence fence = device->createFenceUnique({});
        vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
        queue.submit(submitInfo, *fence);

        if (device->waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            std::cerr << "[ERROR] Fence wait failed during one-time submit." << std::endl;
        
    } catch (const vk::Error& e) {
        std::cerr << "[ERROR] Vulkan error during one-time submit: " << e.what() << std::endl;
    }
}

vk::SurfaceFormatKHR Context::chooseSwapSurfaceFormat() const {
    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface.get());

    for (const auto& availableFormat : availableFormats)
        if (availableFormat.format == vk::Format::eR8G8B8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;

    throw std::runtime_error("No suitable swap surface format found! Expected eR8G8B8A8Unorm with SrgbNonlinear color space.");
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