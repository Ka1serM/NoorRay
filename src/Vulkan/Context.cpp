#include "Context.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <cstdlib>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(const int width, const int height, const bool isHeadless)
    : windowWidth(width), windowHeight(height), headless(isHeadless)
{
    try {
        if (headless) {
            // Headless mode: use SDL offscreen driver so we can still load the Vulkan
            // library and create a hidden window for surface-based device selection.
            SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
        } else {
#ifdef __linux__
            SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "1");
            SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, "1");

            if (std::getenv("WAYLAND_DISPLAY") && !std::getenv("SDL_VIDEO_DRIVER")) {
                SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
            }
#endif
        }

        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "[FATAL] Failed to initialize SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to initialize SDL.");
        }

        if (!SDL_Vulkan_LoadLibrary(nullptr)) {
            std::cerr << "[FATAL] Failed to load Vulkan library via SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to load Vulkan library.");
        }

        if (!headless) {
            const float dpiScaleFloat = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
            if (dpiScaleFloat != 0.0f) {
                dpiScale = dpiScaleFloat;
                windowWidth  = static_cast<int>(static_cast<float>(windowWidth)  * dpiScale);
                windowHeight = static_cast<int>(static_cast<float>(windowHeight) * dpiScale);
            }
        }

        const SDL_WindowFlags windowFlags = headless
            ? (SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN)
            : (SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

        window = SDL_CreateWindow("NoorRay by Marcel K.", windowWidth, windowHeight, windowFlags);
        if (!window) {
            std::cerr << "[FATAL] Failed to create SDL window: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to create SDL window.");
        }

        const auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
        if (!vkGetInstanceProcAddr) {
            std::cerr << "[FATAL] Failed to get vkGetInstanceProcAddr: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to get vkGetInstanceProcAddr.");
        }

        // Initialize the dispatcher with the function pointer loader. This is needed to find vkCreateInstance.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        createVulkanInstance();

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

#ifdef DEBUG
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
#endif

        VkSurfaceKHR _surface;
        if (!SDL_Vulkan_CreateSurface(window, instance.get(), nullptr, &_surface)) {
            std::cerr << "[FATAL] Failed to create window surface with SDL: " << SDL_GetError() << std::endl;
            throw std::runtime_error("Failed to create window surface.");
        }
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});
        
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

// --- Create the Debug Messenger Info ---
// We will chain this to both the instance creation and create a persistent one later.
vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;

#ifdef DEBUG
std::cout << "[INFO] Enabling Vulkan validation layers." << std::endl;
layers.push_back("VK_LAYER_KHRONOS_validation");
extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

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
validationFeatures.setEnabledValidationFeatures(enabledFeatures);
validationFeatures.pNext = &messengerInfo;
instanceInfo.pNext = &validationFeatures;
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

    // In headless mode the swapchain extension is unnecessary (we never present).
    if (headless) {
        auto& exts = RequiredDeviceExtensions;
        exts.erase(std::remove_if(exts.begin(), exts.end(),
            [](const char* s) { return std::string_view(s) == VK_KHR_SWAPCHAIN_EXTENSION_NAME; }),
            exts.end());
    }

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
    if (rtxSupported) {
        std::cout << "[INFO] Ray tracing extensions are supported. Enabling them." << std::endl;
        enabledExtensions.insert(enabledExtensions.end(), RayTracingExtensions.begin(), RayTracingExtensions.end());
    } else
        std::cout << "[INFO] Ray tracing extensions are not supported. Proceeding without them." << std::endl;
    
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    vk::PhysicalDeviceVulkan12Features features12{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};

    // Build the pNext chain. This creates a linked list of feature structs.
    void** nextFeature = nullptr;

    features12.sType = vk::StructureType::ePhysicalDeviceVulkan12Features;
    nextFeature = &features12.pNext;

    dynamicRenderingFeatures.sType = vk::StructureType::ePhysicalDeviceDynamicRenderingFeatures;
    *nextFeature = &dynamicRenderingFeatures;
    nextFeature = &dynamicRenderingFeatures.pNext;

    if (rtxSupported) {
        accelFeatures.sType = vk::StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR;
        *nextFeature = &accelFeatures;
        nextFeature = &accelFeatures.pNext;

        rayQueryFeatures.sType = vk::StructureType::ePhysicalDeviceRayQueryFeaturesKHR;
        *nextFeature = &rayQueryFeatures;
        nextFeature = &rayQueryFeatures.pNext;
    }

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
    if (rtxSupported) {
        accelFeatures.accelerationStructure = VK_TRUE;
        rayQueryFeatures.rayQuery           = VK_TRUE;
    }

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
    else
        severity = "[VULKAN WARN]";
    
    std::cerr << severity << ": " << pCallbackData->pMessage << std::endl;
    return vk::False;
}

Context::~Context() {
    std::cout << "[INFO] Destroying Context..." << std::endl;
    try {
        if (device)
            device->waitIdle();
    } catch (const vk::Error& e) {
        std::cerr << "[ERROR] Vulkan error during device->waitIdle(): " << e.what() << std::endl;
    }

    SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}
