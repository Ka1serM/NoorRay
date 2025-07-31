#include "Context.h"
#include <iostream>
#include <set>
#include <map>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

// Add hints for the driver to select the dedicated GPU
extern "C" {
  __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
  __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(int width, int height) {
    // Create window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Vulkan Toy Pathtracer by Marcel K.", nullptr, nullptr);

    // Initialize the dynamic loader
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create Vulkan instance
    createInstance();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    // Create debug messenger
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
    messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);
    messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
    messengerInfo.pfnUserCallback = &debugUtilsMessengerCallback;
    messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);

    // Create surface
    VkSurfaceKHR _surface;
    VkResult res = glfwCreateWindowSurface(instance.get(), window, nullptr, &_surface);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }
    surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});

    // Select the best physical device
    pickPhysicalDevice();

    // Create logical device
    createLogicalDevice();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    // Get the device queue
    queue = device->getQueue(queueFamilyIndices.front(), 0);

    // Create command pool
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIndices.front());
    commandPool = device->createCommandPoolUnique(commandPoolInfo);

    // Create descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eSampler, 64 },
        { vk::DescriptorType::eCombinedImageSampler, 10000 }, // Increased size
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
    
    // Conditionally add acceleration structure to the pool
    if(rtxSupported) {
        poolSizes.push_back({ vk::DescriptorType::eAccelerationStructureKHR, 16 });
    }

    uint32_t maxSets = 210;
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    poolInfo.maxSets = maxSets;
    poolInfo.setPoolSizes(poolSizes);
    descriptorPool = device->createDescriptorPoolUnique(poolInfo);
}

void Context::createInstance() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    vk::ApplicationInfo appInfo("Vulkan Pathtracer", 1, "No Engine", 1, VK_API_VERSION_1_3);
    std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};

    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&appInfo)
                .setPEnabledLayerNames(layers)
                .setPEnabledExtensionNames(extensions);
    instance = createInstanceUnique(instanceInfo);
}

void Context::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::multimap<int, vk::PhysicalDevice> candidates;

    std::cout << "Available GPUs:" << std::endl;
    for (const auto& device : devices) {
        int score = 0;
        vk::PhysicalDeviceProperties properties = device.getProperties();
        vk::PhysicalDeviceMemoryProperties memProperties = device.getMemoryProperties();

        // Strongly prefer discrete GPUs to ensure they are picked over integrated ones
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            score += 100000;
        }

        uint64_t vramSize = 0;
        // Add VRAM size to score (in MB)
        for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
            if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                vramSize += memProperties.memoryHeaps[i].size;
            }
        }
        score += vramSize / (1024 * 1024);
        
        std::cout << "  - " << properties.deviceName 
                  << " (Type: " << vk::to_string(properties.deviceType) 
                  << ", VRAM: " << (vramSize / (1024*1024)) << "MB"
                  << ", Score: " << score << ")" << std::endl;

        candidates.insert(std::make_pair(score, device));
    }

    if (candidates.rbegin()->first > 0) {
        physicalDevice = candidates.rbegin()->second;
    } else {
        throw std::runtime_error("Failed to find a suitable GPU!");
    }
    
    vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
    std::cout << "\nPicked GPU: " << props.deviceName << std::endl;
}

void Context::createLogicalDevice() {
    // Find queue family with support for graphics, compute, and presentation
    std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& flags = queueFamilies[i].queueFlags;
        bool hasGraphics = bool(flags & vk::QueueFlagBits::eGraphics);
        bool hasCompute = bool(flags & vk::QueueFlagBits::eCompute);
        bool hasPresent = physicalDevice.getSurfaceSupportKHR(i, surface.get());

        if (hasGraphics && hasCompute && hasPresent) {
            queueFamilyIndices.push_back(i);
            break;
        }
    }
    if (queueFamilyIndices.empty()) {
        throw std::runtime_error("Could not find a suitable queue family!");
    }

    // --- Device Creation ---
    std::vector<const char*> deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    };
    const std::vector<const char*> rtxDeviceExtensions{
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    
    rtxSupported = checkDeviceExtensionSupport(rtxDeviceExtensions);
    if (rtxSupported) {
        std::cout << "Ray tracing extensions are supported. Enabling them." << std::endl;
        deviceExtensions.insert(deviceExtensions.end(), rtxDeviceExtensions.begin(), rtxDeviceExtensions.end());
    } else {
        std::cout << "Ray tracing extensions are not supported. Proceeding without them." << std::endl;
    }

    // Prepare feature structures for pNext chain
    vk::PhysicalDeviceFeatures2 features2{};
    vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    // Build the pNext chain
    features2.pNext = &indexingFeatures;
    indexingFeatures.pNext = &bufferDeviceAddressFeatures;
    void** pNextTail = &bufferDeviceAddressFeatures.pNext;

    if (rtxSupported) {
        rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
        *pNextTail = &rayTracingPipelineFeatures;
        pNextTail = &rayTracingPipelineFeatures.pNext;
        
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;
        *pNextTail = &accelerationStructureFeatures;
        pNextTail = &accelerationStructureFeatures.pNext;
    }

    constexpr float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndices.front(), 1, &queuePriority);
    vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, deviceExtensions, nullptr, &features2);

    device = physicalDevice.createDeviceUnique(deviceCreateInfo);
}

bool Context::checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
    std::set<std::string> required(requiredExtensions.begin(), requiredExtensions.end());
    // Corrected function call
    std::vector<vk::ExtensionProperties> available = physicalDevice.enumerateDeviceExtensionProperties();
    for (const auto& ext : available) {
        required.erase(ext.extensionName.data());
    }
    return required.empty();
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
    vk::CommandBufferAllocateInfo allocInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo).front());
    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(*commandBuffer);
    commandBuffer->end();
    vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo);
    queue.waitIdle();
}

vk::UniqueDescriptorSet Context::allocateDescSet(vk::DescriptorSetLayout descSetLayout) {
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool.get(), descSetLayout);
    return std::move(device->allocateDescriptorSetsUnique(allocInfo).front());
}

vk::PresentModeKHR Context::chooseSwapPresentMode() const {
    const std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    // Prefer Immediate for unlocked framerate (may cause tearing)
    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eImmediate) {
            std::cout << "Present Mode: Immediate (Unlocked, Tearing)" << std::endl;
            return mode;
        }
    }
    
    // Fallback to Mailbox for low-latency without tearing
    for (const auto& mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            std::cout << "Present Mode: Mailbox (Low-latency, No Tearing)" << std::endl;
            return mode;
        }
    }
    
    // Guaranteed fallback to FIFO (standard V-Sync)
    std::cout << "Present Mode: FIFO (V-Sync)" << std::endl;
    return vk::PresentModeKHR::eFifo;
}

vk::SurfaceFormatKHR Context::chooseSwapSurfaceFormat() const {
    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface.get());

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eR8G8B8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
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
    if (device) {
        device->waitIdle();
    }
    // All vk::Unique handles are automatically destroyed
    glfwDestroyWindow(window);
    glfwTerminate();
}
