#include "Context.h"
#include <iostream>

#include "Globals.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Context::Context(int width, int height) {

    // Create window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Vulkan Toy Pathtracer by Marcel K.", nullptr, nullptr);

    // Prepare extensions and layers
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create instance
    vk::ApplicationInfo appInfo;
    appInfo.setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&appInfo);
    std::vector layers{"VK_LAYER_KHRONOS_validation"};
    instanceInfo.setPEnabledLayerNames(layers);
    instanceInfo.setPEnabledExtensionNames(extensions);
    instance = createInstanceUnique(instanceInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    //TODO Pick best gpu
    physicalDevice = instance->enumeratePhysicalDevices().back();

    // Create debug messenger
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
    messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    messengerInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
    messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);

    // Create surface
    VkSurfaceKHR _surface;
    VkResult res = glfwCreateWindowSurface(instance.get(), window, nullptr, &_surface);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {instance.get()});

    // Find queue family with support for graphics and compute
    std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (int i = 0; i < queueFamilies.size(); i++) {
        auto flags = queueFamilies[i].queueFlags;
        bool supportGraphics = (flags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{};
        bool supportCompute = (flags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{};
        bool supportPresent = physicalDevice.getSurfaceSupportKHR(i, surface.get());

        if (supportGraphics && supportCompute && supportPresent) {
            queueFamilyIndex = i;
            break;
        }
    }


    // Create device
    constexpr float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.setQueueFamilyIndex(queueFamilyIndex);
    queueCreateInfo.setQueuePriorities(queuePriority);

    const std::vector deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME // BINDLESS
    };

    if (!checkDeviceExtensionSupport(deviceExtensions)) {
        throw std::runtime_error("Some required extensions are not supported");
    }

    vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceBufferDeviceAddressFeatures,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceDescriptorIndexingFeatures,
        vk::PhysicalDeviceFeatures2
    > createInfoChain;

    createInfoChain.get<vk::DeviceCreateInfo>()
        .setQueueCreateInfos(queueCreateInfo)
        .setPEnabledExtensionNames(deviceExtensions);

    createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = VK_TRUE;
    createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64 = VK_TRUE;

    createInfoChain.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>().bufferDeviceAddress = VK_TRUE;
    createInfoChain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline = VK_TRUE;
    createInfoChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure = VK_TRUE;

    auto& indexingFeatures = createInfoChain.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    queue = device->getQueue(queueFamilyIndex, 0);

    // Create command pool
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIndex);
    commandPool = device->createCommandPoolUnique(commandPoolInfo);

    // Create descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes{
        {vk::DescriptorType::eAccelerationStructureKHR, 1},
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageBuffer, 3 * MAX_MESHES}, //BINDLESS, Vertices + Indices + Faces = 3
    {vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES}, //BINDLESS
    };


    vk::DescriptorPoolCreateInfo descPoolInfo;
    descPoolInfo.setPoolSizes(poolSizes);
    descPoolInfo.setMaxSets(1);
    descPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind); //BINDLESS
    descPool = device->createDescriptorPoolUnique(descPoolInfo);
}

bool Context::checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<std::string> requiredExtensionNames(requiredExtensions.begin(), requiredExtensions.end());

    for (const auto& extension : availableExtensions)
        std::erase(requiredExtensionNames, extension.extensionName);

    if (requiredExtensionNames.empty()) // All good, return
        return true;

    std::cout << "The following required extensions are not supported by the device:" << std::endl;
    for (const auto& name : requiredExtensionNames)
        std::cout << "\t" << name << std::endl;

    return false;
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(commandPool.get());
    commandBufferInfo.setCommandBufferCount(1);

    vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(commandBufferInfo).front());
    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(*commandBuffer);
    commandBuffer->end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(*commandBuffer);
    queue.submit(submitInfo);
    queue.waitIdle();
}

vk::UniqueDescriptorSet Context::allocateDescSet(vk::DescriptorSetLayout descSetLayout) {
    vk::DescriptorSetAllocateInfo descSetInfo;
    descSetInfo.setDescriptorPool(descPool.get());
    descSetInfo.setSetLayouts(descSetLayout);
    return std::move(device->allocateDescriptorSetsUnique(descSetInfo).front());
}

VKAPI_ATTR VkBool32 VKAPI_CALL Context::debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                                    void* pUserData)
{
    std::cerr << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}