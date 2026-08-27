#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "gpu/gpu.hpp"
#include "gpu/interop.hpp"
#include "internal.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace gpu::detail {

namespace {
[[noreturn]] void throw_vk(const vk::SystemError& error, const char* operation) {
    throw Error(ErrorCode::InvalidState, std::string(operation) + ": " + error.what());
}

vk::Bool32 VKAPI_CALL debug_callback(
    const vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (data && data->pMessage) {
        const char* label =
            severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ? "error"
            : severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning ? "warning"
            : "info";
        std::fprintf(stderr, "[gpu validation %s] %s\n", label, data->pMessage);
    }
    return vk::False;
}

bool has_extension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::string_view(extension.extensionName) == name;
    });
}
}

DeviceImpl::DeviceImpl(const DeviceConfig& config) {
    surface_provider_ = config.surface;
    try {
        create_instance(config);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance());
        // The surface has to exist before the physical device is chosen: a
        // device that cannot present to it is not a candidate.
        create_surface(config);
        select_physical_device();
        create_device(config);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_device());
        create_allocator();
        create_command_state();
    } catch (const vk::SystemError& error) {
        throw_vk(error, "Vulkan device initialization failed");
    }
}

// The mandatory set is what the whole library is built on: buffer device
// addresses for every buffer, a timeline semaphore for all synchronization,
// synchronization2 and dynamic rendering for command recording, the descriptor
// heap for every pipeline's root arguments, and unified image layouts so no
// image ever needs a layout transition.
DeviceImpl::Capabilities DeviceImpl::probe(const vk::PhysicalDevice candidate) {
    Capabilities capabilities{};
    const auto properties = candidate.getProperties();
    if (properties.apiVersion < VK_API_VERSION_1_3)
        return capabilities;

    const auto extensions = candidate.enumerateDeviceExtensionProperties();
    const bool has_heap = has_extension(extensions, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME)
        && has_extension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    const bool has_unified_layouts = has_extension(
        extensions, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
    const bool has_as = has_extension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        && has_extension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool has_query = has_as && has_extension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    const bool has_pipeline = has_as
        && has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);

    vk::PhysicalDeviceVulkan11Features supported11{};
    vk::PhysicalDeviceVulkan12Features supported12{};
    vk::PhysicalDeviceVulkan13Features supported13{};
    vk::PhysicalDeviceMaintenance5Features supported_maintenance5{};
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT supported_heap{};
    vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR supported_unified{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR supported_as{};
    vk::PhysicalDeviceRayQueryFeaturesKHR supported_query{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR supported_rt{};
    vk::PhysicalDeviceFeatures2 supported{};
    // Chain unconditionally: querying a feature struct whose extension is
    // absent simply reports it unsupported.
    supported.pNext = &supported11;
    supported11.pNext = &supported12;
    supported12.pNext = &supported13;
    supported13.pNext = &supported_maintenance5;
    supported_maintenance5.pNext = &supported_heap;
    supported_heap.pNext = &supported_unified;
    supported_unified.pNext = &supported_as;
    supported_as.pNext = &supported_query;
    supported_query.pNext = &supported_rt;
    candidate.getFeatures2(&supported);

    bool has_compute = false;
    for (const auto& queue : candidate.getQueueFamilyProperties())
        has_compute |= static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);

    capabilities.mandatory = has_compute
        && supported12.bufferDeviceAddress && supported12.timelineSemaphore
        && supported12.runtimeDescriptorArray
        && supported13.synchronization2 && supported13.dynamicRendering
        && has_heap && supported_maintenance5.maintenance5 && supported_heap.descriptorHeap
        && has_unified_layouts && supported_unified.unifiedImageLayouts;
    capabilities.acceleration_structure = has_as && supported_as.accelerationStructure;
    capabilities.ray_query = has_query && capabilities.acceleration_structure
        && supported_query.rayQuery;
    capabilities.ray_tracing = has_pipeline && capabilities.acceleration_structure
        && supported_rt.rayTracingPipeline;
    return capabilities;
}

void DeviceImpl::adopt_capabilities(const Capabilities& capabilities) {
    acceleration_structure_supported_ = capabilities.acceleration_structure;
    ray_query_supported_ = capabilities.ray_query;
    ray_tracing_supported_ = capabilities.ray_tracing;
    features_.ray_query = ray_query_supported_;
    features_.ray_tracing = ray_tracing_supported_;
}

void DeviceImpl::initialize_resources() {
    create_descriptor_heap();
    argument_arena_ = create_buffer(argument_arena_size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
}

void DeviceImpl::create_instance(const DeviceConfig& config) {
    // A windowing toolkit loads its own Vulkan library, and a surface created
    // against a different loader is undefined. When one is present, its
    // loader is the one the whole device uses.
    auto loader = vkGetInstanceProcAddr;
    if (config.surface) {
        if (const auto provided = config.surface->instance_proc_address())
            loader = reinterpret_cast<PFN_vkGetInstanceProcAddr>(provided);
    }
    if (!loader)
        throw Error(ErrorCode::InvalidState, "Vulkan loader did not provide vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(loader);

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    // Platform extensions are supplied by the window provider below.
    if (config.surface)
        for (const char* extension : config.surface->instance_extensions())
            extensions.push_back(extension);
    if (config.enable_validation) {
        for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
            if (std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                break;
            }
        }
        if (!layers.empty())
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const std::string application_name(config.application_name);
    const vk::ApplicationInfo appInfo(application_name.c_str(), 1, "gpu", 1, VK_API_VERSION_1_3);
    vk::InstanceCreateInfo createInfo{};
    createInfo.setPApplicationInfo(&appInfo)
        .setPEnabledLayerNames(layers)
        .setPEnabledExtensionNames(extensions);
    instance_ = vk::createInstanceUnique(createInfo);

    if (!layers.empty()) {
        // VK_EXT_debug_utils is requested only with the validation layer. Do
        // not infer this from the platform surface extensions above.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_.get());
        messenger_ = instance_->createDebugUtilsMessengerEXTUnique({{},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debug_callback});
    }
}

void DeviceImpl::create_surface(const DeviceConfig& config) {
    if (!config.surface)
        return;
    const std::uintptr_t raw = config.surface->create_surface(
        reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(vk_instance())));
    if (!raw)
        throw Error(ErrorCode::InvalidState, "SurfaceProvider failed to create a surface");
    surface_ = vk::UniqueSurfaceKHR(vk::SurfaceKHR(reinterpret_cast<VkSurfaceKHR>(raw)),
        vk::detail::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>(vk_instance()));
}

void DeviceImpl::select_physical_device() {
    const auto devices = vk_instance().enumeratePhysicalDevices();
    if (devices.empty())
        throw Error(ErrorCode::UnsupportedFeature, "No Vulkan physical device is available");

    vk::PhysicalDevice best;
    Capabilities best_capabilities{};
    vk::DeviceSize best_memory = 0;
    bool best_is_discrete = false;
    for (const auto candidate : devices) {
        const Capabilities capabilities = probe(candidate);
        if (!capabilities.mandatory)
            continue;
        if (surface_ && !has_extension(candidate.enumerateDeviceExtensionProperties(),
                VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            continue;

        vk::DeviceSize memory = 0;
        const auto memory_properties = candidate.getMemoryProperties();
        for (std::uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
            const auto& heap = memory_properties.memoryHeaps[i];
            if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                memory += heap.size;
        }
        const bool is_discrete =
            candidate.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
        if (!best || (is_discrete && !best_is_discrete)
            || (is_discrete == best_is_discrete && memory > best_memory)) {
            best = candidate;
            best_capabilities = capabilities;
            best_memory = memory;
            best_is_discrete = is_discrete;
        }
    }
    if (!best)
        throw Error(ErrorCode::UnsupportedFeature,
            "No Vulkan 1.3 device supports buffer device address, timeline semaphores, "
            "synchronization2, dynamic rendering, VK_EXT_descriptor_heap and "
            "VK_KHR_unified_image_layouts");
    physical_device_ = best;
    adopt_capabilities(best_capabilities);
}

void DeviceImpl::create_device(const DeviceConfig&) {
    const auto queues = physical_device_.getQueueFamilyProperties();
    std::optional<std::uint32_t> selected;
    for (std::uint32_t i = 0; i < queues.size(); ++i) {
        if (!(queues[i].queueFlags & vk::QueueFlagBits::eCompute))
            continue;
        // A presenting device needs one queue that can do everything: the
        // frame's dispatches, its draws and its present all go through the
        // single queue this library owns.
        if (surface_ && !physical_device_.getSurfaceSupportKHR(i, surface_.get()))
            continue;
        // Prefer a unified graphics+compute queue. The public API exposes
        // both dispatch and raster operations, so selecting a compute-only
        // queue makes Device::render invalid even though compute works.
        if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            selected = i;
            break;
        }
        if (!selected)
            selected = i;
    }
    if (!selected)
        throw Error(ErrorCode::UnsupportedFeature, surface_
            ? "Selected Vulkan device has no queue that supports both compute and presentation"
            : "Selected Vulkan device has no compute queue");
    queue_family_ = *selected;
    constexpr float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo({}, queue_family_, 1, &priority);

    const auto extensions = physical_device_.enumerateDeviceExtensionProperties();
    std::vector<const char*> enabled_extensions{
        VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
        // VK_EXT_descriptor_heap depends on maintenance5, and omitting it is a
        // spec violation the validation layers reject outright.
        VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
        VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
    };
    if (surface_)
        enabled_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    external_memory_fd_enabled_ = has_extension(extensions,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    if (external_memory_fd_enabled_)
        enabled_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    external_semaphore_fd_enabled_ = has_extension(extensions,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    if (external_semaphore_fd_enabled_)
        enabled_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    // Slang lowers descriptor-heap access through untyped pointers. It is not
    // required by this library's own recording, but shaders compiled for the
    // heap ABI need it, so enable it whenever the driver offers it.
    if (has_extension(extensions, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME))
        enabled_extensions.push_back(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);

    const vk::PhysicalDeviceFeatures supported_base = physical_device_.getFeatures();
    vk::PhysicalDeviceFeatures base{};
    if (supported_base.shaderInt64)
        base.shaderInt64 = VK_TRUE;

    // shaderDrawParameters is declared by Slang-compiled vertex shaders; the
    // validation layers reject the module without it.
    vk::PhysicalDeviceVulkan11Features features11{};
    features11.shaderDrawParameters = VK_TRUE;
    vk::PhysicalDeviceVulkan12Features features12{};
    features12.bufferDeviceAddress = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    // Descriptor-heap shaders address resources through unbounded arrays, so
    // the SPIR-V they produce declares RuntimeDescriptorArray.
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    vk::PhysicalDeviceVulkan13Features features13{};
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    vk::PhysicalDeviceMaintenance5Features maintenance5{};
    maintenance5.maintenance5 = VK_TRUE;
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT heap_features{};
    heap_features.descriptorHeap = VK_TRUE;
    vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_layouts{};
    unified_layouts.unifiedImageLayouts = VK_TRUE;
    features11.pNext = &features12;
    features12.pNext = &features13;
    features13.pNext = &maintenance5;
    maintenance5.pNext = &heap_features;
    heap_features.pNext = &unified_layouts;
    void** tail = &unified_layouts.pNext;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    vk::PhysicalDeviceRayQueryFeaturesKHR query_features{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rt_features{};
    if (acceleration_structure_supported_) {
        as_features.accelerationStructure = VK_TRUE;
        *tail = &as_features;
        tail = &as_features.pNext;
        enabled_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enabled_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        if (ray_query_supported_) {
            query_features.rayQuery = VK_TRUE;
            *tail = &query_features;
            tail = &query_features.pNext;
            enabled_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        }
        if (ray_tracing_supported_) {
            rt_features.rayTracingPipeline = VK_TRUE;
            *tail = &rt_features;
            tail = &rt_features.pNext;
            enabled_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        }
    }

    vk::DeviceCreateInfo createInfo{};
    createInfo.setPEnabledFeatures(&base)
        .setQueueCreateInfos(queueInfo)
        .setPEnabledExtensionNames(enabled_extensions)
        .setPNext(&features11);
    device_ = physical_device_.createDeviceUnique(createInfo);
    queue_ = vk_device().getQueue(queue_family_, 0);
}

void DeviceImpl::create_allocator() {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo info{};
    info.instance = vk_instance();
    info.physicalDevice = physical_device_;
    info.device = vk_device();
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    info.pVulkanFunctions = &functions;
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (vmaCreateAllocator(&info, &allocator_) != VK_SUCCESS)
        throw Error(ErrorCode::OutOfMemory, "VMA allocator creation failed");
}

void DeviceImpl::create_command_state() {
    command_pool_ = vk_device().createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                        queue_family_});
    vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, 0);
    timeline_ = vk_device().createSemaphoreUnique({{}, &timelineInfo});
    timestamp_query_pool_ = vk_device().createQueryPoolUnique(
        {{}, vk::QueryType::eTimestamp, 256});
    timestamp_period_ns_ = physical_device_.getProperties().limits.timestampPeriod;
    const auto queue_properties = physical_device_.getQueueFamilyProperties();
    if (queue_family_ < queue_properties.size())
        timestamp_valid_bits_ = queue_properties[queue_family_].timestampValidBits;
}

std::shared_ptr<TimestampQuery::State> DeviceImpl::create_timestamp() {
    if (!timestamp_query_pool_ || next_timestamp_query_ + 2u > 256u)
        throw Error(ErrorCode::OutOfMemory, "GPU timestamp query capacity exhausted");
    auto state = std::make_shared<TimestampQuery::State>();
    state->device = self_.lock();
    state->first_query = next_timestamp_query_;
    next_timestamp_query_ += 2u;
    return state;
}

void DeviceImpl::measure(const std::shared_ptr<TimestampQuery::State>& state,
    const std::function<void()>& commands) {
    if (!state || !timestamp_query_pool_ || state->first_query + 1u >= 256u)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU timestamp query");
    const std::uint32_t first = state->first_query;
    // The bracket is recorded into whatever command buffer the enclosed work
    // goes into, so a measured scope inside a frame times only that scope
    // rather than the whole frame.
    submit([first, this](const vk::CommandBuffer command) {
        command.resetQueryPool(*timestamp_query_pool_, first, 2u);
        command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe,
            *timestamp_query_pool_, first);
    });
    commands();
    submit([first, this](const vk::CommandBuffer command) {
        command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe,
            *timestamp_query_pool_, first + 1u);
    });
}

double DeviceImpl::timestamp_milliseconds(TimestampQuery::State& query) {
    synchronize();
    std::uint64_t values[2]{};
    const VkResult result = vkGetQueryPoolResults(
        static_cast<VkDevice>(vk_device()),
        static_cast<VkQueryPool>(*timestamp_query_pool_), query.first_query, 2u,
        sizeof(values), values, sizeof(values[0]),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS)
        throw Error(ErrorCode::InvalidState, "failed to read GPU timestamps");
    std::uint64_t ticks = values[1] - values[0];
    if (timestamp_valid_bits_ != 0u && timestamp_valid_bits_ < 64u)
    {
        const std::uint64_t mask = (std::uint64_t{1} << timestamp_valid_bits_) - 1u;
        ticks = (values[1] - values[0]) & mask;
    }
    return static_cast<double>(ticks) * timestamp_period_ns_ * 1.0e-6;
}

void DeviceImpl::create_descriptor_heap() {
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &descriptor_properties_;
    if (acceleration_structure_supported_) {
        descriptor_properties_.pNext = &acceleration_structure_properties_;
        acceleration_structure_properties_.pNext = &ray_tracing_properties_;
    }
    physical_device_.getProperties2(&properties);
    descriptor_stride_ = std::max(resource_descriptor_slot_size,
        std::max({descriptor_properties_.bufferDescriptorSize,
            descriptor_properties_.imageDescriptorSize,
            descriptor_properties_.samplerDescriptorSize,
            descriptor_properties_.bufferDescriptorAlignment,
            descriptor_properties_.imageDescriptorAlignment,
            vk::DeviceSize(1)}));
    const vk::DeviceSize alignment = std::max(descriptor_properties_.resourceHeapAlignment,
        vk::DeviceSize(1));
    const vk::DeviceSize reserved = (descriptor_properties_.minResourceHeapReservedRange
        + alignment - 1) / alignment * alignment;
    const vk::DeviceSize capacity = reserved + descriptor_stride_ * descriptor_capacity + alignment;
    descriptor_heap_ = create_buffer(capacity,
        vk::BufferUsageFlagBits::eDescriptorHeapEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    descriptor_heap_offset_ = (alignment - descriptor_heap_->address % alignment) % alignment;

    if (descriptor_properties_.samplerDescriptorSize != 0) {
        sampler_stride_ = std::max(descriptor_properties_.samplerDescriptorSize,
            descriptor_properties_.samplerDescriptorAlignment);
        const vk::DeviceSize sampler_alignment = std::max(
            descriptor_properties_.samplerHeapAlignment, vk::DeviceSize(1));
        sampler_heap_ = create_buffer(
            descriptor_properties_.maxSamplerHeapSize + sampler_alignment,
            vk::BufferUsageFlagBits::eDescriptorHeapEXT
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_CPU_TO_GPU, true);
        sampler_heap_offset_ = (sampler_alignment - sampler_heap_->address % sampler_alignment)
            % sampler_alignment;
    }
}

void DeviceImpl::bind_heaps(const vk::CommandBuffer command) const {
    if (!descriptor_heap_)
        return;
    const vk::DeviceSize descriptor_bytes = descriptor_stride_ * descriptor_capacity;
    const vk::DeviceSize reserved = descriptor_properties_.minResourceHeapReservedRange;
    const vk::DeviceAddressRangeEXT range{
        descriptor_heap_->address + descriptor_heap_offset_, descriptor_bytes + reserved};
    command.bindResourceHeapEXT({range, descriptor_bytes, reserved});
    if (sampler_heap_) {
        const vk::DeviceSize sampler_reserved = descriptor_properties_.minSamplerHeapReservedRange;
        const vk::DeviceSize sampler_heap_size = descriptor_properties_.maxSamplerHeapSize;
        const vk::DeviceSize sampler_bytes = sampler_heap_size > sampler_reserved
            ? sampler_heap_size - sampler_reserved : 0;
        const vk::DeviceAddressRangeEXT sampler_range{
            sampler_heap_->address + sampler_heap_offset_, sampler_heap_size};
        command.bindSamplerHeapEXT({sampler_range, sampler_bytes, sampler_reserved});
    }
}

void DeviceImpl::retain_active(std::shared_ptr<void> resource) {
    if (resource && active_command_)
        active_resources_.push_back(std::move(resource));
}

void DeviceImpl::release_resource(const ResourceHandle handle) {
    if (!handle || handle.value >= descriptor_capacity)
        return;
    std::lock_guard descriptor_lock(descriptor_mutex_);
    free_descriptors_.push_back(handle.value);
}

void DeviceImpl::release_sampler(const SamplerHandle handle) {
    if (!handle)
        return;
    std::lock_guard descriptor_lock(descriptor_mutex_);
    free_sampler_descriptors_.push_back(handle.value);
}

void DeviceImpl::shutdown() noexcept {
    if (shut_down_)
        return;
    shut_down_ = true;
    try {
        synchronize();
    } catch (...) {
        // Destructors cannot report device-loss errors.
    }
    active_resources_.clear();
    pending_.clear();
    // The queue is idle, so everything still deferred can be released.
    std::vector<Retired> retired;
    {
        std::lock_guard retire_lock(retire_mutex_);
        retired.swap(retired_);
    }
    for (auto& entry : retired) {
        if (entry.release)
            entry.release();
    }
    // These internal buffers own VMA allocations and are destroyed before the
    // allocator itself. External resource objects keep DeviceImpl alive, so
    // no user-visible resource should remain here.
    argument_arena_.reset();
    descriptor_heap_.reset();
    sampler_heap_.reset();
}

DeviceImpl::~DeviceImpl() {
    shutdown();
    if (allocator_)
        vmaDestroyAllocator(allocator_);
}

std::shared_ptr<BufferImpl> DeviceImpl::create_buffer(const std::size_t size,
    const vk::BufferUsageFlags usage, const VmaMemoryUsage memory_usage, const bool mapped) {
    if (size == 0)
        throw Error(ErrorCode::InvalidArgument, "GPU buffers cannot have zero bytes");
    vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memory_usage;
    if (mapped)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationResult{};
    VkBuffer rawBuffer = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator_, reinterpret_cast<const VkBufferCreateInfo*>(&bufferInfo),
                        &allocationInfo, &rawBuffer, &allocation, &allocationResult) != VK_SUCCESS)
        throw Error(ErrorCode::OutOfMemory, "VMA buffer allocation failed");

    auto result = std::make_shared<BufferImpl>();
    result->device = self_.lock();
    result->buffer = rawBuffer;
    result->allocation = allocation;
    result->size = size;
    result->mapped = allocationResult.pMappedData;
    if (mapped && !result->mapped) {
        if (vmaMapMemory(allocator_, allocation, &result->mapped) != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, rawBuffer, allocation);
            throw Error(ErrorCode::OutOfMemory, "VMA could not map a host-visible buffer");
        }
    }
    result->host_visible = result->mapped != nullptr;
    result->mapped_by_api = mapped && allocationResult.pMappedData == nullptr;
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        result->address = buffer_address(result->buffer);
        // A freed buffer's device address can be handed straight back to the
        // next allocation, so the new owner must replace any dead entry rather
        // than losing to it.
        buffers_.insert_or_assign(result->address, result);
    }
    return result;
}

vk::DeviceAddress DeviceImpl::buffer_address(const vk::Buffer buffer) const {
    if (!buffer)
        return 0;
    return vk_device().getBufferAddress({buffer});
}

std::pair<vk::Buffer, vk::DeviceSize> DeviceImpl::find_buffer(const vk::DeviceAddress address) const {
    const auto buffer = find_buffer_resource(address);
    return {buffer->buffer, address - buffer->address};
}

std::shared_ptr<BufferImpl> DeviceImpl::find_buffer_resource(const vk::DeviceAddress address) const {
    // buffers_ is keyed by base address, so the candidate is the last buffer
    // that starts at or before `address`.
    const auto candidate = buffers_.upper_bound(address);
    if (candidate != buffers_.begin()) {
        const auto buffer = std::prev(candidate)->second.lock();
        if (buffer && address >= buffer->address && address < buffer->address + buffer->size)
            return buffer;
    }
    throw Error(ErrorCode::InvalidResource, "GPU address does not refer to a live buffer");
}

interop::DeviceHandles DeviceImpl::native_handles() const noexcept {
    return {
        reinterpret_cast<std::uintptr_t>(static_cast<VkInstance>(vk_instance())),
        reinterpret_cast<std::uintptr_t>(static_cast<VkPhysicalDevice>(physical_device_)),
        reinterpret_cast<std::uintptr_t>(static_cast<VkDevice>(vk_device())),
        reinterpret_cast<std::uintptr_t>(static_cast<VkQueue>(queue_)),
        queue_family_,
    };
}

std::shared_ptr<ImageImpl> DeviceImpl::find_image(const ImageHandle handle) const {
    if (!handle)
        return {};
    for (const auto& weak : images_) {
        if (const auto image = weak.lock(); image
            && (image->handle.value == handle.value
                || image->sampled_handle.value == handle.value
                || image->storage_handle.value == handle.value))
            return image;
    }
    return {};
}

std::shared_ptr<ShaderImpl> DeviceImpl::create_shader(const std::span<const std::byte> spirv,
    const std::string_view entry_point) {
    if (spirv.empty() || spirv.size_bytes() % sizeof(std::uint32_t) != 0)
        throw Error(ErrorCode::InvalidShader, "SPIR-V must be non-empty and 4-byte aligned");
    const auto* words = reinterpret_cast<const std::uint32_t*>(spirv.data());
    if (reinterpret_cast<std::uintptr_t>(spirv.data()) % alignof(std::uint32_t) != 0)
        throw Error(ErrorCode::InvalidShader, "SPIR-V data is not 4-byte aligned");
    if (words[0] != 0x07230203u)
        throw Error(ErrorCode::InvalidShader, "SPIR-V magic number is invalid");
    if (entry_point.empty())
        throw Error(ErrorCode::InvalidArgument, "shader entry point cannot be empty");

    vk::ShaderModuleCreateInfo info({}, spirv.size_bytes(), words);
    auto result = std::make_shared<ShaderImpl>();
    result->device = self_.lock();
    result->entry_point = entry_point;
    try {
        result->module = vk_device().createShaderModuleUnique(info);
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    return result;
}

std::shared_ptr<ComputePipelineImpl> DeviceImpl::create_compute(const Shader& shader) {
    if (!shader.impl_)
        throw Error(ErrorCode::InvalidResource,
            "cannot create a compute pipeline from an empty shader");
    auto result = std::make_shared<ComputePipelineImpl>();
    result->device = self_.lock();
    const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute,
        *shader.impl_->module, shader.impl_->entry_point.c_str());
    const vk::PipelineCreateFlags2CreateInfo flags2{
        vk::PipelineCreateFlagBits2::eDescriptorHeapEXT};
    const vk::ComputePipelineCreateInfo pipelineInfo{{}, stage, nullptr, {}, {}, &flags2};
    try {
        result->pipeline = vk_device().createComputePipelineUnique({}, pipelineInfo).value;
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }
    return result;
}

std::shared_ptr<SamplerImpl> DeviceImpl::create_sampler(const SamplerDesc& desc) {
    const auto filter = desc.filter == Filter::Linear ? vk::Filter::eLinear : vk::Filter::eNearest;
    const auto address = [](const AddressMode mode) {
        switch (mode) {
        case AddressMode::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
        case AddressMode::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
        case AddressMode::ClampToBorder: return vk::SamplerAddressMode::eClampToBorder;
        default: return vk::SamplerAddressMode::eRepeat;
        }
    };
    vk::SamplerCreateInfo info({}, filter, filter, vk::SamplerMipmapMode::eLinear,
        address(desc.address_u), address(desc.address_v), address(desc.address_w));
    auto result = std::make_shared<SamplerImpl>();
    result->device = self_.lock();
    result->sampler = vk_device().createSamplerUnique(info);
    result->handle = sampler_heap_ ? write_sampler_descriptor(info)
                                   : SamplerHandle{allocate_handle()};
    samplers_.push_back(result);
    return result;
}

GpuToken DeviceImpl::submit(const std::function<void(vk::CommandBuffer)>& record,
    std::vector<std::shared_ptr<void>> resources) {
    std::lock_guard lock(mutex_);
    if (shut_down_)
        throw Error(ErrorCode::InvalidState, "gpu::Device has been shut down");
    if (frame_command_) {
        // A frame is open: batch into its command buffer instead of opening a
        // submission of our own. Every dispatch, trace and render scope issued
        // between begin_frame and end_frame therefore lands in one submission.
        //
        // The ordering barriers around the recorded work are the same ones a
        // standalone submission gets, so batching does not weaken the implicit
        // ordering callers already rely on between consecutive operations.
        vk::MemoryBarrier2 ordering{};
        ordering.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
            .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
            .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
        frame_command_.pipelineBarrier2({{}, ordering, {}, {}});
        active_resources_.clear();
        record(frame_command_);
        frame_command_.pipelineBarrier2({{}, ordering, {}, {}});
        // The frame's submission is what will retire these, so they are held
        // until end_frame hands them to the pending queue.
        frame_resources_.insert(frame_resources_.end(),
            std::make_move_iterator(resources.begin()), std::make_move_iterator(resources.end()));
        frame_resources_.insert(frame_resources_.end(),
            active_resources_.begin(), active_resources_.end());
        active_resources_.clear();
        return frame_token_;
    }
    reap_completed();
    auto commands = vk_device().allocateCommandBuffersUnique(
        {*command_pool_, vk::CommandBufferLevel::ePrimary, 1});
    if (commands.empty())
        throw Error(ErrorCode::OutOfMemory, "Vulkan command-buffer allocation failed");
    auto command = std::move(commands.front());
    command->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    bind_heaps(command.get());
    vk::MemoryBarrier2 ordering{};
    ordering.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    command->pipelineBarrier2({{}, ordering, {}, {}});
    active_resources_.clear();
    record(command.get());
    resources.insert(resources.end(), active_resources_.begin(), active_resources_.end());
    active_resources_.clear();
    // Publish writes made by this submission to later queue submissions. This
    // is required for upload copies consumed by externally recorded NoorRay
    // commands, which are submitted directly to the same queue after this
    // API-owned command buffer.
    command->pipelineBarrier2({{}, ordering, {}, {}});
    command->end();

    const GpuToken token{next_timeline_++};
    vk::TimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.setSignalSemaphoreValues(token.value);
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(command.get()).setSignalSemaphores(timeline_.get());
    submitInfo.pNext = &timelineInfo;
    queue_.submit(submitInfo);
    pending_.push_back({token, std::move(command), std::move(resources)});
    return token;
}

void DeviceImpl::abandon_frame(Frame::State& state) {
    std::lock_guard lock(mutex_);
    if (frame_command_ == state.command)
        frame_command_ = nullptr;
    // Nothing was submitted, so nothing is in flight: the command buffer and
    // the resources it referenced can be released as soon as the GPU is idle
    // with respect to earlier work, which the ordinary retire path handles.
    frame_resources_.clear();
    state.open = false;
    state.command = nullptr;
    if (state.owned_command) {
        vk_device().waitIdle();
        state.owned_command.reset();
    }
}

void DeviceImpl::retire(std::function<void()> release) {
    if (!release)
        return;
    {
        std::lock_guard lock(retire_mutex_);
        if (!shut_down_) {
            // next_timeline_ is only ever incremented, so reading it without
            // mutex_ can retire a resource one value later than strictly
            // necessary - never earlier, which is what would be unsafe.
            retired_.push_back({next_timeline_ - 1, std::move(release)});
            return;
        }
    }
    // After shutdown the queue is idle, so releasing immediately is safe.
    release();
}

void DeviceImpl::reap_completed() {
    if (!timeline_)
        return;
    const std::uint64_t completed = vk_device().getSemaphoreCounterValue(timeline_.get());
    while (!pending_.empty() && pending_.front().token.value <= completed)
        pending_.pop_front();

    // Collect first, then release with no lock held: a release can drop the
    // last reference to another resource and re-enter retire().
    std::vector<std::function<void()>> releases;
    {
        std::lock_guard retire_lock(retire_mutex_);
        const auto ready = std::partition(retired_.begin(), retired_.end(),
            [completed](const Retired& entry) { return entry.timeline > completed; });
        for (auto it = ready; it != retired_.end(); ++it)
            releases.push_back(std::move(it->release));
        retired_.erase(ready, retired_.end());
    }
    for (auto& release : releases)
        release();

    if (buffers_.size() > 64) {
        std::erase_if(buffers_, [](const auto& entry) { return entry.second.expired(); });
        std::erase_if(images_, [](const auto& entry) { return entry.expired(); });
        std::erase_if(samplers_, [](const auto& entry) { return entry.expired(); });
        std::erase_if(acceleration_structures_,
            [](const auto& entry) { return entry.expired(); });
    }
}

void DeviceImpl::wait(const GpuToken token) {
    if (token.value == 0)
        return;
    std::lock_guard lock(mutex_);
    vk::SemaphoreWaitInfo info({}, timeline_.get(), token.value);
    const auto result = vk_device().waitSemaphores(info, std::numeric_limits<std::uint64_t>::max());
    if (result != vk::Result::eSuccess)
        throw Error(ErrorCode::DeviceLost, "waiting for the GPU timeline failed");
    reap_completed();
}

GpuToken DeviceImpl::signal() {
    return submit([](vk::CommandBuffer) {});
}

void DeviceImpl::synchronize() {
    std::uint64_t value = 0;
    {
        std::lock_guard lock(mutex_);
        value = next_timeline_ - 1;
    }
    if (value)
        wait({value});
}

vk::DeviceAddress DeviceImpl::stage_arguments(const void* args, const std::size_t size) {
    if (!args || size == 0)
        return 0;
    std::lock_guard lock(argument_mutex_);
    if (!argument_arena_)
        throw Error(ErrorCode::InvalidState, "gpu::Device has been shut down");
    if (size > argument_arena_->size)
        throw Error(ErrorCode::OutOfMemory,
            "root arguments do not fit in the GPU argument arena");
    constexpr std::size_t alignment = 16;
    std::size_t offset = (argument_offset_ + alignment - 1) & ~(alignment - 1);
    // The arena is a ring. Root argument records are tiny relative to its
    // size, so wrapping reuses storage thousands of dispatches later - long
    // after any command buffer that referenced it has retired. This matters
    // for externally recorded command buffers, whose completion this library
    // never observes and so cannot key a reset on.
    if (offset + size > argument_arena_->size)
        offset = 0;
    std::memcpy(static_cast<std::byte*>(argument_arena_->mapped) + offset, args, size);
    vmaFlushAllocation(allocator_, argument_arena_->allocation, offset, size);
    argument_offset_ = offset + size;
    return argument_arena_->address + offset;
}

void DeviceImpl::push_root(const vk::CommandBuffer command, const vk::DeviceAddress root) {
    command.pushDataEXT({0, vk::HostAddressRangeConstEXT{&root, sizeof(root)}});
}

void DeviceImpl::record_compute(const ComputePipelineImpl& pipeline,
    const vk::CommandBuffer command, const DispatchSize groups,
    const void* args, const std::size_t size) {
    if (!pipeline.pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid compute dispatch");

    // NoorRay records dispatches into a command buffer it owns, while uploads
    // and acceleration-structure builds are submitted by this API on the same
    // queue. Make those earlier writes visible at the consuming dispatch;
    // queue submission order alone is not a memory dependency for shader reads.
    vk::MemoryBarrier2 ordering{};
    ordering.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead
            | vk::AccessFlagBits2::eMemoryWrite
            | vk::AccessFlagBits2::eAccelerationStructureWriteKHR)
        .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
        .setDstAccessMask(vk::AccessFlagBits2::eShaderRead
            | vk::AccessFlagBits2::eShaderWrite
            | vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    command.pipelineBarrier2({{}, ordering, {}, {}});
    bind_heaps(command);
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline.pipeline);
    push_root(command, stage_arguments(args, size));
    command.dispatch(groups.x, groups.y, groups.z);
}

vk::PipelineStageFlags2 DeviceImpl::stage_mask(const Stage stage) {
    switch (stage) {
    case Stage::Copy: return vk::PipelineStageFlagBits2::eCopy;
    case Stage::Compute: return vk::PipelineStageFlagBits2::eComputeShader;
    case Stage::Vertex: return vk::PipelineStageFlagBits2::eVertexShader;
    case Stage::Fragment: return vk::PipelineStageFlagBits2::eFragmentShader;
    case Stage::RayTracing: return vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
    case Stage::AccelerationStructure: return vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
    case Stage::Present: return vk::PipelineStageFlagBits2::eAllCommands;
    }
    return vk::PipelineStageFlagBits2::eAllCommands;
}

void DeviceImpl::record_barrier(const vk::CommandBuffer command, const Stage source,
    const Stage destination) const {
    const vk::PipelineStageFlags2 sourceStages = stage_mask(source);
    const vk::PipelineStageFlags2 destinationStages = stage_mask(destination);
    vk::MemoryBarrier2 memory{};
    memory.setSrcStageMask(sourceStages)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(destinationStages)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    command.pipelineBarrier2({{}, memory, {}, {}});
}

void DeviceImpl::barrier(const Stage source, const Stage destination) {
    submit([this, source, destination](const vk::CommandBuffer command) {
        record_barrier(command, source, destination);
    });
}

std::shared_ptr<RayTracingPipelineImpl> DeviceImpl::create_ray_tracing(
    const RayTracingPipelineDesc& desc) {
    if (!ray_tracing_supported_)
        throw Error(ErrorCode::UnsupportedFeature,
            "acceleration structures are not enabled on this gpu::Device");
    if (!desc.raygen.impl_)
        throw Error(ErrorCode::InvalidArgument, "ray-tracing pipelines require a ray-generation shader");

    auto result = std::make_shared<RayTracingPipelineImpl>();
    result->device = self_.lock();

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> groups;
    const vk::DescriptorMappingSourceDataEXT as_mapping_data{
        vk::DescriptorMappingSourceConstantOffsetEXT{
            0, static_cast<std::uint32_t>(descriptor_stride_)}};
    const vk::DescriptorSetAndBindingMappingEXT as_mapping{
        0, 0, 1, vk::SpirvResourceTypeFlagBitsEXT::eAccelerationStructure,
        vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset, as_mapping_data};
    const vk::ShaderDescriptorSetAndBindingMappingInfoEXT as_mapping_info{
        1, &as_mapping};
    auto add_stage = [&](const Shader& shader, const vk::ShaderStageFlagBits stage) {
        if (!shader.impl_)
            throw Error(ErrorCode::InvalidResource, "ray-tracing shader list contains an empty shader");
        const auto index = static_cast<std::uint32_t>(stages.size());
        result->shaders.push_back(shader.impl_);
        vk::PipelineShaderStageCreateInfo stage_info{
            {}, stage, *shader.impl_->module, shader.impl_->entry_point.c_str()};
        // The raygen stage declares where its acceleration-structure binding
        // is read from in the resource heap.
        if (stage == vk::ShaderStageFlagBits::eRaygenKHR)
            stage_info.pNext = &as_mapping_info;
        stages.push_back(stage_info);
        return index;
    };
    const auto raygen_index = add_stage(desc.raygen, vk::ShaderStageFlagBits::eRaygenKHR);
    groups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, raygen_index,
        VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    for (const auto& shader : desc.miss) {
        const auto index = add_stage(shader, vk::ShaderStageFlagBits::eMissKHR);
        groups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, index,
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    }
    const auto hit_count = std::max({desc.closest_hit.size(), desc.any_hit.size(), desc.intersection.size()});
    for (std::size_t i = 0; i < hit_count; ++i) {
        const bool has_intersection = i < desc.intersection.size();
        const auto type = has_intersection ? vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup
                                            : vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
        std::uint32_t closest = VK_SHADER_UNUSED_KHR;
        std::uint32_t any = VK_SHADER_UNUSED_KHR;
        std::uint32_t intersection = VK_SHADER_UNUSED_KHR;
        if (i < desc.closest_hit.size() && desc.closest_hit[i].impl_)
            closest = add_stage(desc.closest_hit[i], vk::ShaderStageFlagBits::eClosestHitKHR);
        if (i < desc.any_hit.size() && desc.any_hit[i].impl_)
            any = add_stage(desc.any_hit[i], vk::ShaderStageFlagBits::eAnyHitKHR);
        if (has_intersection)
            intersection = add_stage(desc.intersection[i], vk::ShaderStageFlagBits::eIntersectionKHR);
        groups.emplace_back(type, VK_SHADER_UNUSED_KHR, closest, any, intersection);
    }
    if (groups.empty())
        throw Error(ErrorCode::InvalidArgument, "ray-tracing pipeline contains no shader groups");

    vk::RayTracingPipelineCreateInfoKHR info({}, stages, groups, 1, nullptr, nullptr,
        nullptr, vk::PipelineLayout{});
    const vk::PipelineCreateFlags2CreateInfo flags2{
        vk::PipelineCreateFlagBits2::eDescriptorHeapEXT};
    info.pNext = &flags2;
    try {
        result->pipeline = vk_device().createRayTracingPipelineKHRUnique({}, {}, info).value;
    } catch (const vk::SystemError& error) {
        throw Error(ErrorCode::ShaderCreationFailed, error.what());
    }

    const std::uint32_t handle_size = ray_tracing_properties_.shaderGroupHandleSize;
    const std::uint32_t alignment = std::max(ray_tracing_properties_.shaderGroupHandleAlignment, 1u);
    const std::uint32_t base_alignment = std::max(
        ray_tracing_properties_.shaderGroupBaseAlignment, 1u);
    const std::uint32_t handle_stride = (handle_size + alignment - 1) / alignment * alignment;
    // Each region begins at a group-base-aligned address. Since miss and hit
    // regions follow raygen in one compact table, the record stride must also
    // preserve that stronger alignment (handle alignment alone is commonly
    // only 32 bytes on NVIDIA hardware).
    const std::uint32_t stride = (handle_stride + base_alignment - 1)
        / base_alignment * base_alignment;
    const std::size_t group_count = groups.size();
    std::vector<std::byte> handle_bytes(group_count * handle_size);
    if (vk_device().getRayTracingShaderGroupHandlesKHR(*result->pipeline, 0,
            static_cast<std::uint32_t>(group_count), handle_bytes.size(), handle_bytes.data()) != vk::Result::eSuccess)
        throw Error(ErrorCode::ShaderCreationFailed, "Vulkan shader binding table handle query failed");
    // VMA does not promise that a storage allocation's device address is
    // aligned to shaderGroupBaseAlignment. Reserve a prefix and publish an
    // aligned address inside the allocation; using the raw allocation base
    // makes vkCmdTraceRaysKHR reject the miss/hit regions (and some drivers
    // report the resulting device loss only at queue submission time).
    result->shader_binding_table = create_buffer(group_count * stride + base_alignment,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memset(result->shader_binding_table->mapped, 0, result->shader_binding_table->size);
    const vk::DeviceAddress base = (result->shader_binding_table->address
        + base_alignment - 1) & ~(static_cast<vk::DeviceAddress>(base_alignment) - 1);
    const std::size_t table_offset = static_cast<std::size_t>(
        base - result->shader_binding_table->address);
    for (std::size_t i = 0; i < group_count; ++i)
        std::memcpy(static_cast<std::byte*>(result->shader_binding_table->mapped)
                + table_offset + i * stride,
            handle_bytes.data() + i * handle_size, handle_size);
    vmaFlushAllocation(allocator_, result->shader_binding_table->allocation, 0,
        result->shader_binding_table->size);

    const std::uint32_t miss_count = static_cast<std::uint32_t>(desc.miss.size());
    const std::uint32_t hit_group_count = static_cast<std::uint32_t>(hit_count);
    result->raygen_region = vk::StridedDeviceAddressRegionKHR{base, stride, stride};
    result->miss_region = miss_count ? vk::StridedDeviceAddressRegionKHR{
        base + stride, stride, static_cast<vk::DeviceSize>(miss_count) * stride} : vk::StridedDeviceAddressRegionKHR{};
    result->hit_region = hit_group_count ? vk::StridedDeviceAddressRegionKHR{
        base + static_cast<vk::DeviceSize>(1 + miss_count) * stride, stride,
        static_cast<vk::DeviceSize>(hit_group_count) * stride} : vk::StridedDeviceAddressRegionKHR{};
    return result;
}

AccelerationStructure DeviceImpl::build_blas(const std::span<const TriangleGeometry> geometry) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature,
            "acceleration structures are not enabled on this gpu::Device");
    if (geometry.empty())
        throw Error(ErrorCode::InvalidArgument, "BLAS requires at least one triangle geometry");

    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<std::shared_ptr<BufferImpl>> source_buffers;
    std::vector<std::uint32_t> primitive_counts;
    geometries.reserve(geometry.size());
    primitive_counts.reserve(geometry.size());
    for (const auto& item : geometry) {
        if (!item.positions.address || !item.indices.address || item.triangle_count == 0)
            throw Error(ErrorCode::InvalidArgument, "BLAS geometry contains an empty GPU address or triangle count");
        const auto positions_buffer = find_buffer_resource(item.positions.address);
        if (positions_buffer->size < sizeof(float3)
            || positions_buffer->size % sizeof(float3) != 0)
            throw Error(ErrorCode::InvalidArgument,
                "BLAS position buffer is not a packed float3 array");
        const std::size_t vertex_count = positions_buffer->size / sizeof(float3);
        if (vertex_count > std::numeric_limits<std::uint32_t>::max())
            throw Error(ErrorCode::InvalidArgument, "BLAS position buffer has too many vertices");
        source_buffers.push_back(positions_buffer);
        source_buffers.push_back(find_buffer_resource(item.indices.address));
        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            vk::Format::eR32G32B32Sfloat,
            vk::DeviceOrHostAddressConstKHR{item.positions.address}, sizeof(float3),
            static_cast<std::uint32_t>(vertex_count - 1),
            vk::IndexType::eUint32,
            vk::DeviceOrHostAddressConstKHR{item.indices.address}, {}};
        geometries.emplace_back(vk::GeometryTypeKHR::eTriangles, triangles,
            item.opaque ? vk::GeometryFlagBitsKHR::eOpaque
                        : vk::GeometryFlagsKHR{});
        primitive_counts.push_back(item.triangle_count);
    }

    const vk::AccelerationStructureBuildGeometryInfoKHR size_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        vk::BuildAccelerationStructureModeKHR::eBuild, {}, {}, geometries, {}, {}};
    const auto sizes = vk_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, size_info, primitive_counts);
    auto result = std::make_shared<AccelerationStructureImpl>();
    result->device = self_.lock();
    acceleration_structures_.push_back(result);
    result->storage = create_buffer(sizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    const vk::AccelerationStructureCreateInfoKHR create_info({}, result->storage->buffer, 0,
        sizes.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel);
    result->acceleration_structure = vk_device().createAccelerationStructureKHRUnique(create_info);
    auto scratch = create_buffer(sizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    const auto acceleration_structure = *result->acceleration_structure;
    const GpuToken token = submit([acceleration_structure, scratch, geometries, primitive_counts]
        (const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eBottomLevel,
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            vk::BuildAccelerationStructureModeKHR::eBuild, {}, acceleration_structure,
            geometries, {}, vk::DeviceOrHostAddressKHR{scratch->address}};
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
        ranges.reserve(primitive_counts.size());
        for (const auto count : primitive_counts)
            ranges.emplace_back(count, 0, 0, 0);
        std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_ptrs{ranges.size()};
        for (std::size_t i = 0; i < ranges.size(); ++i)
            range_ptrs[i] = &ranges[i];
        command.buildAccelerationStructuresKHR(build_info, range_ptrs);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR
                | vk::AccessFlagBits2::eAccelerationStructureWriteKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, {result->storage, scratch});
    wait(token);
    result->address = vk_device().getAccelerationStructureAddressKHR({acceleration_structure});
    result->storage_size = result->storage->size;
    result->handle = write_acceleration_structure_descriptor(acceleration_structure,
        result->address, result->storage_size);
    return AccelerationStructure(std::move(result));
}

AccelerationStructure DeviceImpl::build_tlas(const std::span<const Instance> instances) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature, "ray tracing is not enabled on this gpu::Device");
    if (instances.empty())
        throw Error(ErrorCode::InvalidArgument, "TLAS requires at least one instance");

    std::vector<vk::AccelerationStructureInstanceKHR> records;
    records.reserve(instances.size());
    std::vector<std::shared_ptr<AccelerationStructureImpl>> acceleration_structures;
    acceleration_structures.reserve(instances.size());
    for (const auto& instance : instances) {
        if (!instance.blas.impl_ || !instance.blas.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS instance references an empty BLAS");
        acceleration_structures.push_back(instance.blas.impl_);
        std::array<std::array<float, 4>, 3> matrix{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 4; ++column)
                matrix[row][column] = instance.transform.values[row][column];
        if (instance.custom_index > 0x00ffffffu
            || instance.shader_binding_table_offset > 0x00ffffffu)
            throw Error(ErrorCode::InvalidArgument,
                "TLAS instance custom index or SBT offset exceeds 24 bits");
        records.emplace_back(vk::TransformMatrixKHR{matrix},
            instance.custom_index, instance.mask,
            instance.shader_binding_table_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            vk_device().getAccelerationStructureAddressKHR(
                {*instance.blas.impl_->acceleration_structure}));
    }

    auto instance_buffer = create_buffer(records.size() * sizeof(records[0]),
        vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(instance_buffer->mapped, records.data(), records.size() * sizeof(records[0]));
    vmaFlushAllocation(allocator_, instance_buffer->allocation, 0,
        records.size() * sizeof(records[0]));
    const vk::DeviceAddress address = instance_buffer->address;
    return build_tlas_at(address, static_cast<std::uint32_t>(records.size()),
        std::move(acceleration_structures), std::move(instance_buffer));
}

AccelerationStructure DeviceImpl::build_tlas(const GpuPtr<InstanceRecord> records,
    const std::uint32_t count, const std::span<const AccelerationStructure> referenced) {
    if (!acceleration_structure_supported_)
        throw Error(ErrorCode::UnsupportedFeature, "ray tracing is not enabled on this gpu::Device");
    if (count == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS requires at least one instance");
    if (records.address == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS instance records have no device address");

    // The records are opaque to the host here, so the BLASes they point at are
    // only kept alive by this span. Dropping one would leave the TLAS holding
    // dangling device addresses.
    std::vector<std::shared_ptr<AccelerationStructureImpl>> acceleration_structures;
    acceleration_structures.reserve(referenced.size());
    for (const auto& structure : referenced) {
        if (!structure.impl_ || !structure.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS references an empty BLAS");
        acceleration_structures.push_back(structure.impl_);
    }
    return build_tlas_at(records.address, count, std::move(acceleration_structures), nullptr);
}

AccelerationStructure DeviceImpl::build_tlas_at(const vk::DeviceAddress records,
    const std::uint32_t count,
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references,
    std::shared_ptr<BufferImpl> owned_input) {
    const vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
        VK_FALSE, vk::DeviceOrHostAddressConstKHR{records}};
    const vk::AccelerationStructureGeometryKHR geometry{
        vk::GeometryTypeKHR::eInstances, instances_data, {}};
    const std::vector<std::uint32_t> primitive_counts{count};
    const vk::AccelerationStructureBuildGeometryInfoKHR size_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
            | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        vk::BuildAccelerationStructureModeKHR::eBuild, {}, {}, 1, &geometry};
    const auto sizes = vk_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, size_info, primitive_counts);
    auto result = std::make_shared<AccelerationStructureImpl>();
    result->device = self_.lock();
    result->primitive_count = count;
    result->updateable = true;
    result->references = std::move(references);
    acceleration_structures_.push_back(result);
    result->storage = create_buffer(sizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    result->acceleration_structure = vk_device().createAccelerationStructureKHRUnique({{},
        result->storage->buffer, 0, sizes.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eTopLevel});
    auto scratch = create_buffer(std::max(sizes.buildScratchSize, sizes.updateScratchSize),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY, false);
    const auto acceleration_structure = *result->acceleration_structure;
    const GpuToken token = submit([acceleration_structure, scratch, geometry, count]
        (const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eTopLevel,
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            vk::BuildAccelerationStructureModeKHR::eBuild, {}, acceleration_structure,
            1, &geometry, {}, vk::DeviceOrHostAddressKHR{scratch->address}};
        const vk::AccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
        const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = &range;
        const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> ranges{range_ptr};
        command.buildAccelerationStructuresKHR(build_info, ranges);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, owned_input ? std::vector<std::shared_ptr<void>>{result->storage, scratch, owned_input}
                   : std::vector<std::shared_ptr<void>>{result->storage, scratch});
    wait(token);
    result->address = vk_device().getAccelerationStructureAddressKHR({acceleration_structure});
    result->storage_size = result->storage->size;
    result->handle = write_acceleration_structure_descriptor(acceleration_structure,
        result->address, result->storage_size);
    result->update_input = std::move(owned_input);
    result->update_scratch = scratch;
    return AccelerationStructure(std::move(result));
}

void DeviceImpl::update_tlas(AccelerationStructure& tlas,
    const std::span<const Instance> instances) {
    auto target = tlas.impl_;
    if (!target || !target->acceleration_structure || !target->updateable)
        throw Error(ErrorCode::InvalidResource, "TLAS was not built for updates");
    if (instances.size() != target->primitive_count)
        throw Error(ErrorCode::InvalidArgument, "TLAS update cannot change the instance count");

    std::vector<vk::AccelerationStructureInstanceKHR> records;
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references;
    records.reserve(instances.size());
    references.reserve(instances.size());
    for (const auto& instance : instances) {
        if (!instance.blas.impl_ || !instance.blas.impl_->acceleration_structure)
            throw Error(ErrorCode::InvalidResource, "TLAS instance references an empty BLAS");
        if (instance.custom_index > 0x00ffffffu
            || instance.shader_binding_table_offset > 0x00ffffffu)
            throw Error(ErrorCode::InvalidArgument,
                "TLAS instance custom index or SBT offset exceeds 24 bits");
        references.push_back(instance.blas.impl_);
        std::array<std::array<float, 4>, 3> matrix{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 4; ++column)
                matrix[row][column] = instance.transform.values[row][column];
        records.emplace_back(vk::TransformMatrixKHR{matrix}, instance.custom_index,
            instance.mask, instance.shader_binding_table_offset,
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
            vk_device().getAccelerationStructureAddressKHR(
                {*instance.blas.impl_->acceleration_structure}));
    }

    auto instance_buffer = target->update_input;
    if (!instance_buffer || instance_buffer->size
            < records.size() * sizeof(records[0]))
        throw Error(ErrorCode::InvalidResource, "TLAS update input storage is invalid");
    std::memcpy(instance_buffer->mapped, records.data(), records.size() * sizeof(records[0]));
    vmaFlushAllocation(allocator_, instance_buffer->allocation, 0, instance_buffer->size);
    update_tlas_at(tlas, instance_buffer->address,
        static_cast<std::uint32_t>(records.size()));
    target->references = std::move(references);
}

void DeviceImpl::update_tlas(AccelerationStructure& tlas,
    const GpuPtr<InstanceRecord> records, const std::uint32_t count) {
    if (records.address == 0)
        throw Error(ErrorCode::InvalidArgument, "TLAS instance records have no device address");
    update_tlas_at(tlas, records.address, count);
}

void DeviceImpl::update_tlas_at(AccelerationStructure& tlas,
    const vk::DeviceAddress records, const std::uint32_t count) {
    auto target = tlas.impl_;
    if (!target || !target->acceleration_structure || !target->updateable)
        throw Error(ErrorCode::InvalidResource, "TLAS was not built for updates");
    if (count != target->primitive_count)
        throw Error(ErrorCode::InvalidArgument, "TLAS update cannot change the instance count");
    const vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
        VK_FALSE, vk::DeviceOrHostAddressConstKHR{records}};
    const vk::AccelerationStructureGeometryKHR geometry{
        vk::GeometryTypeKHR::eInstances, instances_data, {}};
    const auto flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
        | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
    auto scratch = target->update_scratch;
    if (!scratch)
        throw Error(ErrorCode::InvalidResource, "TLAS update scratch storage is invalid");
    const auto acceleration_structure = *target->acceleration_structure;
    const GpuToken token = submit([acceleration_structure, scratch,
        geometry, flags, count](const vk::CommandBuffer command) mutable {
        const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
            vk::AccelerationStructureTypeKHR::eTopLevel, flags,
            vk::BuildAccelerationStructureModeKHR::eUpdate, acceleration_structure,
            acceleration_structure, 1, &geometry, {},
            vk::DeviceOrHostAddressKHR{scratch->address}};
        const vk::AccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
        const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = &range;
        command.buildAccelerationStructuresKHR(build_info, range_ptr);
        const vk::MemoryBarrier2 ready{
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eComputeShader
                | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR};
        command.pipelineBarrier2({{}, ready, {}, {}});
    }, {target, scratch});
    wait(token);
}
void DeviceImpl::upload(const std::shared_ptr<BufferImpl>& destination, const void* data,
    const std::size_t bytes, const std::size_t destination_offset) {
    if (!destination || !data || destination_offset > destination->size
        || bytes > destination->size - destination_offset)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU upload range");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferSrc,
                                 VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(staging->mapped, data, bytes);
    vmaFlushAllocation(allocator_, staging->allocation, 0, bytes);
    submit([staging, destination, bytes, destination_offset](const vk::CommandBuffer command) {
        command.copyBuffer(staging->buffer, destination->buffer,
            vk::BufferCopy(0, destination_offset, bytes));
    }, {staging, destination});
}

void DeviceImpl::download(const std::shared_ptr<BufferImpl>& source, void* data,
    const std::size_t bytes) {
    if (!source || !data || bytes > source->size)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU download range");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferDst,
                                 VMA_MEMORY_USAGE_GPU_TO_CPU, true);
    const GpuToken token = submit([staging, source, bytes](const vk::CommandBuffer command) {
        command.copyBuffer(source->buffer, staging->buffer, vk::BufferCopy(0, 0, bytes));
    }, {staging, source});
    wait(token);
    vmaInvalidateAllocation(allocator_, staging->allocation, 0, bytes);
    if (staging->mapped)
        std::memcpy(data, staging->mapped, bytes);
}

void DeviceImpl::record_copy_image(const vk::CommandBuffer command,
    ImageImpl& source, ImageImpl& destination) {
    // VK_KHR_unified_image_layouts lets both images stay in GENERAL for the
    // copy, so only the memory dependency has to be expressed.
    const auto memory_dependency = [&command](const vk::PipelineStageFlags2 source_stage,
        const vk::AccessFlags2 source_access, const vk::PipelineStageFlags2 destination_stage,
        const vk::AccessFlags2 destination_access) {
        vk::MemoryBarrier2 barrier{};
        barrier.setSrcStageMask(source_stage)
            .setSrcAccessMask(source_access)
            .setDstStageMask(destination_stage)
            .setDstAccessMask(destination_access);
        command.pipelineBarrier2({{}, barrier, {}, {}});
    };
    memory_dependency(vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
    if (source.width == destination.width && source.height == destination.height
        && source.format == destination.format) {
        command.copyImage(source.image, vk::ImageLayout::eGeneral,
            destination.image, vk::ImageLayout::eGeneral,
            vk::ImageCopy{{vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                {source.width, source.height, 1}});
    } else {
        // Differing extents or formats need a blit, which also gives us the
        // filtered downscale a viewport composite wants.
        const vk::ImageBlit region{
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(source.width),
                static_cast<std::int32_t>(source.height), 1}},
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(destination.width),
                static_cast<std::int32_t>(destination.height), 1}}};
        command.blitImage(source.image, vk::ImageLayout::eGeneral,
            destination.image, vk::ImageLayout::eGeneral, region, vk::Filter::eLinear);
    }
    memory_dependency(vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

void DeviceImpl::copy_image(const ImageHandle source, const ImageHandle destination) {
    const auto from = find_image(source);
    const auto to = find_image(destination);
    if (!from || !to)
        throw Error(ErrorCode::InvalidResource, "image copy given a handle that is not live");
    submit([this, from, to](const vk::CommandBuffer command) {
        record_copy_image(command, *from, *to);
    }, {from, to});
}

void ComputePipelineImpl::launch(const DispatchSize groups, const void* args,
    const std::size_t size) const {
    if (!device || !pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid compute dispatch");
    auto self = const_cast<ComputePipelineImpl*>(this)->shared_from_this();
    device->submit([self, groups, args, size](const vk::CommandBuffer command) {
        self->device->record_compute(*self, command, groups, args, size);
    }, {self});
}

void ComputePipelineImpl::launch_indirect(const GpuPtr<DispatchArgs> args,
    const void* argument_data, const std::size_t argument_size) const {
    if (!device || !pipeline || args.address == 0 || !argument_data || argument_size == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid indirect compute dispatch");
    const auto resource = device->find_buffer_resource(args.address);
    const vk::Buffer buffer = resource->buffer;
    const vk::DeviceSize offset = args.address - resource->address;
    auto self = const_cast<ComputePipelineImpl*>(this)->shared_from_this();
    device->submit([self, buffer, offset, argument_data, argument_size]
        (const vk::CommandBuffer command) {
        DeviceImpl& device_impl = *self->device;
        // The dispatch dimensions come from the GPU, but the shader still
        // reads its root arguments through the same push-data path.
        const vk::DeviceAddress root = device_impl.stage_arguments(argument_data, argument_size);
        device_impl.bind_heaps(command);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *self->pipeline);
        DeviceImpl::push_root(command, root);
        command.dispatchIndirect(buffer, offset);
    }, std::vector<std::shared_ptr<void>>{self, resource});
}

BufferImpl::~BufferImpl() {
    if (!device || !allocation)
        return;
    device->retire([owner = device, allocator = device->allocator_, buffer = this->buffer,
        allocation = this->allocation, unmap = mapped_by_api, descriptor = handle] {
        if (unmap)
            vmaUnmapMemory(allocator, allocation);
        if (buffer)
            vmaDestroyBuffer(allocator, buffer, allocation);
        owner->release_resource(descriptor);
    });
}

SamplerImpl::~SamplerImpl() {
    if (!device || !sampler)
        return;
    device->retire([owner = device, sampler = sampler.release(), descriptor = handle,
        vk_device = device->device()] {
        if (sampler)
            vk_device.destroySampler(sampler);
        owner->release_sampler(descriptor);
    });
}

AccelerationStructureImpl::~AccelerationStructureImpl() {
    if (!device || !acceleration_structure)
        return;
    device->retire([owner = device, acceleration = acceleration_structure.release(),
        descriptor = handle, storage = std::move(storage), vk_device = device->device()] {
        if (acceleration)
            vk_device.destroyAccelerationStructureKHR(acceleration);
        owner->release_resource(ResourceHandle{descriptor.value});
    });
}

std::shared_ptr<BufferImpl> make_buffer(const std::shared_ptr<DeviceImpl>& device,
    const std::size_t size, const std::size_t) {
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
            | vk::BufferUsageFlagBits::eTransferSrc
            | vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eIndirectBuffer
            | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    if (device->acceleration_structure_supported())
        usage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    return device->create_buffer(size, usage, VMA_MEMORY_USAGE_GPU_ONLY, false);
}

void upload_buffer(const std::shared_ptr<DeviceImpl>& device,
    const std::shared_ptr<BufferImpl>& destination, const void* data,
    const std::size_t bytes, const std::size_t, const std::size_t destination_offset) {
    device->upload(destination, data, bytes, destination_offset);
}

void download_buffer(const std::shared_ptr<DeviceImpl>& device,
    const std::shared_ptr<BufferImpl>& source, void* data,
    const std::size_t bytes, const std::size_t) {
    device->download(source, data, bytes);
}

std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>& buffer) {
    return buffer ? buffer->address : 0;
}

ResourceHandle buffer_handle(const std::shared_ptr<BufferImpl>& buffer) {
    return buffer ? buffer->device->buffer_resource(buffer) : ResourceHandle{};
}

AccelerationStructureHandle acceleration_structure_handle(
    const std::shared_ptr<AccelerationStructureImpl>& acceleration_structure) {
    return acceleration_structure ? acceleration_structure->handle : AccelerationStructureHandle{};
}

} // namespace gpu::detail

namespace gpu {

Device::Device(const DeviceConfig& config)
    : impl_(std::make_shared<detail::DeviceImpl>(config)) {
    impl_->attach_self(impl_);
    impl_->initialize_resources();
}
Device::~Device() {
    if (impl_)
        impl_->shutdown();
}
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

DeviceFeatures Device::features() const { return impl_->features(); }

Shader Device::create_shader(const std::span<const std::byte> spirv) {
    return create_shader(spirv, "main");
}
Shader Device::create_shader(const std::span<const std::byte> spirv, const std::string_view entry_point) {
    auto shader = impl_->create_shader(spirv, entry_point);
    return Shader(shader, shader->entry_point);
}
ComputePipeline Device::compute(const Shader& shader) {
    return ComputePipeline(impl_->create_compute(shader));
}
GraphicsPipeline Device::graphics(const GraphicsPipelineDesc& desc) {
    return GraphicsPipeline(impl_->create_graphics(desc));
}
RayTracingPipeline Device::ray_tracing(const RayTracingPipelineDesc& desc) {
    return RayTracingPipeline(impl_->create_ray_tracing(desc));
}
AccelerationStructure Device::build_blas(const std::span<const TriangleGeometry> geometry) {
    return impl_->build_blas(geometry);
}
AccelerationStructure Device::build_tlas(const std::span<const Instance> instances) {
    return impl_->build_tlas(instances);
}
void Device::update_tlas(AccelerationStructure& tlas,
    const std::span<const Instance> instances) {
    impl_->update_tlas(tlas, instances);
}
Sampler Device::sampler(const SamplerDesc& desc) {
    return Sampler(impl_->create_sampler(desc));
}
void Device::barrier(const Stage source, const Stage destination) { impl_->barrier(source, destination); }
GpuToken Device::signal() { return impl_->signal(); }
void Device::wait(const GpuToken token) { impl_->wait(token); }
void Device::synchronize() { impl_->synchronize(); }
void Device::render(const RenderTarget& target, const std::function<void()>& draw_commands) {
    impl_->render(target, draw_commands);
}
void Device::copy(const ImageHandle source, const ImageHandle destination) {
    impl_->copy_image(source, destination);
}
TimestampQuery Device::timestamp() {
    return TimestampQuery(impl_->create_timestamp());
}
void Device::measure(const TimestampQuery& query, const std::function<void()>& commands) {
    if (!query.impl_)
        throw Error(ErrorCode::InvalidResource, "timestamp query is empty");
    impl_->measure(query.impl_, commands);
}
Swapchain Device::swapchain() { return swapchain(ImageFormat::Auto); }
Swapchain Device::swapchain(const ImageFormat preferred) {
    return Swapchain(impl_->create_swapchain(preferred));
}
Frame Device::begin_frame(Swapchain& swapchain) {
    if (!swapchain)
        throw Error(ErrorCode::InvalidResource, "swapchain is empty");
    return Frame(impl_->begin_frame(swapchain.impl_));
}
void Device::end_frame(Frame&& frame) {
    if (!frame.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot present an unacquired frame");
    impl_->end_frame(*frame.impl_);
}

double TimestampQuery::milliseconds() const {
    if (!impl_ || !impl_->device)
        return 0.0;
    impl_->milliseconds = impl_->device->timestamp_milliseconds(*impl_);
    return impl_->milliseconds;
}

void ComputePipeline::launch_bytes(const DispatchSize groups, const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "compute pipeline is empty");
    impl_->launch(groups, args, size);
}
void ComputePipeline::launch_indirect_bytes(const GpuPtr<DispatchArgs> groups,
    const void* args, const std::size_t size) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "compute pipeline is empty");
    impl_->launch_indirect(groups, args, size);
}

void detail::DeviceImpl::record_ray_tracing(const detail::RayTracingPipelineImpl& pipeline,
    const vk::CommandBuffer command, const DispatchSize groups,
    const void* args, const std::size_t size) {
    if (!pipeline.pipeline || !args || size == 0
        || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid ray-tracing dispatch");
    const vk::DeviceAddress root = stage_arguments(args, size);
    vk::MemoryBarrier2 ordering{};
    ordering.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands
            | vk::PipelineStageFlagBits2::eHost)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead
            | vk::AccessFlagBits2::eMemoryWrite
            | vk::AccessFlagBits2::eHostWrite
            | vk::AccessFlagBits2::eAccelerationStructureWriteKHR)
        .setDstStageMask(vk::PipelineStageFlagBits2::eRayTracingShaderKHR)
        .setDstAccessMask(vk::AccessFlagBits2::eShaderRead
            | vk::AccessFlagBits2::eShaderWrite
            | vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    command.pipelineBarrier2({{}, ordering, {}, {}});
    bind_heaps(command);
    command.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline.pipeline);
    push_root(command, root);
    command.traceRaysKHR(pipeline.raygen_region, pipeline.miss_region,
        pipeline.hit_region, {}, groups.x, groups.y, groups.z);
}

void detail::RayTracingPipelineImpl::trace(const DispatchSize groups, const void* args,
    const std::size_t size) const {
    if (!device || !pipeline || !args || size == 0 || groups.x == 0 || groups.y == 0 || groups.z == 0)
        throw Error(ErrorCode::InvalidArgument, "invalid ray-tracing dispatch");
    auto self = const_cast<RayTracingPipelineImpl*>(this)->shared_from_this();
    device->submit([self, groups, args, size](const vk::CommandBuffer command) {
        self->device->record_ray_tracing(*self, command, groups, args, size);
    }, std::vector<std::shared_ptr<void>>{self, shader_binding_table});
}

void RayTracingPipeline::trace_bytes(const DispatchSize size, const void* args, const std::size_t bytes) const {
    if (!impl_)
        throw Error(ErrorCode::InvalidResource, "ray-tracing pipeline is empty");
    impl_->trace(size, args, bytes);
}

AccelerationStructureHandle AccelerationStructure::handle() const noexcept {
    return detail::acceleration_structure_handle(impl_);
}

AccelerationStructure::operator bool() const noexcept {
    return static_cast<bool>(handle());
}

SamplerHandle Sampler::handle() const noexcept { return impl_ ? impl_->handle : SamplerHandle{}; }

} // namespace gpu

namespace gpu {

namespace interop {

DeviceHandles device_handles(Device& device) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot read handles from an empty gpu::Device");
    return device.impl_->native_handles();
}

std::uintptr_t image_view(Device& device, const ImageHandle handle) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot inspect through an empty gpu::Device");
    const auto image = device.impl_->find_image(handle);
    if (!image)
        throw Error(ErrorCode::InvalidResource, "GPU image handle is not live");
    return reinterpret_cast<std::uintptr_t>(static_cast<VkImageView>(*image->view));
}

ExternalImageMemory export_image_memory(Device& device, const ImageHandle handle) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot export from an empty gpu::Device");
    return device.impl_->export_image_memory(handle);
}

ExternalSemaphore signal_external(Device& device) {
    if (!device.impl_)
        throw Error(ErrorCode::InvalidResource, "cannot export from an empty gpu::Device");
    return device.impl_->signal_external();
}

std::uintptr_t command_buffer(const Frame& frame) {
    if (!frame.impl_ || !frame.impl_->open)
        throw Error(ErrorCode::InvalidState,
            "interop::command_buffer requires a frame between begin_frame and end_frame");
    return reinterpret_cast<std::uintptr_t>(
        static_cast<VkCommandBuffer>(frame.impl_->command));
}

std::uint32_t native_format(const ImageFormat format) {
    return static_cast<std::uint32_t>(detail::to_vulkan_format(format));
}

} // namespace interop
} // namespace gpu
