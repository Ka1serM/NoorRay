#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "gpu/compute.hpp"
#include "gpu/graphics.hpp"
#include "gpu/image.hpp"
#include "gpu/interop.hpp"
#include "gpu/surface.hpp"
#include "gpu/swapchain.hpp"
#include "gpu/raytracing.hpp"
#include "gpu/sampler.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <span>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gpu {

namespace detail { class DeviceImpl; class SwapchainImpl; struct ImageImpl; }

struct TimestampQuery::State {
    std::shared_ptr<detail::DeviceImpl> device;
    // Two consecutive query-pool slots: begin and end.
    std::uint32_t first_query = ~0u;
    double milliseconds = 0.0;
};

// A frame is an open recording scope. While `command` is set, DeviceImpl
// batches every dispatch, trace and render scope into it rather than
// submitting each one on its own.
struct Frame::State {
    std::shared_ptr<detail::DeviceImpl> device;
    std::shared_ptr<detail::SwapchainImpl> swapchain;
    vk::CommandBuffer command;
    vk::UniqueCommandBuffer owned_command;
    std::uint32_t image_index = 0;
    std::uint32_t semaphore_index = 0;
    ImageHandle target{};
    bool open = false;
};

} // namespace gpu

namespace gpu::detail {

class DeviceImpl;

// Descriptor-heap capacities. These bound the opaque handle space, so they are
// shared by the allocator and by every heap write and bind.
inline constexpr std::uint32_t descriptor_capacity = 4096;
inline constexpr std::uint32_t sampler_descriptor_capacity = 1024;
// Slang's ResourceHeapEXT lowering indexes every resource descriptor through a
// 128-byte slot, regardless of the physical descriptor payload size reported
// by VK_EXT_descriptor_heap.
inline constexpr vk::DeviceSize resource_descriptor_slot_size = 128;
inline constexpr std::size_t argument_arena_size = 4u * 1024u * 1024u;

// The library renders depth into a single format; RenderTarget depth images
// and depth-enabled pipelines are both built against it.
inline constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

inline vk::Format to_vulkan_format(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Bgra8Unorm: return vk::Format::eB8G8R8A8Unorm;
    case ImageFormat::Rgba32Float: return vk::Format::eR32G32B32A32Sfloat;
    case ImageFormat::R32Uint: return vk::Format::eR32Uint;
    case ImageFormat::D32Float: return depth_format;
    case ImageFormat::Auto:
    case ImageFormat::Rgba8Unorm:
    default: return vk::Format::eR8G8B8A8Unorm;
    }
}

// Bytes per texel for the small public format vocabulary.
inline std::size_t format_texel_size(const ImageFormat format) {
    switch (format) {
    case ImageFormat::Rgba32Float: return 16;
    case ImageFormat::R32Uint:
    case ImageFormat::D32Float: return 4;
    case ImageFormat::Bgra8Unorm:
    case ImageFormat::Auto:
    case ImageFormat::Rgba8Unorm:
    default: return 4;
    }
}

struct BufferImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::Buffer buffer;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::DeviceAddress address = 0;
    ResourceHandle handle{};
    std::size_t size = 0;
    void* mapped = nullptr;
    bool host_visible = false;
    bool mapped_by_api = false;

    ~BufferImpl();
};

struct ShaderImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::UniqueShaderModule module;
    std::string entry_point = "main";

    ~ShaderImpl() = default;
};

struct ImageImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::Image image;
    vk::UniqueImageView view;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::DeviceMemory external_memory;
    ImageHandle handle{};
    ImageHandle sampled_handle{};
    ImageHandle storage_handle{};
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    std::size_t byte_size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // False for swapchain images: the presentation engine owns the VkImage and
    // there is no VMA allocation, but the view and descriptors are still ours.
    bool owns_image = true;
    bool exportable = false;

    ~ImageImpl();
};

struct SamplerImpl {
    std::shared_ptr<DeviceImpl> device;
    vk::UniqueSampler sampler;
    SamplerHandle handle{};

    ~SamplerImpl();
};

struct AccelerationStructureImpl {
    std::shared_ptr<DeviceImpl> device;
    std::shared_ptr<BufferImpl> storage;
    vk::UniqueAccelerationStructureKHR acceleration_structure;
    vk::DeviceAddress address = 0;
    vk::DeviceSize storage_size = 0;
    AccelerationStructureHandle handle{};
    std::uint32_t primitive_count = 0;
    bool updateable = false;
    std::vector<std::shared_ptr<AccelerationStructureImpl>> references;
    std::shared_ptr<BufferImpl> update_input;
    std::shared_ptr<BufferImpl> update_scratch;

    ~AccelerationStructureImpl();
};

class ComputePipelineImpl : public std::enable_shared_from_this<ComputePipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;

    void launch(DispatchSize groups, const void* args, std::size_t size) const;
    void launch_indirect(GpuPtr<DispatchArgs> args, const void* argument_data,
        std::size_t argument_size) const;
};

class GraphicsPipelineImpl : public std::enable_shared_from_this<GraphicsPipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;
    vk::Format color_format = vk::Format::eR8G8B8A8Unorm;
    bool uses_depth = false;
    GraphicsState state{};

    void draw(std::uint32_t vertex_count, const void* args, std::size_t size) const;
    void draw_indirect(GpuPtr<DrawArgs> commands, std::uint32_t draw_count,
        const void* args, std::size_t size) const;
};

class RayTracingPipelineImpl : public std::enable_shared_from_this<RayTracingPipelineImpl> {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniquePipeline pipeline;
    std::shared_ptr<BufferImpl> shader_binding_table;
    vk::StridedDeviceAddressRegionKHR raygen_region{};
    vk::StridedDeviceAddressRegionKHR miss_region{};
    vk::StridedDeviceAddressRegionKHR hit_region{};
    std::vector<std::shared_ptr<ShaderImpl>> shaders;
    void trace(DispatchSize, const void*, std::size_t) const;
};

// One presentation chain. Rebuilt in place whenever the surface size stops
// matching the provider's, so the public Swapchain handle stays valid across
// resizes and the caller never sees an out-of-date error.
class SwapchainImpl {
public:
    std::shared_ptr<DeviceImpl> device;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<std::shared_ptr<ImageImpl>> images;
    // Acquire semaphores cycle independently of image indices: the index is
    // only known after the acquire that the semaphore belongs to.
    std::vector<vk::UniqueSemaphore> acquire_semaphores;
    std::vector<vk::UniqueSemaphore> present_semaphores;
    // A swapchain image must be transitioned from PRESENT_SRC rather than
    // UNDEFINED once it has been presented at least once, or its contents are
    // discarded. Tracked per image because acquisition order is arbitrary.
    std::vector<bool> presented;
    vk::Format format = vk::Format::eUndefined;
    ImageFormat public_format = ImageFormat::Bgra8Unorm;
    vk::ColorSpaceKHR color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t semaphore_cursor = 0;
    bool stale = true;

    ~SwapchainImpl();
};

class DeviceImpl : public std::enable_shared_from_this<DeviceImpl> {
public:
    explicit DeviceImpl(const DeviceConfig& config);
    ~DeviceImpl();

    void attach_self(const std::shared_ptr<DeviceImpl>& self) { self_ = self; }
    void shutdown() noexcept;
    void initialize_resources();
    DeviceFeatures features() const noexcept { return features_; }
    bool acceleration_structure_supported() const noexcept { return acceleration_structure_supported_; }

    std::shared_ptr<BufferImpl> create_buffer(std::size_t size, vk::BufferUsageFlags usage,
                                               VmaMemoryUsage memory_usage, bool mapped);
    std::shared_ptr<ImageImpl> create_image(std::uint32_t width, std::uint32_t height,
        ImageUsage usage, ImageFormat format);
    std::shared_ptr<ShaderImpl> create_shader(std::span<const std::byte> spirv,
                                              std::string_view entry_point);
    std::shared_ptr<ComputePipelineImpl> create_compute(const Shader& shader);
    std::shared_ptr<GraphicsPipelineImpl> create_graphics(const GraphicsPipelineDesc& desc);
    std::shared_ptr<SamplerImpl> create_sampler(const SamplerDesc& desc);

    void upload(const std::shared_ptr<BufferImpl>& destination, const void* data,
                std::size_t bytes, std::size_t destination_offset = 0);
    void download(const std::shared_ptr<BufferImpl>& source, void* data, std::size_t bytes);
    void upload_image(const std::shared_ptr<ImageImpl>&, const void* data, std::size_t bytes);
    void download_image(const std::shared_ptr<ImageImpl>&, void* data, std::size_t bytes);
    void copy_image(ImageHandle source, ImageHandle destination);
    void record_copy_image(vk::CommandBuffer, ImageImpl& source, ImageImpl& destination);
    void transfer_barrier(vk::CommandBuffer, bool before) const;
    void record_barrier(vk::CommandBuffer, Stage source, Stage destination) const;
    void barrier(Stage source, Stage destination);
    GpuToken signal();
    void wait(GpuToken token);
    void synchronize();

    vk::Device device() const noexcept { return vk_device(); }
    vk::PhysicalDevice physical_device() const noexcept { return physical_device_; }
    vk::Queue queue() const noexcept { return queue_; }
    vk::DeviceAddress buffer_address(vk::Buffer buffer) const;
    std::shared_ptr<BufferImpl> find_buffer_resource(vk::DeviceAddress address) const;
    std::pair<vk::Buffer, vk::DeviceSize> find_buffer(vk::DeviceAddress address) const;
    void bind_heaps(vk::CommandBuffer) const;
    void retain_active(std::shared_ptr<void> resource);
    void release_resource(ResourceHandle handle);
    void release_sampler(SamplerHandle handle);
    // Stage `size` bytes of root arguments in the argument arena and return the
    // GPU address the shader's root pointer should carry.
    vk::DeviceAddress stage_arguments(const void* args, std::size_t size);
    // Every pipeline is a descriptor-heap pipeline, so root arguments always
    // travel through the extension's push-data path.
    static void push_root(vk::CommandBuffer, vk::DeviceAddress root);
    void write_descriptor(ResourceHandle, const vk::ResourceDescriptorInfoEXT&) const;
    void write_image_descriptor(const ImageImpl&, ResourceHandle, vk::DescriptorType) const;
    ResourceHandle allocate_resource();
    ResourceHandle buffer_resource(const std::shared_ptr<BufferImpl>&);
    SamplerHandle write_sampler_descriptor(const vk::SamplerCreateInfo&);
    AccelerationStructureHandle write_acceleration_structure_descriptor(vk::AccelerationStructureKHR,
                                                                         vk::DeviceAddress, vk::DeviceSize);
    std::shared_ptr<ImageImpl> find_image(ImageHandle handle) const;
    void render(const RenderTarget&, const std::function<void()>&);
    // Shared prologue for every in-render draw: validates that the pipeline
    // matches the active render target, binds it, and sets viewport/scissor.
    void begin_draw(const GraphicsPipelineImpl&);
    void record_draw(const GraphicsPipelineImpl&, std::uint32_t, std::uint32_t,
        const void*, std::size_t);
    void record_draw_indirect(const GraphicsPipelineImpl&, GpuPtr<DrawArgs>,
        std::uint32_t, const void*, std::size_t);
    void record_compute(const ComputePipelineImpl&, vk::CommandBuffer, DispatchSize,
        const void*, std::size_t);
    void record_ray_tracing(const RayTracingPipelineImpl&, vk::CommandBuffer,
        DispatchSize, const void*, std::size_t);

    // --- Timestamps
    std::shared_ptr<TimestampQuery::State> create_timestamp();
    void measure(const std::shared_ptr<TimestampQuery::State>&,
        const std::function<void()>&);
    double timestamp_milliseconds(TimestampQuery::State&);

    // --- Presentation
    bool presenting() const noexcept { return surface_provider_ != nullptr; }
    std::shared_ptr<SwapchainImpl> create_swapchain(ImageFormat preferred);
    // Register a presentation-engine image as an ordinary ImageImpl so it can
    // be used as a RenderTarget and a copy destination like any other image.
    std::shared_ptr<ImageImpl> wrap_presentation_image(vk::Image, vk::Format,
        ImageFormat, std::uint32_t width, std::uint32_t height);
    void rebuild_swapchain(SwapchainImpl&);
    std::shared_ptr<Frame::State> begin_frame(const std::shared_ptr<SwapchainImpl>&);
    void end_frame(Frame::State&);
    // Discard a frame that was never ended: release its command buffer and
    // referenced resources without submitting or presenting.
    void abandon_frame(Frame::State&);
    vk::SurfaceKHR surface() const noexcept { return surface_ ? surface_.get() : vk::SurfaceKHR{}; }
    interop::DeviceHandles native_handles() const noexcept;
    interop::ExternalImageMemory export_image_memory(ImageHandle);
    interop::ExternalSemaphore signal_external();
    std::shared_ptr<RayTracingPipelineImpl> create_ray_tracing(const RayTracingPipelineDesc&);
    AccelerationStructure build_blas(std::span<const TriangleGeometry>);
    AccelerationStructure build_tlas(std::span<const Instance>);
    void update_tlas(AccelerationStructure&, std::span<const Instance>);
    AccelerationStructure build_tlas(GpuPtr<InstanceRecord>, std::uint32_t,
        std::span<const AccelerationStructure>);
    void update_tlas(AccelerationStructure&, GpuPtr<InstanceRecord>, std::uint32_t);
    // Shared tail of both build paths: everything after the instance records
    // exist at a device address. owned_input is retained for the host-span
    // overload, whose staging buffer the structure keeps for later refits.
    AccelerationStructure build_tlas_at(vk::DeviceAddress records, std::uint32_t count,
        std::vector<std::shared_ptr<AccelerationStructureImpl>> references,
        std::shared_ptr<BufferImpl> owned_input);
    void update_tlas_at(AccelerationStructure&, vk::DeviceAddress records,
        std::uint32_t count);

    std::uint64_t allocate_handle() noexcept { return next_handle_++; }

private:
    friend class ComputePipelineImpl;
    friend class GraphicsPipelineImpl;
    friend class RayTracingPipelineImpl;
    friend struct BufferImpl;
    friend struct ImageImpl;
    friend struct AccelerationStructureImpl;
    struct Pending {
        GpuToken token;
        vk::UniqueCommandBuffer command;
        std::vector<std::shared_ptr<void>> resources;
    };
    // A resource destroyed while the GPU may still be reading it is retired
    // here and released once the timeline passes the value it was retired at.
    struct Retired {
        std::uint64_t timeline = 0;
        std::function<void()> release;
    };

    struct Capabilities {
        bool mandatory = false;   // BDA, timeline, sync2, dynamic rendering,
                                  // descriptor heap, unified image layouts
        bool acceleration_structure = false;
        bool ray_query = false;
        bool ray_tracing = false;
    };
    static Capabilities probe(vk::PhysicalDevice);
    void adopt_capabilities(const Capabilities&);

    void create_instance(const DeviceConfig& config);
    void create_surface(const DeviceConfig& config);
    void select_physical_device();
    void create_device(const DeviceConfig& config);
    void create_allocator();
    void create_command_state();
    void create_descriptor_heap();
    void reap_completed();
public:
    // Called from resource destructors: defers the Vulkan/VMA release until
    // every submission that could reference the resource has completed.
    void retire(std::function<void()> release);
private:
    GpuToken submit(const std::function<void(vk::CommandBuffer)>& record,
                    std::vector<std::shared_ptr<void>> resources = {});
    static vk::PipelineStageFlags2 stage_mask(Stage stage);
    vk::Instance vk_instance() const noexcept { return instance_.get(); }
    vk::Device vk_device() const noexcept { return device_.get(); }

    std::weak_ptr<DeviceImpl> self_;
    DeviceFeatures features_{};
    vk::UniqueInstance instance_;
    vk::UniqueDebugUtilsMessengerEXT messenger_;
    vk::PhysicalDevice physical_device_;
    vk::UniqueDevice device_;
    vk::Queue queue_;
    SurfaceProvider* surface_provider_ = nullptr;
    vk::UniqueSurfaceKHR surface_;
    // Set between begin_frame and end_frame. Its presence is what makes
    // submit() batch into the frame instead of submitting immediately.
    vk::CommandBuffer frame_command_;
    GpuToken frame_token_{};
    std::vector<std::shared_ptr<void>> frame_resources_;
    std::uint32_t queue_family_ = 0;
    bool acceleration_structure_supported_ = false;
    bool ray_query_supported_ = false;
    bool ray_tracing_supported_ = false;
    bool external_memory_fd_enabled_ = false;
    bool external_semaphore_fd_enabled_ = false;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    vk::UniqueCommandPool command_pool_;
    vk::UniqueSemaphore timeline_;
    vk::UniqueQueryPool timestamp_query_pool_;
    std::uint32_t next_timestamp_query_ = 0;
    float timestamp_period_ns_ = 1.0f;
    std::uint32_t timestamp_valid_bits_ = 64;
    std::uint64_t next_timeline_ = 1;
    std::uint64_t next_handle_ = 1;
    std::shared_ptr<BufferImpl> argument_arena_;
    std::shared_ptr<BufferImpl> descriptor_heap_;
    std::shared_ptr<BufferImpl> sampler_heap_;
    vk::PhysicalDeviceDescriptorHeapPropertiesEXT descriptor_properties_{};
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties_{};
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties_{};
    vk::DeviceSize descriptor_heap_offset_ = 0;
    vk::DeviceSize descriptor_stride_ = 0;
    // Descriptor zero is reserved so opaque handles remain truthy even when
    // the underlying heap index is the first application-visible slot.
    std::uint32_t next_descriptor_ = 1;
    std::vector<std::uint32_t> free_descriptors_;
    vk::DeviceSize sampler_heap_offset_ = 0;
    vk::DeviceSize sampler_stride_ = 0;
    std::uint32_t next_sampler_descriptor_ = 1;
    std::vector<std::uint32_t> free_sampler_descriptors_;
    std::mutex descriptor_mutex_;
    // Ring offset into argument_arena_; see DeviceImpl::stage_arguments.
    std::size_t argument_offset_ = 0;
    std::mutex argument_mutex_;
    std::map<vk::DeviceAddress, std::weak_ptr<BufferImpl>> buffers_;
    std::vector<std::weak_ptr<ImageImpl>> images_;
    std::vector<std::weak_ptr<SamplerImpl>> samplers_;
    std::vector<std::weak_ptr<AccelerationStructureImpl>> acceleration_structures_;
    std::deque<Pending> pending_;
    // Guarded by its own mutex: retiring happens inside resource destructors,
    // which can run while mutex_ is already held (a completed submission
    // dropping its last reference to a buffer, for instance).
    std::vector<Retired> retired_;
    std::mutex retire_mutex_;
    vk::CommandBuffer active_command_;
    vk::Format active_color_format_ = vk::Format::eUndefined;
    bool active_has_depth_ = false;
    bool active_flip_y_ = false;
    std::vector<std::shared_ptr<void>> active_resources_;
    bool shut_down_ = false;
    std::uint32_t active_width_ = 0;
    std::uint32_t active_height_ = 0;
    mutable std::mutex mutex_;
};

std::shared_ptr<BufferImpl> make_buffer(const std::shared_ptr<DeviceImpl>&, std::size_t, std::size_t);
void upload_buffer(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<BufferImpl>&,
                   const void*, std::size_t, std::size_t, std::size_t);
void download_buffer(const std::shared_ptr<DeviceImpl>&, const std::shared_ptr<BufferImpl>&,
                     void*, std::size_t, std::size_t);
std::uint64_t buffer_address(const std::shared_ptr<BufferImpl>&);
ResourceHandle buffer_handle(const std::shared_ptr<BufferImpl>&);
std::shared_ptr<ImageImpl> make_image(const std::shared_ptr<DeviceImpl>&, std::uint32_t,
                                      std::uint32_t, ImageUsage, ImageFormat);
std::uint64_t image_handle(const std::shared_ptr<ImageImpl>&);
std::uint64_t image_sampled_handle(const std::shared_ptr<ImageImpl>&);
std::uint64_t image_storage_handle(const std::shared_ptr<ImageImpl>&);
AccelerationStructureHandle acceleration_structure_handle(
    const std::shared_ptr<AccelerationStructureImpl>&);

} // namespace gpu::detail
