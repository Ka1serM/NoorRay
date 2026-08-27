#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "internal.hpp"
#include "gpu/interop.hpp"

#include <cstring>

namespace gpu::detail {

ResourceHandle DeviceImpl::allocate_resource() {
    std::lock_guard descriptor_lock(descriptor_mutex_);
    if (!free_descriptors_.empty()) {
        const std::uint32_t value = free_descriptors_.back();
        free_descriptors_.pop_back();
        return ResourceHandle{value};
    }
    if (next_descriptor_ >= descriptor_capacity)
        throw Error(ErrorCode::OutOfMemory, "gpu resource descriptor heap exhausted");
    return ResourceHandle{next_descriptor_++};
}

// Every resource descriptor is written the same way: pick the slot, hand the
// driver the descriptor payload, and let it fill the heap for us.
void DeviceImpl::write_descriptor(const ResourceHandle handle,
    const vk::ResourceDescriptorInfoEXT& descriptor) const {
    if (!handle || handle.value >= descriptor_capacity)
        throw Error(ErrorCode::InvalidArgument, "resource descriptor handle is out of range");
    const vk::HostAddressRangeEXT destination{
        static_cast<std::byte*>(descriptor_heap_->mapped)
            + descriptor_heap_offset_ + handle.value * descriptor_stride_,
        static_cast<std::size_t>(descriptor_stride_)};
    if (vk_device().writeResourceDescriptorsEXT(1, &descriptor, &destination)
        != vk::Result::eSuccess)
        throw Error(ErrorCode::InvalidState, "Vulkan descriptor heap write failed");
}

ResourceHandle DeviceImpl::buffer_resource(const std::shared_ptr<BufferImpl>& buffer) {
    if (!buffer || buffer->device.get() != this)
        throw Error(ErrorCode::InvalidResource, "GPU buffer belongs to another device");
    std::lock_guard lock(mutex_);
    if (!buffer->handle) {
        buffer->handle = allocate_resource();
        const vk::DeviceAddressRangeEXT range{buffer->address, buffer->size};
        vk::ResourceDescriptorDataEXT data{};
        data.pAddressRange = &range;
        write_descriptor(buffer->handle,
            {vk::DescriptorType::eStorageBuffer, data});
    }
    return buffer->handle;
}

void DeviceImpl::write_image_descriptor(const ImageImpl& image,
    const ResourceHandle handle, const vk::DescriptorType type) const {
    const vk::ImageViewCreateInfo view_info{
        {}, image.image, vk::ImageViewType::e2D, image.format, {},
        {image.aspect, 0, 1, 0, 1}};
    const vk::ImageDescriptorInfoEXT image_info{&view_info, vk::ImageLayout::eGeneral};
    vk::ResourceDescriptorDataEXT data{};
    data.pImage = &image_info;
    write_descriptor(handle, {type, data});
}

SamplerHandle DeviceImpl::write_sampler_descriptor(const vk::SamplerCreateInfo& info) {
    if (!sampler_heap_)
        throw Error(ErrorCode::UnsupportedFeature,
            "this device reports no sampler descriptor storage");
    const vk::DeviceSize reserved = descriptor_properties_.minSamplerHeapReservedRange;
    const vk::DeviceSize available = descriptor_properties_.maxSamplerHeapSize > reserved
        ? descriptor_properties_.maxSamplerHeapSize - reserved : 0;
    std::lock_guard descriptor_lock(descriptor_mutex_);
    std::uint32_t descriptor = 0;
    if (!free_sampler_descriptors_.empty()) {
        descriptor = free_sampler_descriptors_.back();
        free_sampler_descriptors_.pop_back();
    } else {
        descriptor = next_sampler_descriptor_++;
    }
    if (descriptor >= sampler_descriptor_capacity
        || descriptor > available / sampler_stride_)
        throw Error(ErrorCode::OutOfMemory, "gpu sampler descriptor heap exhausted");
    const vk::DeviceSize offset = sampler_heap_offset_
        + static_cast<vk::DeviceSize>(descriptor) * sampler_stride_;
    const vk::HostAddressRangeEXT destination{
        static_cast<std::byte*>(sampler_heap_->mapped) + offset,
        static_cast<std::size_t>(sampler_stride_)};
    if (vk_device().writeSamplerDescriptorsEXT(1, &info, &destination) != vk::Result::eSuccess)
        throw Error(ErrorCode::InvalidState, "Vulkan descriptor heap sampler write failed");
    return SamplerHandle{descriptor};
}

AccelerationStructureHandle DeviceImpl::write_acceleration_structure_descriptor(
    const vk::AccelerationStructureKHR acceleration_structure, const vk::DeviceAddress address,
    const vk::DeviceSize size) {
    if (!acceleration_structure)
        return {};
    const ResourceHandle resource = allocate_resource();
    const vk::DeviceAddressRangeEXT range{address, size};
    vk::ResourceDescriptorDataEXT data{};
    data.pAddressRange = &range;
    write_descriptor(resource, {vk::DescriptorType::eAccelerationStructureKHR, data});
    return AccelerationStructureHandle{resource.value};
}

ImageImpl::~ImageImpl() {
    if (!device || !image)
        return;
    // A presentation image is owned by the swapchain, but its view, its
    // identity handle and any descriptors written for it are ours - so the
    // retire path runs either way and only the image release is conditional.
    device->retire([owner = device, allocator = device->allocator_, image = this->image,
        allocation = this->allocation, external_memory = this->external_memory,
        owns = owns_image, view = view.release(),
        vk_device = device->device(), identity = handle,
        sampled = sampled_handle, storage = storage_handle] {
        if (view)
            vk_device.destroyImageView(view);
        if (owns && allocation)
            vmaDestroyImage(allocator, image, allocation);
        else if (owns && external_memory) {
            vk_device.destroyImage(image);
            vk_device.freeMemory(external_memory);
        }
        owner->release_resource(ResourceHandle{identity.value});
        if (sampled.value != identity.value)
            owner->release_resource(ResourceHandle{sampled.value});
        if (storage.value != identity.value && storage.value != sampled.value)
            owner->release_resource(ResourceHandle{storage.value});
    });
}

std::shared_ptr<ImageImpl> DeviceImpl::create_image(const std::uint32_t width,
    const std::uint32_t height, const ImageUsage usage, const ImageFormat requested_format) {
    vk::ImageUsageFlags vulkan_usage = vk::ImageUsageFlagBits::eTransferSrc
        | vk::ImageUsageFlagBits::eTransferDst;
    const auto requested = static_cast<std::uint32_t>(usage);
    const auto wants = [requested](const ImageUsage flag) {
        return (requested & static_cast<std::uint32_t>(flag)) != 0;
    };
    const bool depth = wants(ImageUsage::DepthAttachment);
    const bool external_memory = wants(ImageUsage::ExternalMemory);
    if (external_memory && !external_memory_fd_enabled_)
        throw Error(ErrorCode::UnsupportedFeature,
            "VK_KHR_external_memory_fd is unavailable on this Vulkan device");
    if (wants(ImageUsage::Sampled))
        vulkan_usage |= vk::ImageUsageFlagBits::eSampled;
    if (wants(ImageUsage::Storage))
        vulkan_usage |= vk::ImageUsageFlagBits::eStorage;
    if (wants(ImageUsage::ColorAttachment))
        vulkan_usage |= vk::ImageUsageFlagBits::eColorAttachment;
    if (depth)
        vulkan_usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

    const ImageFormat format_choice = requested_format == ImageFormat::Auto
        ? (depth ? ImageFormat::D32Float : ImageFormat::Rgba8Unorm) : requested_format;
    if ((depth && format_choice != ImageFormat::D32Float)
        || (!depth && format_choice == ImageFormat::D32Float))
        throw Error(ErrorCode::InvalidArgument, "image format does not match image usage");
    const vk::Format format = to_vulkan_format(format_choice);
    const vk::ImageAspectFlags aspect = depth
        ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
    vk::ExternalMemoryImageCreateInfo external_info{};
    external_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format,
        {width, height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal, vulkan_usage, vk::SharingMode::eExclusive, {},
        vk::ImageLayout::eUndefined);
    vk::Image image{};
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::DeviceMemory exported_memory{};
    if (external_memory) {
        imageInfo.pNext = &external_info;
        image = vk_device().createImage(imageInfo);
        const vk::MemoryRequirements requirements = vk_device().getImageMemoryRequirements(image);
        const auto properties = physical_device_.getMemoryProperties();
        std::uint32_t memory_type = ~0u;
        for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i))
                && (properties.memoryTypes[i].propertyFlags
                    & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                memory_type = i;
                break;
            }
        if (memory_type == ~0u) {
            vk_device().destroyImage(image);
            throw Error(ErrorCode::OutOfMemory, "no device-local memory type for external image");
        }
        vk::MemoryDedicatedAllocateInfo dedicated{};
        dedicated.image = image;
        vk::ExportMemoryAllocateInfo export_info{};
        export_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
        export_info.pNext = &dedicated;
        exported_memory = vk_device().allocateMemory(
            vk::MemoryAllocateInfo(requirements.size, memory_type, &export_info));
        vk_device().bindImageMemory(image, exported_memory, 0);
    } else {
        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VkImage rawImage = VK_NULL_HANDLE;
        if (vmaCreateImage(allocator_, reinterpret_cast<const VkImageCreateInfo*>(&imageInfo),
                           &allocationInfo, &rawImage, &allocation, nullptr) != VK_SUCCESS)
            throw Error(ErrorCode::OutOfMemory, "VMA image allocation failed");
        image = rawImage;
    }

    auto result = std::make_shared<ImageImpl>();
    result->device = self_.lock();
    result->image = image;
    result->allocation = allocation;
    result->external_memory = exported_memory;
    result->exportable = external_memory;
    result->format = format;
    result->aspect = aspect;
    result->width = width;
    result->height = height;
    result->byte_size = static_cast<std::size_t>(width) * height
        * format_texel_size(format_choice);
    if (wants(ImageUsage::Storage))
        result->storage_handle = ImageHandle{allocate_resource().value};
    if (wants(ImageUsage::Sampled))
        result->sampled_handle = ImageHandle{allocate_resource().value};
    result->handle = result->storage_handle
        ? result->storage_handle : result->sampled_handle;
    // Render targets and depth-only images still need an opaque identity for
    // RenderTarget lookup even when they have no shader descriptor role.
    if (!result->handle)
        result->handle = ImageHandle{allocate_resource().value};
    images_.push_back(result);
    const vk::ImageViewCreateInfo viewInfo({}, result->image, vk::ImageViewType::e2D,
        format, {}, {aspect, 0, 1, 0, 1});
    result->view = vk_device().createImageViewUnique(viewInfo);
    if (wants(ImageUsage::Storage))
        write_image_descriptor(*result, ResourceHandle{result->storage_handle.value},
            vk::DescriptorType::eStorageImage);
    if (wants(ImageUsage::Sampled))
        write_image_descriptor(*result, ResourceHandle{result->sampled_handle.value},
            vk::DescriptorType::eSampledImage);
    // Unified image layouts removes every transition between usages, but an
    // image is still created in UNDEFINED and has to reach GENERAL once. This
    // is the only layout transition left in the library.
    submit([image = result->image, aspect](const vk::CommandBuffer command) {
        vk::ImageMemoryBarrier2 barrier{};
        barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
            .setSrcAccessMask(vk::AccessFlagBits2::eNone)
            .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
            .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
            .setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setImage(image)
            .setSubresourceRange({aspect, 0, 1, 0, 1});
        command.pipelineBarrier2({{}, {}, {}, barrier});
    }, {result});
    return result;
}

// Both transfer directions need the same memory dependency around the copy;
// with unified image layouts that is all they need.
void DeviceImpl::transfer_barrier(const vk::CommandBuffer command,
    const bool before) const {
    vk::MemoryBarrier2 barrier{};
    if (before) {
        barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
            .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setDstAccessMask(vk::AccessFlagBits2::eTransferRead
                | vk::AccessFlagBits2::eTransferWrite);
    } else {
        barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setSrcAccessMask(vk::AccessFlagBits2::eTransferRead
                | vk::AccessFlagBits2::eTransferWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
            .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    }
    command.pipelineBarrier2({{}, barrier, {}, {}});
}

void DeviceImpl::upload_image(const std::shared_ptr<ImageImpl>& image, const void* data,
    const std::size_t bytes) {
    if (!image || !data || bytes == 0 || bytes != image->byte_size)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU image upload");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(staging->mapped, data, bytes);
    vmaFlushAllocation(allocator_, staging->allocation, 0, bytes);
    submit([this, staging, image](const vk::CommandBuffer command) {
        transfer_barrier(command, true);
        command.copyBufferToImage(staging->buffer, image->image,
            vk::ImageLayout::eGeneral,
            vk::BufferImageCopy(0, 0, 0, {image->aspect, 0, 0, 1}, {0, 0, 0},
                {image->width, image->height, 1}));
        transfer_barrier(command, false);
    }, {staging, image});
}

void DeviceImpl::download_image(const std::shared_ptr<ImageImpl>& image, void* data,
    const std::size_t bytes) {
    if (!image || !data || bytes == 0 || bytes != image->byte_size)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU image download");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_TO_CPU, true);
    const GpuToken token = submit([this, staging, image](const vk::CommandBuffer command) {
        transfer_barrier(command, true);
        command.copyImageToBuffer(image->image, vk::ImageLayout::eGeneral,
            staging->buffer, vk::BufferImageCopy(0, 0, 0,
                {image->aspect, 0, 0, 1}, {0, 0, 0}, {image->width, image->height, 1}));
        transfer_barrier(command, false);
    }, {staging, image});
    wait(token);
    vmaInvalidateAllocation(allocator_, staging->allocation, 0, bytes);
    if (staging->mapped)
        std::memcpy(data, staging->mapped, bytes);
}

std::shared_ptr<ImageImpl> make_image(const std::shared_ptr<DeviceImpl>& device,
    const std::uint32_t width, const std::uint32_t height, const ImageUsage usage,
    const ImageFormat format) {
    return device->create_image(width, height, usage, format);
}

interop::ExternalImageMemory DeviceImpl::export_image_memory(const ImageHandle handle) {
    if (!external_memory_fd_enabled_)
        throw Error(ErrorCode::UnsupportedFeature,
            "VK_KHR_external_memory_fd is unavailable on this Vulkan device");
    const auto image = find_image(handle);
    if (!image || !image->exportable || !image->external_memory)
        throw Error(ErrorCode::InvalidArgument,
            "image was not created with gpu::ImageUsage::ExternalMemory");
    vk::MemoryGetFdInfoKHR fd_info{};
    fd_info.memory = image->external_memory;
    fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    const int fd = vk_device().getMemoryFdKHR(fd_info);
    if (fd < 0)
        throw Error(ErrorCode::InvalidState, "Vulkan returned an invalid external-memory FD");
    const vk::MemoryRequirements requirements = vk_device().getImageMemoryRequirements(image->image);
    return {fd, requirements.size, image->width, image->height,
        static_cast<std::uint32_t>(image->format)};
}

interop::ExternalSemaphore DeviceImpl::signal_external()
{
    std::lock_guard lock(mutex_);
    if (frame_command_)
        throw Error(ErrorCode::InvalidState,
            "external semaphore export requires a completed standalone submission");
    if (!external_semaphore_fd_enabled_)
        throw Error(ErrorCode::UnsupportedFeature,
            "VK_KHR_external_semaphore_fd is unavailable on this Vulkan device");

    vk::ExportSemaphoreCreateInfo exportInfo{};
    exportInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.pNext = &exportInfo;
    vk::UniqueSemaphore semaphore = vk_device().createSemaphoreUnique(semaphoreInfo);
    // Queue order makes this signal happen after the renderer's preceding
    // standalone dispatch.  GL_EXT_semaphore_fd consumes the FD and waits on
    // the same binary payload before accessing the shared allocation.
    vk::SubmitInfo submitInfo{};
    submitInfo.setSignalSemaphores(semaphore.get());
    queue_.submit(submitInfo);
    vk::SemaphoreGetFdInfoKHR fdInfo{};
    fdInfo.semaphore = semaphore.get();
    fdInfo.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    const int fd = vk_device().getSemaphoreFdKHR(fdInfo);
    if (fd < 0)
        throw Error(ErrorCode::InvalidState,
            "Vulkan returned an invalid external-semaphore FD");
    // The exported FD owns the external payload. The Vulkan semaphore itself
    // is no longer needed after queue submission and may be released.
    return {fd};
}
std::uint64_t image_handle(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->handle.value : 0;
}
std::uint64_t image_sampled_handle(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->sampled_handle.value : 0;
}
std::uint64_t image_storage_handle(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->storage_handle.value : 0;
}
std::size_t image_byte_size(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->byte_size : 0;
}
void upload_image(const std::shared_ptr<DeviceImpl>& device,
    const std::shared_ptr<ImageImpl>& image, const void* data, const std::size_t bytes) {
    device->upload_image(image, data, bytes);
}
void download_image(const std::shared_ptr<DeviceImpl>& device,
    const std::shared_ptr<ImageImpl>& image, void* data, const std::size_t bytes) {
    device->download_image(image, data, bytes);
}

} // namespace gpu::detail
