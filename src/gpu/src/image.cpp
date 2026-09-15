#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "internal.hpp"
#include "gpu/interop.hpp"

#include <cstring>

namespace gpu::detail {

// The shader-facing handle for an image: a resource-heap slot holding a
// descriptor for the whole image. The view is described inline, so the heap
// entry owns no VkImageView of its own; it stays valid while the image does,
// and ImageImpl's retire path returns the slot only after the GPU is done.
std::uint32_t DeviceImpl::write_image_descriptor(const ImageImpl& image,
    const vk::DescriptorType type) {
    const std::uint32_t slot = allocate_slot(texture_heap_);
    const vk::ImageViewUsageCreateInfo usage{type == vk::DescriptorType::eStorageImage
        ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled};
    const vk::ImageViewCreateInfo view({}, image.image, vk::ImageViewType::e2D,
        image.format, {}, {image.aspect, 0, 1, 0, 1}, &usage);
    const vk::ImageDescriptorInfoEXT image_info{&view, vk::ImageLayout::eGeneral};
    const vk::ResourceDescriptorInfoEXT descriptor{type,
        vk::ResourceDescriptorDataEXT{&image_info}};
    const vk::HostAddressRangeEXT destination = slot_range(texture_heap_, slot);
    if (vk_device().writeResourceDescriptorsEXT(1, &descriptor, &destination)
            != vk::Result::eSuccess) {
        release_slot(texture_heap_, slot);
        throw Error(ErrorCode::InvalidState, "writing an image descriptor failed");
    }
    flush_slot(texture_heap_, slot);
    return slot;
}

ImageImpl::~ImageImpl() {
    if (!device || !image)
        return;
    // A presentation image is owned by the swapchain, but its view is ours, so
    // the retire path runs either way and only the image release is
    // conditional. Heap slots go back to the free list here too, never
    // earlier: in-flight work may still read the descriptors they hold.
    device->retire([allocator = device->allocator_, image = this->image,
        allocation = this->allocation, external_memory = this->external_memory,
        owns = owns_image, view = view.release(),
        vk_device = device->device(), owner = device.get(),
        sampled = sampled_handle.value, storage = storage_handle.value] {
        owner->release_slot(owner->texture_heap_, sampled);
        owner->release_slot(owner->texture_heap_, storage);
        if (view)
            vk_device.destroyImageView(view);
        if (owns && allocation)
            vmaDestroyImage(allocator, image, allocation);
        else if (owns && external_memory) {
            vk_device.destroyImage(image);
            vk_device.freeMemory(external_memory);
        }
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
    // The identity handle names this image to render(), copy() and interop. It
    // is a weak host reference, deliberately unrelated to the heap indices
    // below, which only mean something to shaders.
    result->handle = ImageHandle{result};
    // The view serves render targets and interop; shader descriptors describe
    // their own view inline.
    const vk::ImageViewCreateInfo viewInfo({}, result->image, vk::ImageViewType::e2D,
        format, {}, {aspect, 0, 1, 0, 1});
    result->view = vk_device().createImageViewUnique(viewInfo);
    if (wants(ImageUsage::Storage))
        result->storage_handle = TextureHandle{
            write_image_descriptor(*result, vk::DescriptorType::eStorageImage)};
    // A sampled image and a sampler are separate heap entries; shaders that
    // filter pair this handle with a Sampler::handle().
    if (wants(ImageUsage::Sampled))
        result->sampled_handle = TextureHandle{
            write_image_descriptor(*result, vk::DescriptorType::eSampledImage)};
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
    if (!image || image->device.get() != this || !data || bytes != image->byte_size)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU image upload");
    if (frame_command_)
        throw Error(ErrorCode::InvalidState, "upload images before beginning a frame");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    std::memcpy(staging->mapped, data, bytes);
    if (vmaFlushAllocation(allocator_, staging->allocation, 0, bytes) != VK_SUCCESS)
        throw Error(ErrorCode::DeviceLost, "flushing image upload failed");
    submit([=](vk::CommandBuffer command) {
        command.copyBufferToImage(staging->buffer, image->image, vk::ImageLayout::eGeneral,
            vk::BufferImageCopy(0, 0, 0, {image->aspect, 0, 0, 1}, {0, 0, 0},
                {image->width, image->height, 1}));
    }, {staging, image});
}

void DeviceImpl::download_image(const std::shared_ptr<ImageImpl>& image, void* data,
    const std::size_t bytes) {
    if (!image || image->device.get() != this || !data || bytes != image->byte_size)
        throw Error(ErrorCode::InvalidArgument, "invalid GPU image download");
    if (frame_command_)
        throw Error(ErrorCode::InvalidState, "read back images after ending a frame");
    auto staging = create_buffer(bytes, vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_TO_CPU, true);
    const auto token = submit([=](vk::CommandBuffer command) {
        command.copyImageToBuffer(image->image, vk::ImageLayout::eGeneral, staging->buffer,
            vk::BufferImageCopy(0, 0, 0, {image->aspect, 0, 0, 1}, {0, 0, 0},
                {image->width, image->height, 1}));
    }, {staging, image});
    wait(token);
    if (vmaInvalidateAllocation(allocator_, staging->allocation, 0, bytes) != VK_SUCCESS)
        throw Error(ErrorCode::DeviceLost, "invalidating image readback failed");
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
    auto semaphore = std::make_shared<vk::UniqueSemaphore>(
        vk_device().createSemaphoreUnique(semaphoreInfo));
    // Queue order makes this signal happen after the renderer's preceding
    // standalone dispatch.  GL_EXT_semaphore_fd consumes the FD and waits on
    // the same binary payload before accessing the shared allocation.
    vk::SubmitInfo submitInfo{};
    const GpuToken token{next_timeline_};
    const std::array signals{semaphore->get(), timeline_.get()};
    const std::array<std::uint64_t, 2> values{0, token.value};
    vk::TimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.setSignalSemaphoreValues(values);
    submitInfo.setSignalSemaphores(signals);
    submitInfo.pNext = &timelineInfo;
    pending_.push_back({token, {}, {semaphore}});
    try {
        queue_.submit(submitInfo);
    } catch (...) {
        pending_.pop_back();
        throw;
    }
    ++next_timeline_;
    vk::SemaphoreGetFdInfoKHR fdInfo{};
    fdInfo.semaphore = semaphore->get();
    fdInfo.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    const int fd = vk_device().getSemaphoreFdKHR(fdInfo);
    if (fd < 0)
        throw Error(ErrorCode::InvalidState,
            "Vulkan returned an invalid external-semaphore FD");
    // Keep the Vulkan semaphore until its signal completes. The exported FD
    // independently owns the external payload.
    return {fd};
}
std::uint32_t image_sampled_handle(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->sampled_handle.value : 0;
}
std::uint32_t image_storage_handle(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->storage_handle.value : 0;
}
std::size_t image_byte_size(const std::shared_ptr<ImageImpl>& image) {
    return image ? image->byte_size : 0;
}
void upload_image(const std::shared_ptr<ImageImpl>& image, const void* data, std::size_t bytes) {
    image->device->upload_image(image, data, bytes);
}
void download_image(const std::shared_ptr<ImageImpl>& image, void* data, std::size_t bytes) {
    image->device->download_image(image, data, bytes);
}

} // namespace gpu::detail
