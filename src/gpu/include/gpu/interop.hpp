#pragma once

#include "image.hpp"
#include "swapchain.hpp"
#include "types.hpp"

#include <cstdint>

namespace gpu {
class Device;
}

// Native-handle access for third-party libraries that speak Vulkan directly.
//
// This is not an escape hatch for recording work: the ordinary API owns every
// pipeline, every dispatch and every submission, and nothing here lets a
// caller drive them from the outside. It exists because libraries such as
// Dear ImGui's Vulkan backend are configured with raw handles and render into
// a raw command buffer, and no abstraction can change that.
//
// Everything returned here is owned by the Device and valid only for as long
// as it is. Handles are widened to uintptr_t so the public headers stay free
// of Vulkan types; reinterpret_cast them back to the Vulkan type named in
// each comment.
namespace gpu::interop {

struct DeviceHandles {
    std::uintptr_t instance = 0;         // VkInstance
    std::uintptr_t physical_device = 0;  // VkPhysicalDevice
    std::uintptr_t device = 0;           // VkDevice
    std::uintptr_t queue = 0;            // VkQueue
    std::uint32_t queue_family = 0;
};

// A duplicated POSIX FD for an image allocation created with
// ImageUsage::ExternalMemory. Ownership transfers to the caller, which must
// close it after importing it into the consuming API. The memory remains
// owned by gpu::Device; the consumer must release its imported object before
// the Image is destroyed.
struct ExternalImageMemory {
    int fd = -1;
    std::uint64_t allocation_size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0; // VkFormat
};

// One-shot Vulkan completion semaphore exported as an opaque POSIX FD.
// Import it into GL_EXT_semaphore_fd and wait before accessing an external
// image.  The FD is consumed by glImportSemaphoreFdEXT.
struct ExternalSemaphore {
    int fd = -1;
};

// The handles a Vulkan-native library needs to initialise itself against this
// device.
DeviceHandles device_handles(Device& device);

// The VkCommandBuffer the frame is recording into. Valid between begin_frame
// and end_frame. A library given this buffer may record its own draws, but
// must not begin or end the buffer, submit it, or leave a render scope open.
std::uintptr_t command_buffer(const Frame& frame);

// The VkImageView backing an image the library owns - what ImGui's
// AddTexture wants in order to sample a rendered AOV.
std::uintptr_t image_view(Device& device, ImageHandle image);

// Export the dedicated opaque-FD allocation backing an image. Throws when
// the image was not created with ImageUsage::ExternalMemory or the selected
// Vulkan implementation lacks VK_KHR_external_memory_fd.
ExternalImageMemory export_image_memory(Device& device, ImageHandle image);
ExternalSemaphore signal_external(Device& device);

// The VkFormat for one of the library's public formats, for third-party
// pipelines that must be created against a matching attachment format.
std::uint32_t native_format(ImageFormat format);

} // namespace gpu::interop
