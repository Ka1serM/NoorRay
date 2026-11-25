#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Device/Interop.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Python {

//! Shows NoorRHI images in a Blender GPUTexture on the GPU; every call needs Blender's OpenGL context current, as it is in RenderEngine.view_draw.
class GlInterop {
public:
    //! Vulkan writes one image while OpenGL copies the other, so the CPU does not wait for the copy of the last frame.
    static constexpr std::size_t slotCount = 2;

    GlInterop() = default;
    GlInterop(const GlInterop&) = delete;
    GlInterop& operator=(const GlInterop&) = delete;
    //! The OpenGL objects outlive this until a later call with a current context frees them.
    ~GlInterop();

    //! Imports RGBA16F images opened with ImageUsage::ExternalMemory, and takes as target the one bound RGBA16F texture of their size.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::span<const NoorRHI::Image, slotCount> images);
    //! Waits on the GPU for the Vulkan work submitted so far, then copies the image of the slot into the target texture.
    [[nodiscard]] std::expected<void, Core::Error> present(std::size_t slot);
    //! Waits until OpenGL has read the image of the slot, so that Vulkan can write it again.
    [[nodiscard]] std::expected<void, Core::Error> waitForCopy(std::size_t slot);
    bool isOpen() const { return target != 0; }

private:
    struct Slot {
        std::uint32_t memoryObject = 0;
        std::uint32_t texture = 0;
        //! The fence of the last copy from this slot.
        void* fence = nullptr;
    };

    void release();

    std::array<Slot, slotCount> slots{};
    NoorRHI::Interop::SharedSemaphore signal;
    std::uint32_t semaphore = 0;
    std::uint32_t target = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

}
