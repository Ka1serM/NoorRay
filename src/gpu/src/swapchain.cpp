#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace gpu {
namespace detail {
namespace {

// Presentation images live in GENERAL like every other image this library
// owns; VK_KHR_unified_image_layouts is mandatory, so a render pass, a blit
// and a compute write all accept it. Only the handoff to the presentation
// engine needs a specific layout, and that transition is what these two
// helpers exist for.
void transition(const vk::CommandBuffer command, const vk::Image image,
    const vk::ImageLayout from, const vk::ImageLayout to) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
        .setOldLayout(from)
        .setNewLayout(to)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    command.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
}

ImageFormat to_public_format(const vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Unorm: return ImageFormat::Bgra8Unorm;
    case vk::Format::eR8G8B8A8Unorm: return ImageFormat::Rgba8Unorm;
    case vk::Format::eR32G32B32A32Sfloat: return ImageFormat::Rgba32Float;
    case vk::Format::eR32Uint: return ImageFormat::R32Uint;
    default: return ImageFormat::Bgra8Unorm;
    }
}

// Pick the caller's preferred format when the surface offers it, and fall back
// to whatever the surface does offer rather than failing: a swapchain format
// is a negotiation, not a requirement the caller can insist on.
vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& available,
    const ImageFormat preferred) {
    if (available.empty())
        throw Error(ErrorCode::UnsupportedFeature, "surface reports no formats");
    if (preferred != ImageFormat::Auto) {
        const vk::Format wanted = to_vulkan_format(preferred);
        for (const auto& candidate : available)
            if (candidate.format == wanted
                && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return candidate;
    }
    for (const auto& candidate : available)
        if ((candidate.format == vk::Format::eB8G8R8A8Unorm
                || candidate.format == vk::Format::eR8G8B8A8Unorm)
            && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return candidate;
    return available.front();
}

vk::PresentModeKHR choose_present_mode(const std::vector<vk::PresentModeKHR>& available) {
    // Mailbox keeps latency low without tearing; FIFO is the only mode the
    // spec guarantees, so it is the fallback.
    if (std::ranges::find(available, vk::PresentModeKHR::eMailbox) != available.end())
        return vk::PresentModeKHR::eMailbox;
    return vk::PresentModeKHR::eFifo;
}

} // namespace

SwapchainImpl::~SwapchainImpl() {
    if (!device)
        return;
    // Presentation images and their views may still be referenced by work in
    // flight; the queue is the only thing that can tell us it is done.
    device->device().waitIdle();
    images.clear();
    acquire_semaphores.clear();
    present_semaphores.clear();
    swapchain.reset();
}

std::shared_ptr<ImageImpl> DeviceImpl::wrap_presentation_image(const vk::Image image,
    const vk::Format format, const ImageFormat public_format,
    const std::uint32_t width, const std::uint32_t height) {
    auto result = std::make_shared<ImageImpl>();
    result->device = self_.lock();
    result->image = image;
    result->allocation = VK_NULL_HANDLE;
    result->owns_image = false;
    result->format = format;
    result->aspect = vk::ImageAspectFlagBits::eColor;
    result->width = width;
    result->height = height;
    result->byte_size = static_cast<std::size_t>(width) * height
        * format_texel_size(public_format);
    // A presentation image is a render target and a blit destination, never a
    // shader resource, so it needs an identity handle but no heap descriptor.
    result->handle = ImageHandle{allocate_resource().value};
    const vk::ImageViewCreateInfo view_info({}, image, vk::ImageViewType::e2D, format, {},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    result->view = vk_device().createImageViewUnique(view_info);
    images_.push_back(result);
    return result;
}

void DeviceImpl::rebuild_swapchain(SwapchainImpl& chain) {
    if (!surface_provider_ || !surface_)
        throw Error(ErrorCode::InvalidState, "this device was created without a SurfaceProvider");

    const auto capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_.get());
    std::uint32_t width = surface_provider_->width();
    std::uint32_t height = surface_provider_->height();
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        width = capabilities.currentExtent.width;
        height = capabilities.currentExtent.height;
    }
    width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    if (width == 0 || height == 0) {
        // A minimised window has a zero-sized surface. Leave the chain stale
        // so begin_frame keeps returning nothing until it comes back.
        chain.width = chain.height = 0;
        chain.stale = true;
        return;
    }

    const auto surface_format = choose_format(
        physical_device_.getSurfaceFormatsKHR(surface_.get()),
        chain.format == vk::Format::eUndefined ? chain.public_format : ImageFormat::Auto);
    chain.format = surface_format.format;
    chain.color_space = surface_format.colorSpace;
    chain.public_format = to_public_format(surface_format.format);
    chain.present_mode = choose_present_mode(
        physical_device_.getSurfacePresentModesKHR(surface_.get()));

    std::uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0)
        image_count = std::min(image_count, capabilities.maxImageCount);

    vk::SwapchainCreateInfoKHR info{};
    info.setSurface(surface_.get())
        .setMinImageCount(image_count)
        .setImageFormat(chain.format)
        .setImageColorSpace(chain.color_space)
        .setImageExtent({width, height})
        .setImageArrayLayers(1)
        // Colour attachment for render scopes, transfer destination for the
        // blit that composites a rendered image onto the frame.
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment
            | vk::ImageUsageFlagBits::eTransferDst
            | vk::ImageUsageFlagBits::eTransferSrc)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(chain.present_mode)
        .setClipped(VK_TRUE)
        .setOldSwapchain(chain.swapchain.get());

    auto replacement = vk_device().createSwapchainKHRUnique(info);

    // Drop the old images only after the new chain exists, and only once the
    // GPU is done with them - views retire through the ordinary deferred path,
    // but the VkImages belong to the chain we are about to destroy.
    vk_device().waitIdle();
    chain.images.clear();
    chain.swapchain = std::move(replacement);

    const auto raw_images = vk_device().getSwapchainImagesKHR(chain.swapchain.get());
    chain.images.reserve(raw_images.size());
    for (const vk::Image image : raw_images)
        chain.images.push_back(
            wrap_presentation_image(image, chain.format, chain.public_format, width, height));
    chain.presented.assign(raw_images.size(), false);

    if (chain.acquire_semaphores.size() < raw_images.size()) {
        chain.acquire_semaphores.clear();
        chain.present_semaphores.clear();
        for (std::size_t i = 0; i < raw_images.size(); ++i) {
            chain.acquire_semaphores.push_back(vk_device().createSemaphoreUnique({}));
            chain.present_semaphores.push_back(vk_device().createSemaphoreUnique({}));
        }
    }
    chain.semaphore_cursor = 0;
    chain.width = width;
    chain.height = height;
    chain.stale = false;
}

std::shared_ptr<SwapchainImpl> DeviceImpl::create_swapchain(const ImageFormat preferred) {
    if (!presenting())
        throw Error(ErrorCode::InvalidState,
            "Device::swapchain requires a DeviceConfig with a SurfaceProvider");
    auto chain = std::make_shared<SwapchainImpl>();
    chain->device = self_.lock();
    chain->public_format = preferred == ImageFormat::Auto ? ImageFormat::Bgra8Unorm : preferred;
    std::lock_guard lock(mutex_);
    rebuild_swapchain(*chain);
    return chain;
}

std::shared_ptr<Frame::State> DeviceImpl::begin_frame(
    const std::shared_ptr<SwapchainImpl>& chain) {
    if (!chain)
        throw Error(ErrorCode::InvalidArgument, "begin_frame requires a valid Swapchain");
    std::lock_guard lock(mutex_);
    if (shut_down_)
        throw Error(ErrorCode::InvalidState, "gpu::Device has been shut down");
    if (frame_command_)
        throw Error(ErrorCode::InvalidState, "a frame is already open on this device");
    reap_completed();

    const bool size_changed = surface_provider_
        && (chain->width != surface_provider_->width()
            || chain->height != surface_provider_->height());
    if (chain->stale || size_changed)
        rebuild_swapchain(*chain);
    if (chain->stale || chain->images.empty())
        return {};

    const std::uint32_t semaphore_index = chain->semaphore_cursor;
    chain->semaphore_cursor =
        (chain->semaphore_cursor + 1) % static_cast<std::uint32_t>(chain->acquire_semaphores.size());

    std::uint32_t image_index = 0;
    const vk::Result acquired = vk_device().acquireNextImageKHR(chain->swapchain.get(),
        std::numeric_limits<std::uint64_t>::max(),
        chain->acquire_semaphores[semaphore_index].get(), {}, &image_index);
    if (acquired == vk::Result::eErrorOutOfDateKHR) {
        chain->stale = true;
        return {};
    }
    if (acquired != vk::Result::eSuccess && acquired != vk::Result::eSuboptimalKHR)
        throw Error(ErrorCode::DeviceLost, "swapchain image acquisition failed");
    // A suboptimal chain still presents correctly, so this frame is rendered
    // and the rebuild happens on the next one.
    if (acquired == vk::Result::eSuboptimalKHR)
        chain->stale = true;

    auto commands = vk_device().allocateCommandBuffersUnique(
        {*command_pool_, vk::CommandBufferLevel::ePrimary, 1});
    if (commands.empty())
        throw Error(ErrorCode::OutOfMemory, "Vulkan command-buffer allocation failed");

    auto state = std::make_shared<Frame::State>();
    state->device = self_.lock();
    state->swapchain = chain;
    state->owned_command = std::move(commands.front());
    state->command = state->owned_command.get();
    state->image_index = image_index;
    state->semaphore_index = semaphore_index;
    state->target = chain->images[image_index]->handle;
    state->open = true;

    state->command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    bind_heaps(state->command);
    transition(state->command, chain->images[image_index]->image,
        chain->presented[image_index] ? vk::ImageLayout::ePresentSrcKHR
                                      : vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral);

    frame_command_ = state->command;
    frame_token_ = GpuToken{next_timeline_};
    frame_resources_.clear();
    return state;
}

void DeviceImpl::end_frame(Frame::State& state) {
    std::lock_guard lock(mutex_);
    if (!state.open || !frame_command_ || frame_command_ != state.command)
        throw Error(ErrorCode::InvalidState, "end_frame called without a matching begin_frame");
    SwapchainImpl& chain = *state.swapchain;

    transition(state.command, chain.images[state.image_index]->image,
        vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    state.command.end();

    const GpuToken token{next_timeline_++};
    const vk::Semaphore acquire = chain.acquire_semaphores[state.semaphore_index].get();
    const vk::Semaphore presented = chain.present_semaphores[state.image_index].get();
    const std::array signal_semaphores{timeline_.get(), presented};
    // Only the timeline entry carries a value; the binary semaphore's slot is
    // ignored but must still be present for the arrays to line up.
    const std::array<std::uint64_t, 2> signal_values{token.value, 0};
    constexpr vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::TimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.setSignalSemaphoreValues(signal_values);
    vk::SubmitInfo submit_info{};
    submit_info.setCommandBuffers(state.command)
        .setWaitSemaphores(acquire)
        .setWaitDstStageMask(wait_stage)
        .setSignalSemaphores(signal_semaphores);
    submit_info.pNext = &timeline_info;
    queue_.submit(submit_info);

    pending_.push_back({token, std::move(state.owned_command), std::move(frame_resources_)});
    frame_resources_.clear();
    frame_command_ = nullptr;
    state.open = false;
    state.command = nullptr;
    chain.presented[state.image_index] = true;

    vk::PresentInfoKHR present_info{};
    const vk::SwapchainKHR raw_swapchain = chain.swapchain.get();
    present_info.setWaitSemaphores(presented)
        .setSwapchains(raw_swapchain)
        .setPImageIndices(&state.image_index);
    const vk::Result presented_result = queue_.presentKHR(&present_info);
    if (presented_result == vk::Result::eErrorOutOfDateKHR
        || presented_result == vk::Result::eSuboptimalKHR)
        chain.stale = true;
    else if (presented_result != vk::Result::eSuccess)
        throw Error(ErrorCode::DeviceLost, "swapchain presentation failed");
}

} // namespace detail

// --- Public swapchain surface -------------------------------------------

ImageFormat Swapchain::format() const noexcept {
    return impl_ ? impl_->public_format : ImageFormat::Auto;
}
std::uint32_t Swapchain::width() const noexcept { return impl_ ? impl_->width : 0; }
std::uint32_t Swapchain::height() const noexcept { return impl_ ? impl_->height : 0; }
std::uint32_t Swapchain::image_count() const noexcept {
    return impl_ ? static_cast<std::uint32_t>(impl_->images.size()) : 0;
}
void Swapchain::invalidate() {
    if (impl_)
        impl_->stale = true;
}

Frame::Frame(Frame&&) noexcept = default;
Frame& Frame::operator=(Frame&&) noexcept = default;

Frame::~Frame() {
    // A frame dropped without end_frame discards its work rather than
    // presenting a half-recorded image. The command buffer and any resources
    // it referenced are released with it.
    if (impl_ && impl_->open && impl_->device) {
        impl_->command.end();
        impl_->device->abandon_frame(*impl_);
    }
}

ImageHandle Frame::target() const noexcept { return impl_ ? impl_->target : ImageHandle{}; }
std::uint32_t Frame::index() const noexcept { return impl_ ? impl_->image_index : 0; }
std::uint32_t Frame::width() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->width : 0;
}
std::uint32_t Frame::height() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->height : 0;
}
ImageFormat Frame::format() const noexcept {
    return impl_ && impl_->swapchain ? impl_->swapchain->public_format : ImageFormat::Auto;
}

} // namespace gpu
