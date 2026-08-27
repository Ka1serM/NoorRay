#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

namespace gpu {

class Device;

class Shader;
class ComputePipeline;
class GraphicsPipeline;
class RayTracingPipeline;
class AccelerationStructure;
class Swapchain;
class Frame;
class SurfaceProvider;
struct GraphicsPipelineDesc;
struct RenderTarget;
struct RayTracingPipelineDesc;
struct TriangleGeometry;
struct Instance;
class Sampler;
template<class T> class Buffer;
template<class T> class Image;
enum class ImageUsage : std::uint32_t;
enum class ImageFormat;
struct SamplerDesc;

namespace detail { class DeviceImpl; }

namespace interop {
struct DeviceHandles;
struct ExternalImageMemory;
struct ExternalSemaphore;
DeviceHandles device_handles(Device&);
std::uintptr_t image_view(Device&, ImageHandle);
ExternalImageMemory export_image_memory(Device&, ImageHandle);
ExternalSemaphore signal_external(Device&);
}

// A GPU-side timing scope. The query is written by the GPU, so its result is
// available once the submission that recorded it has completed - after
// synchronize(), or after the frame that contained it has been presented and
// its token waited on.
class TimestampQuery {
public:
    TimestampQuery() = default;

    // Elapsed GPU time for the last completed measurement, in milliseconds.
    // Returns 0 before the first measurement completes.
    double milliseconds() const;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
    friend class Device;
    friend class detail::DeviceImpl;
    struct State;
    explicit TimestampQuery(std::shared_ptr<State> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<State> impl_;
};

// Buffer device addresses, dynamic rendering, the descriptor heap and unified
// image layouts are mandatory: a device that lacks any of them is rejected at
// construction, so only the optional ray-tracing capabilities are reported.
struct DeviceFeatures {
    bool ray_query = false;
    bool ray_tracing = false;
};

struct DeviceConfig {
    bool enable_validation = false;
    std::string_view application_name = "gpu";
    // Supplying a provider makes this a presenting device: it enables the
    // swapchain extension, selects a present-capable queue, and lets
    // swapchain() and begin_frame() be used. Leaving it null gives a headless
    // device, which is what offline rendering and the tests want.
    SurfaceProvider* surface = nullptr;
};

class Device {
public:
    explicit Device(const DeviceConfig& config = {});
    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    DeviceFeatures features() const;

    template<class T> Buffer<T> buffer(std::size_t count);
    template<class T> Image<T> image(std::uint32_t width, std::uint32_t height, ImageUsage usage);
    template<class T> Image<T> image(std::uint32_t width, std::uint32_t height,
        ImageUsage usage, ImageFormat format);

    Shader create_shader(std::span<const std::byte> spirv);
    Shader create_shader(std::span<const std::byte> spirv, std::string_view entry_point);
    ComputePipeline compute(const Shader& shader);
    GraphicsPipeline graphics(const GraphicsPipelineDesc& desc);
    RayTracingPipeline ray_tracing(const RayTracingPipelineDesc& desc);
    AccelerationStructure build_blas(std::span<const TriangleGeometry> geometry);
    AccelerationStructure build_tlas(std::span<const Instance> instances);
    void update_tlas(AccelerationStructure& tlas, std::span<const Instance> instances);
    // Builds from records already resident in device memory. The caller owns
    // the buffer and must keep every referenced BLAS alive through the span.
    AccelerationStructure build_tlas(GpuPtr<InstanceRecord> records, std::uint32_t count,
        std::span<const AccelerationStructure> referenced);
    void update_tlas(AccelerationStructure& tlas, GpuPtr<InstanceRecord> records,
        std::uint32_t count);
    Sampler sampler(const SamplerDesc& desc);

    template<class T> void upload(Buffer<T>& destination, std::span<const T> data);
    template<class T> void upload(Buffer<T>& destination, std::span<const T> data,
        std::size_t destination_offset);
    template<class T> void download(std::span<T> destination, const Buffer<T>& source);
    template<class T> void upload(Image<T>& destination, std::span<const T> data);
    template<class T> void download(std::span<T> destination, const Image<T>& source);

    // Blit one image onto another, scaling if the extents differ. Both images
    // may belong to this device or be swapchain images; layout handling is
    // internal, as everywhere else in this API.
    void copy(ImageHandle source, ImageHandle destination);

    void barrier(Stage source, Stage destination);
    GpuToken signal();
    void wait(GpuToken token);
    void synchronize();

    // Draw into `target`. Draws issued by the callback are recorded into the
    // enclosing frame when one is open, and into a private submission when
    // none is.
    void render(const RenderTarget& target, const std::function<void()>& draw_commands);

    TimestampQuery timestamp();
    // Bracket `commands` with GPU timestamps written into `query`.
    void measure(const TimestampQuery& query, const std::function<void()>& commands);

    // --- Presentation. Valid only on a device built with a SurfaceProvider.

    // Create the presentation chain. The overload taking a format honours it
    // when the surface supports it and quietly replaces it with a supported
    // one otherwise; read the result back with Swapchain::format().
    Swapchain swapchain();
    Swapchain swapchain(ImageFormat preferred);

    // Acquire the next presentation image and open a recording scope. Returns
    // a falsy Frame when the chain was rebuilt and no image was acquired - the
    // caller should skip the frame and try again.
    Frame begin_frame(Swapchain& swapchain);

    // Close the frame, submit everything recorded into it, and present.
    void end_frame(Frame&& frame);

private:
    friend interop::DeviceHandles interop::device_handles(Device&);
    friend std::uintptr_t interop::image_view(Device&, ImageHandle);
    friend interop::ExternalImageMemory interop::export_image_memory(Device&, ImageHandle);
    friend interop::ExternalSemaphore interop::signal_external(Device&);
    std::shared_ptr<detail::DeviceImpl> impl_;
};

} // namespace gpu
