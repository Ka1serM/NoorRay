#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Readback.h"
#include "NoorRay.Python/Blender/SceneSync.h"
#include "NoorRay.Python/GlInterop.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/RenderOptions.h"
#include "NoorRay/Session/Session.h"

namespace NoorRay::Python {

//! The Blender add-on passes every setting, so the defaults matter only to the app.
using EngineSettings = Render::RenderOptions;

//! A camera as a view and projection matrix give it: the shift is the image centre in normalized device coordinates.
struct CameraView {
    Core::Transform transform;
    bool orthographic = false;
    float verticalFov = 0.6981317f;
    float orthographicHeight = 10.0f;
    Core::float2 shift;
    float near = 0.01f;
    float far = 10000.0f;
    //! A view through the camera of the scene has its depth of field.
    bool throughCamera = false;
    //! The part of the image of the scene camera that the film covers, for a panoramic camera.
    std::optional<Core::float4> window;
};

//! A geometry pass of the first camera sample of each pixel.
enum class FilmPass : std::uint8_t { Depth, Normal, DiffuseColor };

struct FrameRange {
    int first = 0;
    int last = 0;
};

//! Writes a depsgraph as a .nrs file and returns the warnings: its current frame, or the animation of the frames. The caller holds the GIL.
[[nodiscard]] std::expected<std::vector<std::string>, Core::Error> exportScene(const Object& depsgraph, const std::filesystem::path& path, bool approximate,
                                                                               std::optional<FrameRange> frames);

//! The device and pipelines of an engine. A Vulkan device takes seconds to make, so the destructor keeps one idle device for the next engine.
class KeptDevice {
public:
    KeptDevice() = default;
    KeptDevice(const KeptDevice&) = delete;
    KeptDevice& operator=(const KeptDevice&) = delete;
    ~KeptDevice();
    //! Takes the idle device when there is one, else makes a device.
    [[nodiscard]] std::expected<void, Core::Error> open();

    NoorRHI::Device device;
    std::shared_ptr<Render::PipelineCache> pipelines;
};

//! Makes the idle device on a thread, so that the first engine does not wait for it; an engine that opens meanwhile waits for the thread.
void prepareIdleDevice();
//! Frees the idle device; the module calls it when Python ends.
void releaseIdleDevice();

//! A device and a session for one Python object: the Blender render engine creates one per viewport and one per final render.
class Engine {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(std::uint32_t width, std::uint32_t height);
    //! Brings the scene to the Blender depsgraph; it reads the depsgraph through bpy, so the caller holds the GIL.
    [[nodiscard]] std::expected<void, Core::Error> sync(const Object& depsgraph, bool approximate, bool accumulates);
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void configure(const EngineSettings& settings);
    //! A call with a new view restarts the accumulation; later syncs keep the view.
    [[nodiscard]] std::expected<void, Core::Error> setCamera(const CameraView& view);
    //! Sets the part of the image that the film covers: left, top, right and bottom from 0 to 1.
    [[nodiscard]] std::expected<void, Core::Error> crop(Core::float4 window);
    [[nodiscard]] std::expected<void, Core::Error> render();
    //! Denoises a final render that stopped before its samples; a complete one is denoised already.
    [[nodiscard]] std::expected<void, Core::Error> denoise();
    //! Writes the linear RGBA film, top row first.
    [[nodiscard]] std::expected<void, Core::Error> read(std::span<float> pixels);
    //! Writes a geometry pass in Blender's space, top row first: one float per pixel for depth, else three.
    [[nodiscard]] std::expected<void, Core::Error> readPass(FilmPass pass, std::span<float> values);
    //! Shares two film images with the current OpenGL context, which render and presentGl then need; the target is the last drawn RGBA16F texture.
    [[nodiscard]] std::expected<void, Core::Error> openGl();
    //! Copies the last rendered frame into the OpenGL target on the GPU, top row first; without a new frame the target keeps the last one.
    [[nodiscard]] std::expected<void, Core::Error> presentGl();

    //! A change restarts the film in the next render; a final render completes when its accumulation does, the viewport after samples frames.
    bool isComplete() const { return !changed && (session.getRenderer().isComplete() || (!accumulates && samples != 0 && frames >= samples)); }
    std::uint32_t getSampleCount() const { return accumulates ? session.getRenderer().getSampleCount() : frames; }
    std::string getCompileNotice() const { return session.getRenderer().getCompileNotice(); }
    std::uint32_t getWidth() const { return session.getRenderer().getWidth(); }
    std::uint32_t getHeight() const { return session.getRenderer().getHeight(); }
    //! The warnings of the last sync.
    std::span<const std::string> getWarnings() const { return warnings; }

private:
    void setProjection(Scene::Camera& camera, const CameraView& view) const;
    [[nodiscard]] std::expected<void, Core::Error> submit(NoorRHI::CommandList&& list);

    //! The first member, so that it ends after the session.
    KeptDevice kept;
    Session::Session session;
    NoorRHI::Readback<NoorRHI::float4> film;
    //! The display-size copy of the FSR or DLSS-RR image, which read downloads.
    NoorRHI::Buffer<NoorRHI::float4> upscaled;
    //! The render writes one image while OpenGL copies the other.
    std::array<NoorRHI::Image, GlInterop::slotCount> shared;
    GlInterop gl;
    std::size_t sharedSlot = 0;
    //! True when the image of sharedSlot holds a frame that presentGl has not copied yet.
    bool sharedPending = false;
    //! The frames that render submitted without a wait, oldest first.
    std::deque<NoorRHI::Token> inFlight;
    SceneSync blender;
    std::vector<std::string> warnings;
    std::optional<CameraView> view;
    std::optional<Core::float4> window;
    bool changed = true;
    bool accumulates = true;
    std::uint32_t samples = 0;
    //! The frames since the last change, for DLSS-RR.
    std::uint32_t frames = 0;
};

}
