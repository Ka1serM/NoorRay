#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include "Backend/Vulkan/Runtime/Buffer.h"
#include "Backend/Vulkan/Runtime/Image.h"
#include "Rendering/Camera/Sensor.h"
#include "Scene/Handle.h"

class Context;
class Raytracer;
class Scene;
class Viewport;

namespace nr {

// Relates the render target to the region the host displays it in.
enum class ViewportSizing
{
    // The render resolution follows the display region, so the rendered aspect
    // ratio always matches what is on screen. This is what an interactive
    // viewport wants.
    MatchTarget,
    // The render resolution is whatever the camera sensor asks for and the
    // image is letterboxed into the display region. This is what framing a
    // final render wants, because the composition must not change with the
    // size of the window.
    FixedResolution,
};

// The CUDA/Vulkan handoff for a frame that record() consumed. Submit the host's
// command buffer waiting on `waitSemaphore` at `value` and signalling
// `signalSemaphore` at `value`, both as timeline semaphores.
struct ViewportFrameSync
{
    vk::Semaphore waitSemaphore{};
    vk::Semaphore signalSemaphore{};
    uint64_t value{};

    explicit operator bool() const
    {
        return waitSemaphore && signalSemaphore && value != 0;
    }
};

// What to do when the render target's aspect ratio does not match the display
// region, which under MatchTarget only happens for the few frames while a
// resize is settling.
enum class ViewportFill
{
    // Fill the whole region and crop the overflow. No bars and no distortion;
    // the transient mismatch costs a sliver of the image instead.
    Cover,
    // Letterbox inside the region, preserving every rendered pixel.
    Fit,
};

// Where the rendered image belongs inside the display region, in pixels
// relative to its top-left corner, plus the sub-rectangle of the display image
// to sample. The image is never stretched: under Cover the UV range narrows,
// under Fit the destination rectangle shrinks.
struct ViewportLayout
{
    float offsetX{};
    float offsetY{};
    float width{};
    float height{};
    float uvMinX{};
    float uvMinY{};
    float uvMaxX{1.0f};
    float uvMaxY{1.0f};
};

struct ViewportPick
{
    bool hit{};
    SceneObjectHandle object{};
    glm::vec3 position{};
};

// The reusable half of an interactive raytraced viewport: the progressive
// render loop, the CUDA/Vulkan frame handoff, resolution and sensor-fit
// management, the presentation image, and pixel picking.
//
// Deliberately free of any windowing or UI dependency, so the NoorRay
// application, Hydra, Python, and embedding applications can all share it and
// supply their own input handling and image display. A host needs only:
//
//   session.setTargetSize(regionWidth, regionHeight);  // when the layout changes
//   session.update();                                  // once per frame
//   const ViewportFrameSync sync = session.record(cmd);
//   // draw session.getDisplayImage() into session.layout()
//
// getDisplayImage() stays valid across resizes, so a descriptor written once
// keeps working; only its contents and extent change.
class ViewportSession
{
public:
    struct CreateInfo
    {
        uint32_t width{1280};
        uint32_t height{720};
        // Format of the image the host samples. Usually the swapchain format.
        vk::Format displayFormat{vk::Format::eB8G8R8A8Unorm};
        ViewportSizing sizing{ViewportSizing::MatchTarget};
        // Only consulted while a resize is settling, when the rendered aspect
        // ratio briefly differs from the region's.
        ViewportFill fill{ViewportFill::Cover};
        // How the physical sensor is fitted when the render target has a
        // different aspect ratio than the sensor.
        SensorFit sensorFit{SensorFit::Horizontal};
        // Samples per progressive step and the budget at which accumulation
        // stops. Both are also readable from the scene's render settings; these
        // only seed them.
        bool applyRenderSettings{true};
        int samplesPerStep{4};
        int maxSamples{512};
        // Cryptomatte, position and normal AOVs. Required for picking and for
        // the compositor's selection outline.
        bool aovEnabled{true};
        // How long a requested size must hold still before it is applied.
        // Reallocating the render targets stalls the device, so applying every
        // intermediate size of a splitter or window drag would stutter badly.

    };

    ViewportSession(Context& context, Scene& scene, const CreateInfo& createInfo);
    ~ViewportSession();

    ViewportSession(const ViewportSession&) = delete;
    ViewportSession& operator=(const ViewportSession&) = delete;

    // ── Layout ──────────────────────────────────────────────────────────────
    // Pixel size of the region the host will draw the image into. Safe to call
    // every frame; a resize is only applied once the value settles.
    void setTargetSize(uint32_t width, uint32_t height);
    // Renders at a fraction of the target size and lets the host upscale.
    void setResolutionScale(float scale);
    void setSizing(ViewportSizing sizing);
    void setFill(ViewportFill fill);
    void setSensorFit(SensorFit fit);
    ViewportLayout layout() const;
    uint32_t renderWidth() const;
    uint32_t renderHeight() const;

    // ── Frame loop ──────────────────────────────────────────────────────────
    // Applies pending resizes and submits the next progressive step. Call once
    // per host frame, before record().
    void update();
    // Records the compositing pass and the copy into the display image.
    // Returns the frame handoff when a completed render was consumed, and a
    // falsy sync when this was only an overlay refresh or nothing was due.
    ViewportFrameSync record(vk::CommandBuffer commandBuffer);

    // The image the host samples. The Image object is stable, but a resize
    // reallocates it, so its view handle changes. Hosts that cache a descriptor
    // should rewrite it whenever displayImageRevision() changes.
    Image& getDisplayImage();
    uint64_t displayImageRevision() const;
    // False until the first completed render has been presented.
    bool hasPresentedFrame() const;
    // Progressive accumulation progress, for a host that wants to show it.
    uint32_t accumulatedSamples() const;
    bool isAccumulationComplete() const;
    void requestAccumulationReset();

    // ── Picking ─────────────────────────────────────────────────────────────
    // All coordinates are pixels relative to the top-left of the display
    // region, i.e. the same space layout() reports in. Points outside the
    // image return a miss rather than clamping to the edge.
    //
    // Selects the object under the point, preferring light billboards when
    // overlays are drawn, and makes it the scene's active object.
    ViewportPick pickObject(float targetX, float targetY);
    // World position under the point, for seeding an arcball pivot. Does not
    // change the selection.
    std::optional<glm::vec3> pickPosition(float targetX, float targetY);

    // ── Presentation ────────────────────────────────────────────────────────
    void setOverlaysEnabled(bool enabled);
    bool overlaysEnabled() const;
    // Index of the Gaussian to outline, or ~0u for none.
    void setSelectedGaussianIndex(uint32_t index);
    uint32_t selectedGaussianIndex() const;

    Raytracer& getRaytracer();
    const Raytracer& getRaytracer() const;
    Scene& getScene();

private:
    void applyPendingResize();
    void resizeRenderTarget(uint32_t width, uint32_t height);
    void applySensorFit();
    bool readbackAovTexel(Image* image, const Buffer& staging, glm::ivec2 pixel);
    // Converts display-region pixels to render-target texels. Returns nullopt
    // when the point is outside the image.
    std::optional<glm::ivec2> targetToPixel(float targetX, float targetY) const;
    bool pickBillboard(glm::ivec2 pixel, ViewportPick& result) const;

    Context& context;
    Scene& scene;
    std::unique_ptr<Raytracer> raytracer;
    std::unique_ptr<Viewport> compositor;

    // The host samples this rather than the compositor's output, so a resize
    // can reallocate the compositor without invalidating draw commands the host
    // already recorded this frame.
    // Always exactly the render extent. A resize reallocates it, and the
    // outgoing image is retired rather than dropped so its contents can be
    // rescaled into the new one - that carry-over is what keeps a resize from
    // flashing black while the first render at the new size is still in flight.
    Image displayImage;
    Image retiredDisplayImage;
    bool carryOverPending{};
    bool displayImageCleared{};
    uint64_t displayImageGeneration{1};
    void ensureDisplayImage(uint32_t width, uint32_t height);

    Buffer cryptoStagingBuffer;
    void* cryptoStagingMapped{};
    Buffer positionStagingBuffer;
    void* positionStagingMapped{};

    vk::Format displayFormat{};
    ViewportSizing sizing{};
    ViewportFill fill{};
    SensorFit sensorFit{};
    float resolutionScale{1.0f};

    uint32_t targetWidth{};
    uint32_t targetHeight{};
    uint32_t requestedWidth{};
    uint32_t requestedHeight{};

    uint32_t frameIndex{};
    uint32_t submittedSamples{};
    bool renderComplete{};
    bool firstFrame{true};
    // The raytracer keeps reporting the last completed frame as ready, so a
    // frame must only ever be adopted once. Without this the host presents a
    // single frame forever while overlays keep animating.
    uint64_t displayedRenderValue{};
    uint32_t displayedCryptomatteId{~0u};
    uint64_t pendingReadyValue{};
    vk::Semaphore pendingRenderReady{};
    vk::Semaphore pendingBufferReleased{};

    bool showOverlays{true};
    uint32_t gaussianSelection{~0u};
};

}  // namespace nr
