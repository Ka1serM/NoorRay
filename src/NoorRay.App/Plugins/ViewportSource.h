#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Camera.h"

namespace NoorRay::App {

//! The viewport shows it above the image, so the user sees that the image is not the scene.
struct ViewportBanner {
    std::string title;
    std::string detail;
    std::string action;
    bool actionEnabled = false;

    bool operator==(const ViewportBanner&) const = default;
};

//! An image that the viewport shows in place of the path tracer, for the camera of the viewport.
class ViewportSource {
public:
    virtual ~ViewportSource() = default;

    //! Records the frame of the camera at this size into the display; true asks for another update.
    [[nodiscard]] virtual std::expected<bool, Core::Error> record(NoorRHI::CommandList& list, const Scene::Camera& camera, std::uint32_t width,
                                                                  std::uint32_t height) = 0;
    //! An sRGB image with premultiplied alpha; it may be smaller than the viewport and have another aspect while a new frame is on its way.
    virtual const NoorRHI::Image& getDisplay() const = 0;
    //! What the viewport frames, together with the scene.
    virtual std::optional<Core::Bounds> getBounds() const = 0;
    virtual ViewportBanner getBanner() const = 0;
    virtual void runAction() = 0;
    //! The user goes back to the scene; the source gives the viewport back.
    virtual void leave() = 0;
};

}
