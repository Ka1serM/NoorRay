#pragma once

#include <functional>
#include <optional>

#include "NoorRay.App/Plugins/ViewportSource.h"
#include "NoorRay.Reconstruction/Alignment.h"
#include "NoorRay.Reconstruction/Trainer.h"

namespace NoorRay::Reconstruction {

//! Shows the method's own render in the viewport, from the viewport camera.
class Source final : public App::ViewportSource {
public:
    struct Events {
        std::function<App::ViewportBanner()> banner;
        std::function<void()> action;
        std::function<void()> leave;
    };

    Source(NoorRHI::Device& device, Trainer& trainer, Events events) : device(device), trainer(trainer), events(std::move(events)) {}

    [[nodiscard]] std::expected<bool, Core::Error> record(NoorRHI::CommandList& list, const Scene::Camera& camera, std::uint32_t width,
                                                          std::uint32_t height) override;
    const NoorRHI::Image& getDisplay() const override { return display; }
    std::optional<Core::Bounds> getBounds() const override { return bounds; }
    App::ViewportBanner getBanner() const override { return events.banner(); }
    void runAction() override { events.action(); }
    void leave() override { events.leave(); }

    void open(const Alignment& alignment, const Core::Bounds& cameraBounds);
    void close();

private:
    struct Request {
        Core::Transform transform;
        float verticalFov = 0.0f;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        bool operator==(const Request&) const = default;
    };

    [[nodiscard]] std::expected<void, Core::Error> upload(const TrainerFrame& frame);

    NoorRHI::Device& device;
    Trainer& trainer;
    Events events;
    Alignment alignment;
    std::optional<Core::Bounds> bounds;
    std::optional<Request> requested;
    bool waiting = false;
    NoorRHI::Image display;
};

}
