#include "NoorRay.Reconstruction/Source.h"

namespace NoorRay::Reconstruction {

void Source::open(const Alignment& chosen, const Core::Bounds& cameraBounds) {
    alignment = chosen;
    bounds = cameraBounds;
    requested.reset();
    waiting = false;
}

void Source::close() {
    bounds.reset();
    requested.reset();
    waiting = false;
    display = {};
}

std::expected<bool, Core::Error> Source::record(NoorRHI::CommandList&, const Scene::Camera& camera, const std::uint32_t width, const std::uint32_t height) {
    const Request request{camera.transform, camera.verticalFov, width, height};
    if (request != requested) {
        requested = request;
        trainer.requestFrame(alignment.toMethod(camera.transform, camera.verticalFov, width, height));
        waiting = true;
    }
    if (std::optional<TrainerFrame> frame = trainer.takeFrame()) {
        waiting = false;
        if (std::expected<void, Core::Error> uploaded = upload(*frame); !uploaded)
            return std::unexpected(std::move(uploaded.error()));
    }
    if (!display) {
        const TrainerFrame empty{1, 1, {0xff000000u}};
        if (std::expected<void, Core::Error> uploaded = upload(empty); !uploaded)
            return std::unexpected(std::move(uploaded.error()));
    }
    return waiting;
}

std::expected<void, Core::Error> Source::upload(const TrainerFrame& frame) {
    NoorRHI::Image opened;
    if (auto created = opened.open(device, {.width = frame.width,
                                            .height = frame.height,
                                            .format = NoorRHI::ImageFormat::Rgba8Unorm,
                                            .usage = NoorRHI::ImageUsage::Sampled,
                                            .name = "reconstruction display"});
        !created)
        return Core::fail(std::move(created.error()));
    if (auto uploaded = opened.upload(std::as_bytes(std::span<const std::uint32_t>(frame.pixels))); !uploaded)
        return Core::fail(std::move(uploaded.error()));
    display = std::move(opened);
    return {};
}

}
