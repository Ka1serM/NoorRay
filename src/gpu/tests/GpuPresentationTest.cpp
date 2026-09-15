#include "TestWindow.hpp"

#include <gpu/gpu.hpp>
#include <SDL3/SDL.h>
#include <cstdlib>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("gpu presentation survives resize and abandoned frames", "[gpu][window]") {
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        SKIP("presentation test requires a desktop display");
    TestWindow window(320, 240);
    gpu::Device device({.enable_validation = true, .surface = &window});
    auto swapchain = device.swapchain();
    gpu::Shared<unsigned> record(device);
    record.commit();
    device.synchronize();
    const auto allocation_bytes = device.memory_report().allocation_bytes;

    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        const int width = iteration % 2 ? 320 : 480;
        const int height = iteration % 2 ? 240 : 360;
        REQUIRE(SDL_SetWindowSize(window.nativeHandle(), width, height));
        REQUIRE(SDL_SyncWindow(window.nativeHandle()));
        SDL_Event event{};
        while (window.pollEvent(event)) {}

        auto frame = device.begin_frame(swapchain);
        if (!frame) frame = device.begin_frame(swapchain);
        REQUIRE(frame);
        CHECK(frame.width() == window.width());
        CHECK(frame.height() == window.height());
        record.data = iteration + 1;
        // Uploads are rejected while a frame's command buffer is open.
        CHECK_THROWS_AS(record.commit(), gpu::Error);

        if (iteration % 3 == 0) {
            frame = {}; // Discard the acquire and command buffer through RAII.
            record.commit();
            frame = device.begin_frame(swapchain);
            REQUIRE(frame);
        }
        device.render({frame.target(), {}}, [] {});
        device.end_frame(std::move(frame));
        record.commit();
    }
    device.synchronize();
    CHECK(device.memory_report().allocation_bytes == allocation_bytes);
}
