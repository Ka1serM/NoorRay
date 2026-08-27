#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include <gpu/surface.hpp>

class Window final : public gpu::SurfaceProvider
{
public:
    Window(uint32_t requestedWidth = 0, uint32_t requestedHeight = 0);
    ~Window() override;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    std::uintptr_t instance_proc_address() const override;
    std::vector<const char*> instance_extensions() const override;
    std::uintptr_t create_surface(std::uintptr_t instance) const override;
    uint32_t width() const override { return pixelWidth; }
    uint32_t height() const override { return pixelHeight; }

    SDL_Window* nativeHandle() const { return window; }
    float getDpiScale() const { return dpiScale; }
    bool pollEvent(SDL_Event& event) const;
    void setFullscreen(bool fullscreen) const;
    bool setRelativeMouseMode(bool enabled) const;
    bool isRelativeMouseMode() const;
    bool hasInputFocus() const;
    void warpMouse(float x, float y) const;

private:
    SDL_Window* window{};
    uint32_t pixelWidth{1920};
    uint32_t pixelHeight{810};
    float dpiScale{1.0f};
};
