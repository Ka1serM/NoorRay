#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "Backend/Vulkan/Runtime/Context.h"

class Window final : public VulkanSurfaceProvider
{
public:
    Window(uint32_t requestedWidth = 0, uint32_t requestedHeight = 0);
    ~Window() override;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    PFN_vkGetInstanceProcAddr getVulkanInstanceProcAddr() const override;
    std::vector<const char*> getRequiredVulkanInstanceExtensions() const override;
    vk::SurfaceKHR createVulkanSurface(vk::Instance instance) const override;
    uint32_t getWidth() const override { return width; }
    uint32_t getHeight() const override { return height; }

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
    uint32_t width{1920};
    uint32_t height{810};
    float dpiScale{1.0f};
};
