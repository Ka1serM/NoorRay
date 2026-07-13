#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "Vulkan/Context.h"

class Window final : public VulkanSurfaceProvider
{
public:
    Window();
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
    void setRelativeMouseMode(bool enabled) const;
    std::pair<float, float> getRelativeMouseDelta() const;
    void warpMouse(float x, float y) const;

private:
    SDL_Window* window{};
    uint32_t width{1920};
    uint32_t height{810};
    float dpiScale{1.0f};
};
