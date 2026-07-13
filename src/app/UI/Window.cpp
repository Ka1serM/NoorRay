#include "Window.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

Window::Window()
{
#ifdef __linux__
    SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "1");
    SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, "1");
    if (std::getenv("WAYLAND_DISPLAY") && !std::getenv("SDL_VIDEO_DRIVER"))
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
#endif

    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(std::string("Failed to initialize SDL: ") + SDL_GetError());
    if (!SDL_Vulkan_LoadLibrary(nullptr))
    {
        SDL_Quit();
        throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());
    }

    const float displayScale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    if (displayScale > 0.0f)
        dpiScale = displayScale;
    width = static_cast<uint32_t>(static_cast<float>(width) * dpiScale);
    height = static_cast<uint32_t>(static_cast<float>(height) * dpiScale);
    window = SDL_CreateWindow("NoorRay by Marcel K.", static_cast<int>(width),
        static_cast<int>(height), SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Vulkan_UnloadLibrary();
        SDL_Quit();
        throw std::runtime_error(std::string("Failed to create SDL window: ") + SDL_GetError());
    }
}

Window::~Window()
{
    if (window)
        SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

PFN_vkGetInstanceProcAddr Window::getVulkanInstanceProcAddr() const
{
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
}

std::vector<const char*> Window::getRequiredVulkanInstanceExtensions() const
{
    unsigned int count{};
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!extensions)
        throw std::runtime_error(std::string("Failed to query SDL Vulkan extensions: ") + SDL_GetError());
    return {extensions, extensions + count};
}

vk::SurfaceKHR Window::createVulkanSurface(const vk::Instance instance) const
{
    VkSurfaceKHR surface{};
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface))
        throw std::runtime_error(std::string("Failed to create SDL Vulkan surface: ") + SDL_GetError());
    return vk::SurfaceKHR(surface);
}

bool Window::pollEvent(SDL_Event& event) const
{
    return SDL_PollEvent(&event);
}

void Window::setFullscreen(const bool fullscreen) const
{
    SDL_SetWindowFullscreen(window, fullscreen);
}

void Window::setRelativeMouseMode(const bool enabled) const
{
    SDL_SetWindowRelativeMouseMode(window, enabled);
}

std::pair<float, float> Window::getRelativeMouseDelta() const
{
    float x{};
    float y{};
    SDL_GetRelativeMouseState(&x, &y);
    return {x, y};
}

void Window::warpMouse(const float x, const float y) const
{
    SDL_WarpMouseInWindow(window, x, y);
}
