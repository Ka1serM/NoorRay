#include "Window.h"

#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

Window::Window(const uint32_t requestedWidth, const uint32_t requestedHeight)
{
#ifdef __linux__
    SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "1");
    SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, "1");
    if (std::getenv("WAYLAND_DISPLAY")) {
        if (!std::getenv("SDL_VIDEO_DRIVER"))
            SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland,x11");
    }
#endif

    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(std::string("Failed to initialize SDL: ") + SDL_GetError());
    if (!SDL_Vulkan_LoadLibrary(nullptr))
    {
        SDL_Quit();
        throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());
    }

    const SDL_DisplayID primaryDisplay = SDL_GetPrimaryDisplay();

    if (requestedWidth > 0 && requestedHeight > 0) {
        pixelWidth = requestedWidth;
        pixelHeight = requestedHeight;
    } else {
        // Choose a logical startup size from the usable monitor area, like
        // other desktop applications. The actual drawable pixel size is
        // queried below.
        SDL_Rect usableBounds{};
        if (SDL_GetDisplayUsableBounds(primaryDisplay, &usableBounds)
            && usableBounds.w > 0 && usableBounds.h > 0) {
            constexpr float startupAreaFraction = 2.0f / 3.0f;
            pixelWidth = static_cast<uint32_t>(usableBounds.w * startupAreaFraction);
            pixelHeight = static_cast<uint32_t>(usableBounds.h * startupAreaFraction);
        }
    }

    window = SDL_CreateWindow("NoorRay by Marcel K.", static_cast<int>(pixelWidth),
        static_cast<int>(pixelHeight),
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
    {
        SDL_Vulkan_UnloadLibrary();
        SDL_Quit();
        throw std::runtime_error(std::string("Failed to create SDL window: ") + SDL_GetError());
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED_DISPLAY(primaryDisplay),
        SDL_WINDOWPOS_CENTERED_DISPLAY(primaryDisplay));

    const float windowScale = SDL_GetWindowDisplayScale(window);
    if (windowScale > 0.0f)
        dpiScale = windowScale;

    // Vulkan needs the drawable pixel extent, not the logical SDL window size.
    // SDL applies the display scale when reporting pixels on high-DPI displays.
    int pixelWidth = 0;
    int pixelHeight = 0;
    if (!SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight))
        throw std::runtime_error(std::string("Failed to query SDL window pixel size: ") + SDL_GetError());
    this->pixelWidth = static_cast<uint32_t>(std::max(pixelWidth, 0));
    this->pixelHeight = static_cast<uint32_t>(std::max(pixelHeight, 0));
}

Window::~Window()
{
    if (window)
        SDL_DestroyWindow(window);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

std::uintptr_t Window::instance_proc_address() const
{
    return reinterpret_cast<std::uintptr_t>(SDL_Vulkan_GetVkGetInstanceProcAddr());
}

std::vector<const char*> Window::instance_extensions() const
{
    unsigned int count{};
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!extensions)
        throw std::runtime_error(std::string("Failed to query SDL Vulkan extensions: ") + SDL_GetError());
    return {extensions, extensions + count};
}

std::uintptr_t Window::create_surface(const std::uintptr_t instance) const
{
    VkSurfaceKHR surface{};
    if (!SDL_Vulkan_CreateSurface(window, reinterpret_cast<VkInstance>(instance), nullptr, &surface))
        throw std::runtime_error(std::string("Failed to create SDL Vulkan surface: ") + SDL_GetError());
    return reinterpret_cast<std::uintptr_t>(surface);
}

bool Window::pollEvent(SDL_Event& event) const
{
    return SDL_PollEvent(&event);
}

void Window::setFullscreen(const bool fullscreen) const
{
    SDL_SetWindowFullscreen(window, fullscreen);
}

bool Window::setRelativeMouseMode(const bool enabled) const
{
    return SDL_SetWindowRelativeMouseMode(window, enabled);
}

bool Window::isRelativeMouseMode() const
{
    return SDL_GetWindowRelativeMouseMode(window);
}

bool Window::hasInputFocus() const
{
    return (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) != 0;
}

void Window::warpMouse(const float x, const float y) const
{
    SDL_WarpMouseInWindow(window, x, y);
}
