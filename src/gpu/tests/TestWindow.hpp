#pragma once

// A minimal SDL3 SurfaceProvider for the presentation test. The gpu library
// never links a windowing toolkit, so its tests carry their own instead of
// borrowing an application's window class.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <gpu/surface.hpp>

class TestWindow final : public gpu::SurfaceProvider {
public:
    TestWindow(std::uint32_t width, std::uint32_t height) {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());
        if (!SDL_Vulkan_LoadLibrary(nullptr)) {
            SDL_Quit();
            throw std::runtime_error(std::string("SDL_Vulkan_LoadLibrary: ") + SDL_GetError());
        }
        window_ = SDL_CreateWindow("gpu tests", static_cast<int>(width),
            static_cast<int>(height), SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!window_) {
            SDL_Vulkan_UnloadLibrary();
            SDL_Quit();
            throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());
        }
    }

    ~TestWindow() override {
        if (window_) SDL_DestroyWindow(window_);
        SDL_Vulkan_UnloadLibrary();
        SDL_Quit();
    }

    TestWindow(const TestWindow&) = delete;
    TestWindow& operator=(const TestWindow&) = delete;

    std::uintptr_t instance_proc_address() const override {
        return reinterpret_cast<std::uintptr_t>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    }

    std::vector<const char*> instance_extensions() const override {
        unsigned int count{};
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
        return std::vector<const char*>(extensions, extensions + count);
    }

    std::uintptr_t create_surface(std::uintptr_t instance) const override {
        VkSurfaceKHR surface{};
        if (!SDL_Vulkan_CreateSurface(window_, reinterpret_cast<VkInstance>(instance),
                nullptr, &surface))
            throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError());
        return reinterpret_cast<std::uintptr_t>(surface);
    }

    std::uint32_t width() const override { return size().first; }
    std::uint32_t height() const override { return size().second; }

    SDL_Window* nativeHandle() const { return window_; }

    bool pollEvent(SDL_Event& event) const { return SDL_PollEvent(&event); }

private:
    std::pair<std::uint32_t, std::uint32_t> size() const {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(window_, &w, &h);
        return {static_cast<std::uint32_t>(std::max(w, 0)),
                static_cast<std::uint32_t>(std::max(h, 0))};
    }

    SDL_Window* window_{};
};
