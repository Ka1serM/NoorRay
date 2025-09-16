#pragma once

#include "UI/RmlRenderInterface.h"
#include "UI/RmlUi_Platform_SDL.h"
#include "Vulkan/Context.h"
#include "Vulkan/Renderer.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>

#include "Vulkan/Image.h"

class RmlUiManager {
public:
    RmlUiManager(Context& context, const Renderer& renderer);
    void bindViewportImage(const Image& image);
    void updateDisplayImage(vk::CommandBuffer cmd, Image& srcImage);
    ~RmlUiManager();

    void render(vk::CommandBuffer command_buffer, vk::ImageView target_image_view, vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fenc);
    void processEvent(SDL_Window* window, SDL_Event& event) const;
    void resize(int width, int height) const;

private:
    SystemInterface_SDL rmlSystemInterface;
    RmlRenderInterface rmlRenderInterface;

    Rml::Context* rmlContext = nullptr;
    Rml::ElementDocument* document = nullptr;

    Image customImage;
    Rml::TextureHandle customTextureId = -1;
};