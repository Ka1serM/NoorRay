#include "RmlUiManager.h"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>

#include "Vulkan/Image.h"

RmlUiManager::RmlUiManager(Context& context, const Renderer& renderer)
    : rmlRenderInterface(
          context.getDevice(),
          context.getGraphicsQueue(),
          context.getAllocator(),
          context.getCommandPool(),
          context.getDescriptorPool(),
          renderer.getColorImageFormat(),
          renderer.getDepthImageFormat()
          ),
    customImage(context, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
{
    // Setup system interface
    rmlSystemInterface.SetWindow(context.getWindow());

    // Initialize RmlUi
    Rml::SetSystemInterface(&rmlSystemInterface);
    Rml::SetRenderInterface(&rmlRenderInterface);
    Rml::Initialise();

    // Create context
    rmlContext = Rml::CreateContext(
        "main",
        {
            static_cast<int>(context.getWindowWidth()),
            static_cast<int>(context.getWindowHeight())
        }
    );

    if (!rmlContext)
        throw std::runtime_error("Failed to create RmlUi context!");

    // Load font(s)
    Rml::LoadFontFace("../assets/fonts/Inter-Regular.ttf");

    // Load initial document
    document = rmlContext->LoadDocument("../assets/rml/hello_world.rml");
    if (document)
        document->Show();
}
void RmlUiManager::bindViewportImage(const Image& image) {
    const Rml::String texture_name = "viewport_image";
    rmlRenderInterface.registerVulkanTexture(texture_name,  image.getImageView(),  Rml::Vector2i{static_cast<int>(image.getWidth()), static_cast<int>(image.getHeight())});

    const Rml::String uri = "vulkan://" + texture_name;

    Rml::Element* viewportElem = document->GetElementById("viewport");
    if (viewportElem)
        viewportElem->SetAttribute("src", uri);
}

void RmlUiManager::updateDisplayImage(const vk::CommandBuffer cmd, Image& srcImage) {
    srcImage.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
    customImage.setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    vk::ImageCopy copyRegion{};
    copyRegion.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.extent = vk::Extent3D{srcImage.getWidth(), srcImage.getHeight(), 1};

    cmd.copyImage(srcImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, customImage.getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegion);

    srcImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    customImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}


RmlUiManager::~RmlUiManager()
{
    if (document) {
        document->Close();
        document = nullptr;
    }

    if (rmlContext) {
        Rml::RemoveContext("main");
        rmlContext = nullptr;
    }

    Rml::Shutdown();
}

void RmlUiManager::render(const vk::CommandBuffer command_buffer, const vk::ImageView target_image_view, const vk::ImageView depthImageView, const vk::Extent2D target_extent, const vk::Fence in_flight_fenc)
{
    rmlRenderInterface.beginFrame(
        command_buffer,
        target_image_view,
        depthImageView,
        target_extent,
        in_flight_fenc
    );

    rmlContext->Update();
    rmlContext->Render();

    rmlRenderInterface.endFrame();
}

void RmlUiManager::processEvent(SDL_Window* window, SDL_Event& event) const
{
    RmlSDL::InputEventHandler(rmlContext, window, event);
}

void RmlUiManager::resize(int width, int height) const
{
    if (width == 0 || height == 0)
        return;

    rmlContext->SetDimensions({ width,height });
}