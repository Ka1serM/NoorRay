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
    customImage(context, 1, 1, renderer.getColorImageFormat(), vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
{
    // Register all embedded files
    static constexpr unsigned char inter_regular_ttf[] = {
    #embed "../../../assets/fonts/Inter-Regular.ttf"
    };
    static constexpr unsigned char editor_html[] = {
    #embed "../../../assets/rml/Editor.html"
    };
    static constexpr unsigned char editor_css[] = {
    #embed "../../../assets/rml/Editor.css"
    };
    static constexpr unsigned char menubar_html[] = {
    #embed "../../../assets/rml/MenuBar.html"
    };
    static constexpr unsigned char menubar_css[] = {
    #embed "../../../assets/rml/MenuBar.css"
    };
    static constexpr unsigned char details_panel_html[] = {
    #embed "../../../assets/rml/DetailsPanel.html"
    };
    static constexpr unsigned char details_panel_css[] = {
    #embed "../../../assets/rml/DetailsPanel.css"
    };
    static constexpr unsigned char scene_graph_html[] = {
    #embed "../../../assets/rml/SceneGraph.html"
    };
    static constexpr unsigned char scene_graph_css[] = {
    #embed "../../../assets/rml/SceneGraph.css"
    };
    static constexpr unsigned char viewport_html[] = {
    #embed "../../../assets/rml/Viewport.html"
    };
    static constexpr unsigned char viewport_css[] = {
    #embed "../../../assets/rml/Viewport.css"
    };

    rmlFileInterface.RegisterFile("Inter-Regular.ttf", inter_regular_ttf, sizeof(inter_regular_ttf));
    rmlFileInterface.RegisterFile("Editor.html", editor_html, sizeof(editor_html));
    rmlFileInterface.RegisterFile("Editor.css", editor_css, sizeof(editor_css));
    rmlFileInterface.RegisterFile("MenuBar.html", menubar_html, sizeof(menubar_html));
    rmlFileInterface.RegisterFile("MenuBar.css", menubar_css, sizeof(menubar_css));
    rmlFileInterface.RegisterFile("DetailsPanel.html", details_panel_html, sizeof(details_panel_html));
    rmlFileInterface.RegisterFile("DetailsPanel.css", details_panel_css, sizeof(details_panel_css));
    rmlFileInterface.RegisterFile("SceneGraph.html", scene_graph_html, sizeof(scene_graph_html));
    rmlFileInterface.RegisterFile("SceneGraph.css", scene_graph_css, sizeof(scene_graph_css));
    rmlFileInterface.RegisterFile("Viewport.html", viewport_html, sizeof(viewport_html));
    rmlFileInterface.RegisterFile("Viewport.css", viewport_css, sizeof(viewport_css));
    
    // Setup system interface
    rmlSystemInterface.SetWindow(context.getWindow());

    // Initialize RmlUi
    Rml::SetFileInterface(&rmlFileInterface);
    Rml::SetSystemInterface(&rmlSystemInterface);
    Rml::SetRenderInterface(&rmlRenderInterface);
    Rml::Initialise();

    // Create context
    rmlContext = Rml::CreateContext("main",{static_cast<int>(context.getWindowWidth()), static_cast<int>(context.getWindowHeight())});
    rmlContext->SetDensityIndependentPixelRatio(context.getDPIScale());

    // Load font(s)
    Rml::LoadFontFace("Inter-Regular.ttf");

    // Load initial document
    document = rmlContext->LoadDocument("Editor.html");
    if (document)
        document->Show();
}

void RmlUiManager::bindViewportImage(const Image& image) {
    const Rml::String texture_name = "viewport-img";
    rmlRenderInterface.registerVulkanTexture(texture_name,  image.getImageView(),  Rml::Vector2i{static_cast<int>(image.getWidth()), static_cast<int>(image.getHeight())});

    Rml::Element* viewportElem = document->GetElementById("viewport-img");
    if (viewportElem)
        viewportElem->SetAttribute("src", "vulkan://" + texture_name);
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
    rmlRenderInterface.beginFrame(command_buffer, target_image_view, depthImageView, target_extent, in_flight_fenc);

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