#include "RmlUiManager.h"
#include <RmlUi/Core.h>
#include <RmlUi/Lua.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Debugger/Debugger.h>

#include "ViewModels/SceneGraphViewModel.h"
#include "ViewModels/ViewportViewModel.h"
#include "Vulkan/Image.h"

RmlUiManager::RmlUiManager(Context& context, Scene& scene, const Renderer& renderer, Image& renderImage, Image& cryptoImage, Image& positionImage)
: rmlRenderInterface(
      context.getDevice(),
      context.getGraphicsQueue(),
      context.getAllocator(),
      context.getCommandPool(),
      context.getDescriptorPool(),
      renderer.getColorImageFormat(),
      renderer.getDepthImageFormat()
    )
{
#ifdef NDEBUG
    // Register all embedded files
    static constexpr unsigned char inter_ttf[] = {
    #embed "../../../assets/fonts/Inter.ttf"
    };
    static constexpr unsigned char theme_css[] = {
    #embed "../../../assets/rml/Theme.css"
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
    static constexpr unsigned char details_html[] = {
    #embed "../../../assets/rml/Details.html"
    };
    static constexpr unsigned char details_css[] = {
    #embed "../../../assets/rml/Details.css"
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
    static constexpr unsigned char content_browser_html[] = {
    #embed "../../../assets/rml/ContentBrowser.html"
    };
    static constexpr unsigned char content_browser_css[] = {
    #embed "../../../assets/rml/ContentBrowser.css"
    };
    static constexpr unsigned char material_editor_html[] = {
    #embed "../../../assets/rml/MaterialEditor.html"
    };
    static constexpr unsigned char material_editor_css[] = {
    #embed "../../../assets/rml/MaterialEditor.css"
    };

    rmlFileInterface.RegisterFile("../assets/fonts/Inter.ttf", inter_ttf, sizeof(inter_ttf));
    rmlFileInterface.RegisterFile("../assets/rml/Theme.css", theme_css, sizeof(theme_css));
    rmlFileInterface.RegisterFile("../assets/rml/Editor.html", editor_html, sizeof(editor_html));
    rmlFileInterface.RegisterFile("../assets/rml/Editor.css", editor_css, sizeof(editor_css));
    rmlFileInterface.RegisterFile("../assets/rml/MenuBar.html", menubar_html, sizeof(menubar_html));
    rmlFileInterface.RegisterFile("../assets/rml/MenuBar.css", menubar_css, sizeof(menubar_css));
    rmlFileInterface.RegisterFile("../assets/rml/Details.html", details_html, sizeof(details_html));
    rmlFileInterface.RegisterFile("../assets/rml/Details.css", details_css, sizeof(details_css));
    rmlFileInterface.RegisterFile("../assets/rml/SceneGraph.html", scene_graph_html, sizeof(scene_graph_html));
    rmlFileInterface.RegisterFile("../assets/rml/SceneGraph.css", scene_graph_css, sizeof(scene_graph_css));
    rmlFileInterface.RegisterFile("../assets/rml/Viewport.html", viewport_html, sizeof(viewport_html));
    rmlFileInterface.RegisterFile("../assets/rml/Viewport.css", viewport_css, sizeof(viewport_css));
    rmlFileInterface.RegisterFile("../assets/rml/ContentBrowser.html", content_browser_html, sizeof(content_browser_html));
    rmlFileInterface.RegisterFile("../assets/rml/ContentBrowser.css", content_browser_css, sizeof(content_browser_css));
    rmlFileInterface.RegisterFile("../assets/rml/MaterialEditor.html", material_editor_html, sizeof(material_editor_html));
    rmlFileInterface.RegisterFile("../assets/rml/MaterialEditor.css", material_editor_css, sizeof(material_editor_css));
    Rml::SetFileInterface(&rmlFileInterface);
#endif

    // Setup system interface
    rmlSystemInterface.SetWindow(context.getWindow());

    // Initialize RmlUi
    Rml::SetSystemInterface(&rmlSystemInterface);
    Rml::SetRenderInterface(&rmlRenderInterface);
    Rml::Initialise();
    Rml::Lua::Initialise();

    // Create context
    rmlContext = Rml::CreateContext("main",{static_cast<int>(context.getWindowWidth()), static_cast<int>(context.getWindowHeight())});
    rmlContext->SetDensityIndependentPixelRatio(context.getDPIScale());

    // Load font(s)
    Rml::LoadFontFace("../assets/fonts/Inter.ttf");

    viewModels.emplace_back(std::make_unique<SceneGraphViewModel>(scene, rmlContext, "scenegraph_vm"));
    viewModels.emplace_back(std::make_unique<MenuBarViewModel>(scene, rmlContext, "menubar_vm"));
    
    // Load initial document
    editorDocument = rmlContext->LoadDocument("../assets/rml/Editor.html");
    editorDocument->SetId("main");
    if (editorDocument)
        editorDocument->Show();

    // Create viewport VM and store pointer
    auto vpVM = std::make_unique<ViewportViewModel>(context, scene, rmlContext, rmlRenderInterface, renderImage, cryptoImage, positionImage, "viewport");
    viewportVM = vpVM.get();  // non-owning pointer
    viewModels.emplace_back(std::move(vpVM));
    
    materialEditorDocument =  rmlContext->LoadDocument("../assets/rml/MaterialEditor.html");
    if (materialEditorDocument)
        materialEditorDocument->Show();

    Rml::Debugger::Initialise(rmlContext);
}

RmlUiManager::~RmlUiManager()
{
    if (rmlContext) {
        Rml::RemoveContext("main");
        rmlContext = nullptr;
    }

    Rml::Shutdown();
}

void RmlUiManager::render(const vk::CommandBuffer command_buffer, const vk::Image target_image,  const vk::ImageView target_image_view, const vk::ImageView depthImageView, const vk::Extent2D target_extent, const vk::Fence in_flight_fenc)
{
    rmlRenderInterface.beginFrame(command_buffer, target_image, target_image_view, depthImageView, target_extent, in_flight_fenc);

    rmlContext->Update();
    for (const auto& vm : viewModels)
        vm->Update();

    rmlContext->Render();

    rmlRenderInterface.endFrame();

    if (viewportVM)
        viewportVM->render(command_buffer);
}

void RmlUiManager::processEvent(SDL_Window* window, SDL_Event& event) const
{
    RmlSDL::InputEventHandler(rmlContext, window, event);
    if (viewportVM)
        viewportVM->processEvent(event);
}

void RmlUiManager::resize(int width, int height) const
{
    if (width != 0 && height != 0)
        rmlContext->SetDimensions({ width,height });
}

void RmlUiManager::reload()
{
    if (editorDocument) {
        editorDocument->Close();
        editorDocument = nullptr;
    }

    if (materialEditorDocument) {
        materialEditorDocument->Close();
        materialEditorDocument = nullptr;
    }
    
    Rml::Factory::ClearStyleSheetCache();

    editorDocument = rmlContext->LoadDocument("../assets/rml/Editor.html");
    if (editorDocument)
        editorDocument->Show();

    materialEditorDocument = rmlContext->LoadDocument("../assets/rml/MaterialEditor.html");
    if (materialEditorDocument)
        materialEditorDocument->Show();
}
