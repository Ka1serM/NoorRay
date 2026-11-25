#include "NoorRay.App/Application.h"

#include <format>

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>

#include "NoorRay/Render/Denoise/DenoiserPlugin.h"
#include "NoorRay/Render/Upscale/DlssRr.h"
#include "RmlKit/UI/Backend/Fatal.h"

namespace NoorRay::App {

namespace {

constexpr std::uint32_t initialSize = 64;

const Render::DlssRrExtensions dlssRrExtensions;

}

Application::Application()
    : RmlKit::UI::Application({.extensions = &dlssRrExtensions}), model(getModel()), app(model), inspector(model, app, session, viewports),
      materials(model, app, getWindow(), session, viewports), nodeGraph(model, app, session, viewports, nodeGraphs),
      scene(model, app, getWindow(), session, viewports, inspector, materials, nodeGraph), renderSettings(model, app, session, viewports),
      status(model, app, session, viewports), timeline(model, app, *getContext().GetRootElement(), session, viewports) {
    Render::prepareDenoiserPlugins();
    Rml::Factory::RegisterElementInstancer("viewport", &viewports);
    Rml::Factory::RegisterElementInstancer("materialpreview", &materialPreviews);
    Rml::Factory::RegisterElementInstancer("nodegraph", &nodeGraphs);
    materialPreviews.addContext(&getContext(), getRenderer());
    model.bindEvent("materialOpen", &Application::onOpenMaterial, this);
    model.bindEvent("toggleMaterialEditor", &Application::toggleMaterialEditor, this);
    model.bindEvent("toggleNodeEditor", &Application::toggleNodeEditor, this);
    model.bindEvent("materialNodes", &Application::onMaterialNodes, this);
    model.bindEvent("materialUseNodes", &Application::onUseNodes, this);
    nodeGraph.retitled = [this] { retitleNodeEditor(); };
    scene.nodeMaterialCreated = [this](const Scene::MaterialHandle material) { openMaterial(material); };
    model.bindEvent("showPlugin", &Application::onShowPlugin, this);
    model.bindEvent("viewportSourceAction", &Application::onViewportSourceAction, this);
    model.bindEvent("viewportSourceLeave", &Application::onViewportSourceLeave, this);
    for (const Plugin* plugin : getPlugins())
        pluginNames.emplace_back(plugin->getName());
    model.bindData([this](Rml::DataModelConstructor& constructor) { constructor.Bind("plugins", &pluginNames); });
    viewports.getEvents().ticks.push_back([this] { return updatePlugins(); });
}

Application::~Application() {
    closeDocuments();
}

std::expected<void, RmlKit::Core::Error> Application::startup() {
    if (auto opened = session.open(getRenderer().getDevice(), initialSize, initialSize); !opened)
        return RmlKit::Core::fail("gpu", opened.error().message);
    if (auto opened = materialPreviews.open(); !opened)
        return RmlKit::Core::fail("gpu", opened.error().message);
    inspector.showEnvironment();
    getRouter().open();
    getRouter().show("editor");
    if (app.materialEditorOpen)
        openMaterialEditor();
    for (Plugin* plugin : getPlugins())
        plugin->open(*this);
    return {};
}

std::expected<void, RmlKit::Core::Error> Application::shutdown() {
    viewports.setSource(nullptr);
    for (Plugin* plugin : getPlugins())
        plugin->close();
    if (auto idle = getRenderer().getDevice().waitIdle(); !idle)
        return RmlKit::Core::fail("gpu", idle.error().message);
    return {};
}

void Application::openMaterialEditor() {
    if (materialWindow) {
        SDL_RaiseWindow(materialWindow->getNative());
        return;
    }
    RmlKit::UI::SecondaryWindow& window =
        openWindow({.title = "Material editor", .width = 478, .height = 590, .minimumWidth = 360, .minimumHeight = 320, .parent = getWindow()});
    model.share(window.getModel());
    materialPreviews.addContext(&window.getContext(), window.getRenderer());
    constexpr const char* path = "UI/Documents/material-editor.rml";
    Rml::ElementDocument* document = window.getContext().LoadDocument(path);
    if (!document)
        RmlKit::UI::fatal(std::format("RmlUi cannot load {}", path));
    document->Show();
    materialFields.emplace(*window.getContext().GetRootElement());
    materialWindow = &window;
    app.materialEditorOpen = true;
    model.dirty("materialEditorOpen");
}

void Application::openNodeEditor() {
    if (nodeWindow) {
        SDL_RaiseWindow(nodeWindow->getNative());
        return;
    }
    RmlKit::UI::SecondaryWindow& window =
        openWindow({.title = nodeGraph.getOwnerName().empty() ? Rml::String("Node editor") : std::format("{} - Node editor", nodeGraph.getOwnerName()),
                    .width = 1280,
                    .height = 760,
                    .minimumWidth = 640,
                    .minimumHeight = 400,
                    .parent = getWindow()});
    model.share(window.getModel());
    constexpr const char* path = "UI/Documents/node-editor.rml";
    Rml::ElementDocument* document = window.getContext().LoadDocument(path);
    if (!document)
        RmlKit::UI::fatal(std::format("RmlUi cannot load {}", path));
    document->Show();
    nodeFields.emplace(*window.getContext().GetRootElement());
    nodeWindow = &window;
    app.nodeEditorOpen = true;
    model.dirty("nodeEditorOpen");
}

void Application::retitleNodeEditor() {
    if (!nodeWindow)
        return;
    const std::string owner = nodeGraph.getOwnerName();
    SDL_SetWindowTitle(nodeWindow->getNative(), owner.empty() ? "Node editor" : std::format("{} - Node editor", owner).c_str());
}

void Application::openMaterial(const Scene::MaterialHandle material) {
    const Scene::Material* found = session.getScene().find(material);
    if (!found)
        return;
    materials.edit(material);
    nodeGraph.showMaterial(material);
    if (found->nodeTree)
        return openNodeEditor();
    openMaterialEditor();
}

void Application::onOpenMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const Core::SlotMap<Scene::Material>& all = session.getScene().getMaterials();
    const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "materialOpen", all.size());
    openMaterial(all.keyAt(static_cast<std::uint32_t>(index)));
}

void Application::onMaterialNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "materialNodes");
    nodeGraph.showMaterial(materials.getMaterial());
    openNodeEditor();
}

void Application::onUseNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "materialUseNodes");
    if (!materials.useNodes())
        return;
    scene.sceneChanged();
    openMaterial(materials.getMaterial());
}

void Application::toggleMaterialEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "toggleMaterialEditor");
    if (!materialWindow)
        return openMaterialEditor();
    materialWindow->close();
}

void Application::toggleNodeEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "toggleNodeEditor");
    if (!nodeWindow)
        return openNodeEditor();
    nodeWindow->close();
}

void Application::moveCamera(const Core::Transform& transform, const float verticalFov) {
    Scene::Scene& scene = session.getScene();
    const Scene::Camera* active = scene.findActiveCamera();
    if (!active)
        return;
    Scene::Camera moved = *active;
    moved.previousTransform = moved.transform;
    moved.transform = transform;
    moved.verticalFov = verticalFov;
    scene.setCamera(scene.getActiveCamera(), moved);
    viewports.followActiveCamera();
    viewports.invalidate();
}

void Application::onShowPlugin(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    getPlugins()[RmlKit::UI::Model::indexArgument(arguments, "showPlugin", getPlugins().size())]->show();
}

bool Application::updatePlugins() {
    bool more = false;
    for (Plugin* plugin : getPlugins())
        more = plugin->update() || more;
    showBanner();
    return more;
}

void Application::showBanner() {
    ViewportSourceModel shown;
    if (const ViewportSource* source = viewports.getSource()) {
        ViewportBanner banner = source->getBanner();
        shown = {true, std::move(banner.title), std::move(banner.detail), std::move(banner.action), banner.actionEnabled};
    }
    if (shown == app.viewportSource)
        return;
    app.viewportSource = std::move(shown);
    model.dirty("viewportSource");
}

void Application::onViewportSourceAction(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "viewportSourceAction");
    if (ViewportSource* source = viewports.getSource())
        source->runAction();
}

void Application::onViewportSourceLeave(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "viewportSourceLeave");
    if (ViewportSource* source = viewports.getSource())
        source->leave();
}

RmlKit::UI::SecondaryWindow& Application::openWindow(const RmlKit::UI::WindowOptions& options) {
    RmlKit::UI::SecondaryWindow& window = RmlKit::UI::Application::openWindow(options);
    windowScrubs.emplace(&window, std::make_unique<NumberScrub>(*window.getContext().GetRootElement()));
    return window;
}

RmlKit::UI::SecondaryWindow& Application::openWindow(const Plugin& plugin) {
    const PluginWindowSize size = plugin.getWindowSize();
    return openWindow(RmlKit::UI::WindowOptions{.title = Rml::String(plugin.getName()),
                                                .width = size.width,
                                                .height = size.height,
                                                .minimumWidth = size.minimumWidth,
                                                .minimumHeight = size.minimumHeight,
                                                .parent = getWindow()});
}

void Application::windowClosed(RmlKit::UI::SecondaryWindow& window) {
    for (Plugin* plugin : getPlugins())
        plugin->windowClosed(window);
    windowScrubs.erase(&window);
    if (&window == nodeWindow) {
        nodeFields.reset();
        nodeWindow = nullptr;
        app.nodeEditorOpen = false;
        model.dirty("nodeEditorOpen");
        return;
    }
    if (&window != materialWindow)
        return;
    materialPreviews.removeContext(&window.getContext());
    materialFields.reset();
    materialWindow = nullptr;
    app.materialEditorOpen = false;
    model.dirty("materialEditorOpen");
}

}
