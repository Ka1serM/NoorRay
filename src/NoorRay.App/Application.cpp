#include "NoorRay.App/Application.h"

#include <format>

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>

#include "RmlKit/UI/Backend/Fatal.h"

namespace NoorRay::App {

namespace {

constexpr std::uint32_t initialSize = 64;

}

Application::Application()
    : model(getModel()), app(model), inspector(model, app, session, viewports),
      materials(model, app, getWindow(), session, viewports), scene(model, app, getWindow(), session, viewports, inspector, materials),
      renderSettings(model, app, session, viewports), status(model, app, session, viewports), timeline(model, app, *getContext().GetRootElement(), viewports),
      training(model, app, getWindow()) {
    Rml::Factory::RegisterElementInstancer("viewport", &viewports);
    Rml::Factory::RegisterElementInstancer("materialpreview", &materialPreviews);
    materialPreviews.addContext(&getContext(), getRenderer());
    model.bindEvent("materialOpen", &Application::onOpenMaterial, this);
    model.bindEvent("toggleMaterialEditor", &Application::toggleMaterialEditor, this);
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
    return {};
}

std::expected<void, RmlKit::Core::Error> Application::shutdown() {
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
    materialScrub.emplace(*window.getContext().GetRootElement());
    materialWindow = &window;
    app.materialEditorOpen = true;
    model.dirty("materialEditorOpen");
}

void Application::onOpenMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    materials.edit(RmlKit::UI::Model::indexArgument(arguments, "materialOpen", session.getScene().getMaterials().size()));
    openMaterialEditor();
}

void Application::toggleMaterialEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "toggleMaterialEditor");
    if (!materialWindow)
        return openMaterialEditor();
    materialWindow->close();
}

void Application::windowClosed(RmlKit::UI::SecondaryWindow& window) {
    if (&window != materialWindow)
        return;
    materialPreviews.removeContext(&window.getContext());
    materialScrub.reset();
    materialFields.reset();
    materialWindow = nullptr;
    app.materialEditorOpen = false;
    model.dirty("materialEditorOpen");
}

}
