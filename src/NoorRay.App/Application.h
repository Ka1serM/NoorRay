#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "NoorRay.App/Plugins/Plugin.h"
#include "NoorRay.App/UI/Controllers/InspectorController.h"
#include "NoorRay.App/UI/Controllers/MaterialController.h"
#include "NoorRay.App/UI/Controllers/NodeGraphController.h"
#include "NoorRay.App/UI/Controllers/RenderSettingsController.h"
#include "NoorRay.App/UI/Controllers/SceneController.h"
#include "NoorRay.App/UI/Controllers/StatusController.h"
#include "NoorRay.App/UI/Controllers/TimelineController.h"
#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/HoldMenu.h"
#include "NoorRay.App/UI/Controls/MaterialPreviewElement.h"
#include "NoorRay.App/UI/Controls/NodeGraphElement.h"
#include "NoorRay.App/UI/Controls/NumberScrub.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Session/Session.h"
#include "RmlKit/UI/Backend/Application.h"

namespace NoorRay::App {

class Application final : public RmlKit::UI::Application, public PluginHost {
public:
    Application();
    ~Application() override;

    std::expected<void, RmlKit::Core::Error> startup() override;
    std::expected<void, RmlKit::Core::Error> shutdown() override;

protected:
    bool handleEvent(const SDL_Event& event) override { return holdMenu.handle(event) || viewports.handleEvent(event); }
    void windowClosed(RmlKit::UI::SecondaryWindow& window) override;

private:
    Session::Session& getSession() override { return session; }
    SDL_Window* getWindow() const override { return RmlKit::UI::Application::getWindow(); }
    //! Every secondary window gets number fields and sliders, so a plugin window has them too.
    RmlKit::UI::SecondaryWindow& openWindow(const RmlKit::UI::WindowOptions& options) override;
    RmlKit::UI::SecondaryWindow& openWindow(const Plugin& plugin) override;
    void setViewportSource(ViewportSource* source) override { viewports.setSource(source); }
    void redraw() override { viewports.invalidate(); }
    void sceneChanged() override { scene.sceneChanged(); }
    void frameViewport() override { viewports.frameScene(); }
    void moveCamera(const Core::Transform& transform, float verticalFov) override;
    void reportError(const Rml::String& source, const Rml::String& message) override { model.report(RmlKit::UI::ReportLevel::Error, source, message); }
    void reportDone(const Rml::String& source, const Rml::String& message) override { model.report(RmlKit::UI::ReportLevel::Done, source, message); }

    void openMaterialEditor();
    void openNodeEditor();
    //! A material with a node tree opens in the node editor, and one without in the material editor.
    void openMaterial(Scene::MaterialHandle material);
    void retitleNodeEditor();
    void onShowPlugin(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onViewportSourceAction(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onViewportSourceLeave(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    bool updatePlugins();
    void showBanner();
    void onOpenMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void toggleMaterialEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void toggleNodeEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onMaterialNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onUseNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    RmlKit::UI::Model& model;
    FieldCommit fieldCommit{*getContext().GetRootElement()};
    NumberScrub numberScrub{*getContext().GetRootElement()};
    AppModel app;
    HoldMenu holdMenu{*getContext().GetRootElement(), [this](const Rml::String& key) {
                          app.menu = key;
                          model.dirty("menu");
                      }};
    Session::Session session;
    ViewportElementInstancer viewports{getRenderer(), session, getWindow()};
    MaterialPreviewInstancer materialPreviews{session};
    NodeGraphInstancer nodeGraphs;
    InspectorController inspector;
    MaterialController materials;
    NodeGraphController nodeGraph;
    SceneController scene;
    RenderSettingsController renderSettings;
    StatusController status;
    TimelineController timeline;
    std::vector<Rml::String> pluginNames;
    RmlKit::UI::SecondaryWindow* materialWindow = nullptr;
    std::optional<FieldCommit> materialFields;
    RmlKit::UI::SecondaryWindow* nodeWindow = nullptr;
    std::optional<FieldCommit> nodeFields;
    std::unordered_map<const RmlKit::UI::SecondaryWindow*, std::unique_ptr<NumberScrub>> windowScrubs;
};

}
