#pragma once

#include <expected>
#include <optional>

#include "NoorRay.App/UI/Controllers/InspectorController.h"
#include "NoorRay.App/UI/Controllers/MaterialController.h"
#include "NoorRay.App/UI/Controllers/RenderSettingsController.h"
#include "NoorRay.App/UI/Controllers/SceneController.h"
#include "NoorRay.App/UI/Controllers/StatusController.h"
#include "NoorRay.App/UI/Controllers/TimelineController.h"
#include "NoorRay.App/UI/Controllers/TrainingController.h"
#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/HoldMenu.h"
#include "NoorRay.App/UI/Controls/MaterialPreviewElement.h"
#include "NoorRay.App/UI/Controls/NumberScrub.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Session/Session.h"
#include "RmlKit/UI/Backend/Application.h"

namespace NoorRay::App {

class Application final : public RmlKit::UI::Application {
public:
    Application();
    ~Application() override;

    std::expected<void, RmlKit::Core::Error> startup() override;
    std::expected<void, RmlKit::Core::Error> shutdown() override;

protected:
    bool handleEvent(const SDL_Event& event) override { return holdMenu.handle(event) || viewports.handleEvent(event); }
    void windowClosed(RmlKit::UI::SecondaryWindow& window) override;

private:
    void openMaterialEditor();
    void onOpenMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void toggleMaterialEditor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

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
    InspectorController inspector;
    MaterialController materials;
    SceneController scene;
    RenderSettingsController renderSettings;
    StatusController status;
    TimelineController timeline;
    TrainingController training;
    RmlKit::UI::SecondaryWindow* materialWindow = nullptr;
    std::optional<FieldCommit> materialFields;
    std::optional<NumberScrub> materialScrub;
};

}
