#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay/Render/RendererSettings.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

class RenderSettingsController {
public:
    RenderSettingsController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports);
    RenderSettingsController(const RenderSettingsController&) = delete;
    RenderSettingsController& operator=(const RenderSettingsController&) = delete;

private:
    void onMode(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onApply(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onRestart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    void show(const Render::RendererSettings& settings);
    Render::RendererSettings read() const;
    void apply(const Render::RendererSettings& settings);

    RmlKit::UI::Model& model;
    AppModel& app;
    Session::Session& session;
    ViewportElementInstancer& viewports;
};

}
