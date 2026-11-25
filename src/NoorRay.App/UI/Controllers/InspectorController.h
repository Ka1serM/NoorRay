#pragma once

#include <cstdint>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Controllers/Selection.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

class InspectorController {
public:
    InspectorController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports);
    InspectorController(const InspectorController&) = delete;
    InspectorController& operator=(const InspectorController&) = delete;

    void show(const Selection& selected);
    void showEnvironment();
    const Selection& getSelection() const { return selection; }
    //! Shows the transform of the selection again after something else moved it.
    void showSelectedTransform();

private:
    void onApply(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onColor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onEnvironmentColor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onApplyEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onClearEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onMakeActiveCamera(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    void showTransform(const Core::Transform& transform);
    void showMaterials(const Scene::Instance& instance, const Scene::Mesh* mesh);
    Core::Transform readTransform() const;
    void changed();
    void storeLight(Scene::LightHandle handle, const Scene::Light& light);
    void storeEnvironment(const Scene::Environment& environment);

    RmlKit::UI::Model& model;
    AppModel& app;
    Session::Session& session;
    ViewportElementInstancer& viewports;
    Selection selection;
};

}
