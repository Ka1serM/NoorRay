#pragma once

#include "NoorRay.App/UI/Models/InspectorModel.h"
#include "NoorRay.App/UI/Models/MaterialEditorModel.h"
#include "NoorRay.App/UI/Models/NodeGraphModel.h"
#include "NoorRay.App/UI/Models/OutlinerModel.h"
#include "NoorRay.App/UI/Models/RenderSettingsModel.h"
#include "NoorRay.App/UI/Models/StatusModel.h"
#include "NoorRay.App/UI/Models/TimelineModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

//! What the viewport shows in place of the scene.
struct ViewportSourceModel {
    bool shown = false;
    Rml::String title;
    Rml::String detail;
    Rml::String action;
    bool actionEnabled = false;

    bool operator==(const ViewportSourceModel&) const = default;
};

class AppModel {
public:
    explicit AppModel(RmlKit::UI::Model& model);
    AppModel(const AppModel&) = delete;
    AppModel& operator=(const AppModel&) = delete;

    OutlinerModel outliner;
    InspectorModel inspector;
    EnvironmentModel environment;
    MaterialEditorModel materialEditor;
    NodeGraphModel nodeGraph;
    RenderSettingsModel renderSettings;
    StatusModel status;
    TimelineModel timeline;
    ViewportSourceModel viewportSource;
    bool canSave = !PLATFORM_WEB;
    //! The open menu of any menu bar, or empty.
    Rml::String menu;
    //! The transform tool of the viewport gizmo: translate, rotate or scale.
    Rml::String tool = "translate";
    //! The last object and light picked from the toolbar groups, which a click on the group adds again.
    Rml::String addShape = "cube";
    Rml::String addLight = "point";
    bool materialEditorOpen = false;
    bool nodeEditorOpen = false;
};

}
