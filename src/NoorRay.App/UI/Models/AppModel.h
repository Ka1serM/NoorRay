#pragma once

#include "NoorRay.App/UI/Models/InspectorModel.h"
#include "NoorRay.App/UI/Models/MaterialEditorModel.h"
#include "NoorRay.App/UI/Models/OutlinerModel.h"
#include "NoorRay.App/UI/Models/RenderSettingsModel.h"
#include "NoorRay.App/UI/Models/StatusModel.h"
#include "NoorRay.App/UI/Models/TimelineModel.h"
#include "NoorRay.App/UI/Models/TrainingModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

class AppModel {
public:
    explicit AppModel(RmlKit::UI::Model& model);
    AppModel(const AppModel&) = delete;
    AppModel& operator=(const AppModel&) = delete;

    OutlinerModel outliner;
    InspectorModel inspector;
    EnvironmentModel environment;
    MaterialEditorModel materialEditor;
    RenderSettingsModel renderSettings;
    StatusModel status;
    TimelineModel timeline;
    TrainingModel training;
    bool canSave = !PLATFORM_WEB;
    //! The open menu of any menu bar, or empty.
    Rml::String menu;
    //! The transform tool of the viewport gizmo: move, scale or rotate.
    Rml::String tool = "move";
    //! The last object and light picked from the toolbar groups, which a click on the group adds again.
    Rml::String addShape = "cube";
    Rml::String addLight = "point";
    bool materialEditorOpen = true;
};

}
