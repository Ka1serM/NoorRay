#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>
#include <SDL3/SDL_video.h>

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;

class TrainingController {
public:
    TrainingController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window);
    TrainingController(const TrainingController&) = delete;
    TrainingController& operator=(const TrainingController&) = delete;

private:
    void onChooseDataset(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onStart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onStop(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    RmlKit::UI::Model& model;
    AppModel& app;
    SDL_Window* window;
};

}
