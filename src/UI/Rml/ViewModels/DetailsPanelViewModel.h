#pragma once
#include "IViewModel.h"
#include "../Observable/ObservableCollection.h"
#include "../Observable/ObservableProperty.h"
#include "../Observable/IObserver.h"
#include <Scene/Scene.h>
#include <string>

// A simple structure to represent a single property for display in the UI.
// This is the "flat" data format for the RML view.
struct PropertyData {
    Rml::String name;
    Rml::String value;
    Rml::String type; // Used to decide which UI control to show (e.g., text, slider)
};

class DetailsPanelViewModel : 
    public IViewModel, 
    public IObserver<SceneEvent> 
{
public:
    DetailsPanelViewModel(Scene& scene); 
    void BindToModel(const Rml::String& model_name, Rml::Context* context) override;

    // The key method that reacts to selection changes in the Scene
    void OnNotified(const SceneEvent& event) override;

private:
    // Updates the internal `properties` list based on the currently selected object.
    void updateProperties();

    Scene& scene_model;
    ObservableCollection<PropertyData> properties;
    ObservableProperty<Rml::String> object_name;
};
