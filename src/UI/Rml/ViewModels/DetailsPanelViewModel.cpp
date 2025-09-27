#include "DetailsPanelViewModel.h"
#include <Scene/SceneObject.h>
#include <sstream>

DetailsPanelViewModel::DetailsPanelViewModel(Scene& scene) 
    : scene_model(scene) 
{
    // The ViewModel must register as an observer of the Scene model to receive notifications
    // about selection changes. This is the crucial step.
    scene_model.AddObserver(this);
}

void DetailsPanelViewModel::BindToModel(const Rml::String& model_name, Rml::Context* context) {
    Rml::DataModelConstructor constructor = context->CreateDataModel(model_name);
    if (!constructor) return;

    // Register our PropertyData struct
    if (auto handle = constructor.RegisterStruct<PropertyData>()) {
        handle.RegisterMember("name", &PropertyData::name);
        handle.RegisterMember("value", &PropertyData::value);
        handle.RegisterMember("type", &PropertyData::type);
    }
    constructor.RegisterArray<std::vector<PropertyData>>();

    // Bind Observable properties/collections
    properties.Bind(constructor, "properties");
    object_name.Bind(constructor, "object_name");

    // Initially populate the view with data from the scene's current selection.
    updateProperties();
}

void DetailsPanelViewModel::OnNotified(const SceneEvent& event) {
    // We only care about selection changes
    if (event.type == SceneEvent::SelectionChanged) {
        updateProperties();
    }
}

void DetailsPanelViewModel::updateProperties() {
    properties.clear();

    const SceneObject* selectedObject = scene_model.getActiveObject();
    if (!selectedObject) {
        object_name.Set("No Object Selected");
        return;
    }

    // Set the object name property at the top of the panel
    object_name.Set(selectedObject->getName());

    // Populate the properties list with data from the selected object.
    // This replaces the old ImGui render() methods.
    properties.push_back({"ID", std::to_string(selectedObject->getId()), "text"});
    properties.push_back({"Name", selectedObject->getName(), "text"});
    
    // Example: Display Transform properties
    const Transform& transform = selectedObject->getTransform();
    std::stringstream ss;
    
    // Position
    ss.str("");
    ss << "x=" << transform.getPosition().x << ", y=" << transform.getPosition().y << ", z=" << transform.getPosition().z;
    properties.push_back({"getPosition()", ss.str(), "text"});

    // getRotation() (as a placeholder)
    ss.str("");
    ss << "x=" << transform.getRotation().x << ", y=" << transform.getRotation().y << ", z=" << transform.getRotation().z;
    properties.push_back({"Rotation", ss.str(), "text"});
    
    // Scale
    ss.str("");
    ss << "x=" << transform.getScale().x << ", y=" << transform.getScale().y << ", z=" << transform.getScale().z;
    properties.push_back({"Scale", ss.str(), "text"});

    // This is where you would handle subclass-specific properties.
    // E.g., if you have a MeshInstance, you could cast and add more properties:
    // if (const MeshInstance* meshInstance = dynamic_cast<const MeshInstance*>(selectedObject)) {
    //     properties.push_back({"Mesh Asset", meshInstance->getMeshAsset()->getName(), "text"});
    // }

    // RmlUi automatically handles the update now because we pushed new data
    // to the ObservableCollection.
}