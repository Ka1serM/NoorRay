#pragma once
#include "IViewModel.h"
#include "../Observable/ObservableCollection.h"
#include "../Observable/ObservableProperty.h"
#include "../Observable/IObserver.h"
#include <Scene/Scene.h>

// The data structure we send to the View: flat with depth info.
struct FlatNodeData {
    int id;
    Rml::String name;
    int depth = 0;
    bool has_children = false;
    bool is_open = false; // State managed by the ViewModel
};

class SceneGraphViewModel : 
    public IViewModel, 
    public IObserver<SceneEvent> 
{
public:
    // Constructor now takes the Scene reference
    SceneGraphViewModel(Scene& scene); 
    void BindToModel(const Rml::String& model_name, Rml::Context* context) override;

    // Implementation of IObserver<Scene::SceneEvent>::OnNotified
    void OnNotified(const SceneEvent& event) override;

    // Public methods for the UI to call
    void ToggleNode(int node_id);
    void SelectNode(int node_id);

private:
    Scene& scene_model;
    
    void RefreshFlatList();
    void BuildFlatListRecursive(const SceneObject* node, int depth);

    // ViewModel state to track which nodes are open, since this state is
    // purely for the UI and shouldn't be in the SceneObject itself.
    std::map<int, bool> node_open_states; 
    ObservableCollection<FlatNodeData> flat_nodes;
    ObservableProperty<int> selected_node_id;
};