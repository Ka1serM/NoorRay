#pragma once
#include "ViewModelBase.h"
#include "../Observable/ObservableCollection.h"
#include "../Observable/ObservableProperty.h"
#include "../Observable/IObserver.h"
#include <Scene/Scene.h>
#include <map>

// The data structure sent to the view: flat with depth info
struct FlatNodeData {
    int id;
    Rml::String name;
    int depth = 0;
    bool has_children = false;
    bool is_open = false; // State managed by the ViewModel
};

class SceneGraphViewModel : 
    public ViewModelBase, 
    public IObserver<SceneEvent> 
{
public:
    SceneGraphViewModel(Scene& scene, Rml::Context* context, const Rml::String& model_name);

    void OnNotified(const SceneEvent& event) override;

    // UI actions
    void ToggleNode(int node_id);
    void SelectNode(int node_id);

private:
    Scene& scene_model;

    void RefreshFlatList();
    void BuildFlatListRecursive(const SceneObject* node, int depth);

    // ViewModel state
    std::map<int, bool> node_open_states;

    // Observable properties
    ObservableCollection<FlatNodeData> flat_nodes;
    ObservableProperty<int> selected_node_id;
};