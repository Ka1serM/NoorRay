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

class SceneGraphViewModel :  public ViewModelBase,  public IObserver<SceneEvent> 
{
public:
    SceneGraphViewModel(Scene& scene, Rml::Context* context, const Rml::String& modelName);
    void ProcessEvent(Rml::Event& event);
    void deleteNode(int node_id);

    void OnNotified(const SceneEvent& event) override;

    // UI actions
    void toggleNode(int nodeId);
    void selectNode(int nodeId) const;

private:
    Scene& sceneModel;

    void refreshFlatList();
    void buildFlatListRecursive(const SceneObject* node, int depth);

    // ViewModel state
    std::map<int, bool> nodeOpenStates;

    // Observable properties (bound to UI)
    ObservableCollection<FlatNodeData> flatNodes;
    ObservableProperty<int> selectedNodeId{-1};
};