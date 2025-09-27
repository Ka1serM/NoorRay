#include "SceneGraphViewModel.h"
#include <map>
#include <Scene/SceneObject.h>

SceneGraphViewModel::SceneGraphViewModel(Scene& scene) 
    : scene_model(scene), selected_node_id(-1) 
{
    // 1. Register as an observer of the Scene
    scene_model.AddObserver(this); 
    // 2. Initial population of the view list
    RefreshFlatList();
}

void SceneGraphViewModel::BindToModel(const Rml::String& model_name, Rml::Context* context) {
    Rml::DataModelConstructor constructor = context->CreateDataModel(model_name);
    if (!constructor) return;

    // Register FlatNodeData structure
    if (auto handle = constructor.RegisterStruct<FlatNodeData>()) {
        handle.RegisterMember("id", &FlatNodeData::id);
        handle.RegisterMember("name", &FlatNodeData::name);
        handle.RegisterMember("depth", &FlatNodeData::depth);
        handle.RegisterMember("has_children", &FlatNodeData::has_children);
        handle.RegisterMember("is_open", &FlatNodeData::is_open);
    }
    constructor.RegisterArray<std::vector<FlatNodeData>>();

    // Bind Observable properties/collections
    flat_nodes.Bind(constructor, "flat_nodes");
    selected_node_id.Bind(constructor, "selected_node_id");

    // Bind UI events
    constructor.BindEventCallback("toggle_node", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (!args.empty()) this->ToggleNode(args[0].Get<int>());
    });

    constructor.BindEventCallback("select_node", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (!args.empty()) this->SelectNode(args[0].Get<int>());
    });
}

void SceneGraphViewModel::OnNotified(const SceneEvent& event) {
    // We only care about changes to the object hierarchy
    if (event.type == SceneEvent::HierarchyChanged) {
        // The Model has changed, so rebuild the ViewModel's flat list.
        RefreshFlatList(); 
    }
}

//UI Event Callbacks
void SceneGraphViewModel::ToggleNode(int node_id) {
    // 1. Flip the open state in the ViewModel's internal map
    node_open_states[node_id] = !node_open_states[node_id];
    // 2. Rebuild the flat list to reflect the new open/closed state
    RefreshFlatList();
}

void SceneGraphViewModel::SelectNode(int node_id) {
    selected_node_id.Set(node_id);
    scene_model.setActiveObjectIndex(node_id); 
}

void SceneGraphViewModel::RefreshFlatList() {
    flat_nodes.clear();
    // Traverse all root objects in the actual Scene model
    for (const auto* root_object : scene_model.getRootObjects())
        BuildFlatListRecursive(root_object, 0);
}

// Recursively traverses the real Scene hierarchy to populate the flat list.
void SceneGraphViewModel::BuildFlatListRecursive(const SceneObject* node, int depth) {
    if (!node)
        return;
    
    const int id = node->getId();
    const bool has_children = !node->getChildren().empty();
    
    // Get the 'is_open' state from the ViewModel's map, defaulting to true if not found
    bool is_open = true;
    if (node_open_states.count(id))
        is_open = node_open_states.at(id);
    else
        node_open_states[id] = true;  // Initialize state for a new node
    
    // Add the current node to the flat list
    flat_nodes.push_back({id, node->getName(), depth, has_children, is_open});
    
    // Recurse only if the node is open and has children
    if (is_open && has_children)
        for (const auto* child : node->getChildren())
            BuildFlatListRecursive(child, depth + 1);
}