#include "SceneGraphViewModel.h"
#include <map>
#include <Scene/SceneObject.h>

#include "RmlUi/Core/Element.h"
#include "RmlUi/Core/Event.h"
#include "RmlUi/Core/EventListener.h"

SceneGraphViewModel::SceneGraphViewModel(Scene& scene, Rml::Context* context, const Rml::String& model_name)
    : ViewModelBase(context, model_name), sceneModel(scene)
{
    // Register as an observer to receive updates from the Scene model.
    sceneModel.AddObserver(this);

    // Register the FlatNodeData struct for data binding with RmlUi.
    if (auto handle = data_model_.RegisterStruct<FlatNodeData>()) {
        handle.RegisterMember("id", &FlatNodeData::id);
        handle.RegisterMember("name", &FlatNodeData::name);
        handle.RegisterMember("depth", &FlatNodeData::depth);
        handle.RegisterMember("has_children", &FlatNodeData::has_children);
        handle.RegisterMember("is_open", &FlatNodeData::is_open);
    }

    // Register the array type for our observable collection.
    data_model_.RegisterArray<std::vector<FlatNodeData>>();

    // Bind the observable properties and collections to the data model.
    Bind("flat_nodes", flatNodes);
    Bind("selected_node_id", selectedNodeId);

    // Bind UI actions to member functions.
    BindAction("toggle_node", [this](Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& args) {
        if (!args.empty()) toggleNode(args[0].Get<int>());
        // Stop the event from bubbling up to the parent's 'select_node' handler.
        event.StopPropagation();
    });

    BindAction("select_node", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        if (!args.empty()) selectNode(args[0].Get<int>());
    });

    // Perform the initial population of the tree view.
    refreshFlatList();
    // Sync initial selection state
    selectedNodeId = sceneModel.getActiveObjectId();
    
    context->GetRootElement()->AddEventListener("keydown", this);
}

void SceneGraphViewModel::ProcessEvent(Rml::Event& event) {
    if (event.GetType() == "keydown") {
        const int key = event.GetParameter<int>("key_identifier", 0);
        if (key == Rml::Input::KI_DELETE && selectedNodeId.Get() != INVALID_INSTANCE) {
            deleteNode(selectedNodeId.Get());
            event.StopPropagation();
        }
    }
}

void SceneGraphViewModel::deleteNode(int node_id) {
    if (auto* obj = sceneModel.getObject(node_id)) {
        sceneModel.remove(obj);  // remove from scene

        // Remove it from the open state map
        nodeOpenStates.erase(node_id);

        // Refresh UI
        refreshFlatList();

        // Reset selection if this node was selected
        if (selectedNodeId.Get() == node_id)
            selectedNodeId = INVALID_INSTANCE;
    }
}


void SceneGraphViewModel::OnNotified(const SceneEvent& event) {
    // This function is called whenever the Scene model changes.
    switch (event.type) {
        case SceneEvent::HierarchyChanged:
            // If objects were added, removed, or reparented, rebuild the entire list.
            refreshFlatList();
            break;

        case SceneEvent::SelectionChanged:
            selectedNodeId = sceneModel.getActiveObjectId();
            break;
        
        default:
            break;
    }
}

void SceneGraphViewModel::toggleNode(const int node_id) {
    // Flip the open/closed state for the given node and refresh the list.
    nodeOpenStates[node_id] = !nodeOpenStates[node_id];
    refreshFlatList();
}

void SceneGraphViewModel::selectNode(const int node_id) const
{
    sceneModel.setActiveObject(node_id);
}

void SceneGraphViewModel::refreshFlatList() {
    flatNodes.clear();
    for (const auto* root_object : sceneModel.getRootObjects())
        buildFlatListRecursive(root_object, 0);
}

void SceneGraphViewModel::buildFlatListRecursive(const SceneObject* node, int depth) {
    if (!node) return;

    const int id = node->getId();
    const bool has_children = !node->getChildren().empty();

    // Check the open state for this node, defaulting to 'true' if not found.
    const auto it = nodeOpenStates.find(id);
    bool is_open = true;
    if (it != nodeOpenStates.end()) {
        is_open = it->second;
    } else
        nodeOpenStates[id] = true; // If the node wasn't in our map, add it with the default open state.

    // Add the node to our flat list for the UI.
    flatNodes.push_back({id, Rml::String(node->getName()), depth, has_children, is_open});

    // If the node is open and has children, recurse into them.
    if (is_open && has_children)
        for (const auto* child : node->getChildren())
            buildFlatListRecursive(child, depth + 1);
}

