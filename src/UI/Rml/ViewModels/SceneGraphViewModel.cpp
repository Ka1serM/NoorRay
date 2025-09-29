#include "SceneGraphViewModel.h"
#include <map>
#include <Scene/SceneObject.h>

SceneGraphViewModel::SceneGraphViewModel(Scene& scene, Rml::Context* context, const Rml::String& model_name)
    : ViewModelBase(context, model_name), scene_model(scene), selected_node_id(-1)
{
    // Register as an observer
    scene_model.AddObserver(this);

    // Register FlatNodeData struct
    if (auto handle = data_model_.RegisterStruct<FlatNodeData>()) {
        handle.RegisterMember("id", &FlatNodeData::id);
        handle.RegisterMember("name", &FlatNodeData::name);
        handle.RegisterMember("depth", &FlatNodeData::depth);
        handle.RegisterMember("has_children", &FlatNodeData::has_children);
        handle.RegisterMember("is_open", &FlatNodeData::is_open);
    }

    // Register array for ObservableCollection
    data_model_.RegisterArray<std::vector<FlatNodeData>>();

    // Bind Observables using your base class helpers
    Bind("flat_nodes", flat_nodes);
    Bind("selected_node_id", selected_node_id);

    // Bind UI actions
    BindAction("toggle_node", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        int node_id = args[0].Get<int>();
        ToggleNode(node_id);
    });

    BindAction("select_node", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
        int node_id = args[0].Get<int>();
        SelectNode(node_id);
    });

    // Initial population
    RefreshFlatList();
}

void SceneGraphViewModel::OnNotified(const SceneEvent& event) {
    if (event.type == SceneEvent::HierarchyChanged)
        RefreshFlatList();
}

void SceneGraphViewModel::ToggleNode(int node_id) {
    node_open_states[node_id] = !node_open_states[node_id];
    RefreshFlatList();
}

void SceneGraphViewModel::SelectNode(int node_id) {
    selected_node_id = node_id;
    scene_model.setActiveObjectIndex(node_id);
}

void SceneGraphViewModel::RefreshFlatList() {
    flat_nodes.clear();
    for (const auto* root_object : scene_model.getRootObjects()) {
        BuildFlatListRecursive(root_object, 0);
    }
}

void SceneGraphViewModel::BuildFlatListRecursive(const SceneObject* node, int depth) {
    if (!node) return;

    const int id = node->getId();
    const bool has_children = !node->getChildren().empty();

    bool is_open = true;
    if (node_open_states.contains(id))
        is_open = node_open_states.at(id);
    else
        node_open_states[id] = true;

    flat_nodes.push_back({id, node->getName(), depth, has_children, is_open});

    if (is_open && has_children)
        for (const auto* child : node->getChildren())
            BuildFlatListRecursive(child, depth + 1);
}