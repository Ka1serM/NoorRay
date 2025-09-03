#include "OutlinerDetailsPanel.h"
#include "Scene/Scene.h"
#include "Scene/SceneObject.h"
#include <imgui.h>
#include <algorithm>
#include <iterator>

OutlinerDetailsPanel::OutlinerDetailsPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void OutlinerDetailsPanel::renderUi() {
    // Outliner Panel
    ImGui::Begin("Outliner");

    // Recursively draw all root-level objects
    for (SceneObject* rootObject : scene.getRootObjects())
        drawNode(rootObject);

    // Drop target for unparenting that fills the remaining space.
    ImGui::Dummy(ImGui::GetContentRegionAvail());
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
            IM_ASSERT(payload->DataSize == sizeof(SceneObject*));
            SceneObject* payload_node = *static_cast<SceneObject**>(payload->Data);
            scene.reparent(payload_node, nullptr);
        }
        ImGui::EndDragDropTarget();
    }

    // Handle deletion with key
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Delete))
        if (SceneObject* activeObject = scene.getActiveObject())
            if (scene.remove(activeObject))
                scene.setActiveObjectIndex(-1);

    ImGui::End();

    // Details Panel
    ImGui::Begin("Details");
    if (ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp)) {
        
        if (SceneObject* activeObject = scene.getActiveObject())
            activeObject->renderUi(); // Object renders its own details
        else
            ImGui::TextUnformatted("No Object Selected.");
        
        ImGui::EndTable();
    }
    ImGui::End();
}

void OutlinerDetailsPanel::drawNode(SceneObject* node) {
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;

    if (node == scene.getActiveObject())
        flags |= ImGuiTreeNodeFlags_Selected;

    if (node->getChildren().empty())
        flags |= ImGuiTreeNodeFlags_Leaf;

    const bool node_open = ImGui::TreeNodeEx(node, flags, "%s", node->getName().c_str());

    // Handle selection by finding the object's index
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        const auto& allObjects = scene.getSceneObjects();
        const auto it = std::ranges::find_if(allObjects, [node](const auto& ptr) { return ptr.get() == node; });
        if (it != allObjects.end()) {
            const int index = std::distance(allObjects.begin(), it);
            scene.setActiveObjectIndex(index);
        }
    }
    
    // Drag & Drop
    if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload("SCENE_OBJECT", &node, sizeof(SceneObject*));
        ImGui::Text("Reparent %s", node->getName().c_str());
        ImGui::EndDragDropSource();
    }

    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
            IM_ASSERT(payload->DataSize == sizeof(SceneObject*));
            SceneObject* payload_node = *static_cast<SceneObject**>(payload->Data);
            scene.reparent(payload_node, node);
        }
        ImGui::EndDragDropTarget();
    }

    // If the node is open, recursively draw its children
    if (node_open) {
        
        for (SceneObject* child : node->getChildren())
            drawNode(child);
        
        ImGui::TreePop();
    }
}