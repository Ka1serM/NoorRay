#include "OutlinerDetailsPanel.h"
#include "Scene/Scene.h"
#include "Scene/SceneObject.h"
#include <imgui.h>
#include <algorithm>
#include <iterator>

OutlinerDetailsPanel::OutlinerDetailsPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void OutlinerDetailsPanel::renderUi() {
    ImGui::Begin("Outliner");

    // Recursively draw all root-level objects
    for (SceneObject* rootObject : scene.getRootObjects())
        drawNode(rootObject);

    // Drop target for unparenting that fills the remaining space.
    ImGui::Dummy(ImGui::GetContentRegionAvail());
    if (ImGui::IsItemClicked())
        scene.resetActiveObjectIndex();

    // The Dummy also serves as the drop target.
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
            IM_ASSERT(payload->DataSize == sizeof(SceneObject*));
            SceneObject* payload_node = *static_cast<SceneObject**>(payload->Data);
            scene.reparent(payload_node, nullptr);
        }
        ImGui::EndDragDropTarget();
    }

    // Handle Copy, Paste, and Delete
    if (ImGui::IsWindowFocused()) {
        const bool isCtrlDown = ImGui::IsKeyDown(ImGuiMod_Ctrl);
        SceneObject* activeObject = scene.getActiveObject();

        // Copy (Ctrl+C)
        if (isCtrlDown && ImGui::IsKeyPressed(ImGuiKey_C))
            scene.copy(activeObject);
        
        // Paste (Ctrl+V)
        if (isCtrlDown && ImGui::IsKeyPressed(ImGuiKey_V))
            scene.paste();

        // Deletion (Delete key)
        if (ImGui::IsKeyPressed(ImGuiKey_Delete))
            scene.remove(activeObject);
    }

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

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
    
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
            const uint32_t index = std::distance(allObjects.begin(), it);
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