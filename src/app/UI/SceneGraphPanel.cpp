#include "SceneGraphPanel.h"
#include "Scene/SceneObject.h"
#include "Rendering/Camera/CameraInstance.h"
#include <imgui.h>
#include <algorithm>
#include <iterator>

SceneGraphPanel::SceneGraphPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void SceneGraphPanel::renderUi() {
    ImGui::Begin(name.c_str());

    // Recursively draw all root-level objects
    for (const auto& rootObject : scene.getRootObjects())
        drawNode(rootObject);

    // Drop target for unparenting that fills the remaining space.
    ImGui::Dummy(ImGui::GetContentRegionAvail());
    if (ImGui::IsItemClicked())
        scene.clearActiveObject();

    // The Dummy also serves as the drop target.
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
            IM_ASSERT(payload->DataSize == sizeof(SceneObjectHandle));
            scene.reparentObject(
                *static_cast<const SceneObjectHandle*>(payload->Data));
        }
        ImGui::EndDragDropTarget();
    }

    // Handle Copy, Paste, and Delete only when this window is focused
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))
    {
        const bool isCtrlDown = ImGui::IsKeyDown(ImGuiMod_Ctrl);
        const SceneObjectHandle activeObject = scene.getActiveObjectHandle();

        if (activeObject.isValid()) {
            // Copy (Ctrl+C)
            if (isCtrlDown && ImGui::IsKeyPressed(ImGuiKey_C))
                scene.copyObject(activeObject);

            // Deletion (Delete key)
            if (ImGui::IsKeyPressed(ImGuiKey_Delete))
                scene.removeObject(activeObject);
        }

        // Paste (Ctrl+V) - can happen even if no object is selected
        if (isCtrlDown && ImGui::IsKeyPressed(ImGuiKey_V))
            scene.paste();
    }

    ImGui::End();
}

void SceneGraphPanel::drawNode(const std::shared_ptr<SceneObject>& node) {
    if (!node)
        return;

    const SceneObjectHandle objectHandle = node->getHandle();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;

    if (objectHandle == scene.getActiveObjectHandle())
        flags |= ImGuiTreeNodeFlags_Selected;
    
    if (node->getChildren().empty())
        flags |= ImGuiTreeNodeFlags_Leaf;

    ImGui::PushID(node.get());
    if (auto* camera = dynamic_cast<CameraInstance*>(node.get())) {
        const bool isActiveCamera = camera == scene.getActiveCamera();
        if (ImGui::RadioButton("##ActiveCamera", isActiveCamera) && !isActiveCamera)
            scene.setActiveCamera(camera);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(isActiveCamera ? "Active render camera" : "Set as active render camera");
        ImGui::SameLine();
    }

    const std::string label = node->getName() + "##SceneObject"
        + std::to_string(objectHandle.index());
    const bool node_open = ImGui::TreeNodeEx(label.c_str(), flags);

    // Handle selection by finding the object's index
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        scene.setActiveObject(objectHandle);
    }
    
    // Drag & Drop Source
    if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload("SCENE_OBJECT", &objectHandle, sizeof(objectHandle));
        ImGui::Text("Reparent %s", node->getName().c_str());
        ImGui::EndDragDropSource();
    }

    // Drag & Drop Target
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
            IM_ASSERT(payload->DataSize == sizeof(SceneObjectHandle));
            scene.reparentObject(
                *static_cast<const SceneObjectHandle*>(payload->Data), objectHandle);
        }
        ImGui::EndDragDropTarget();
    }

    // If the node is open, recursively draw its children
    if (node_open) {
        for (const auto& child : node->getChildren())
            drawNode(child);
        ImGui::TreePop();
    }
    ImGui::PopID();
}
