#include "OutlinerDetailsPanel.h"
#include <imgui.h>

#include "Vulkan/Renderer.h"

class InputTracker;

OutlinerDetailsPanel::OutlinerDetailsPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene)
{
}

void OutlinerDetailsPanel::renderUi() {
    int activeIndex = scene.getActiveObjectIndex();

    // Outliner list
    ImGui::Begin("Outliner");
    for (size_t i = 0; i < scene.getSceneObjects().size(); ++i) {
        const auto& obj = scene.getSceneObjects()[i];
        if (ImGui::Selectable((obj->name + "###" + std::to_string(i)).c_str(), activeIndex == static_cast<int>(i)))
            scene.setActiveObjectIndex(static_cast<int>(i));
    }

    // Delete key
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Delete) && activeIndex != -1)
        if (SceneObject* activeObject = scene.getActiveObject())
            if (scene.remove(activeObject))
                scene.setActiveObjectIndex(-1);

    ImGui::End();

    // Details panel
    ImGui::Begin("Details");

    if (ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp)) {
        if (SceneObject* activeObject = scene.getActiveObject())
            activeObject->renderUi();//object renders its own details
        else
            ImGui::TextUnformatted("No Object Selected.");
        ImGui::EndTable();
    }

    ImGui::End();
}