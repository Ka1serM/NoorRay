#include "OutlinerDetailsPanel.h"
#include <imgui.h>

#include "Vulkan/Renderer.h"

class InputTracker;

OutlinerDetailsPanel::OutlinerDetailsPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene)
{
}

void OutlinerDetailsPanel::renderUi() {
    int activeIndex = scene.getActiveObjectIndex();

    ImGui::Begin("Outliner");
    
    for (size_t i = 0; i < scene.getSceneObjects().size(); ++i) {
        const auto& obj = scene.getSceneObjects()[i];
        if (ImGui::Selectable((obj->name + "###" + std::to_string(i)).c_str(), activeIndex == static_cast<int>(i)))
            scene.setActiveObjectIndex(static_cast<int>(i)); // Set the active index in the scene
    }

    // Check for delete key press only when the outliner window is focused
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Delete) && activeIndex != -1)
        if (SceneObject* activeObject = scene.getActiveObject())
            if (scene.remove(activeObject))
                scene.setActiveObjectIndex(-1);

    ImGui::End();

    ImGui::Begin("Details");
    
    if (ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("Label");
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        // Use the scene's active object for rendering details
        if (SceneObject* activeObject = scene.getActiveObject())
            activeObject->renderUi();
        else 
            ImGui::TextUnformatted("No Object Selected.");
        ImGui::EndTable();
    }
    ImGui::End();
}