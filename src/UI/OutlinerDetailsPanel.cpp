#include "OutlinerDetailsPanel.h"
#include <imgui.h>

#include "Vulkan/Renderer.h"

class InputTracker;

OutlinerDetailsPanel::OutlinerDetailsPanel(Scene& scene) : scene(scene)
{
}

void OutlinerDetailsPanel::renderUi() {
    ImGui::Begin("Outliner");
    bool hovered = ImGui::IsWindowFocused();
    int activeIndex = scene.getActiveObjectIndex();

    for (size_t i = 0; i < scene.getSceneObjects().size(); ++i) {
        const auto& obj = scene.getSceneObjects()[i];
        if (ImGui::Selectable((obj->name + "###" + std::to_string(i)).c_str(), activeIndex == (int)i))
            scene.setActiveObjectIndex(static_cast<int>(i)); // Set the active index in the scene
    }

    ImGui::End();

    ImGui::Begin("Details");
    
    if (ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupColumn("Label");
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        // Use the scene's active object for rendering details
        SceneObject* activeObject = scene.getActiveObject();
        if (activeObject) {
            activeObject->renderUi();

            if (hovered && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
                if (scene.remove(activeObject))
                    scene.setActiveObjectIndex(-1);
            }
        } else {
            ImGui::TextUnformatted("No Object Selected.");
        }
        ImGui::EndTable();
    }
    ImGui::End();
}