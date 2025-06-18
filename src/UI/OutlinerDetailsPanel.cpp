#include "OutlinerDetailsPanel.h"
#include <imgui.h>

#include "Camera/InputTracker.h"
#include "Vulkan/Renderer.h"

class InputTracker;

OutlinerDetailsPanel::OutlinerDetailsPanel(Renderer& renderer, InputTracker& inputTracker) : renderer(renderer), inputTracker(inputTracker), selectedObjectIndex(-1)
{
}

void OutlinerDetailsPanel::renderUi() {
    ImGui::Begin("Outliner");

    for (size_t i = 0; i < renderer.sceneObjects.size(); ++i) {
        const auto& obj = renderer.sceneObjects[i];
        if (ImGui::Selectable((obj->name + "###" + std::to_string(i)).c_str(), selectedObjectIndex == (int)i))
            selectedObjectIndex = static_cast<int>(i);
    }

    ImGui::End();

    ImGui::Begin("Details");
    ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV);
    ImGui::TableSetupColumn("Label");
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

    if (selectedObjectIndex >= 0 && selectedObjectIndex < (int)renderer.sceneObjects.size() && renderer.sceneObjects[selectedObjectIndex]) {
        renderer.sceneObjects[selectedObjectIndex]->renderUi();

        // Example: Remove selected object if Delete key is pressed
        if (inputTracker.isKeyPressed(GLFW_KEY_DELETE)) {
            SceneObject* objPtr = renderer.sceneObjects[selectedObjectIndex].get();
            
            if (renderer.remove(objPtr)) // Clear selection to avoid invalid index
                selectedObjectIndex = -1;
        }
    } else {
        ImGui::TextUnformatted("No Object Selected.");
    }

    ImGui::EndTable();
    ImGui::End();
}

void OutlinerDetailsPanel::setSelectedIndex(int index)
{
    selectedObjectIndex = index;
}
