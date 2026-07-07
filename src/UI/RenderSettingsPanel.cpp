#include "UI/RenderSettingsPanel.h"

#include <cfloat>
#include <utility>

#include <imgui.h>

#include "Scene/RenderSettings.h"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

RenderSettingsPanel::RenderSettingsPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene)
{
}

void RenderSettingsPanel::renderUi()
{
    RenderSettings& settings = scene.getRenderSettings();
    bool changed = false;

    ImGui::Begin(name.c_str());
    if (ImGui::BeginTable(
        "RenderSettingsTable", 2,
        ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoBordersInBody))
    {
        ImGui::TableSetupColumn("Label");
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);

        ImGuiManager::dragFloatRow(
            "Exposure", settings.exposure, 0.01f, -100.f, 100.f,
            [&](const float value) { settings.exposure = value; changed = true; });

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Tonemapping");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox(
            "##Tonemapping", reinterpret_cast<bool*>(&settings.tonemappingEnabled));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Samples Per Pixel");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragInt("##SamplesDrag", &settings.samples, 0.1f, 1, 64, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Max Samples");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragInt("##MaxSamples", &settings.maxSamples, 1.f, 1, 100000, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Max Bounces");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragInt("##MaxBounces", &settings.maxBounces, 0.1f, 1, 64, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("RR Start Bounce");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragInt("##RRBounce", &settings.russianRouletteStartBounce, 0.1f, 0, 16, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Transparent Background");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##TransparentBg", reinterpret_cast<bool*>(&settings.transparentBackground));

        static constexpr const char* kBufferVisNames[] = {
            "Beauty", "Albedo", "Normal", "Cryptomatte", "Position"
        };
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Buffer View");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        int bufVis = static_cast<int>(settings.bufferVisualization);
        if (ImGui::Combo("##BufferVis", &bufVis, kBufferVisNames, IM_ARRAYSIZE(kBufferVisNames)))
        {
            settings.bufferVisualization = static_cast<BufferVisualization>(bufVis);
            changed = true;
        }

        ImGui::EndTable();
    }

    if (changed)
        scene.setDirtyFlag(Accumulation);
    ImGui::End();
}
