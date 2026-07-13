#include "UI/RenderSettingsPanel.h"

#include <cfloat>
#include <utility>

#include <imgui.h>

#include "Raytracing/GaussianProxyBlas.h"
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
        changed |= ImGui::Checkbox("##Tonemapping", &settings.tonemappingEnabled);

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
        ImGui::TextUnformatted("Noise Limit");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##NoiseLimitEnabled", &settings.noiseLimitEnabled);

        if (settings.noiseLimitEnabled)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Noise Level");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= ImGui::SliderFloat(
                "##NoiseLevel", &settings.noiseLevel, 0.000001f, 0.1f, "%.6f",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        }

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

        static constexpr const char* kProxyNames[] =
            { "Icosahedron", "Octahedron", "Triangular Bipyramid" };
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Gaussian Proxy");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        int proxyType = static_cast<int>(settings.gaussianProxyType);
        if (ImGui::Combo("##GaussianProxy", &proxyType, kProxyNames, IM_ARRAYSIZE(kProxyNames)))
        {
            settings.gaussianProxyType = static_cast<GaussianProxyType>(proxyType);
            changed = true;
            scene.setDirtyFlag(TLAS);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Cutoff Sigma");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::SliderFloat("##CutoffSigma", &settings.gaussianCutoffSigma, 1.0f, 6.0f, "%.1f"))
        {
            changed = true;
            scene.setDirtyFlag(TLAS);
        }

        static constexpr const char* kGaussianShadingNames[] =
            { "Global Illumination", "Direct Color" };
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Gaussian Shading");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        int gaussianShading = static_cast<int>(settings.gaussianShadingMode);
        if (ImGui::Combo(
            "##GaussianShading", &gaussianShading,
            kGaussianShadingNames, IM_ARRAYSIZE(kGaussianShadingNames)))
        {
            settings.gaussianShadingMode =
                static_cast<GaussianShadingMode>(gaussianShading);
            changed = true;
        }

        static constexpr const char* kSphericalHarmonicsNames[] =
            { "Degree 0", "Degree 1", "Degree 2", "Degree 3" };
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("SH Render Limit");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        int renderOrder = static_cast<int>(settings.gaussianRenderSphericalHarmonics);
        if (ImGui::Combo("##GaussianShRender", &renderOrder, kSphericalHarmonicsNames,
            IM_ARRAYSIZE(kSphericalHarmonicsNames)))
        {
            settings.gaussianRenderSphericalHarmonics = static_cast<SphericalHarmonicsOrder>(renderOrder);
            changed = true;
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Proxy Overdraw");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##ProxyOverdraw", &settings.gaussianProxyOverdrawVisualization);

        if (settings.gaussianProxyOverdrawVisualization)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Overdraw Range");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= ImGui::SliderInt(
                "##ProxyOverdrawMax", &settings.gaussianProxyOverdrawMax, 1, 1024);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Transparent Background");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##TransparentBg", &settings.transparentBackground);

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
