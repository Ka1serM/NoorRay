#include "UI/RenderSettingsPanel.h"

#include <cfloat>
#include <utility>

#include <imgui.h>

#include "Scene/RenderSettings.h"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"
#include "UI/MathInput.h"

namespace
{
BufferVisualization restoreBuffer(
    const std::optional<BufferVisualization> previous,
    const RenderSettings& settings)
{
    const BufferVisualization candidate =
        previous.value_or(BufferVisualization::Beauty);
    if (candidate == BufferVisualization::ProxyOverdraw
        && !settings.gaussianProxyOverdrawVisualization)
        return BufferVisualization::Beauty;
    return candidate;
}
}

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
        changed |= MathInput::DragInt("##SamplesDrag", &settings.samples, 0.1f, 1, 64, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Max Samples");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= MathInput::DragInt("##MaxSamples", &settings.maxSamples, 1.f, 1, 100000, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("AOVs During Camera Motion");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##AovEnabled", &settings.aovEnabled);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Max Bounces");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= MathInput::DragInt("##MaxBounces", &settings.maxBounces, 0.1f, 1, 64, "%d");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Indirect Light Clamp");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= MathInput::DragFloat(
            "##IndirectLightClamp", &settings.indirectLightClamp,
            0.1f, 0.0f, 100000.0f, "%.2f",
            ImGuiSliderFlags_AlwaysClamp);

        static constexpr const char* kProxyNames[] =
            { "Icosphere", "Octahedron", "Icosahedron", "Icosphere (Level 2)" };
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
        if (MathInput::SliderFloat("##CutoffSigma", &settings.gaussianCutoffSigma, 1.0f, 6.0f, "%.1f"))
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
            scene.setDirtyFlag(GaussianData);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Proxy Overdraw");
        ImGui::TableSetColumnIndex(1);
        if (ImGui::Checkbox("##ProxyOverdraw",
                &settings.gaussianProxyOverdrawVisualization))
        {
            if (settings.gaussianProxyOverdrawVisualization) {
                previousProxyOverdrawBuffer =
                    settings.bufferVisualization
                        == BufferVisualization::ProxyOverdraw
                    ? BufferVisualization::Beauty
                    : settings.bufferVisualization;
                settings.bufferVisualization =
                    BufferVisualization::ProxyOverdraw;
            } else {
                if (settings.bufferVisualization
                    == BufferVisualization::ProxyOverdraw)
                    settings.bufferVisualization = restoreBuffer(
                        previousProxyOverdrawBuffer, settings);
                previousProxyOverdrawBuffer.reset();
            }
            changed = true;
        }

        if (settings.gaussianProxyOverdrawVisualization)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Overdraw Range");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= MathInput::SliderInt(
                "##ProxyOverdrawMax", &settings.gaussianProxyOverdrawMax, 1, 1024);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Transparent Background");
        ImGui::TableSetColumnIndex(1);
        changed |= ImGui::Checkbox("##TransparentBg", &settings.transparentBackground);

        static constexpr const char* kBufferVisNames[] = {
            "Beauty", "Albedo", "Normal", "Cryptomatte", "Position",
            "Proxy Overdraw"
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
