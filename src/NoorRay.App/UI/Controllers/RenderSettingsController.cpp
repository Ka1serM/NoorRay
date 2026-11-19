#include "NoorRay.App/UI/Controllers/RenderSettingsController.h"

#include <algorithm>
#include <array>
#include <format>
#include <string_view>

#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

constexpr std::array<std::string_view, 2> gaussianShadingNames{"emissive", "scattering"};
constexpr std::array<std::string_view, 3> toneMappingNames{"linear", "neutral", "agx"};

template <class Enum, std::size_t count>
Rml::String nameOf(const std::array<std::string_view, count>& names, const Enum value) {
    return Rml::String(names[static_cast<std::size_t>(value)]);
}

template <class Enum, std::size_t count>
Enum valueOf(const std::array<std::string_view, count>& names, const Rml::String& name) {
    const auto found = std::ranges::find(names, std::string_view(name));
    if (found == names.end())
        RmlKit::UI::fatal(std::format("the render setting value {} is unknown", name));
    return static_cast<Enum>(found - names.begin());
}

}

RenderSettingsController::RenderSettingsController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports)
    : model(model), app(app), session(session), viewports(viewports) {
    model.bindEvent("renderMode", &RenderSettingsController::onMode, this);
    model.bindEvent("renderApply", &RenderSettingsController::onApply, this);
    model.bindEvent("renderRestart", &RenderSettingsController::onRestart, this);
    show(session.getRenderer().getSettings());
}

void RenderSettingsController::show(const Render::RendererSettings& settings) {
    RenderSettingsModel& shown = app.renderSettings;
    shown.mode = settings.shading == Render::Shading::Raster ? "raster" : "pathTraced";
    shown.tracedCameraRays = settings.primaryVisibility == Render::PrimaryVisibility::Traced;
    shown.pathReuse = settings.reuse != Render::Reuse::Off;
    shown.accumulate = settings.accumulate;
    shown.maxSamples = static_cast<int>(settings.maxSamples);
    shown.maxDepth = static_cast<int>(settings.wavefront.maxDepth);
    shown.rouletteStart = static_cast<int>(settings.wavefront.rouletteStart);
    shown.indirectClamp = settings.wavefront.indirectClamp;
    shown.gaussianShading = nameOf(gaussianShadingNames, settings.wavefront.gaussianShading);
    shown.toneMapping = nameOf(toneMappingNames, settings.display.toneMapping);
    shown.exposure = settings.display.exposure;
    model.dirty("renderSettings");
}

Render::RendererSettings RenderSettingsController::read() const {
    const RenderSettingsModel& shown = app.renderSettings;
    Render::RendererSettings settings = session.getRenderer().getSettings();
    const bool raster = shown.mode == "raster";
    settings.shading = raster ? Render::Shading::Raster : Render::Shading::PathTraced;
    settings.primaryVisibility = !raster && shown.tracedCameraRays ? Render::PrimaryVisibility::Traced : Render::PrimaryVisibility::Raster;
    settings.reuse = !raster && shown.pathReuse ? Render::Reuse::DiPt : Render::Reuse::Off;
    settings.accumulate = shown.accumulate;
    settings.maxSamples = static_cast<std::uint32_t>(std::max(shown.maxSamples, 0));
    settings.wavefront.maxDepth = static_cast<std::uint32_t>(std::clamp(shown.maxDepth, 1, 256));
    settings.wavefront.rouletteStart = static_cast<std::uint32_t>(std::max(shown.rouletteStart, 0));
    settings.wavefront.indirectClamp = std::max<float>(shown.indirectClamp, 0.0f);
    settings.wavefront.gaussianShading = valueOf<Render::GaussianShading>(gaussianShadingNames, shown.gaussianShading);
    settings.display.toneMapping = valueOf<Render::ToneMapping>(toneMappingNames, shown.toneMapping);
    settings.display.exposure = shown.exposure;
    return settings;
}

void RenderSettingsController::apply(const Render::RendererSettings& settings) {
    session.getRenderer().setSettings(settings);
    show(settings);
    viewports.invalidate();
}

void RenderSettingsController::onMode(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "renderMode");
    const Render::RendererSettings current = session.getRenderer().getSettings();
    Render::RendererSettings settings = app.renderSettings.mode == "raster" ? Render::RendererSettings::viewport() : Render::RendererSettings::pathTraced();
    settings.display = current.display;
    settings.maxSamples = current.maxSamples;
    apply(settings);
}

void RenderSettingsController::onApply(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "renderApply");
    viewports.setMotionDivisor(app.renderSettings.motionDivisor);
    apply(read());
}

void RenderSettingsController::onRestart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "renderRestart");
    session.getRenderer().restartAccumulation();
    viewports.invalidate();
}

}
