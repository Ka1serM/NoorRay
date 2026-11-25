#include "NoorRay.App/UI/Controllers/RenderSettingsController.h"

#include <algorithm>
#include <array>
#include <format>
#include <string_view>

#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Gpu/GaussianAccelerationStructure.h"
#include "NoorRay/Render/RenderOptions.h"
#include "NoorRay/Render/Upscale/DlssRr.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

//! The denoiser names of the add-on, in the order of Render::Denoiser.
constexpr std::array<std::string_view, 3> denoiserNames{"off", "nrd", "openDlssRr"};
constexpr std::array<std::string_view, 3> reuseNames{"off", "di", "pt"};
constexpr std::array<std::string_view, 3> gaussianShadingNames{"off", "emissive", "scattering"};
constexpr std::array<std::string_view, 5> upscaleQualityNames{"native", "quality", "balanced", "performance", "ultraPerformance"};

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
    model.bindEvent("renderMode", &RenderSettingsController::onApply, this);
    model.bindEvent("renderApply", &RenderSettingsController::onApply, this);
    model.bindEvent("renderRestart", &RenderSettingsController::onRestart, this);
    app.renderSettings.openDlssRrBuilt = Render::DlssRr::isBuilt();
    if (!app.renderSettings.openDlssRrBuilt)
        app.renderSettings.denoiser = "nrd";
    app.renderSettings.gaussianTracingBuilt = Gpu::GaussianAccelerationStructure::isAvailable();
    //! The editor starts with the defaults of the Blender add-on and of Cycles.
    apply(read());
}

Render::RenderOptions RenderSettingsController::readOptions() const {
    const RenderSettingsModel& shown = app.renderSettings;
    Render::RenderOptions options;
    options.samples = static_cast<std::uint32_t>(std::max(shown.maxFrames, 0));
    options.denoiser = valueOf<Render::Denoiser>(denoiserNames, shown.denoiser);
    options.lightTree = shown.lightTree;
    options.visibilityCache = shown.visibilityCache;
    options.gaussianShading = valueOf<Render::GaussianShading>(gaussianShadingNames, shown.gaussianShading);
    options.seed = static_cast<std::uint32_t>(std::max(shown.seed, 0));
    options.temporal = shown.temporal;
    options.reuse = valueOf<Render::Reuse>(reuseNames, shown.reuse);
    options.radianceCache = shown.radianceCache;
    options.upscale = shown.upscale;
    options.renderScale = Render::upscaleQualityScales[static_cast<std::size_t>(valueOf<Render::UpscaleQuality>(upscaleQualityNames, shown.upscaleQuality))];
    options.sharpness = std::clamp<float>(shown.sharpness, 0.0f, 1.0f);
    const auto bounces = [](const int value) { return static_cast<std::uint32_t>(std::clamp(value, 0, 1024)); };
    options.maxBounces = bounces(shown.maxBounces);
    options.diffuseBounces = bounces(shown.diffuseBounces);
    options.glossyBounces = bounces(shown.glossyBounces);
    options.transmissionBounces = bounces(shown.transmissionBounces);
    options.volumeBounces = bounces(shown.volumeBounces);
    options.transparentBounces = bounces(shown.transparentBounces);
    options.clampDirect = std::max<float>(shown.clampDirect, 0.0f);
    options.clampIndirect = std::max<float>(shown.clampIndirect, 0.0f);
    options.filterGlossy = std::max<float>(shown.filterGlossy, 0.0f);
    options.exposure = std::max<float>(shown.filmExposure, 0.0f);
    options.filterWidth = std::clamp<float>(shown.pixelFilter, 0.01f, 10.0f);
    return options;
}

Render::RendererSettings RenderSettingsController::read() const {
    Render::RendererSettings settings = readOptions().toRendererSettings();
    settings.display = session.getRenderer().getSettings().display;
    if (app.renderSettings.mode == "raster")
        settings.shading = Render::Shading::Raster;
    return settings;
}

void RenderSettingsController::apply(const Render::RendererSettings& settings) {
    //! RELAX and FSR keep their history over camera moves, so a smaller viewport while moving only discards it.
    const Render::RendererSettings validated = settings.validated(false);
    const bool temporal = validated.temporal && (validated.denoiser != Render::Denoiser::Off || validated.upscaler != Render::Upscaler::Off);
    viewports.setMotionDivisor(temporal ? 1 : app.renderSettings.motionDivisor);
    session.getRenderer().setSettings(settings);
    model.dirty("renderSettings");
    viewports.invalidate();
}

void RenderSettingsController::onApply(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "renderApply");
    apply(read());
}

void RenderSettingsController::onRestart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "renderRestart");
    session.getRenderer().restartAccumulation();
    viewports.invalidate();
}

}
