#include "NoorRay.App/UI/Controllers/StatusController.h"

#include <format>
#include <utility>

#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

constexpr double smoothing = 0.1;

//! Every change to a bound text lays out the whole document again, so the status bar takes new values at a fixed rate and not on every frame.
constexpr std::chrono::milliseconds publishInterval{100};

constexpr std::chrono::milliseconds timingInterval{500};

Rml::String formatBytes(const std::size_t bytes) {
    if (bytes >= std::size_t{1} << 30)
        return std::format("{:.2f} GiB", static_cast<double>(bytes) / static_cast<double>(std::size_t{1} << 30));
    return std::format("{:.1f} MiB", static_cast<double>(bytes) / static_cast<double>(std::size_t{1} << 20));
}

Rml::String formatCount(const std::uint64_t count) {
    if (count >= 1'000'000)
        return std::format("{:.2f} M", static_cast<double>(count) / 1e6);
    return std::format("{}", count);
}

}

StatusController::StatusController(RmlKit::UI::Model& model, AppModel& app, const Session::Session& session, ViewportElementInstancer& viewports)
    : model(model), app(app), session(session) {
    viewports.getEvents().frame = [this] { frame(); };
    viewports.getEvents().error = [this](const Rml::String& message) { fail(message); };
}

void StatusController::frame() {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - std::exchange(lastFrame, now)).count();
    if (seconds < 1.0)
        smoothedSeconds = smoothedSeconds == 0.0 ? seconds : smoothedSeconds + (seconds - smoothedSeconds) * smoothing;
    const Render::Renderer& renderer = session.getRenderer();
    StatusModel& status = app.status;
    const int samples = static_cast<int>(renderer.getSampleCount());
    const bool restarted = samples < status.samples;
    std::string compiling = renderer.getCompileNotice();
    if (now - lastPublish < publishInterval && !restarted && renderer.isComplete() == status.complete && compiling == status.compiling)
        return;
    lastPublish = now;
    const Gpu::GpuSceneStats stats = session.getGpuScene().getStats();
    status.samples = samples;
    status.maxSamples = static_cast<int>(renderer.getSettings().maxSamples);
    status.complete = renderer.isComplete();
    status.compiling = std::move(compiling);
    if (now - lastTiming >= timingInterval) {
        lastTiming = now;
        status.frameTime = smoothedSeconds > 0.0 ? std::format("{:.1f} ms", smoothedSeconds * 1e3) : Rml::String();
        status.framesPerSecond = smoothedSeconds > 0.0 ? std::format("{:.0f} fps", 1.0 / smoothedSeconds) : Rml::String();
    }
    status.resolution = std::format("{} × {}", renderer.getWidth(), renderer.getHeight());
    status.meshes = static_cast<int>(stats.meshes);
    status.instances = static_cast<int>(stats.instances);
    status.materials = static_cast<int>(stats.materials);
    status.lights = static_cast<int>(stats.lights);
    status.gaussians = formatCount(stats.gaussians);
    status.meshMemory = formatBytes(stats.meshBytes);
    status.textureMemory = formatBytes(stats.textureBytes);
    status.error.clear();
    model.dirty("status");
}

void StatusController::fail(const Rml::String& message) {
    app.status.error = message;
    model.report(RmlKit::UI::ReportLevel::Error, "renderer", message);
    model.dirty("status");
}

}
