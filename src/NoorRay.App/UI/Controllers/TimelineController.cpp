#include "NoorRay.App/UI/Controllers/TimelineController.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

constexpr int maxLabels = 12;
constexpr int minorPerMajor = 5;

//! The smallest of 1, 2, 5, 10, 20, 50, ... frames that keeps the labels at most maxLabels.
int labelStep(const int span) {
    for (int scale = 1;; scale *= 10)
        for (const int base : std::array{1, 2, 5})
            if (span / (base * scale) <= maxLabels)
                return base * scale;
}

}

TimelineController::TimelineController(RmlKit::UI::Model& model, AppModel& app, Rml::Element& root, ViewportElementInstancer& viewports)
    : model(model), app(app), scrub(root, [this](const float fraction) { seek(fraction); }) {
    model.bindEvent("timelinePlay", &TimelineController::onPlay, this);
    model.bindEvent("timelineFirst", &TimelineController::onFirst, this);
    model.bindEvent("timelineLast", &TimelineController::onLast, this);
    model.bindEvent("timelinePrevious", &TimelineController::onPrevious, this);
    model.bindEvent("timelineNext", &TimelineController::onNext, this);
    model.bindEvent("timelineApply", &TimelineController::onApply, this);
    viewports.getEvents().tick = [this] { return tick(); };
    show();
}

void TimelineController::onPlay(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelinePlay");
    TimelineModel& timeline = app.timeline;
    timeline.playing = !timeline.playing;
    lastTick = std::chrono::steady_clock::now();
    pendingFrames = 0.0;
    if (timeline.playing && timeline.frame >= timeline.end)
        timeline.frame = timeline.start;
    show();
}

void TimelineController::onFirst(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelineFirst");
    setFrame(app.timeline.start);
}

void TimelineController::onLast(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelineLast");
    setFrame(app.timeline.end);
}

void TimelineController::onPrevious(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelinePrevious");
    setFrame(app.timeline.frame - 1);
}

void TimelineController::onNext(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelineNext");
    setFrame(app.timeline.frame + 1);
}

void TimelineController::onApply(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "timelineApply");
    show();
}

bool TimelineController::tick() {
    TimelineModel& timeline = app.timeline;
    if (!timeline.playing)
        return false;
    const auto now = std::chrono::steady_clock::now();
    pendingFrames += std::chrono::duration<double>(now - std::exchange(lastTick, now)).count() * timeline.fps;
    const auto frames = static_cast<int>(std::floor(pendingFrames));
    if (frames == 0)
        return true;
    pendingFrames -= frames;
    //! Playback loops over the range.
    const int length = timeline.end - timeline.start + 1;
    timeline.frame = timeline.start + ((timeline.frame - timeline.start + frames) % length + length) % length;
    show();
    return true;
}

void TimelineController::seek(const float fraction) {
    const TimelineModel& timeline = app.timeline;
    setFrame(timeline.start + static_cast<int>(std::lround(fraction * static_cast<float>(timeline.end - timeline.start))));
}

void TimelineController::setFrame(const int frame) {
    app.timeline.frame = frame;
    show();
}

void TimelineController::show() {
    TimelineModel& timeline = app.timeline;
    timeline.start = std::max(timeline.start, 0);
    timeline.end = std::max(timeline.end, timeline.start + 1);
    timeline.fps = std::clamp(timeline.fps, 1, 240);
    timeline.frame = std::clamp(timeline.frame, timeline.start, timeline.end);
    const int span = timeline.end - timeline.start;
    const auto place = [&](const int frame) { return 100.0f * static_cast<float>(frame - timeline.start) / static_cast<float>(span); };
    timeline.playhead = place(timeline.frame);
    const int major = labelStep(span);
    const int minor = major >= minorPerMajor ? major / minorPerMajor : 1;
    timeline.ticks.clear();
    for (int frame = (timeline.start + minor - 1) / minor * minor; frame <= timeline.end; frame += minor)
        timeline.ticks.push_back({frame, place(frame), frame % major == 0});
    model.dirty("timeline");
}

}
