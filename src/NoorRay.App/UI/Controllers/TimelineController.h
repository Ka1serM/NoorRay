#pragma once

#include <chrono>
#include <optional>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Controls/TimelineScrub.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::Session {
class Session;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

//! The timeline under the viewport: the frame and its range, scrubbing on the ruler, and playback; the frame drives the session animation and camera.
class TimelineController {
public:
    TimelineController(RmlKit::UI::Model& model, AppModel& app, Rml::Element& root, Session::Session& session, ViewportElementInstancer& viewports);
    TimelineController(const TimelineController&) = delete;
    TimelineController& operator=(const TimelineController&) = delete;

private:
    void onPlay(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onFirst(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onLast(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onPrevious(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onNext(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onApply(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    //! Called on every viewport update; true while playing, so the viewport keeps updating.
    bool tick();
    void seek(float fraction);
    void setFrame(int frame);
    //! Takes the frame range and rate of an opened animation.
    void opened();
    //! Shows the current frame of the animation in the scene and the view.
    void applyFrame();
    //! Clamps the range and the frame, then rebuilds the ruler and the playhead.
    void show();
    //! Reads the ruler width and marks a narrow transport; true when the width changed.
    bool measure();

    RmlKit::UI::Model& model;
    AppModel& app;
    Rml::Element& root;
    Session::Session& session;
    ViewportElementInstancer& viewports;
    TimelineScrub scrub;
    //! The frame the scene shows, so a frame is applied once.
    std::optional<int> appliedFrame;
    std::chrono::steady_clock::time_point lastTick;
    double pendingFrames = 0.0;
    Rml::Element* ruler = nullptr;
    //! The ruler width in dp that the ticks were made for.
    float rulerWidth = 480.0f;
};

}
