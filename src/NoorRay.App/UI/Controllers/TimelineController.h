#pragma once

#include <chrono>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Controls/TimelineScrub.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

//! The timeline under the viewport: the current frame, the frame range, scrubbing on the ruler, and playback at the frame rate.
class TimelineController {
public:
    TimelineController(RmlKit::UI::Model& model, AppModel& app, Rml::Element& root, ViewportElementInstancer& viewports);
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
    //! Clamps the range and the frame, then rebuilds the ruler and the playhead.
    void show();

    RmlKit::UI::Model& model;
    AppModel& app;
    TimelineScrub scrub;
    std::chrono::steady_clock::time_point lastTick;
    double pendingFrames = 0.0;
};

}
