#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

namespace NoorRay::App {

//! A ruler mark: the frame, its place on the ruler in percent, and whether it carries a label.
struct TimelineTick {
    int frame = 0;
    float position = 0.0f;
    bool major = false;
};

//! The frame range and playback of the timeline. Nothing in the scene is animated yet, so the frame only moves the playhead.
struct TimelineModel {
    int frame = 1;
    int start = 1;
    int end = 250;
    int fps = 24;
    bool playing = false;
    //! The playhead on the ruler in percent.
    float playhead = 0.0f;
    std::vector<TimelineTick> ticks;
};

void bind(Rml::DataModelConstructor& model, TimelineModel& state);

}
