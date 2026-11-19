#pragma once

#include <chrono>

#include <RmlUi/Core/Types.h>

#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

//! Shows the frame, sample and scene statistics in the status bar and the statistics tab after every rendered frame.
class StatusController {
public:
    StatusController(RmlKit::UI::Model& model, AppModel& app, const Session::Session& session, ViewportElementInstancer& viewports);
    StatusController(const StatusController&) = delete;
    StatusController& operator=(const StatusController&) = delete;

private:
    void frame();
    void fail(const Rml::String& message);

    RmlKit::UI::Model& model;
    AppModel& app;
    const Session::Session& session;
    std::chrono::steady_clock::time_point lastFrame;
    std::chrono::steady_clock::time_point lastPublish;
    std::chrono::steady_clock::time_point lastTiming;
    double smoothedSeconds = 0.0;
};

}
