#include "NoorRay.App/UI/Models/TimelineModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, TimelineModel& state) {
    if (auto handle = model.RegisterStruct<TimelineTick>()) {
        handle.RegisterMember("frame", &TimelineTick::frame);
        handle.RegisterMember("position", &TimelineTick::position);
        handle.RegisterMember("major", &TimelineTick::major);
    }
    model.RegisterArray<std::vector<TimelineTick>>();
    auto timeline = model.RegisterStruct<TimelineModel>();
    timeline.RegisterMember("frame", &TimelineModel::frame);
    timeline.RegisterMember("start", &TimelineModel::start);
    timeline.RegisterMember("end", &TimelineModel::end);
    timeline.RegisterMember("fps", &TimelineModel::fps);
    timeline.RegisterMember("playing", &TimelineModel::playing);
    timeline.RegisterMember("playhead", &TimelineModel::playhead);
    timeline.RegisterMember("ticks", &TimelineModel::ticks);
    model.Bind("timeline", &state);
}

}
