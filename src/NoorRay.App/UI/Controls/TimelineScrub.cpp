#include "NoorRay.App/UI/Controls/TimelineScrub.h"

#include <algorithm>
#include <utility>

namespace NoorRay::App {

namespace {

constexpr Rml::EventId events[] = {Rml::EventId::Mousedown, Rml::EventId::Mousemove, Rml::EventId::Mouseup};

Rml::Element* findRuler(Rml::Element* element) {
    for (; element; element = element->GetParentNode())
        if (element->IsClassSet("timeline-ruler"))
            return element;
    return nullptr;
}

}

TimelineScrub::TimelineScrub(Rml::Element& root, Moved moved) : root(root), moved(std::move(moved)) {
    for (const Rml::EventId id : events)
        root.AddEventListener(id, this, true);
}

TimelineScrub::~TimelineScrub() {
    for (const Rml::EventId id : events)
        root.RemoveEventListener(id, this, true);
}

void TimelineScrub::ProcessEvent(Rml::Event& event) {
    switch (event.GetId()) {
    case Rml::EventId::Mousedown:
        if (event.GetParameter<int>("button", 0) != 0 || !(ruler = findRuler(event.GetTargetElement())))
            return;
        seek(event);
        event.StopPropagation();
        return;
    case Rml::EventId::Mousemove:
        if (ruler)
            seek(event);
        return;
    default:
        ruler = nullptr;
        return;
    }
}

void TimelineScrub::seek(const Rml::Event& event) const {
    const float left = ruler->GetAbsoluteLeft() + ruler->GetClientLeft();
    const float width = ruler->GetClientWidth();
    if (width <= 0.0f)
        return;
    moved(std::clamp((event.GetParameter<float>("mouse_x", 0.0f) - left) / width, 0.0f, 1.0f));
}

}
