#pragma once

#include <functional>

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>

namespace NoorRay::App {

//! Press and drag on a .timeline-ruler element: gives the pointer's place along the ruler, 0 at its left edge and 1 at its right.
class TimelineScrub final : public Rml::EventListener {
public:
    using Moved = std::function<void(float fraction)>;

    TimelineScrub(Rml::Element& root, Moved moved);
    ~TimelineScrub() override;
    TimelineScrub(const TimelineScrub&) = delete;
    TimelineScrub& operator=(const TimelineScrub&) = delete;

    void ProcessEvent(Rml::Event& event) override;

private:
    void seek(const Rml::Event& event) const;

    Rml::Element& root;
    Moved moved;
    Rml::Element* ruler = nullptr;
};

}
