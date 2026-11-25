#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>

namespace NoorRay::App {

//! Number fields: a drag on a .num changes it, a click edits it, .num-up, .num-down and a .slider set it, by step (0.1), speed (step / 10), min, max, integer.
class NumberScrub final : public Rml::EventListener {
public:
    explicit NumberScrub(Rml::Element& root);
    ~NumberScrub() override;
    NumberScrub(const NumberScrub&) = delete;
    NumberScrub& operator=(const NumberScrub&) = delete;

    void ProcessEvent(Rml::Event& event) override;

private:
    void press(Rml::Event& event, Rml::Element& field);
    void move(Rml::Event& event);
    void release();
    void slide(float x);
    void step(Rml::Event& event, Rml::Element& field, float direction);
    void commit(Rml::Element& field, float value);

    Rml::Element& root;
    Rml::Element* pressed = nullptr;
    Rml::Element* slider = nullptr;
    float startX = 0.0f;
    float startY = 0.0f;
    float lastX = 0.0f;
    float value = 0.0f;
    bool scrubbing = false;
};

}
