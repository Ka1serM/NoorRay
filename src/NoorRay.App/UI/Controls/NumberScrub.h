#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>

namespace NoorRay::App {

//! ImGui-style number fields: an up or down drag on a .num changes the value, a click edits the text, .num-up and .num-down step it, and a .slider beside a .num sets it from its min to its max; the attributes step (default 0.1), speed (change per pixel, default step / 10), min, max and integer set the behaviour; Shift is ten times faster and Ctrl ten times slower.
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
    float lastY = 0.0f;
    float value = 0.0f;
    bool scrubbing = false;
};

}
