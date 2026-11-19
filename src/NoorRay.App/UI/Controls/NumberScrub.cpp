#include "NoorRay.App/UI/Controls/NumberScrub.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <limits>
#include <utility>

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>

namespace NoorRay::App {

namespace {

constexpr float clickSlop = 3.0f;
constexpr Rml::EventId events[] = {Rml::EventId::Mousedown, Rml::EventId::Mousemove, Rml::EventId::Mouseup, Rml::EventId::Blur};

Rml::Element* findClass(Rml::Element* element, const char* name) {
    for (; element; element = element->GetParentNode())
        if (element->IsClassSet(name))
            return element;
    return nullptr;
}

Rml::Element* inputOf(Rml::Element& field) {
    return field.QuerySelector("input");
}

float attribute(const Rml::Element& field, const char* name, const float fallback) {
    return field.GetAttribute<float>(name, fallback);
}

float parse(const Rml::String& text) {
    const std::size_t first = text.find_first_not_of(' ');
    if (first == Rml::String::npos)
        return 0.0f;
    const char* begin = text.data() + first + (text[first] == '+' ? 1 : 0);
    float parsed = 0.0f;
    if (const auto [end, error] = std::from_chars(begin, text.data() + text.size(), parsed); error != std::errc{} || !std::isfinite(parsed))
        return 0.0f;
    return parsed;
}

Rml::String format(const float value, const bool integer) {
    if (integer)
        return std::format("{}", std::lround(value));
    Rml::String text = std::format("{:.3f}", value);
    const std::size_t dot = text.find('.');
    const std::size_t last = text.find_last_not_of('0');
    text.erase(last == dot ? dot : last + 1);
    return text == "-0" ? "0" : text;
}

float modifier(const Rml::Event& event) {
    if (event.GetParameter<int>("shift_key", 0) != 0)
        return 10.0f;
    if (event.GetParameter<int>("ctrl_key", 0) != 0)
        return 0.1f;
    return 1.0f;
}

}

NumberScrub::NumberScrub(Rml::Element& root) : root(root) {
    for (const Rml::EventId id : events)
        root.AddEventListener(id, this, true);
}

NumberScrub::~NumberScrub() {
    for (const Rml::EventId id : events)
        root.RemoveEventListener(id, this, true);
}

void NumberScrub::ProcessEvent(Rml::Event& event) {
    switch (event.GetId()) {
    case Rml::EventId::Mousedown: {
        if (event.GetParameter<int>("button", 0) != 0)
            return;
        Rml::Element* target = event.GetTargetElement();
        if (Rml::Element* track = findClass(target, "slider"); track && track->GetParentNode()) {
            Rml::Element* field = track->GetParentNode()->QuerySelector(".num");
            if (!field)
                return;
            pressed = field;
            slider = track;
            scrubbing = true;
            return slide(event.GetParameter<float>("mouse_x", 0.0f));
        }
        Rml::Element* field = findClass(target, "num");
        if (!field || field->IsClassSet("editing"))
            return;
        if (findClass(target, "num-up"))
            return step(event, *field, 1.0f);
        if (findClass(target, "num-down"))
            return step(event, *field, -1.0f);
        return press(event, *field);
    }
    case Rml::EventId::Mousemove:
        return move(event);
    case Rml::EventId::Mouseup:
        return release();
    case Rml::EventId::Blur:
        if (Rml::Element* field = findClass(event.GetTargetElement(), "num"))
            field->SetClass("editing", false);
        return;
    default:
        return;
    }
}

void NumberScrub::press(Rml::Event& event, Rml::Element& field) {
    Rml::Element* input = inputOf(field);
    if (!input)
        return;
    pressed = &field;
    scrubbing = false;
    startX = event.GetParameter<float>("mouse_x", 0.0f);
    startY = lastY = event.GetParameter<float>("mouse_y", 0.0f);
    value = parse(input->GetAttribute<Rml::String>("value", ""));
}

void NumberScrub::move(Rml::Event& event) {
    if (!pressed)
        return;
    const float x = event.GetParameter<float>("mouse_x", 0.0f);
    const float y = event.GetParameter<float>("mouse_y", 0.0f);
    if (slider)
        return slide(x);
    if (!scrubbing && std::abs(x - startX) <= clickSlop && std::abs(y - startY) <= clickSlop)
        return;
    if (!scrubbing) {
        scrubbing = true;
        pressed->SetClass("scrubbing", true);
        lastY = startY;
    }
    const float density = root.GetContext() ? root.GetContext()->GetDensityIndependentPixelRatio() : 1.0f;
    const float stepSize = attribute(*pressed, "step", 0.1f);
    const float speed = attribute(*pressed, "speed", stepSize * 0.1f);
    //! Up raises the value, as in Cinema 4D.
    value += (lastY - y) / density * speed * modifier(event);
    lastY = y;
    commit(*pressed, value);
}

void NumberScrub::release() {
    Rml::Element* field = std::exchange(pressed, nullptr);
    slider = nullptr;
    if (!field)
        return;
    field->SetClass("scrubbing", false);
    if (std::exchange(scrubbing, false))
        return;
    Rml::Element* input = inputOf(*field);
    if (!input)
        return;
    field->SetClass("editing", true);
    input->Focus();
    if (auto* text = rmlui_dynamic_cast<Rml::ElementFormControlInput*>(input))
        text->Select();
}

void NumberScrub::slide(const float x) {
    const float left = slider->GetAbsoluteOffset(Rml::BoxArea::Border).x;
    const float width = slider->GetBox().GetSize(Rml::BoxArea::Border).x;
    const float low = attribute(*slider, "min", 0.0f);
    const float high = attribute(*slider, "max", 1.0f);
    const float t = width > 0.0f ? std::clamp((x - left) / width, 0.0f, 1.0f) : 0.0f;
    commit(*pressed, low + t * (high - low));
}

void NumberScrub::step(Rml::Event& event, Rml::Element& field, const float direction) {
    Rml::Element* input = inputOf(field);
    if (!input)
        return;
    const float current = parse(input->GetAttribute<Rml::String>("value", ""));
    commit(field, current + direction * attribute(field, "step", 0.1f) * modifier(event));
    event.StopPropagation();
}

void NumberScrub::commit(Rml::Element& field, const float next) {
    Rml::Element* input = inputOf(field);
    if (!input)
        return;
    const float low = attribute(field, "min", std::numeric_limits<float>::lowest());
    const float high = attribute(field, "max", std::numeric_limits<float>::max());
    const bool integer = field.HasAttribute("integer");
    const Rml::String text = format(std::clamp(next, low, high), integer);
    if (text == input->GetAttribute<Rml::String>("value", ""))
        return;
    input->SetAttribute("value", text);
    input->DispatchEvent(Rml::EventId::Change, {{"value", Rml::Variant(text)}, {"linebreak", Rml::Variant(true)}});
}

}
