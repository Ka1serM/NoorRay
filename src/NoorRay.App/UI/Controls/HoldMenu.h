#pragma once

#include <cstdint>
#include <functional>

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

namespace NoorRay::App {

//! Cinema 4D's tool groups: a press held on an element with hold-menu="key" opens that menu, a quick click keeps the element's own click, and a release over a .flyout-item of the open menu clicks it.
class HoldMenu final : public Rml::EventListener {
public:
    HoldMenu(Rml::Element& root, std::function<void(const Rml::String&)> open);
    ~HoldMenu() override;
    HoldMenu(const HoldMenu&) = delete;
    HoldMenu& operator=(const HoldMenu&) = delete;

    void ProcessEvent(Rml::Event& event) override;
    //! Takes the timer event that ends a hold.
    bool handle(const SDL_Event& event);

private:
    static Uint32 SDLCALL expire(void* menu, SDL_TimerID timer, Uint32 interval);
    void cancelTimer();

    Rml::Element& root;
    std::function<void(const Rml::String&)> open;
    Uint32 holdEvent = 0;
    SDL_TimerID timer = 0;
    std::uint32_t press = 0;
    Rml::Element* pressed = nullptr;
    bool opened = false;
    bool swallowClick = false;
};

}
