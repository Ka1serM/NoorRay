#include "NoorRay.App/UI/Controls/HoldMenu.h"

#include <cstdint>
#include <utility>

#include <RmlUi/Core/Context.h>

#include "RmlKit/UI/Backend/Fatal.h"

namespace NoorRay::App {

namespace {

constexpr Uint32 holdMilliseconds = 300;

Rml::Element* holdTarget(Rml::Element* element) {
    for (; element; element = element->GetParentNode())
        if (element->HasAttribute("hold-menu"))
            return element;
    return nullptr;
}

Rml::Element* flyoutItem(Rml::Element* element) {
    for (; element; element = element->GetParentNode())
        if (element->IsClassSet("flyout-item"))
            return element;
    return nullptr;
}

}

HoldMenu::HoldMenu(Rml::Element& root, std::function<void(const Rml::String&)> open) : root(root), open(std::move(open)) {
    holdEvent = SDL_RegisterEvents(1);
    if (holdEvent == 0)
        RmlKit::UI::fatal("SDL_RegisterEvents failed for the hold menu");
    root.AddEventListener(Rml::EventId::Mousedown, this);
    root.AddEventListener(Rml::EventId::Mouseup, this);
    root.AddEventListener(Rml::EventId::Click, this, true);
}

HoldMenu::~HoldMenu() {
    cancelTimer();
    root.RemoveEventListener(Rml::EventId::Mousedown, this);
    root.RemoveEventListener(Rml::EventId::Mouseup, this);
    root.RemoveEventListener(Rml::EventId::Click, this, true);
}

void HoldMenu::cancelTimer() {
    if (timer)
        SDL_RemoveTimer(std::exchange(timer, 0));
}

Uint32 SDLCALL HoldMenu::expire(void* menu, SDL_TimerID, Uint32) {
    const HoldMenu& self = *static_cast<HoldMenu*>(menu);
    SDL_Event event{};
    event.type = self.holdEvent;
    event.user.code = static_cast<Sint32>(self.press);
    SDL_PushEvent(&event);
    return 0;
}

bool HoldMenu::handle(const SDL_Event& event) {
    if (event.type != holdEvent)
        return false;
    timer = 0;
    if (!pressed || static_cast<std::uint32_t>(event.user.code) != press)
        return true;
    opened = true;
    open(pressed->GetAttribute<Rml::String>("hold-menu", ""));
    if (Rml::Context* context = root.GetContext())
        context->RequestNextUpdate(0);
    return true;
}

void HoldMenu::ProcessEvent(Rml::Event& event) {
    const bool left = event.GetParameter<int>("button", 0) == 0;
    switch (event.GetId()) {
    case Rml::EventId::Mousedown:
        if (!left)
            return;
        cancelTimer();
        pressed = holdTarget(event.GetTargetElement());
        opened = false;
        if (!pressed)
            return;
        ++press;
        timer = SDL_AddTimer(holdMilliseconds, expire, this);
        return;
    case Rml::EventId::Mouseup: {
        if (!left)
            return;
        cancelTimer();
        swallowClick = std::exchange(opened, false);
        pressed = nullptr;
        if (!swallowClick)
            return;
        //! A release over an item of the open menu picks it, as a press, drag and release does in Cinema 4D.
        Rml::Context* context = root.GetContext();
        if (Rml::Element* item = flyoutItem(context ? context->GetHoverElement() : nullptr))
            item->Click();
        return;
    }
    case Rml::EventId::Click:
        if (std::exchange(swallowClick, false) && holdTarget(event.GetTargetElement()))
            event.StopPropagation();
        return;
    default:
        return;
    }
}

}
