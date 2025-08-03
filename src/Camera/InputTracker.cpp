#include "InputTracker.h"
#include <cstring>

#include "SDL3/SDL_keyboard.h"
#include "SDL3/SDL_mouse.h"

InputTracker::InputTracker(SDL_Window* window) : window(window), deltaX(0.0f), deltaY(0.0f) {}

void InputTracker::update() {
    // Copy the current key states to the previous state buffer
    memcpy(prevKeyStates, keyStates, sizeof(keyStates));
    
    // FIX: Use the correct return type, const Uint8*
    const bool* currentKeyStates = SDL_GetKeyboardState(nullptr);
    for (int i = 0; i < SDL_SCANCODE_COUNT; ++i)
        keyStates[i] = currentKeyStates[i];

    // Copy the current mouse button states to the previous state buffer
    memcpy(prevMouseButtonStates, mouseButtonStates, sizeof(mouseButtonStates));
    
    const Uint32 currentButtonMask = SDL_GetMouseState(nullptr, nullptr);
    for (int i = 1; i < 16; ++i) // Button indexes are 1-based
        mouseButtonStates[i] = (currentButtonMask & SDL_BUTTON_MASK(i));

    // Get the relative mouse motion since the last poll
    SDL_GetRelativeMouseState(&deltaX, &deltaY);
}

bool InputTracker::isKeyPressed(SDL_Scancode scancode) const {
    return keyStates[scancode] && !prevKeyStates[scancode];
}

bool InputTracker::isKeyHeld(SDL_Scancode scancode) const {
    return keyStates[scancode];
}

bool InputTracker::isKeyReleased(SDL_Scancode scancode) const {
    return !keyStates[scancode] && prevKeyStates[scancode];
}

bool InputTracker::isMouseButtonPressed(Uint8 button) const {
    if (button >= 16) return false;
    return mouseButtonStates[button] && !prevMouseButtonStates[button];
}

bool InputTracker::isMouseButtonHeld(Uint8 button) const {
    if (button >= 16) return false;
    return mouseButtonStates[button];
}

bool InputTracker::isMouseButtonReleased(Uint8 button) const {
    if (button >= 16) return false;
    return !mouseButtonStates[button] && prevMouseButtonStates[button];
}

void InputTracker::getMouseDelta(float& outDeltaX, float& outDeltaY) const {
    outDeltaX = deltaX;
    outDeltaY = deltaY;
}
