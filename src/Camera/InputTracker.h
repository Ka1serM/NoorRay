#pragma once

#include "SDL3/SDL_scancode.h"
#include "SDL3/SDL_video.h"

union SDL_Event;

class InputTracker {
public:
    InputTracker(SDL_Window* window);
    
    void update();
    
    bool isKeyPressed(SDL_Scancode scancode) const;
    bool isKeyHeld(SDL_Scancode scancode) const;
    bool isKeyReleased(SDL_Scancode scancode) const;

    bool isMouseButtonPressed(Uint8 button) const;
    bool isMouseButtonHeld(Uint8 button) const;
    bool isMouseButtonReleased(Uint8 button) const;

    void getMouseDelta(float& outDeltaX, float& outDeltaY) const;

private:
    SDL_Window* window;

    Uint8 keyStates[SDL_SCANCODE_COUNT]{};
    Uint8 prevKeyStates[SDL_SCANCODE_COUNT]{};

    bool mouseButtonStates[16]{};
    bool prevMouseButtonStates[16]{};
    
    float deltaX;
    float deltaY;
};