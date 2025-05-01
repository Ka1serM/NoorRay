#pragma once
#include "shaders/SharedStructs.h"

struct PushConstants {
    PushData push;
    CameraData camera;

    PushConstants(int frame, int isPathtracing, const CameraData& camera) : camera(camera) {
        push.frame = frame;
        push.isPathtracing = isPathtracing;
    }
};