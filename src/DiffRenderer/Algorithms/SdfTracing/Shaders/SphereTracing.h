#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace SphereTracing {

using namespace NoorRHI;

static const uint albedoOffset = 0;
static const uint translationOffset = 3;
static const uint gridOffset = 6;

struct Arguments {
    float4 cameraOrigin;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    float4 light;
    Ptr<float> parameters;
    Ptr<float> weights;
    Ptr<float> image;
    AtomicPtr<float> gradients;
    uint width;
    uint height;
    uint resolution;
    uint samples;
    uint seed;
    uint linesPerPixel;
    uint pitch;
    uint padding;
};

}
