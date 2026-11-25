#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace RayMarching {

using namespace NoorRHI;

static const uint viewCount = 4;

struct Arguments {
    Ptr<float> density;
    Ptr<float> tumor;
    Ptr<float4> transferFunction;
    AtomicPtr<float> transferFunctionGradients;
    Ptr<float4> image;
    Ptr<float4> imageGradients;
    float4 eyes[viewCount];
    float4 background;
    float tanHalfFov;
    float stepSize;
    float extinctionScale;
    uint transferFunctionSize;
    uint volumeSize;
    uint width;
    uint height;
};

}
