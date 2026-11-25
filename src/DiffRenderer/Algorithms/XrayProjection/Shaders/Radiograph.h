#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Radiograph {

using namespace NoorRHI;

struct Arguments {
    Ptr<float> density;
    Ptr<float> image;
    Ptr<float> imageGradients;
    AtomicPtr<float> poseGradients;
    float4 angles;
    float4 translation;
    float sourceToDetector;
    float pixelSpacing;
    float alphaMin;
    float alphaMax;
    uint pointCount;
    uint volumeSize;
    uint width;
    uint height;
};

}
