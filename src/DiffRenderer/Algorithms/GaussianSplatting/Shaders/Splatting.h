#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Splatting {

using namespace NoorRHI;

struct Parameters {
    Ptr<float> means;
    Ptr<float> quaternions;
    Ptr<float> scales;
    Ptr<float> opacities;
    Ptr<float> colors;
};

struct Arguments {
    float4 view[3];
    float4 intrinsics;
    float4 background;
    Parameters parameters;
    Parameters gradients;
    Ptr<float4> projected;
    Ptr<uint> order;
    Ptr<float> image;
    Ptr<float> transmittance;
    Ptr<uint> contributors;
    Ptr<float> imageGradient;
    AtomicPtr<float> projectedGradient;
    uint count;
    uint width;
    uint height;
    float nearPlane;
    float farPlane;
    float blur;
};

}
