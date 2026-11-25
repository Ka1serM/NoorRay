#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Optimizing {

using namespace NoorRHI;

struct AdamArguments {
    Ptr<float> parameters;
    Ptr<float> gradients;
    Ptr<float> first;
    Ptr<float> second;
    uint count;
    uint threads;
    float stepSize;
    float beta1;
    float beta2;
    float epsilon;
    //! One over the square root of the bias correction of the second moment.
    float secondCorrection;
};

struct GatherArguments {
    Ptr<float> source;
    Ptr<float> destination;
    Ptr<uint> indices;
    uint count;
    uint threads;
    uint stride;
};

}
