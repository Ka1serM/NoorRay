#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Instances {

using namespace NoorRHI;

//! The raw parameters of 3DGS: unnormalized quaternions (w, x, y, z), logarithms of the scales and logits of the opacities. GaussianProxies sets the rest.
struct Arguments {
    Ptr<float> means;
    Ptr<float> quaternions;
    Ptr<float> logScales;
    Ptr<float> opacityLogits;
    Ptr<uint> records;
    AccelerationStructureHandle proxy;
    uint count;
    uint threads;
};

}
