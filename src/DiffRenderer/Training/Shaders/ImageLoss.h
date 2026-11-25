#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace ImageLoss {

using namespace NoorRHI;

struct Arguments {
    Ptr<float> image;
    Ptr<uint> target;
    Ptr<float> gradient;
    //! The horizontal filter sums, one plane of width x height for each sum and channel.
    Ptr<float> rows;
    //! dLoss/dMean, dLoss/dSquares and dLoss/dProducts of every pixel, one plane for each and channel.
    Ptr<float> partials;
    Ptr<float> groupLosses;
    Ptr<float> value;
    uint width;
    uint height;
    uint groupCount;
    //! One over the number of values, width x height x 3.
    float scale;
};

}
